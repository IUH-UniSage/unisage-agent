"""Validates that a list of already-built `Chunk`s has internally
consistent fields.

This is NOT the place that verifies "was a table row actually cut
correctly" - that is `TableRowChunker`'s own responsibility (self-check +
exhaustive tests, see `app/rag/chunking/table_row.py`), because it is the
only place left holding the table's raw `list[list[str]]`. Once a `Chunk`
is built, only its own summary fields survive (`row_start`/`row_end`/
`row_count`, `column_names`, ...) - this validator's job is to catch
round-trip/mapping bugs (DB, Qdrant, legacy-vs-new `chunking_version`
mixing, a client tampering with fields before `POST /ingestion/embedding`),
not to re-derive table structure truth from `content` text.

Rules (a)-(m) below; (k)/(l) are the inverse checks of (f)/(g)/(d).
"""

from app.core.errors.exceptions import ChunkValidationException
from app.schemas.ingestion import Chunk, HeaderSource, RegionType, SourceType

_TABLE_REGION_TYPES = (RegionType.TABLE, RegionType.EXCEL_ROW)


def validate_chunks(chunks: list[Chunk]) -> None:
    """Raise `ChunkValidationException` (with one entry per offending chunk,
    keyed by `str(chunk_index)`) if any chunk violates rules (a)-(l).
    Collects ALL failing chunks/reasons before raising - callers get a full
    picture in one round trip, not just the first failure.
    """

    errors: dict[str, str] = {}
    for chunk in chunks:
        reasons = _validate_one(chunk)
        if reasons:
            errors[str(chunk.chunk_index)] = "; ".join(reasons)

    if errors:
        raise ChunkValidationException(errors)


def _validate_one(chunk: Chunk) -> list[str]:
    reasons: list[str] = []

    # (a) heading_path is always a list (Pydantic already enforces the
    # type at construction, but a defensive check costs nothing and guards
    # against a future relaxation of the field's type).
    if not isinstance(chunk.heading_path, list):
        reasons.append("heading_path phải là list")

    # (e) every chunk that went through dispatch() must carry these.
    if chunk.source_type is None:
        reasons.append("source_type không được None")
    if chunk.block_index is None or chunk.block_index < 0:
        reasons.append("block_index phải là số nguyên >= 0, không được None")

    # (c) PDF: page_start/page_end mandatory for EVERY region_type, not just TABLE.
    if chunk.source_type == SourceType.PDF:
        if chunk.page_start is None or chunk.page_end is None:
            reasons.append("chunk nguồn PDF phải có page_start/page_end")
        elif chunk.page_end < chunk.page_start:
            reasons.append("page_end phải >= page_start")

    locator = chunk.source_locator

    # (b) [v5: mandatory, not "if present"] TABLE/EXCEL_ROW chunks must
    # always carry row_start/row_end/row_count, all non-None and consistent.
    if chunk.region_type in _TABLE_REGION_TYPES:
        if locator is None:
            reasons.append("chunk bảng/hàng phải có source_locator")
        else:
            if locator.row_start is None or locator.row_end is None or locator.row_count is None:
                reasons.append("chunk bảng/hàng phải có row_start/row_end/row_count đều khác None")
            else:
                if locator.row_start > locator.row_end:
                    reasons.append("row_start phải <= row_end")
                if locator.row_end - locator.row_start + 1 != locator.row_count:
                    reasons.append("row_count không khớp row_end - row_start + 1")
        # (j) chunk bảng luôn phải định vị được thuộc bảng nào.
        if locator is None or locator.table_id is None:
            reasons.append("chunk bảng/hàng phải có source_locator.table_id")

    # (d)/(i)/(l) is_partial_row and its inverse.
    if locator is not None:
        if locator.is_partial_row:
            row_part = locator.row_part
            row_part_count = locator.row_part_count
            if row_part is None or row_part_count is None or locator.table_id is None:
                reasons.append(
                    "is_partial_row=True phải có row_part/row_part_count/table_id đều khác None"
                )
            else:
                if not (1 <= row_part <= row_part_count):
                    reasons.append("row_part phải nằm trong khoảng [1, row_part_count]")
                # (l) [v5] a row split into just "1 of 1" is a meaningless fallback.
                if row_part_count < 2:
                    reasons.append("is_partial_row=True nhưng row_part_count < 2")
        else:
            # (i) inverse of (d): a normal row must not carry partial-row markers.
            if locator.row_part is not None or locator.row_part_count is not None:
                reasons.append("is_partial_row=False nhưng row_part/row_part_count không phải None")

    # (f)/(h) header_source == MISSING implications.
    if chunk.header_source == HeaderSource.MISSING:
        if chunk.has_header is not False:
            reasons.append("header_source=MISSING nhưng has_header không phải False")
        if chunk.column_names is not None:
            reasons.append("header_source=MISSING nhưng column_names không phải None")
        if chunk.header_confidence != 0.0:
            reasons.append("header_source=MISSING nhưng header_confidence khác 0.0")
    else:
        # (k) [v5] inverse of (f): a determined header source must mean has_header=True.
        if chunk.has_header is not True:
            reasons.append(
                "header_source khác MISSING (explicit/inferred) nhưng has_header không phải True"
            )

    # (g) has_header=False can never carry column names, regardless of header_source.
    if chunk.has_header is False and chunk.column_names is not None:
        reasons.append("has_header=False nhưng column_names không phải None")

    # (b) has_header=True implies non-empty column_names (self-declared consistency,
    # not a re-count of real cells).
    if chunk.has_header is True and not chunk.column_names:
        reasons.append("has_header=True nhưng column_names rỗng")

    return reasons
