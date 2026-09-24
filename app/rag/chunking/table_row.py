import hashlib
import math
from dataclasses import dataclass

import tiktoken

from app.core.config import settings
from app.core.exceptions import ChunkingConfigException
from app.rag.ingestion.canonical_table import TableBlock, TableRow
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk, RegionType, SourceLocator

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Row content beyond `usable_budget` but within this multiple of it is kept
# intact as its own (over-budget) chunk rather than being split - only a row
# that exceeds even this hard ceiling gets the "Phần k/n" fallback split.
_HARD_CAP_FACTOR = 2

# Reserved token budget for the "(Phần k/n)" marker line prepended to an
# oversized-row fallback chunk - a generous fixed constant (real marker
# text is a handful of tokens) rather than an exact per-chunk calculation,
# to keep the fallback's own budget math simple and conservative.
_PARTIAL_ROW_MARKER_RESERVE_TOKENS = 20

# A row's breadcrumb may take at most this share of the usable chunk budget;
# a longer ancestor path is shortened from its far (root) end.
_BREADCRUMB_MAX_SHARE = 0.4

# Line opening a group of rows that have no known ancestors right after a
# group that did - keeps them from reading as children of the previous group.
_NO_ANCESTORS_MARK = "[-]"


class TableStructureError(RuntimeError):
    """Raised by `TableRowChunker`'s own internal self-check when a chunk it
    just built does not faithfully represent the table row(s) it came from
    (e.g. a row's cell count doesn't match the table's header). This is the
    ONLY place left holding the table's raw `list[list[str]]` - the
    self-check here is what actually catches "a row got cut/misaligned",
    not `validate_chunks` downstream (Phase 4), which only checks summary
    fields for internal consistency, not real cell content.

    This is an internal INVARIANT failure (a bug in the chunker itself),
    not a user input/config problem - unlike `ChunkingConfigException`.
    Deliberately a plain `RuntimeError` subclass, not `assert`: `assert` is
    stripped entirely under `python -O`/`PYTHONOPTIMIZE`, which would make
    this "only" defense silently vanish in a misconfigured production/CI
    run.

    Security note (do NOT log `.raw_row` in production): `str(self)` and
    `.args` deliberately carry only a SHA-256 digest and per-cell lengths of
    the offending row, never any of its actual text (not even a short
    truncated prefix - a short cell's whole content could fit inside one),
    because a row may contain sensitive/PII data pulled straight from an
    uploaded document. `.raw_row` (the full, unredacted row) is exposed as
    a property for tests/debug tooling only - callers must gate access to
    it behind an explicit debug/test flag, never pass it to a production
    log statement or exception response body.
    """

    def __init__(
        self,
        *,
        table_id: str | None,
        block_index: int | None,
        row_index: int,
        expected_cell_count: int,
        actual_cell_count: int,
        row: list[str],
    ) -> None:
        self.table_id = table_id
        self.block_index = block_index
        self.row_index = row_index
        self.expected_cell_count = expected_cell_count
        self.actual_cell_count = actual_cell_count
        self._raw_row = list(row)

        # Deliberately NOT a text preview of the row: even a short truncated
        # prefix can reveal the entire cell content for a short cell (e.g. a
        # name or an ID), which may be sensitive/PII pulled straight from an
        # uploaded document. The digest lets the SAME row be recognized
        # across log lines/retries without revealing what it says; only
        # `.raw_row` (test/debug-only, see property docstring) exposes the
        # actual text.
        row_text = " | ".join(row)
        self.row_sample_digest = hashlib.sha256(row_text.encode("utf-8")).hexdigest()[:12]

        message = (
            "Table structure invariant violated: "
            f"table_id={table_id} block_index={block_index} row_index={row_index} "
            f"expected_cell_count={expected_cell_count} actual_cell_count={actual_cell_count} "
            f"row_sample_digest={self.row_sample_digest} row_cell_lengths={[len(c) for c in row]}"
        )
        super().__init__(message)

    @property
    def raw_row(self) -> list[str]:
        """Full, unredacted row content. TEST/DEBUG USE ONLY - never pass
        this to a production log statement or an API error response; see
        the class docstring's security note."""

        return list(self._raw_row)


def _escape_cell(value: str) -> str:
    """Escape/normalize one cell's text for markdown pipe-table rendering.

    `|` must be escaped (`\\|`) or it silently creates a phantom extra
    column, corrupting every cell's alignment after it - the same class of
    "row cut wrong" bug this plan sets out to fix, just happening at render
    time instead of at split time. Newlines are normalized to `<br>`
    (pipe-table rows cannot contain a live newline without breaking the
    table into an extra row).
    """

    return (
        value.replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .replace("\r", "<br>")
    )


def _render_row(cells: list[str]) -> str:
    return "| " + " | ".join(_escape_cell(cell) for cell in cells) + " |"


def _render_header_block(header_row: list[str]) -> str:
    header_line = _render_row(header_row)
    separator_line = "| " + " | ".join("---" for _ in header_row) + " |"
    return f"{header_line}\n{separator_line}"


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass(frozen=True)
class TableRowChunker:
    """Chunks TABLE regions row-by-row directly from the canonical
    `TableBlock` rows (see `canonical_table`), each rendered with its own full
    ancestor breadcrumb,
    never re-parsing rendered markdown - a normal row is NEVER split across
    chunks; only a row that exceeds even the hard cap is split, and always
    with explicit `is_partial_row`/`row_part`/`row_part_count` markers (see
    `SourceLocator`), never silently.
    """

    max_tokens: int = 400

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ChunkingConfigException(
                f"max_tokens={self.max_tokens} must be positive for TableRowChunker."
            )

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for region in regions:
            if region.region_type != RegionType.TABLE:
                continue
            chunks.extend(self._split_region(region))
        return [
            chunk.model_copy(update={"chunk_index": index}) for index, chunk in enumerate(chunks)
        ]

    def _split_region(self, region: ParsedRegion) -> list[Chunk]:
        table = region.table
        if table is None:
            # Should not happen for a real TABLE region produced by
            # `table_aware_parser` (Phase 1/2 always attach a `TableBlock`,
            # even a MISSING-header one) - treat as an empty table rather
            # than crashing on a malformed/hand-built region.
            return []
        rows = table.canonical_rows()
        if not rows:
            return []

        table_id = table.table_id or (
            f"table-{region.block_index}" if region.block_index is not None else "table-0"
        )
        heading_prefix = " > ".join(region.heading_path)
        header_block = _render_header_block(table.header_row) if table.header_row else ""

        prefix_parts: list[str] = []
        if heading_prefix:
            prefix_parts.append(f"{heading_prefix}\n\n")
        if header_block:
            prefix_parts.append(f"{header_block}\n")
        prefix_str = "".join(prefix_parts)

        usable_budget = self.max_tokens - _token_len(prefix_str)
        if usable_budget <= 0:
            raise ChunkingConfigException(
                "heading_path/table header consumes "
                f"{_token_len(prefix_str)} tokens, leaving no room under "
                f"max_tokens={self.max_tokens}. Increase max_tokens or shorten heading_path."
            )
        hard_cap = usable_budget * _HARD_CAP_FACTOR
        breadcrumb_budget = max(1, int(usable_budget * _BREADCRUMB_MAX_SHARE))

        chunks: list[Chunk] = []
        pending: list[_Entry] = []

        def flush_pending() -> None:
            nonlocal pending
            if pending:
                chunks.append(self._build_chunk(region, table, prefix_str, pending, table_id))
                pending = []

        for row in rows:
            if table.header_row is not None and len(row.cells) != len(table.header_row):
                raise TableStructureError(
                    table_id=table_id,
                    block_index=region.block_index,
                    row_index=row.row_index,
                    expected_cell_count=len(table.header_row),
                    actual_cell_count=len(row.cells),
                    row=row.cells,
                )

            crumb, truncated = _breadcrumb_line(row.ancestors, breadcrumb_budget)
            entry = _Entry(row, crumb, truncated)
            alone_tokens = _token_len(_render_body([entry]))

            if alone_tokens > usable_budget:
                flush_pending()
                if alone_tokens <= hard_cap:
                    chunks.append(self._build_chunk(region, table, prefix_str, [entry], table_id))
                else:
                    chunks.extend(
                        self._split_oversized_row(
                            region, table, prefix_str, entry, usable_budget, table_id
                        )
                    )
                continue

            if pending and _token_len(_render_body([*pending, entry])) > usable_budget:
                flush_pending()
            pending.append(entry)

        flush_pending()
        return chunks

    def _chunk_quality(
        self, table: TableBlock, entries: list["_Entry"]
    ) -> tuple[float | None, list[str]]:
        """Lowest row confidence and the de-duplicated warnings of the rows in
        a chunk, plus the table's own warnings. Plain hand-built tables (no
        provenance) yield `(None, [])`."""

        if not table.rows:
            return None, []
        warnings: list[str] = []
        for entry in entries:
            for warning in entry.row.warnings:
                if warning not in warnings:
                    warnings.append(warning)
            if entry.truncated and "breadcrumb_truncated" not in warnings:
                warnings.append("breadcrumb_truncated")
        for warning in table.warnings:
            if warning not in warnings:
                warnings.append(warning)
        return min(entry.row.confidence for entry in entries), warnings

    def _build_chunk(
        self,
        region: ParsedRegion,
        table: TableBlock,
        prefix_str: str,
        entries: list["_Entry"],
        table_id: str,
    ) -> Chunk:
        content = prefix_str + _render_body(entries)
        first, last = entries[0].row, entries[-1].row
        locator = SourceLocator(
            section=heading_section(region.heading_path),
            table_id=table_id,
            row_start=first.row_index,
            row_end=last.row_index,
            row_count=len(entries),
        )
        confidence, warnings = self._chunk_quality(table, entries)
        page_start, page_end = _page_range(entries, region)
        return Chunk(
            chunk_index=0,
            content=content,
            region_type=RegionType.TABLE,
            source_type=region.source_type,
            block_index=region.block_index,
            heading_path=list(region.heading_path),
            page_start=page_start,
            page_end=page_end,
            source_locator=locator,
            column_names=list(table.header_row) if table.header_row else None,
            has_header=table.header_row is not None,
            header_source=table.header_source,
            header_confidence=table.header_confidence,
            chunking_version=settings.INGEST_CHUNKING_VERSION,
            structure_confidence=confidence,
            parse_warnings=warnings,
        )

    def _split_oversized_row(
        self,
        region: ParsedRegion,
        table: TableBlock,
        prefix_str: str,
        entry: "_Entry",
        usable_budget: int,
        table_id: str,
    ) -> list[Chunk]:
        """A single row so large even the hard cap can't hold it intact -
        the declared, explicitly-marked exception to "never cut a row" (see
        module docstring): split the row's longest cell across N chunks,
        every one carrying `is_partial_row=True`/`row_part`/
        `row_part_count`/the same `table_id`/`row_start=row_end=row_index`
        and the row's full breadcrumb, so all parts can be reassembled later.
        """

        row = entry.row
        cells = row.cells
        crumb_prefix = f"{entry.crumb}\n" if entry.crumb else ""
        longest_index = max(range(len(cells)), key=lambda i: _token_len(cells[i]))
        other_cells = [("" if i == longest_index else cell) for i, cell in enumerate(cells)]
        overhead_tokens = _token_len(crumb_prefix + _render_row(other_cells))
        per_part_budget = usable_budget - overhead_tokens - _PARTIAL_ROW_MARKER_RESERVE_TOKENS
        if per_part_budget <= 0:
            raise ChunkingConfigException(
                "Row too wide to split even with the oversized-row fallback: "
                f"table_id={table_id} row_index={row.row_index}, usable_budget={usable_budget}. "
                "Increase max_tokens."
            )

        long_text = cells[longest_index]
        long_tokens = _ENCODING.encode(long_text)
        total_parts = max(1, math.ceil(len(long_tokens) / per_part_budget))
        confidence, warnings = self._chunk_quality(table, [entry])
        page_start, page_end = _page_range([entry], region)

        parts: list[Chunk] = []
        for part in range(1, total_parts + 1):
            start = (part - 1) * per_part_budget
            piece_tokens = long_tokens[start : start + per_part_budget]
            piece_text = _ENCODING.decode(piece_tokens)
            part_row = list(cells)
            part_row[longest_index] = piece_text
            marker = f"(Phần {part}/{total_parts})\n"
            content = prefix_str + marker + crumb_prefix + _render_row(part_row)
            locator = SourceLocator(
                section=heading_section(region.heading_path),
                table_id=table_id,
                row_start=row.row_index,
                row_end=row.row_index,
                row_count=1,
                row_part=part,
                row_part_count=total_parts,
                is_partial_row=True,
            )
            parts.append(
                Chunk(
                    chunk_index=0,
                    content=content,
                    region_type=RegionType.TABLE,
                    source_type=region.source_type,
                    block_index=region.block_index,
                    heading_path=list(region.heading_path),
                    page_start=page_start,
                    page_end=page_end,
                    source_locator=locator,
                    column_names=list(table.header_row) if table.header_row else None,
                    has_header=table.header_row is not None,
                    header_source=table.header_source,
                    header_confidence=table.header_confidence,
                    chunking_version=settings.INGEST_CHUNKING_VERSION,
                    structure_confidence=confidence,
                    parse_warnings=warnings,
                )
            )
        return parts


@dataclass(frozen=True)
class _Entry:
    """One row queued for a chunk, with its rendered breadcrumb line."""

    row: TableRow
    crumb: str
    truncated: bool = False


def _breadcrumb_line(ancestors: list[str], max_tokens: int) -> tuple[str, bool]:
    """`[A > B > C]` for a row's FULL ancestor path. Only when the path alone
    would eat too much of the chunk budget are the farthest ancestors dropped
    (with a leading `…`), nearest kept - and the caller flags the chunk."""

    if not ancestors:
        return "", False
    line = "[" + " > ".join(ancestors) + "]"
    if _token_len(line) <= max_tokens:
        return line, False
    remaining = list(ancestors)
    while len(remaining) > 1:
        remaining = remaining[1:]
        line = "[… > " + " > ".join(remaining) + "]"
        if _token_len(line) <= max_tokens:
            return line, True
    kept = _ENCODING.decode(_ENCODING.encode(remaining[0])[: max(1, max_tokens - 6)])
    return f"[… > {kept}…]", True


def _render_body(entries: list[_Entry]) -> str:
    """Rows in order; every change of ancestors opens a new group headed by
    that row's full breadcrumb (`[-]` when it has none but the previous group
    did), so two rows with different parents can never read as siblings."""

    lines: list[str] = []
    previous: str | None = None
    for entry in entries:
        if entry.crumb != previous:
            if entry.crumb:
                lines.append(entry.crumb)
            elif previous:
                lines.append(_NO_ANCESTORS_MARK)
            previous = entry.crumb
        lines.append(_render_row(entry.row.cells))
    return "\n".join(lines)


def _page_range(entries: list[_Entry], region: ParsedRegion) -> tuple[int | None, int | None]:
    starts = [e.row.page_start for e in entries if e.row.page_start is not None]
    ends = [e.row.page_end for e in entries if e.row.page_end is not None]
    if not starts or not ends:
        return region.page_start, region.page_end
    return min(starts), max(ends)


def heading_section(heading_path: list[str]) -> str | None:
    joined = " > ".join(heading_path)
    return joined or None
