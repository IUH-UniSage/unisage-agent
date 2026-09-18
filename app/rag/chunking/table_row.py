import hashlib
import math
from dataclasses import dataclass

import tiktoken

from app.core.config import settings
from app.core.exceptions import ChunkingConfigException
from app.rag.ingestion.table_aware_parser import ParsedRegion, TableBlock
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
    """Chunks TABLE regions row-by-row directly from `TableBlock.data_rows`,
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
        if not table.data_rows:
            return []

        table_id = f"table-{region.block_index}" if region.block_index is not None else "table-0"
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

        chunks: list[Chunk] = []
        pending_rows: list[list[str]] = []
        pending_start: int | None = None

        def flush_pending(row_end: int) -> None:
            nonlocal pending_rows, pending_start
            if not pending_rows:
                return
            chunks.append(
                self._build_chunk(
                    region,
                    table,
                    prefix_str,
                    pending_rows,
                    row_start=pending_start if pending_start is not None else row_end,
                    row_end=row_end,
                    table_id=table_id,
                )
            )
            pending_rows = []
            pending_start = None

        for row_index, row in enumerate(table.data_rows, start=1):
            if table.header_row is not None and len(row) != len(table.header_row):
                raise TableStructureError(
                    table_id=table_id,
                    block_index=region.block_index,
                    row_index=row_index,
                    expected_cell_count=len(table.header_row),
                    actual_cell_count=len(row),
                    row=row,
                )

            row_line = _render_row(row)
            row_tokens = _token_len(row_line)

            if row_tokens > usable_budget:
                flush_pending(row_index - 1)
                if row_tokens <= hard_cap:
                    chunks.append(
                        self._build_chunk(
                            region,
                            table,
                            prefix_str,
                            [row],
                            row_start=row_index,
                            row_end=row_index,
                            table_id=table_id,
                        )
                    )
                else:
                    chunks.extend(
                        self._split_oversized_row(
                            region, table, prefix_str, row, row_index, usable_budget, table_id
                        )
                    )
                continue

            candidate_lines = [_render_row(r) for r in (*pending_rows, row)]
            candidate_tokens = _token_len("\n".join(candidate_lines))
            if pending_rows and candidate_tokens > usable_budget:
                flush_pending(row_index - 1)

            if pending_start is None:
                pending_start = row_index
            pending_rows.append(row)

        flush_pending(len(table.data_rows))
        return chunks

    def _build_chunk(
        self,
        region: ParsedRegion,
        table: TableBlock,
        prefix_str: str,
        rows: list[list[str]],
        *,
        row_start: int,
        row_end: int,
        table_id: str,
    ) -> Chunk:
        lines = [_render_row(row) for row in rows]
        content = prefix_str + "\n".join(lines)
        locator = SourceLocator(
            section=heading_section(region.heading_path),
            table_id=table_id,
            row_start=row_start,
            row_end=row_end,
            row_count=len(rows),
        )
        return Chunk(
            chunk_index=0,
            content=content,
            region_type=RegionType.TABLE,
            source_type=region.source_type,
            block_index=region.block_index,
            heading_path=list(region.heading_path),
            page_start=region.page_start,
            page_end=region.page_end,
            source_locator=locator,
            column_names=list(table.header_row) if table.header_row else None,
            has_header=table.header_row is not None,
            header_source=table.header_source,
            header_confidence=table.header_confidence,
            chunking_version=settings.CHUNKING_VERSION,
        )

    def _split_oversized_row(
        self,
        region: ParsedRegion,
        table: TableBlock,
        prefix_str: str,
        row: list[str],
        row_index: int,
        usable_budget: int,
        table_id: str,
    ) -> list[Chunk]:
        """A single row so large even the hard cap can't hold it intact -
        the declared, explicitly-marked exception to "never cut a row" (see
        module docstring): split the row's longest cell across N chunks,
        every one carrying `is_partial_row=True`/`row_part`/
        `row_part_count`/the same `table_id`/`row_start=row_end=row_index`
        so all parts can be reassembled later.
        """

        longest_index = max(range(len(row)), key=lambda i: _token_len(row[i]))
        other_cells = [("" if i == longest_index else cell) for i, cell in enumerate(row)]
        overhead_tokens = _token_len(_render_row(other_cells))
        per_part_budget = usable_budget - overhead_tokens - _PARTIAL_ROW_MARKER_RESERVE_TOKENS
        if per_part_budget <= 0:
            raise ChunkingConfigException(
                "Row too wide to split even with the oversized-row fallback: "
                f"table_id={table_id} row_index={row_index}, usable_budget={usable_budget}. "
                "Increase max_tokens."
            )

        long_text = row[longest_index]
        long_tokens = _ENCODING.encode(long_text)
        total_parts = max(1, math.ceil(len(long_tokens) / per_part_budget))

        parts: list[Chunk] = []
        for part in range(1, total_parts + 1):
            start = (part - 1) * per_part_budget
            piece_tokens = long_tokens[start : start + per_part_budget]
            piece_text = _ENCODING.decode(piece_tokens)
            part_row = list(row)
            part_row[longest_index] = piece_text
            marker = f"(Phần {part}/{total_parts})\n"
            content = prefix_str + marker + _render_row(part_row)
            locator = SourceLocator(
                section=heading_section(region.heading_path),
                table_id=table_id,
                row_start=row_index,
                row_end=row_index,
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
                    page_start=region.page_start,
                    page_end=region.page_end,
                    source_locator=locator,
                    column_names=list(table.header_row) if table.header_row else None,
                    has_header=table.header_row is not None,
                    header_source=table.header_source,
                    header_confidence=table.header_confidence,
                    chunking_version=settings.CHUNKING_VERSION,
                )
            )
        return parts


def heading_section(heading_path: list[str]) -> str | None:
    joined = " > ".join(heading_path)
    return joined or None
