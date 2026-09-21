from dataclasses import dataclass
from io import BytesIO

import openpyxl

from app.core.config import settings
from app.schemas.ingestion import Chunk, HeaderSource, RegionType, SourceLocator, SourceType


def _format_row(values: tuple[object, ...]) -> str:
    return " | ".join("" if value is None else str(value) for value in values)


@dataclass(frozen=True)
class ExcelRowChunker:
    """Row-based `.xlsx` chunker: each chunk covers `rows_per_chunk` data rows,
    with the header row prepended as context.

    `header_source=EXPLICIT` here is a PRODUCT CONTRACT ("the first row of
    an uploaded .xlsx is always its header"), not a structural signal
    verified from `openpyxl` the way HTML `<th>`/DOCX `<w:tblHeader>` are -
    unlike those two, nothing in the file format confirms the first row is
    really a header. Behavior is unchanged from before this field existed;
    only the label's true meaning is now documented (see plan known-gaps:
    an .xlsx whose first row isn't really a header is still treated as one).
    """

    rows_per_chunk: int = 1

    def __post_init__(self) -> None:
        if self.rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")

    def split(self, content: bytes) -> list[Chunk]:
        workbook = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
        try:
            sheet = workbook.active
            assert sheet is not None
            sheet_name = sheet.title
            rows = sheet.iter_rows(values_only=True)
            header = next(rows, None)
            column_names = (
                ["" if value is None else str(value) for value in header] if header else None
            )
            header_line = _format_row(header) if header else ""

            chunks: list[Chunk] = []
            buffer: list[tuple[object, ...]] = []
            row_start = 1
            for row in rows:
                buffer.append(row)
                if len(buffer) == self.rows_per_chunk:
                    chunks.append(
                        self._make_chunk(
                            header_line, column_names, sheet_name, buffer, row_start, len(chunks)
                        )
                    )
                    row_start += len(buffer)
                    buffer = []
            if buffer:
                chunks.append(
                    self._make_chunk(
                        header_line, column_names, sheet_name, buffer, row_start, len(chunks)
                    )
                )
            return chunks
        finally:
            workbook.close()

    @staticmethod
    def _make_chunk(
        header_line: str,
        column_names: list[str] | None,
        sheet_name: str,
        rows: list[tuple[object, ...]],
        row_start: int,
        index: int,
    ) -> Chunk:
        lines = [header_line] if header_line else []
        lines.extend(_format_row(row) for row in rows)
        row_end = row_start + len(rows) - 1
        return Chunk(
            chunk_index=index,
            content="\n".join(lines),
            region_type=RegionType.EXCEL_ROW,
            source_type=SourceType.XLSX,
            block_index=0,
            heading_path=[],
            source_locator=SourceLocator(
                sheet_name=sheet_name,
                table_id="table-0",
                row_start=row_start,
                row_end=row_end,
                row_count=len(rows),
            ),
            column_names=column_names,
            has_header=column_names is not None,
            header_source=(
                HeaderSource.EXPLICIT if column_names is not None else HeaderSource.MISSING
            ),
            header_confidence=1.0 if column_names is not None else 0.0,
            chunking_version=settings.CHUNKING_VERSION,
        )
