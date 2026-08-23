from dataclasses import dataclass
from io import BytesIO

import openpyxl

from app.schemas.ingestion import Chunk, RegionType


def _format_row(values: tuple[object, ...]) -> str:
    return " | ".join("" if value is None else str(value) for value in values)


@dataclass(frozen=True)
class ExcelRowChunker:
    """Row-based `.xlsx` chunker: each chunk covers `rows_per_chunk` data rows,
    with the header row prepended as context."""

    rows_per_chunk: int = 1

    def __post_init__(self) -> None:
        if self.rows_per_chunk <= 0:
            raise ValueError("rows_per_chunk must be positive")

    def split(self, content: bytes) -> list[Chunk]:
        workbook = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
        try:
            sheet = workbook.active
            assert sheet is not None
            rows = sheet.iter_rows(values_only=True)
            header = next(rows, None)
            header_line = _format_row(header) if header else ""

            chunks: list[Chunk] = []
            buffer: list[tuple[object, ...]] = []
            for row in rows:
                buffer.append(row)
                if len(buffer) == self.rows_per_chunk:
                    chunks.append(self._make_chunk(header_line, buffer, len(chunks)))
                    buffer = []
            if buffer:
                chunks.append(self._make_chunk(header_line, buffer, len(chunks)))
            return chunks
        finally:
            workbook.close()

    @staticmethod
    def _make_chunk(header_line: str, rows: list[tuple[object, ...]], index: int) -> Chunk:
        lines = [header_line] if header_line else []
        lines.extend(_format_row(row) for row in rows)
        return Chunk(
            chunk_index=index,
            content="\n".join(lines),
            region_type=RegionType.EXCEL_ROW,
        )
