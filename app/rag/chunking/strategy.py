from typing import Any, Protocol

from app.core.config import settings
from app.core.errors.exceptions import StrategyFileTypeMismatchException
from app.rag.chunking.excel_rows import ExcelRowChunker
from app.rag.chunking.markdown_aware import MarkdownAwareChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.semantic import SemanticChunker
from app.rag.chunking.table_row import TableRowChunker
from app.rag.chunking.token_based import TokenBasedChunker
from app.rag.ingestion.parser import get_extension
from app.rag.ingestion.table_aware_parser import ParsedRegion, split_regions
from app.schemas.ingestion import Chunk, ChunkingStrategyName, RegionType


class _RegionChunker(Protocol):
    def split(self, regions: list[ParsedRegion]) -> list[Chunk]: ...


def dispatch(
    strategy: ChunkingStrategyName,
    params: dict[str, Any],
    content: bytes,
    filename: str,
) -> list[Chunk]:
    """Route a chunking request to the selected strategy, validating file-type fit.

    TABLE regions ALWAYS go through `TableRowChunker`, regardless of the
    requested text strategy - a table is never handed to
    `RecursiveChunker`/`TokenBasedChunker`/`SemanticChunker`/
    `MarkdownAwareChunker`, all of which only know about paragraph/token
    boundaries and would happily cut a table mid-row. `table_chunks` and
    `text_chunks` are then merged back together ordered by each chunk's
    `block_index` (the position of its originating region in the document),
    then `chunk_index` is renumbered 0..n-1 over the merged, ordered list -
    `block_index` itself is left untouched by the renumbering.
    """

    extension = get_extension(filename)

    if strategy == ChunkingStrategyName.EXCEL_ROW:
        if extension != "xlsx":
            raise StrategyFileTypeMismatchException(strategy, filename)
        rows_per_chunk = int(params.get("rows_per_chunk", 1))
        return ExcelRowChunker(rows_per_chunk=rows_per_chunk).split(content)

    if extension == "xlsx":
        raise StrategyFileTypeMismatchException(strategy, filename)

    regions = split_regions(content, filename, extension)
    table_regions = [region for region in regions if region.region_type == RegionType.TABLE]
    text_regions = [region for region in regions if region.region_type != RegionType.TABLE]

    table_max_tokens = int(params.get("table_max_tokens", settings.INGEST_TABLE_CHUNK_MAX_TOKENS))
    table_chunks = TableRowChunker(max_tokens=table_max_tokens).split(table_regions)

    chunker: _RegionChunker = _build_text_chunker(strategy, params)
    text_chunks = chunker.split(text_regions)

    merged = sorted(
        [*table_chunks, *text_chunks],
        key=lambda chunk: (chunk.block_index is None, chunk.block_index or 0),
    )
    return [chunk.model_copy(update={"chunk_index": i}) for i, chunk in enumerate(merged)]


def _build_text_chunker(strategy: ChunkingStrategyName, params: dict[str, Any]) -> _RegionChunker:
    if strategy == ChunkingStrategyName.RECURSIVE:
        return RecursiveChunker(
            chunk_size=int(params.get("chunk_size", 800)),
            overlap=int(params.get("overlap", 120)),
        )
    if strategy == ChunkingStrategyName.TOKEN_BASED:
        return TokenBasedChunker(
            chunk_size=int(params.get("chunk_size", 400)),
            overlap=int(params.get("overlap", 40)),
        )
    if strategy == ChunkingStrategyName.SEMANTIC:
        return SemanticChunker(
            target_tokens=int(params.get("target_tokens", 400)),
            overlap_ratio=float(params.get("overlap_ratio", 0.2)),
            similarity_threshold=float(params.get("similarity_threshold", 0.5)),
        )
    return MarkdownAwareChunker(
        text_chunker=RecursiveChunker(
            chunk_size=int(params.get("chunk_size", 800)),
            overlap=int(params.get("overlap", 120)),
        )
    )
