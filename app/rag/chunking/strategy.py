from typing import Any, Protocol

from app.core.exceptions import StrategyFileTypeMismatchException
from app.rag.chunking.excel_rows import ExcelRowChunker
from app.rag.chunking.markdown_aware import MarkdownAwareChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.semantic import SemanticChunker
from app.rag.chunking.token_based import TokenBasedChunker
from app.rag.ingestion.parser import get_extension
from app.rag.ingestion.table_aware_parser import ParsedRegion, split_regions
from app.schemas.ingestion import Chunk, ChunkingStrategyName


class _RegionChunker(Protocol):
    def split(self, regions: list[ParsedRegion]) -> list[Chunk]: ...


def dispatch(
    strategy: ChunkingStrategyName,
    params: dict[str, Any],
    content: bytes,
    filename: str,
) -> list[Chunk]:
    """Route a chunking request to the selected strategy, validating file-type fit."""

    extension = get_extension(filename)

    if strategy == ChunkingStrategyName.EXCEL_ROW:
        if extension != "xlsx":
            raise StrategyFileTypeMismatchException(strategy, filename)
        rows_per_chunk = int(params.get("rows_per_chunk", 1))
        return ExcelRowChunker(rows_per_chunk=rows_per_chunk).split(content)

    if extension == "xlsx":
        raise StrategyFileTypeMismatchException(strategy, filename)

    regions = split_regions(content, filename, extension)
    chunker: _RegionChunker = _build_text_chunker(strategy, params)
    return chunker.split(regions)


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
