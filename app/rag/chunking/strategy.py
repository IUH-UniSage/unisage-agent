import logging
from collections.abc import Callable
from typing import Any, Protocol

from app.core.config import settings
from app.core.errors.exceptions import (
    ChunkingConfigException,
    StrategyFileTypeMismatchException,
)
from app.rag.chunking.excel_rows import ExcelRowChunker
from app.rag.chunking.markdown_aware import MarkdownAwareChunker
from app.rag.chunking.recursive import RecursiveChunker
from app.rag.chunking.semantic import SemanticChunker
from app.rag.chunking.table_row import TableRowChunker
from app.rag.chunking.token_based import TokenBasedChunker
from app.rag.ingestion.parser import get_extension, parse_or_raise
from app.rag.ingestion.table_aware_parser import ParsedRegion, split_regions
from app.schemas.ingestion import Chunk, ChunkingStrategyName, RegionType

logger = logging.getLogger(__name__)


class _RegionChunker(Protocol):
    def split(self, regions: list[ParsedRegion]) -> list[Chunk]: ...


async def dispatch(
    strategy: ChunkingStrategyName,
    params: dict[str, Any],
    content: bytes,
    filename: str,
    *,
    document_id: str,
    user_id: str | None = None,
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

    `async def` only for `SemanticChunker` (the one strategy that calls an embedding
    provider): its embedding calls for this request are grouped under one
    `purpose="SEMANTIC_CHUNKING"` `UsageRecorder` via `SemanticChunker.split_tracked()`,
    keyed by `document_id`/`user_id` - every other chunker's plain sync `.split()` is
    called as-is, no behavior change.
    """

    extension = get_extension(filename)

    if strategy == ChunkingStrategyName.EXCEL_ROW:
        if extension != "xlsx":
            raise StrategyFileTypeMismatchException(strategy, filename)
        excel_chunker = _with_valid_params(
            lambda: ExcelRowChunker(rows_per_chunk=int(params.get("rows_per_chunk", 1)))
        )
        return parse_or_raise(lambda: excel_chunker.split(content), filename)

    if extension == "xlsx":
        raise StrategyFileTypeMismatchException(strategy, filename)

    # Build (and so validate) every chunker from the client's params BEFORE parsing - a bad
    # param is the client's input error, reported as such, not a 500.
    table_chunker = _with_valid_params(
        lambda: TableRowChunker(
            max_tokens=int(params.get("table_max_tokens", settings.INGEST_TABLE_CHUNK_MAX_TOKENS))
        )
    )
    chunker: _RegionChunker = _with_valid_params(lambda: _build_text_chunker(strategy, params))

    regions = parse_or_raise(lambda: split_regions(content, filename, extension), filename)
    table_regions = [region for region in regions if region.region_type == RegionType.TABLE]
    text_regions = [region for region in regions if region.region_type != RegionType.TABLE]

    table_chunks = table_chunker.split(table_regions)

    if isinstance(chunker, SemanticChunker):
        text_chunks = await chunker.split_tracked(
            text_regions, document_id=document_id, user_id=user_id
        )
    else:
        text_chunks = chunker.split(text_regions)

    merged = sorted(
        [*table_chunks, *text_chunks],
        key=lambda chunk: (chunk.block_index is None, chunk.block_index or 0),
    )
    return [chunk.model_copy(update={"chunk_index": i}) for i, chunk in enumerate(merged)]


def _with_valid_params[T](build: Callable[[], T]) -> T:
    """Runs a chunker constructor, turning a bad client param (non-numeric value, a
    non-positive size, overlap >= size, ...) into a 422 instead of a generic 500."""

    try:
        return build()
    except (TypeError, ValueError) as exc:
        raise ChunkingConfigException(f"Tham số chia đoạn không hợp lệ: {exc}") from exc


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
