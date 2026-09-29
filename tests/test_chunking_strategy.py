import pytest

from app.core.errors.exceptions import StrategyFileTypeMismatchException
from app.rag.chunking.strategy import dispatch
from app.schemas.ingestion import ChunkingStrategyName, RegionType
from tests.fixtures.documents import make_pdf_bytes, make_xlsx_bytes

_DOC_ID = "doc-test"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategyName.RECURSIVE,
        ChunkingStrategyName.TOKEN_BASED,
        ChunkingStrategyName.MARKDOWN_AWARE,
    ],
)
async def test_dispatch_reaches_text_based_strategies_on_a_pdf(
    strategy: ChunkingStrategyName,
) -> None:
    content = make_pdf_bytes("A reasonably long sentence for chunking purposes.")

    chunks = await dispatch(strategy, {}, content, "handbook.pdf", document_id=_DOC_ID)

    assert chunks


@pytest.mark.asyncio
async def test_dispatch_reaches_semantic_strategy_without_live_openai_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    async def _fake_embed_tracked(
        self: OpenAIEmbedder, texts: list[str], usage_recorder: object, budget_tracker: object
    ) -> list[list[float]]:
        del usage_recorder, budget_tracker
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(OpenAIEmbedder, "embed_tracked", _fake_embed_tracked)
    content = make_pdf_bytes("A reasonably long sentence for chunking purposes.")

    chunks = await dispatch(
        ChunkingStrategyName.SEMANTIC, {}, content, "handbook.pdf", document_id=_DOC_ID
    )

    assert chunks


@pytest.mark.asyncio
async def test_dispatch_reaches_excel_row_strategy_on_an_xlsx() -> None:
    content = make_xlsx_bytes(header=["Name"], rows=[["Alice"], ["Bob"]])

    chunks = await dispatch(
        ChunkingStrategyName.EXCEL_ROW, {}, content, "roster.xlsx", document_id=_DOC_ID
    )

    assert len(chunks) == 2


@pytest.mark.asyncio
async def test_dispatch_rejects_excel_row_strategy_on_non_xlsx_file() -> None:
    content = make_pdf_bytes("Not a spreadsheet.")

    with pytest.raises(StrategyFileTypeMismatchException):
        await dispatch(
            ChunkingStrategyName.EXCEL_ROW, {}, content, "handbook.pdf", document_id=_DOC_ID
        )


@pytest.mark.asyncio
async def test_dispatch_rejects_non_excel_strategy_on_xlsx_file() -> None:
    content = make_xlsx_bytes(header=["Name"], rows=[["Alice"]])

    with pytest.raises(StrategyFileTypeMismatchException):
        await dispatch(
            ChunkingStrategyName.RECURSIVE, {}, content, "roster.xlsx", document_id=_DOC_ID
        )


@pytest.mark.asyncio
async def test_table_chunk_budget_defaults_to_the_setting_and_ignores_text_target_tokens() -> None:
    from pathlib import Path

    import tiktoken

    from app.core.config import settings

    sample = (
        Path(__file__).parent.parent
        / "_to_delete"
        / "Quyet dinh 1035 QD DHCN Hoc phi 2025-2026.pdf"
    )
    if not sample.exists():
        pytest.skip("sample PDF not available")
    content = sample.read_bytes()
    encoding = tiktoken.get_encoding("cl100k_base")

    async def table_chunks(params: dict) -> list:
        chunks = await dispatch(
            ChunkingStrategyName.RECURSIVE, params, content, sample.name, document_id=_DOC_ID
        )
        return [c for c in chunks if c.region_type == RegionType.TABLE]

    default = await table_chunks({})
    assert settings.INGEST_TABLE_CHUNK_MAX_TOKENS == 800
    assert max(len(encoding.encode(c.content)) for c in default) <= 800
    assert len(default) < len(await table_chunks({"table_max_tokens": 400}))
    # the text strategy's own knobs never touch table chunks
    assert len(await table_chunks({"chunk_size": 300, "overlap": 10})) == len(default)
