import pytest

from app.core.exceptions import StrategyFileTypeMismatchException
from app.rag.chunking.strategy import dispatch
from app.schemas.ingestion import ChunkingStrategyName
from tests.fixtures.documents import make_pdf_bytes, make_xlsx_bytes


@pytest.mark.parametrize(
    "strategy",
    [
        ChunkingStrategyName.RECURSIVE,
        ChunkingStrategyName.TOKEN_BASED,
        ChunkingStrategyName.MARKDOWN_AWARE,
    ],
)
def test_dispatch_reaches_text_based_strategies_on_a_pdf(
    strategy: ChunkingStrategyName,
) -> None:
    content = make_pdf_bytes("A reasonably long sentence for chunking purposes.")

    chunks = dispatch(strategy, {}, content, "handbook.pdf")

    assert chunks


def test_dispatch_reaches_semantic_strategy_without_live_openai_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.rag.embeddings.openai_embedder import OpenAIEmbedder

    def _fake_embed(self: OpenAIEmbedder, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(OpenAIEmbedder, "embed", _fake_embed)
    content = make_pdf_bytes("A reasonably long sentence for chunking purposes.")

    chunks = dispatch(ChunkingStrategyName.SEMANTIC, {}, content, "handbook.pdf")

    assert chunks


def test_dispatch_reaches_excel_row_strategy_on_an_xlsx() -> None:
    content = make_xlsx_bytes(header=["Name"], rows=[["Alice"], ["Bob"]])

    chunks = dispatch(ChunkingStrategyName.EXCEL_ROW, {}, content, "roster.xlsx")

    assert len(chunks) == 2


def test_dispatch_rejects_excel_row_strategy_on_non_xlsx_file() -> None:
    content = make_pdf_bytes("Not a spreadsheet.")

    with pytest.raises(StrategyFileTypeMismatchException):
        dispatch(ChunkingStrategyName.EXCEL_ROW, {}, content, "handbook.pdf")


def test_dispatch_rejects_non_excel_strategy_on_xlsx_file() -> None:
    content = make_xlsx_bytes(header=["Name"], rows=[["Alice"]])

    with pytest.raises(StrategyFileTypeMismatchException):
        dispatch(ChunkingStrategyName.RECURSIVE, {}, content, "roster.xlsx")
