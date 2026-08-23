import json
from unittest.mock import MagicMock

from app.rag.enrichment.multi_representation import MultiRepresentationEnricher
from app.schemas.ingestion import Chunk, RegionType


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    return client


def test_enrich_returns_summary_and_configured_question_count() -> None:
    payload = json.dumps(
        {
            "summary": "A short summary.",
            "questions": ["Q1?", "Q2?", "Q3?"],
        }
    )
    client = _mock_client(payload)
    enricher = MultiRepresentationEnricher(model="gpt-4o-mini", question_count=3, client=client)
    chunk = Chunk(chunk_index=0, content="Some chunk content.", region_type=RegionType.TEXT)

    enriched = enricher.enrich(chunk)

    assert enriched.summary == "A short summary."
    assert enriched.questions == ["Q1?", "Q2?", "Q3?"]
    assert enriched.chunk == chunk


def test_enrich_falls_back_to_empty_on_malformed_json() -> None:
    client = _mock_client("not valid json")
    enricher = MultiRepresentationEnricher(model="gpt-4o-mini", question_count=3, client=client)
    chunk = Chunk(chunk_index=0, content="Some chunk content.", region_type=RegionType.TEXT)

    enriched = enricher.enrich(chunk)

    assert enriched.summary == ""
    assert enriched.questions == []


def test_enrich_falls_back_to_empty_on_short_question_list() -> None:
    payload = json.dumps({"summary": "A short summary.", "questions": ["Only one?"]})
    client = _mock_client(payload)
    enricher = MultiRepresentationEnricher(model="gpt-4o-mini", question_count=3, client=client)
    chunk = Chunk(chunk_index=0, content="Some chunk content.", region_type=RegionType.TEXT)

    enriched = enricher.enrich(chunk)

    assert enriched.summary == ""
    assert enriched.questions == []
