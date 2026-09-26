import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import settings
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.model_registry import require_top_priority_credential
from app.rag.prompting.loader import get_templates
from app.schemas.ingestion import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichedChunk:
    """A chunk enriched with a summary and hypothetical questions for multi-representation
    search."""

    chunk: Chunk
    summary: str
    questions: list[str]


@dataclass(frozen=True)
class MultiRepresentationEnricher:
    """Add a summary + hypothetical questions to a chunk via one LLM call.

    Model name, API key and base URL come from the model registry's ACTIVE EXTRACTION
    credential (plan.md "Cutover khỏi cấu hình `.env` tĩnh") - never `.env`. Both are resolved
    lazily on first `enrich()` call, not at construction time - mirrors `OpenAIEmbedder`; tests
    inject `model`/`client` directly to skip the registry entirely (see
    tests/test_multi_representation.py).
    """

    model: str | None = None
    question_count: int = field(default_factory=lambda: settings.INGEST_MULTI_REP_QUESTION_COUNT)
    client: OpenAI | None = None

    def enrich(self, chunk: Chunk) -> EnrichedChunk:
        """Enrich one chunk; a malformed/short LLM response falls back to an empty
        result with a logged warning rather than raising, so one bad chunk doesn't
        kill the whole embedding batch."""

        model = self.model
        client = self.client
        if model is None or client is None:
            credential = require_top_priority_credential("EXTRACTION")
            model = model or credential.model_name or ""
            client = client or OpenAI(
                api_key=credential.api_key,
                base_url=credential.api_base_url or None,
                http_client=build_provider_http_client_sync(
                    ProviderConnectionInfo(api_base_url=credential.api_base_url or "")
                ),
            )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": get_templates().agent_multi_representation_enricher.format(
                        question_count=self.question_count
                    ),
                },
                {"role": "user", "content": chunk.content},
            ],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(raw_content)
            summary = str(data["summary"])
            questions = [str(question) for question in data["questions"]]
            if not summary or len(questions) != self.question_count:
                raise ValueError("incomplete multi-representation response")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning(
                "Malformed multi-representation response for chunk %s; falling back to empty.",
                chunk.chunk_index,
            )
            return EnrichedChunk(chunk=chunk, summary="", questions=[])

        return EnrichedChunk(chunk=chunk, summary=summary, questions=questions)
