import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from app.core.config import settings
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.schemas.ingestion import Chunk

logger = logging.getLogger(__name__)

# Static .env-sourced credential for now — registry-driven provider selection
# (per-purpose base URL/provider from the model registry snapshot) is Task 5/9's job.
_OPENAI_API_BASE_URL = "https://api.openai.com/v1"

_SYSTEM_PROMPT = (
    "You summarize a document chunk and propose hypothetical questions it answers. "
    "Always respond in Vietnamese, regardless of the input chunk's language - the "
    "summary and questions must match the language a Vietnamese student would "
    "actually ask in, so they embed close to real user queries. "
    'Respond with JSON only: {{"summary": string, "questions": string[]}}. '
    "Produce exactly {question_count} questions."
)


@dataclass(frozen=True)
class EnrichedChunk:
    """A chunk enriched with a summary and hypothetical questions for multi-representation
    search."""

    chunk: Chunk
    summary: str
    questions: list[str]


@dataclass(frozen=True)
class MultiRepresentationEnricher:
    """Add a summary + hypothetical questions to a chunk via one LLM call."""

    model: str = field(default_factory=lambda: settings.INGEST_MULTI_REP_LLM_MODEL)
    question_count: int = field(default_factory=lambda: settings.INGEST_MULTI_REP_QUESTION_COUNT)
    client: OpenAI | None = None

    def enrich(self, chunk: Chunk) -> EnrichedChunk:
        """Enrich one chunk; a malformed/short LLM response falls back to an empty
        result with a logged warning rather than raising, so one bad chunk doesn't
        kill the whole embedding batch."""

        client = self.client or OpenAI(
            api_key=settings.OPENAI_API_KEY,
            http_client=build_provider_http_client_sync(
                ProviderConnectionInfo(api_base_url=_OPENAI_API_BASE_URL)
            ),
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(question_count=self.question_count),
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
