import re
from dataclasses import dataclass, field

import tiktoken

from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.embeddings.provider import EmbeddingProvider
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENTENCE_BOUNDARY.split(text.strip()) if sentence]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@dataclass(frozen=True)
class SemanticChunker:
    """Group sentences toward a token target, breaking early at semantic shifts.

    Heuristic, not state-of-the-art: adjacent sentences are embedded via
    `EmbeddingProvider` and grouped by cosine similarity, targeting
    `target_tokens` per chunk with `overlap_ratio` of the previous chunk
    carried into the next. Acceptance is judged on boundary/size behavior,
    not semantic quality.
    """

    target_tokens: int = 400
    overlap_ratio: float = 0.2
    similarity_threshold: float = 0.5
    embedder: EmbeddingProvider = field(default_factory=OpenAIEmbedder)

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for region in regions:
            sentences = _split_sentences(region.content)
            if not sentences:
                continue
            embeddings = self.embedder.embed(sentences)
            for group in self._group_sentences(sentences, embeddings):
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=" ".join(group),
                        region_type=region.region_type,
                    )
                )
        return chunks

    def _group_sentences(
        self, sentences: list[str], embeddings: list[list[float]]
    ) -> list[list[str]]:
        overlap_tokens = int(self.target_tokens * self.overlap_ratio)
        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for index, sentence in enumerate(sentences):
            sentence_tokens = len(_ENCODING.encode(sentence))
            is_semantic_break = (
                current
                and index > 0
                and _cosine_similarity(embeddings[index - 1], embeddings[index])
                < self.similarity_threshold
            )
            exceeds_target = current and current_tokens + sentence_tokens > self.target_tokens

            if current and (is_semantic_break or exceeds_target):
                groups.append(current)
                current = self._carry_overlap(current, overlap_tokens)
                current_tokens = sum(len(_ENCODING.encode(s)) for s in current)

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _carry_overlap(previous_group: list[str], overlap_tokens: int) -> list[str]:
        """Carry trailing sentences from the previous group up to `overlap_tokens`."""

        carried: list[str] = []
        carried_tokens = 0
        for sentence in reversed(previous_group):
            sentence_tokens = len(_ENCODING.encode(sentence))
            if carried and carried_tokens + sentence_tokens > overlap_tokens:
                break
            carried.insert(0, sentence)
            carried_tokens += sentence_tokens
        return carried
