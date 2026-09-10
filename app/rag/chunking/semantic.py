import math
import re
from dataclasses import dataclass, field

import tiktoken

from app.core.config import settings
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.embeddings.provider import EmbeddingProvider
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk

_ENCODING = tiktoken.get_encoding("cl100k_base")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_LIST_MARKER = re.compile(r"^[A-Za-z0-9][.)]$")


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation, then glue lone list markers forward.

    The boundary regex treats `a.`, `1.` or `b)` at the start of a list
    item as a full sentence of its own (the marker ends in `.`/`)` and is
    followed by whitespace). Those one-token fragments are meaningless to
    embed and only produce tiny chunks, so a fragment that is *only* a
    marker is merged into the sentence that follows it.
    """

    raw = [sentence for sentence in _SENTENCE_BOUNDARY.split(text.strip()) if sentence]
    merged: list[str] = []
    pending_marker = ""
    for sentence in raw:
        if pending_marker:
            merged.append(f"{pending_marker} {sentence}")
            pending_marker = ""
        elif _LIST_MARKER.match(sentence):
            pending_marker = sentence
        else:
            merged.append(sentence)
    if pending_marker:
        merged.append(pending_marker)
    return merged


def _split_oversized(sentence: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Break a single sentence longer than `max_tokens` into token windows.

    Fallback for content the sentence splitter cannot divide - e.g. a long
    bullet list flattened to one line with no `.`/`!`/`?` in it. Windows
    carry `overlap_tokens` of context into the next piece.
    """

    tokens = _ENCODING.encode(sentence)
    if len(tokens) <= max_tokens:
        return [sentence]
    step = max(1, max_tokens - overlap_tokens)
    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        piece = _ENCODING.decode(window).strip()
        if piece:
            pieces.append(piece)
        if start + max_tokens >= len(tokens):
            break
    return pieces or [sentence]


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

    `target_tokens` is a soft target. The hard ceiling is derived from it
    via `settings.SEMANTIC_MAX_TOKEN_FACTOR` (see `max_tokens`) and is used
    to hard-split overlong sentences and force group breaks. Chunks that
    end up below `min_tokens` are folded into an adjacent chunk of the
    same region type so the strategy never emits tiny fragments.
    """

    target_tokens: int = 400
    overlap_ratio: float = 0.2
    similarity_threshold: float = 0.5
    min_tokens: int = 48
    embedder: EmbeddingProvider = field(default_factory=OpenAIEmbedder)

    @property
    def max_tokens(self) -> int:
        return math.ceil(self.target_tokens * settings.SEMANTIC_MAX_TOKEN_FACTOR)

    @property
    def _overlap_tokens(self) -> int:
        return int(self.target_tokens * self.overlap_ratio)

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for region in regions:
            sentences = self._prepare_sentences(region.content)
            if not sentences:
                continue
            embeddings = self.embedder.embed(sentences)
            for group in self._group_sentences(sentences, embeddings):
                chunks.append(
                    Chunk(
                        chunk_index=0,
                        content=" ".join(group),
                        region_type=region.region_type,
                    )
                )
        return self._merge_small_chunks(chunks)

    def _prepare_sentences(self, content: str) -> list[str]:
        prepared: list[str] = []
        for sentence in _split_sentences(content):
            prepared.extend(
                _split_oversized(sentence, self.max_tokens, self._overlap_tokens)
            )
        return prepared

    def _group_sentences(
        self, sentences: list[str], embeddings: list[list[float]]
    ) -> list[list[str]]:
        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for index, sentence in enumerate(sentences):
            sentence_tokens = _token_len(sentence)
            is_semantic_break = (
                current
                and index > 0
                and _cosine_similarity(embeddings[index - 1], embeddings[index])
                < self.similarity_threshold
            )
            exceeds_target = current and current_tokens + sentence_tokens > self.target_tokens

            if current and (is_semantic_break or exceeds_target):
                groups.append(current)
                carried = self._carry_overlap(current, self._overlap_tokens)
                if sum(_token_len(s) for s in carried) + sentence_tokens > self.max_tokens:
                    carried = []
                current = carried
                current_tokens = sum(_token_len(s) for s in current)

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

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Fold any sub-`min_tokens` chunk into an adjacent same-type chunk.

        A chunk below the floor is glued onto the preceding chunk when they
        share a region type and the result still fits `max_tokens`; a small
        leading chunk is instead glued onto the one that follows it. Chunks
        that cannot be merged either way (a lone small region between two
        tables, say) are left as-is. `chunk_index` is renumbered afterward.
        """

        merged: list[Chunk] = []
        for chunk in chunks:
            previous = merged[-1] if merged else None
            can_merge_back = (
                previous is not None
                and previous.region_type == chunk.region_type
                and (
                    _token_len(previous.content) < self.min_tokens
                    or _token_len(chunk.content) < self.min_tokens
                )
                and _token_len(previous.content) + _token_len(chunk.content)
                <= self.max_tokens
            )
            if can_merge_back and previous is not None:
                merged[-1] = previous.model_copy(
                    update={"content": f"{previous.content} {chunk.content}"}
                )
            else:
                merged.append(chunk)

        return [
            chunk.model_copy(update={"chunk_index": index})
            for index, chunk in enumerate(merged)
        ]
