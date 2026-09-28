import math
import re
from dataclasses import dataclass, field

import tiktoken

from app.core.config import settings
from app.core.errors.exceptions import ChunkingConfigException
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.embeddings.provider import EmbeddingProvider
from app.rag.ingestion.table_aware_parser import ParsedRegion
from app.schemas.ingestion import Chunk, SourceLocator

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


def _heading_prefix(heading_path: list[str]) -> str:
    joined = " > ".join(heading_path)
    return f"{joined}\n\n" if joined else ""


def _section(heading_path: list[str]) -> str | None:
    joined = " > ".join(heading_path)
    return joined or None


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
    via `settings.INGEST_SEMANTIC_MAX_TOKEN_FACTOR` (see `max_tokens`) and is used
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
        return math.ceil(self.target_tokens * settings.INGEST_SEMANTIC_MAX_TOKEN_FACTOR)

    @property
    def _overlap_tokens(self) -> int:
        return int(self.target_tokens * self.overlap_ratio)

    def split(self, regions: list[ParsedRegion]) -> list[Chunk]:
        """Group each region's sentences toward `target_tokens`, then merge
        tiny trailing chunks. `region.heading_path` is prepended into EVERY
        final chunk's content (not just the region's first), with its token
        cost subtracted from `max_tokens` BEFORE grouping/splitting -
        `usable_max_tokens` - so a chunk's final prefixed content never
        exceeds `max_tokens`. Small-chunk merging (`_merge_small_chunks`) is
        restricted to chunks from the SAME region (`block_index`) so two
        different regions' headings/pages are never silently glued
        together.
        """

        raw_chunks: list[Chunk] = []
        prefixes: dict[int | None, str] = {}
        usable_max_tokens_by_block: dict[int | None, int] = {}
        for region in regions:
            prefix = _heading_prefix(region.heading_path)
            prefix_tokens = _token_len(prefix) if prefix else 0
            usable_max_tokens = self.max_tokens - prefix_tokens
            if usable_max_tokens <= 0:
                raise ChunkingConfigException(
                    f"heading_path prefix is {prefix_tokens} tokens, leaving "
                    f"usable_max_tokens={usable_max_tokens} <= 0 for max_tokens={self.max_tokens}. "
                    "Increase target_tokens or shorten heading_path."
                )
            prefixes[region.block_index] = prefix
            usable_max_tokens_by_block[region.block_index] = usable_max_tokens

            sentences = self._prepare_sentences(region.content, usable_max_tokens)
            if not sentences:
                continue
            embeddings = self.embedder.embed(sentences)
            for group in self._group_sentences(sentences, embeddings, usable_max_tokens):
                raw_chunks.append(
                    Chunk(
                        chunk_index=0,
                        content=" ".join(group),
                        region_type=region.region_type,
                        source_type=region.source_type,
                        block_index=region.block_index,
                        heading_path=list(region.heading_path),
                        page_start=region.page_start,
                        page_end=region.page_end,
                    )
                )

        merged = self._merge_small_chunks(raw_chunks, usable_max_tokens_by_block)
        final_chunks: list[Chunk] = []
        for index, chunk in enumerate(merged):
            prefix = prefixes.get(chunk.block_index, "")
            final_chunks.append(
                chunk.model_copy(
                    update={
                        "chunk_index": index,
                        "content": f"{prefix}{chunk.content}",
                        "source_locator": SourceLocator(section=_section(chunk.heading_path)),
                        "chunking_version": settings.INGEST_CHUNKING_VERSION,
                    }
                )
            )
        return final_chunks

    def _prepare_sentences(self, content: str, usable_max_tokens: int) -> list[str]:
        prepared: list[str] = []
        for sentence in _split_sentences(content):
            prepared.extend(_split_oversized(sentence, usable_max_tokens, self._overlap_tokens))
        return prepared

    def _group_sentences(
        self, sentences: list[str], embeddings: list[list[float]], usable_max_tokens: int
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
                if sum(_token_len(s) for s in carried) + sentence_tokens > usable_max_tokens:
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

    def _merge_small_chunks(
        self, chunks: list[Chunk], usable_max_tokens_by_block: dict[int | None, int]
    ) -> list[Chunk]:
        """Fold any sub-`min_tokens` chunk into an adjacent same-type chunk.

        A chunk below the floor is glued onto the preceding chunk when they
        share a region type AND the same originating region (`block_index`)
        and the result still fits `max_tokens`; a small leading chunk is
        instead glued onto the one that follows it. Chunks that cannot be
        merged either way (a lone small region between two tables, say) are
        left as-is. `chunk_index` is renumbered afterward. Requiring the
        same `block_index` (not just the same `region_type`) prevents two
        different regions - with different `heading_path`/`page_start` -
        from being silently glued into one chunk that could only truthfully
        carry one of their headings.
        """

        merged: list[Chunk] = []
        for chunk in chunks:
            previous = merged[-1] if merged else None
            usable_max_tokens = usable_max_tokens_by_block.get(chunk.block_index, self.max_tokens)
            can_merge_back = (
                previous is not None
                and previous.region_type == chunk.region_type
                and previous.block_index == chunk.block_index
                and (
                    _token_len(previous.content) < self.min_tokens
                    or _token_len(chunk.content) < self.min_tokens
                )
                and _token_len(previous.content) + _token_len(chunk.content) <= usable_max_tokens
            )
            if can_merge_back and previous is not None:
                merged[-1] = previous.model_copy(
                    update={"content": f"{previous.content} {chunk.content}"}
                )
            else:
                merged.append(chunk)

        return [
            chunk.model_copy(update={"chunk_index": index}) for index, chunk in enumerate(merged)
        ]
