"""CrossEncoderRerankNode (09a) - each sub-query's chunks re-scored by a cross-encoder.

The LLM rerank judges 16 chunks with one model call of 3-6 s. A cross-encoder
(bge-reranker-v2-m3, `reranker/` service) reads every (question, chunk) pair in a few
tens of milliseconds and ranks them; this node keeps each sub-query's top
`CHAT_CROSS_ENCODER_KEEP` by that score. It never empties a sub-query: a sub-query with
candidates keeps at least one, so the "no chunk answers it -> web search" decision
stays with the LLM rerank (mode `cross_encoder_then_llm`) or does not happen (mode
`cross_encoder`).

The chunk's `score` (cosine, or the lexical constant) is left alone - the rescue rule,
the dense floor and web search's ordering read it - and the cross-encoder score goes
to `metadata["ce_score"]` for the trace and the benchmark. Any service failure is
returned as `failure` with the input result untouched: the graph then runs the LLM
rerank as in mode `llm` (fail-over, docs/specs/SPEC-cross-encoder-rerank.md).
"""

import asyncio
import logging
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import settings
from app.graph.nodes.post_retrieval_rerank import TurnRerankResult, turn_result
from app.integrations.cross_encoder_client import (
    CrossEncoderClient,
    CrossEncoderUnavailableError,
)
from app.rag.prompting.citations import source_title
from app.rag.retrieval.service import EXPANDED_FLAG
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

CE_SCORE_KEY = "ce_score"
# Prefix of the AI-admin warning raised when the service failed and the LLM rerank ran.
CROSS_ENCODER_SKIPPED = "Bỏ qua bước xếp hạng cross-encoder (chạy lọc bằng LLM thay thế): "


@dataclass(frozen=True)
class CrossEncoderOutcome:
    result: TurnRerankResult
    elapsed_ms: int
    # Why the input result was returned unchanged (the service failed).
    failure: str | None = None


def passage_text(chunk: RetrievedChunk) -> str:
    """What the cross-encoder reads for a chunk: the document's title, its heading path,
    then its content. The heading carries the table's subject (which programme level a
    fee row is for); the title carries what the chunk itself rarely repeats - the
    organisation, the school year, the system ("Tổng đài 18008000" is only the answer
    for the Ministry's registration system). Measured 29/10/2026 on the demo set: the
    answer chunk reached the top 8 for 154/179 questions with the title, 144 without."""

    title = source_title(chunk.source) if chunk.source else ""
    parts = [f"Tài liệu: {title}" if title else "", " > ".join(chunk.heading_path or [])]
    return "\n".join([*(part for part in parts if part), chunk.content])


async def cross_encoder_rerank(
    client: CrossEncoderClient,
    questions: Sequence[str],
    rerank_result: TurnRerankResult,
    *,
    keep: int | None = None,
    asked: str = "",
) -> CrossEncoderOutcome:
    """`questions[i]` is sub-query i's standalone question, in the order of
    `rerank_result.per_query`. Sub-queries are scored concurrently (one request each).
    `asked` is the user's own message: a person's name the standalone rewrite dropped is
    still read from it (see `named_in_title`)."""

    keep = keep or settings.CHAT_CROSS_ENCODER_KEEP
    started = time.perf_counter()
    per_query_chunks = [list(result.chunks) for result in rerank_result.per_query]
    try:
        scored = await asyncio.gather(
            *(
                _score_query(client, question, chunks)
                for question, chunks in zip(questions, per_query_chunks, strict=True)
            )
        )
    except CrossEncoderUnavailableError as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.warning(
            "cross-encoder rerank failed after %d ms - falling over to the LLM rerank: %s",
            elapsed_ms,
            exc,
        )
        return CrossEncoderOutcome(result=rerank_result, elapsed_ms=elapsed_ms, failure=str(exc))

    per_query_kept: list[list[RetrievedChunk]] = []
    for query_index, (question, chunks, scores) in enumerate(
        zip(questions, per_query_chunks, scored, strict=True)
    ):
        named = named_in_title(f"{question}\n{asked}", chunks)
        order = _cap_expanded(
            sorted(range(len(chunks)), key=lambda i: (not named[i], -scores[i], i)), chunks, keep
        )
        kept = [
            chunks[i].model_copy(
                update={"metadata": {**chunks[i].metadata, CE_SCORE_KEY: scores[i]}}
            )
            for i in order[:keep]
        ]
        per_query_kept.append(kept)
        logger.info(
            "cross-encoder SQ%d %r: kept [%s]; dropped [%s]",
            query_index + 1,
            question,
            "; ".join(_label(chunks[i], scores[i]) for i in order[:keep]),
            "; ".join(_label(chunks[i], scores[i]) for i in order[keep:]),
        )

    narrowed = turn_result(per_query_kept, best_scores=rerank_result.best_scores)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "cross-encoder rerank kept %d of %d chunk(s) in %d ms (keep=%d per sub-query)",
        sum(len(kept) for kept in per_query_kept),
        sum(len(chunks) for chunks in per_query_chunks),
        elapsed_ms,
        keep,
    )
    return CrossEncoderOutcome(result=narrowed, elapsed_ms=elapsed_ms)


# The cross-encoder reads at most 512 tokens of (question + passage) and drops the rest. A
# course-plan chunk is a wide table (17 columns, ~1 600 chars): its header alone fills a third
# of that, so the rows at the end of the chunk were never read - "Anh văn 2" at char 1 382 of
# chunk :7 scored below every other chunk of the programme and was cut (q0100, 04/11/2026).
# A table chunk longer than this is scored as row windows that each repeat the title, heading
# and column header, and the chunk takes its best window's score.
CE_MAX_PASSAGES_PER_REQUEST = 64


def passage_windows(chunk: RetrievedChunk, budget: int | None = None) -> list[str]:
    """`passage_text(chunk)`, or for a long table chunk its row windows (see above)."""

    budget = budget or settings.CHAT_CROSS_ENCODER_WINDOW_CHARS
    whole = passage_text(chunk)
    lines = chunk.content.split("\n")
    first = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|")), None)
    if first is None or len(chunk.content) <= budget:
        return [whole]
    has_separator = first + 1 < len(lines) and set(
        lines[first + 1].replace("|", "").strip()
    ) <= set("-: ")
    body_start = first + 2 if has_separator else first + 1
    rows = [line for line in lines[body_start:] if line.strip()]
    if len(rows) < 2:
        return [whole]
    title = source_title(chunk.source) if chunk.source else ""
    lead = [
        f"Tài liệu: {title}" if title else "",
        " > ".join(chunk.heading_path or []),
        *[line for line in lines[:first] if line.strip()],
        lines[first],
    ]
    prefix = "\n".join(part for part in lead if part)
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for row in rows:
        if current and size + len(row) > budget:
            windows.append("\n".join([prefix, *current]))
            current, size = [], 0
        current.append(row)
        size += len(row) + 1
    if current:
        windows.append("\n".join([prefix, *current]))
    return windows


async def _score_query(
    client: CrossEncoderClient, question: str, chunks: list[RetrievedChunk]
) -> list[float]:
    if not chunks:
        return []
    windows = [passage_windows(chunk) for chunk in chunks]
    flat = [text for group in windows for text in group]
    # the service takes at most RERANKER_MAX_PASSAGES (64) passages per request
    batches = [
        flat[start : start + CE_MAX_PASSAGES_PER_REQUEST]
        for start in range(0, len(flat), CE_MAX_PASSAGES_PER_REQUEST)
    ]
    responses = await asyncio.gather(*(client.rerank(question, batch) for batch in batches))
    flat_scores = [score for response in responses for score in response.scores]
    scores: list[float] = []
    offset = 0
    for group in windows:
        scores.append(max(flat_scores[offset : offset + len(group)]))
        offset += len(group)
    return scores


# A question about one person's document ("portfolio của sinh viên Nguyễn Chí Thanh", "hồ sơ
# của ứng viên Nguyễn Phúc Hùng") competes with the same kind of document of other people: the
# cross-encoder reads the shared template and ranked another student's portfolio first (q0133,
# d0146, 04/11/2026). When the question names a person and some - not all - candidates come
# from a document whose title carries that name (accents ignored: "Nguyen Chi Thanh"), those
# go first, in cross-encoder order.
_SURNAMES = frozenset(
    unicodedata.normalize("NFC", name).casefold()
    for name in "Nguyễn Trần Lê Phạm Hoàng Huỳnh Phan Vũ Võ Đặng Bùi Đỗ Hồ Ngô Dương Lý Đinh "
    "Trương Lương Lâm Mai Tô Hà Đào Cao Châu Trịnh Đoàn Quách Thái Tăng Kiều Lưu Tạ Khúc "
    "Giang Tiêu Thạch Hứa".split()
)
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()


def person_names(question: str) -> list[str]:
    """Folded full names (surname + 2-3 capitalised syllables) the question mentions."""

    words = [match.group() for match in _WORD.finditer(question)]
    names: list[str] = []
    for index, word in enumerate(words):
        if not word[:1].isupper() or unicodedata.normalize("NFC", word).casefold() not in _SURNAMES:
            continue
        given = []
        for follower in words[index + 1 : index + 4]:
            if not follower[:1].isupper():
                break
            given.append(follower)
        parts = [_fold(part) for part in [word, *given]]
        # every reading of 3+ syllables: a capitalised word after the name ("... Hiếu Phó
        # trưởng khoa") may or may not belong to it
        names.extend(" ".join(parts[:size]) for size in range(3, len(parts) + 1))
    return names


def named_in_title(question: str, chunks: Sequence[RetrievedChunk]) -> list[bool]:
    names = person_names(question)
    if not names:
        return [False] * len(chunks)
    titles = [
        " ".join(_WORD.findall(_fold(source_title(c.source) if c.source else ""))) for c in chunks
    ]
    marks = [any(f" {name} " in f" {title} " for name in names) for title in titles]
    return marks if 0 < sum(marks) < len(marks) else [False] * len(chunks)


def _cap_expanded(order: list[int], chunks: Sequence[RetrievedChunk], keep: int) -> list[int]:
    """`order` with at most CHAT_RETRIEVAL_EXPAND_KEEP chunks added by same-document expansion
    (app/rag/retrieval/service.py) in its first `keep`: they are there to supply the one row
    the search missed, not to fill the context with one document and push out another
    source (d0091, 04/11/2026). The ones over the cap move behind the rest."""

    cap = settings.CHAT_RETRIEVAL_EXPAND_KEEP
    head: list[int] = []
    over: list[int] = []
    taken = 0
    for index in order:
        if chunks[index].metadata.get(EXPANDED_FLAG):
            if taken >= cap:
                over.append(index)
                continue
            taken += 1
        head.append(index)
    return head + over


def _label(chunk: RetrievedChunk, ce_score: float) -> str:
    section = f" > {chunk.heading_path[-1]}" if chunk.heading_path else ""
    return f"{source_title(chunk.source)}{section} (ce={ce_score:.2f}, s={chunk.score:.2f})"
