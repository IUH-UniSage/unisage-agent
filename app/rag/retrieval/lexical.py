"""Lexical (BM25) side of hybrid retrieval - docs/specs/SPEC-hybrid-retrieval.md.

Dense embeddings treat a course code ("2105618"), a decision number ("1035") or a cohort
label ("K18") as near-noise; a BM25 sparse vector keeps them as exact terms. The sparse
vector here is only term frequencies: Qdrant applies IDF itself (`Modifier.IDF` on the
`bm25` sparse vector), so the same function serves a chunk at ingest/backfill time and a
question at query time. No model, no dependency.

Vietnamese is tokenized by syllable (runs of letters/digits): what has to match is a code,
a number or a proper name, not a compound word.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter
from collections.abc import Hashable, Sequence

# Name of the sparse vector on the chunk collection (`ensure_collection`).
BM25_VECTOR = "bm25"
# Reciprocal rank fusion constant (`CHAT_RETRIEVAL_RRF_K` overrides). The textbook 60 was
# tuned for fusing many comparable runs; here, fusing two lists before a reranker, it let any
# chunk present in both lists outrank a BM25 rank-1 hit missing from dense (measured
# 21/10/2026, see config). With 1 a rank-1 hit scores 1/2 and only chunks near the top of
# both lists pass it.
RRF_K_DEFAULT = 1

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """NFC (composed and decomposed Vietnamese must match), casefold, split on anything that
    is not a letter or digit. One-character tokens stay only when they are digits, so "a"
    and "-" vanish but a course code or a year never does."""

    folded = unicodedata.normalize("NFC", text).casefold()
    return [token for token in _TOKEN.findall(folded) if len(token) > 1 or token.isdigit()]


def lexical_text(content: str, heading_path: Sequence[str] = ()) -> str:
    """What a chunk contributes to BM25: its heading path (course codes and section names
    often live only there) followed by its content."""

    headings = " ".join(h for h in heading_path if h)
    return f"{headings}\n{content}" if headings else content


# BM25 term-frequency saturation (Robertson k1, b) and the average chunk length in tokens
# the length normalization is relative to. Qdrant's `Modifier.IDF` supplies the idf; what we
# store is the rest of the BM25 term weight, so a chunk that repeats "học phần" fifty times
# (a course table) scores at most ~k1+1 for that term, not fifty.
BM25_K1 = 1.2
BM25_B = 0.75
# Measured on the eval corpus (2 323 chunks, 21/10/2026): mean 165, median 135, p90 323 tokens.
BM25_AVG_LENGTH = 165.0

# Vietnamese function and question words dropped from the QUESTION side only. Measured on the
# eval corpus: "học phần 2107555 có bao nhiêu tín chỉ?" put the one chunk holding the code at
# BM25 rank 24 because "có", "bao", "nhiêu" alone carry ~9 idf points - more than the code -
# and regulation chunks repeat them. Chunks keep every term (nothing to rebuild when this list
# changes); the dense side still reads the question whole. Codes, numbers, "điều", "khoản",
# "mục" and the like are never here.
QUERY_STOPWORDS = frozenset(
    """
    ai ạ bao bạn bị biết bởi cái các cần cho chưa có còn cũng của cứ do đã đang đâu đây để đến
    đều đó được em gì giúp hay hỏi hoặc khi không là làm lại mà mình mới một nào này nên nếu
    những như nhiêu nhỉ ơi phải rằng rồi ra sao sẽ tại thế thì tôi từ và vào vậy về với xin
    """.split()
)


def sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """A chunk's `bm25` vector, `(indices, values)`: index = CRC32 of the token (u32, a
    collision between two real tokens is negligible), value = the saturated,
    length-normalized BM25 term frequency `tf (k1 + 1) / (tf + k1 (1 - b + b len/avg))`.
    Sorted by index; empty text gives two empty lists."""

    tokens = tokenize(text)
    counts = Counter(tokens)
    if not counts:
        return [], []
    length_norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(tokens) / BM25_AVG_LENGTH)
    pairs = sorted(
        (
            zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF,
            n * (BM25_K1 + 1.0) / (n + length_norm),
        )
        for token, n in counts.items()
    )
    return [index for index, _ in pairs], [value for _, value in pairs]


def query_tokens(text: str) -> list[str]:
    """`tokenize` minus `QUERY_STOPWORDS`; what a question asks BM25 for."""

    return [token for token in tokenize(text) if token not in QUERY_STOPWORDS]


# Query weight of a token that carries a digit (course code, decision number, cohort "k18",
# article "5"). Measured on the eval corpus: even after stopwords, "học phần <code> tín chỉ"
# scores a regulation chunk holding "học", "phần", "tín", "chỉ" at ~13.6 and the one chunk
# holding the code at ~10 (idf 7.35 x saturated tf 0.86, plus "học"), because four mid-idf
# words beat one rare one. Identifiers are what this vector exists for, so they count x3.
IDENTIFIER_WEIGHT = 3.0


def sparse_query_vector(text: str) -> tuple[list[int], list[float]]:
    """A question's side of the BM25 dot product: one weight per distinct non-stopword term -
    1.0 for a word, `IDENTIFIER_WEIGHT` for a token with a digit. Qdrant multiplies each by the
    term's idf, so a rare code counts for far more than "học phần"."""

    weights: dict[int, float] = {}
    for token in query_tokens(text):
        index = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
        weight = IDENTIFIER_WEIGHT if any(ch.isdigit() for ch in token) else 1.0
        weights[index] = max(weights.get(index, 0.0), weight)
    indices = sorted(weights)
    return indices, [weights[index] for index in indices]


def rrf_merge(
    ranked_lists: Sequence[Sequence[Hashable]], *, k: int = RRF_K_DEFAULT
) -> list[tuple[Hashable, float]]:
    """Reciprocal rank fusion of several ranked id lists: score(id) = sum over lists of
    1 / (k + rank), rank starting at 1. Ids present in more lists, and higher up, win.
    Ties keep first-appearance order, so the result is deterministic."""

    scores: dict[Hashable, float] = {}
    order: dict[Hashable, int] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            order.setdefault(item, len(order))
    return sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))
