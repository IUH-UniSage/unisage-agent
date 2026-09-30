"""Fixed probe sentences for the embedding identity guard — plan.md "Embedding identity guard".

Embedding a small, fixed set of sentences and comparing the resulting vectors ("fingerprint")
catches a silent provider-side model swap that a `provider`/`modelName`/`dimension` string/int
comparison alone would miss (e.g. a provider renaming or re-serving a model behind the same
name and dimension) — two different models almost never produce near-identical vectors for the
same input, even when they happen to share a dimension.

`PROBE_SENTENCES` must NEVER change once any identity has been established against them. The
registered fingerprint in Java's `embedding_index_identity` table is immutable (no UPDATE/DELETE
path — see plan.md) and is only ever comparable against a fingerprint measured from these exact
sentences, in this exact order. Changing this tuple would silently invalidate every
already-registered collection's fingerprint comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

# 3 short, stable Vietnamese sentences — plan.md doesn't mandate specific text, just "3 câu probe
# cố định". Deliberately generic academic-domain sentences (this is the RAG corpus's domain) with
# no numbers/dates/proper nouns that might tempt a future edit "to make them more realistic".
PROBE_SENTENCES: tuple[str, str, str] = (
    "Sinh viên nộp học phí trước khi đăng ký môn học.",
    "Thư viện của trường mở cửa từ bảy giờ sáng đến chín giờ tối.",
    "Giảng viên chấm bài thi cuối kỳ trong vòng hai tuần.",
)


@dataclass(frozen=True)
class EmbeddingFingerprint:
    """One credential's measured identity — `dimension` is the length of each probe vector,
    `vectors` holds one vector per `PROBE_SENTENCES` entry, same order."""

    dimension: int
    vectors: tuple[tuple[float, ...], ...]

    def flattened(self) -> list[float]:
        """Concatenates all probe vectors into a single flat list — this is the wire shape
        `backend-java` actually stores/exchanges (`InternalEmbeddingIndexIdentityRequest.
        fingerprint`/`...Response.fingerprint` are both a flat `Float[]`, never a nested array),
        in `PROBE_SENTENCES` order."""

        return [value for vector in self.vectors for value in vector]


def measure_fingerprint(embed: Callable[[list[str]], list[list[float]]]) -> EmbeddingFingerprint:
    """Embeds `PROBE_SENTENCES` with `embed` (normally the underlying provider call, injected so
    this stays pure/testable) and returns the fingerprint.

    Raises `ValueError` if the provider returns a different vector count, or vectors of
    inconsistent length across the probes — never silently accepts a malformed response as an
    identity measurement.
    """

    vectors = embed(list(PROBE_SENTENCES))
    if len(vectors) != len(PROBE_SENTENCES):
        raise ValueError(
            f"embedding probe expected {len(PROBE_SENTENCES)} vectors, provider returned "
            f"{len(vectors)}"
        )
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"embedding probe vectors have inconsistent dimensions: {dimensions}")
    return EmbeddingFingerprint(
        dimension=dimensions.pop(),
        vectors=tuple(tuple(float(value) for value in vector) for vector in vectors),
    )


def unflatten_fingerprint(flat: Sequence[float], dimension: int) -> tuple[tuple[float, ...], ...]:
    """Inverse of `EmbeddingFingerprint.flattened()` — splits a flat array (as read back from
    Java) into `len(PROBE_SENTENCES)` vectors of `dimension` values each."""

    values = list(flat)
    expected_len = dimension * len(PROBE_SENTENCES)
    if len(values) != expected_len:
        raise ValueError(
            f"flat fingerprint has {len(values)} values, expected {expected_len} "
            f"({len(PROBE_SENTENCES)} probes x dimension {dimension})"
        )
    return tuple(
        tuple(values[index * dimension : (index + 1) * dimension])
        for index in range(len(PROBE_SENTENCES))
    )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors — `0.0` if either is a zero vector."""

    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def fingerprints_match(
    measured: EmbeddingFingerprint,
    registered: Sequence[Sequence[float]],
    *,
    threshold: float = 0.999,
) -> bool:
    """True if every probe vector in `measured` matches the same-position vector in `registered`
    within `threshold` cosine similarity (plan.md "Embedding identity guard": "khớp danh tính
    index (cùng chiều, cosine mỗi probe >= 0.999)")."""

    if len(measured.vectors) != len(registered):
        return False
    return all(
        cosine_similarity(v1, v2) >= threshold for v1, v2 in zip(measured.vectors, registered)
    )
