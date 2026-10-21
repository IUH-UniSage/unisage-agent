"""The BM25 side of hybrid retrieval (docs/specs/SPEC-hybrid-retrieval.md): tokenizer,
sparse vector and reciprocal rank fusion - all pure, no Qdrant."""

from __future__ import annotations

import unicodedata

from app.rag.retrieval.lexical import (
    BM25_VECTOR,
    IDENTIFIER_WEIGHT,
    lexical_text,
    query_tokens,
    rrf_merge,
    sparse_query_vector,
    sparse_vector,
    tokenize,
)


def test_tokenize_keeps_codes_numbers_and_syllables_drops_punctuation() -> None:
    tokens = tokenize("Học phần 2105618 (mã) - K18, năm 2025-2026.")
    assert tokens == ["học", "phần", "2105618", "mã", "k18", "năm", "2025", "2026"]
    assert "(" not in tokens and "-" not in tokens


def test_tokenize_is_case_and_normalization_insensitive() -> None:
    composed = "Quyết định 1035/QĐ-ĐHCN"
    decomposed = unicodedata.normalize("NFD", composed)
    assert tokenize(composed) == tokenize(decomposed) == tokenize(composed.upper())
    assert tokenize(composed) == ["quyết", "định", "1035", "qđ", "đhcn"]


def test_tokenize_drops_single_letters_but_not_single_digits() -> None:
    assert tokenize("a b 5 x y") == ["5"]
    assert tokenize("") == []


def test_sparse_vector_saturates_term_frequency_with_stable_indices() -> None:
    indices, values = sparse_vector("học học phí 2105618")
    assert len(indices) == 3 and len(values) == 3
    assert indices == sorted(indices)
    assert all(0 <= index <= 0xFFFFFFFF for index in indices)
    by_index = dict(zip(indices, values, strict=True))
    (học,) = sparse_query_vector("học")[0]
    (phí,) = sparse_query_vector("phí")[0]
    # tf 2 weighs more than tf 1, but far less than twice: BM25 saturation.
    assert by_index[phí] < by_index[học] < 2 * by_index[phí]
    assert sparse_vector("học học phí 2105618") == sparse_vector("HỌC học phí, 2105618!")
    assert sparse_vector("   ") == ([], [])


def test_a_term_repeated_fifty_times_scores_near_k1_plus_1_not_fifty() -> None:
    (_index,), (value,) = sparse_vector(" ".join(["học"] * 50))
    assert 1.5 < value < 2.2  # k1 + 1 = 2.2 is the ceiling


def test_sparse_query_vector_is_one_per_distinct_term_and_weights_identifiers() -> None:
    indices, values = sparse_query_vector("học phần 2105618 học phần?")
    assert len(indices) == 3
    assert indices == sorted(indices)
    by_index = dict(zip(indices, values, strict=True))
    (code,) = sparse_query_vector("2105618")[0]
    assert by_index.pop(code) == IDENTIFIER_WEIGHT
    assert set(by_index.values()) == {1.0}
    assert sparse_query_vector("K18 điều 5")[1] == [
        IDENTIFIER_WEIGHT,
        IDENTIFIER_WEIGHT,
        1.0,
    ] or sorted(sparse_query_vector("K18 điều 5")[1]) == [1.0, IDENTIFIER_WEIGHT, IDENTIFIER_WEIGHT]
    assert sparse_query_vector("") == ([], [])


def test_the_question_drops_vietnamese_function_words_but_chunks_keep_them() -> None:
    question = "Học phần 2107555 có bao nhiêu tín chỉ?"
    assert query_tokens(question) == ["học", "phần", "2107555", "tín", "chỉ"]
    assert "có" in tokenize(question)
    # Only stopwords -> no sparse request at all; codes and article numbers always survive.
    assert sparse_query_vector("có được không ạ?") == ([], [])
    assert query_tokens("điều 5 khoản 2 QĐ 1035") == ["điều", "5", "khoản", "2", "qđ", "1035"]
    # The chunk side is untouched: its vector still carries "có".
    (có,) = sparse_query_vector("có")[0] if sparse_query_vector("có")[0] else (None,)
    assert có is None
    assert len(sparse_vector("có 3 tín chỉ")[0]) == 4


def test_lexical_text_prefixes_the_heading_path() -> None:
    assert lexical_text("nội dung", ["Chương 2", "Mục 2105618"]) == "Chương 2 Mục 2105618\nnội dung"
    assert lexical_text("nội dung", []) == "nội dung"
    assert lexical_text("nội dung", ["", None]) == "nội dung"  # type: ignore[list-item]


def test_rrf_merge_rewards_presence_in_several_lists() -> None:
    fused = rrf_merge([["x", "y", "z"], ["y", "x"], ["q"]], k=60)
    ids = [item for item, _ in fused]
    # x: 1/61 + 1/62 ; y: 1/62 + 1/61 -> equal, x first by first appearance; q rank-1 of its
    # list beats z (rank 3 of one list).
    assert ids == ["x", "y", "q", "z"]
    scores = dict(fused)
    assert abs(scores["x"] - (1 / 61 + 1 / 62)) < 1e-12
    assert scores["q"] == 1 / 61


def test_rrf_merge_handles_empty_input() -> None:
    assert rrf_merge([]) == []
    assert rrf_merge([[], []]) == []


def test_sparse_vector_name() -> None:
    assert BM25_VECTOR == "bm25"
