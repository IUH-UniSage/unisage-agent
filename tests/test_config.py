import pytest

from app.core.config import Settings


def test_new_settings_have_sane_defaults() -> None:
    """Env-driven settings for Java integration, clarification, and
    retrieval/rerank load with sane defaults."""

    fresh = Settings()

    assert fresh.BACKEND_JAVA_BASE_URL == "http://localhost:8401/api/v1"
    assert fresh.CHAT_CLARIFICATION_MAX_RETRY == 2
    assert fresh.CHAT_RETRIEVAL_MAX_CHUNKS == 8
    assert fresh.CHAT_RERANK_SCORE_THRESHOLD == 0.70


def test_settings_load_overrides_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_CLARIFICATION_MAX_RETRY", "5")
    monkeypatch.setenv("CHAT_RERANK_SCORE_THRESHOLD", "0.5")

    fresh = Settings()

    assert fresh.CHAT_CLARIFICATION_MAX_RETRY == 5
    assert fresh.CHAT_RERANK_SCORE_THRESHOLD == 0.5
