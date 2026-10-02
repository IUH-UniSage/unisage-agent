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


# ── APP_ENV=production safety checks ────────────────────────────────────


def _prod_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("APP_INTERNAL_SECRET_KEY", "a-real-secret-at-least-32-characters-long")
    monkeypatch.setenv("BACKEND_JAVA_BASE_URL", "https://backend-java.internal/api/v1")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def test_production_default_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _prod_env(monkeypatch, APP_INTERNAL_SECRET_KEY="unisage-internal-secret-key-2026")

    with pytest.raises(ValueError, match="APP_INTERNAL_SECRET_KEY"):
        Settings()


def test_production_short_secret_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _prod_env(monkeypatch, APP_INTERNAL_SECRET_KEY="too-short")

    with pytest.raises(ValueError, match="APP_INTERNAL_SECRET_KEY"):
        Settings()


def test_production_http_without_encrypted_flag_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _prod_env(monkeypatch, BACKEND_JAVA_BASE_URL="http://backend-java.internal/api/v1")

    with pytest.raises(ValueError, match="INTERNAL_NETWORK_ENCRYPTED"):
        Settings()


def test_production_http_with_encrypted_flag_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _prod_env(
        monkeypatch,
        BACKEND_JAVA_BASE_URL="http://backend-java.internal/api/v1",
        INTERNAL_NETWORK_ENCRYPTED="true",
    )

    fresh = Settings()

    assert fresh.INTERNAL_NETWORK_ENCRYPTED is True


@pytest.mark.parametrize("host", ["host.docker.internal", "localhost", "127.0.0.1"])
def test_production_dev_only_host_raises(monkeypatch: pytest.MonkeyPatch, host: str) -> None:
    _prod_env(monkeypatch, BACKEND_JAVA_BASE_URL=f"https://{host}/api/v1")

    with pytest.raises(ValueError, match="dev-only"):
        Settings()


def test_production_valid_config_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _prod_env(monkeypatch)

    fresh = Settings()

    assert fresh.APP_ENV == "production"


def test_development_env_skips_all_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("APP_INTERNAL_SECRET_KEY", "unisage-internal-secret-key-2026")
    monkeypatch.setenv("BACKEND_JAVA_BASE_URL", "http://localhost:8401/api/v1")

    fresh = Settings()

    assert fresh.APP_ENV == "development"


def test_web_search_defaults_are_off_and_domain_restricted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("CHAT_WEB_SEARCH_ENABLED", "TAVILY_API_KEY", "TAVILY_INCLUDE_DOMAINS"):
        monkeypatch.delenv(name, raising=False)

    fresh = Settings(_env_file=None)

    assert fresh.CHAT_WEB_SEARCH_ENABLED is False
    assert fresh.TAVILY_API_KEY == ""
    assert fresh.TAVILY_INCLUDE_DOMAINS == ["iuh.edu.vn"]
    assert fresh.CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN == 4


def test_tavily_include_domains_parse_from_comma_separated_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_INCLUDE_DOMAINS", "iuh.edu.vn, pdt.iuh.edu.vn ,")

    fresh = Settings()

    assert fresh.TAVILY_INCLUDE_DOMAINS == ["iuh.edu.vn", "pdt.iuh.edu.vn"]
