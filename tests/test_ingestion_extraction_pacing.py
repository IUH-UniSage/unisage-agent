from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.worker.tasks.ingestion as ingestion
from app.core.config import settings
from app.core.errors.llm_error_classifier import ErrorType
from app.core.registry.errors import NoAvailableCredentialError
from app.schemas.ingestion import Chunk, RegionType


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Records every `asyncio.sleep` the ingest helpers ask for, without actually waiting."""

    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("app.worker.tasks.ingestion.asyncio.sleep", fake_sleep)
    return recorded


def _settings(monkeypatch: pytest.MonkeyPatch, **values: float) -> None:
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)


def _chunk() -> Chunk:
    return Chunk(chunk_index=0, content="Some chunk content.", region_type=RegionType.TEXT)


def _clock(monkeypatch: pytest.MonkeyPatch, now: float) -> None:
    monkeypatch.setattr(ingestion, "time", SimpleNamespace(monotonic=lambda: now))


# ── pacing between extraction calls ──────────────────────


@pytest.mark.asyncio
async def test_first_extraction_is_never_paused(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MIN_INTERVAL_SECONDS=5.0)
    _clock(monkeypatch, 100.0)

    started = await ingestion._pause_between_extractions(None)

    assert started == 100.0
    assert sleeps == []


@pytest.mark.asyncio
async def test_pause_waits_only_for_the_rest_of_the_interval(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MIN_INTERVAL_SECONDS=5.0)
    _clock(monkeypatch, 102.0)

    started = await ingestion._pause_between_extractions(100.0)

    assert sleeps == [3.0]
    assert started == 102.0


@pytest.mark.asyncio
async def test_pause_skipped_when_previous_call_already_took_the_interval(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MIN_INTERVAL_SECONDS=5.0)
    _clock(monkeypatch, 107.0)

    await ingestion._pause_between_extractions(100.0)

    assert sleeps == []


@pytest.mark.asyncio
async def test_pause_is_off_when_interval_is_zero(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MIN_INTERVAL_SECONDS=0.0)
    _clock(monkeypatch, 100.0)

    await ingestion._pause_between_extractions(100.0)

    assert sleeps == []


# ── waiting for a credential instead of failing the job ──


def _enricher(*outcomes: object) -> MagicMock:
    enricher = MagicMock()
    enricher.enrich_tracked = AsyncMock(side_effect=list(outcomes))
    return enricher


def _transient_exhaustion() -> NoAvailableCredentialError:
    return NoAvailableCredentialError("EXTRACTION", last_error=RuntimeError("503 unavailable"))


async def _enrich(enricher: MagicMock) -> object:
    return await ingestion._enrich_waiting_for_credentials(
        enricher, _chunk(), MagicMock(), MagicMock()
    )


@pytest.mark.asyncio
async def test_waits_then_retries_the_same_chunk_when_credentials_cool_down(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(
        monkeypatch,
        INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS=35.0,
        INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS=6,
    )
    enriched = object()
    enricher = _enricher(_transient_exhaustion(), _transient_exhaustion(), enriched)

    assert await _enrich(enricher) is enriched

    assert sleeps == [35.0, 35.0]
    assert enricher.enrich_tracked.await_count == 3


@pytest.mark.asyncio
async def test_gives_up_after_the_configured_number_of_waits(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(
        monkeypatch,
        INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS=35.0,
        INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS=2,
    )
    enricher = _enricher(*(_transient_exhaustion() for _ in range(5)))

    with pytest.raises(NoAvailableCredentialError):
        await _enrich(enricher)

    assert sleeps == [35.0, 35.0]
    assert enricher.enrich_tracked.await_count == 3


@pytest.mark.asyncio
async def test_does_not_wait_when_waits_are_disabled(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS=0)
    enricher = _enricher(_transient_exhaustion())

    with pytest.raises(NoAvailableCredentialError):
        await _enrich(enricher)

    assert sleeps == []
    assert enricher.enrich_tracked.await_count == 1


@pytest.mark.asyncio
async def test_does_not_wait_for_a_permanent_failure(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS=6)
    monkeypatch.setattr(ingestion, "classify_llm_error", lambda _exc: ErrorType.PERMANENT)
    enricher = _enricher(_transient_exhaustion())

    with pytest.raises(NoAvailableCredentialError):
        await _enrich(enricher)

    assert sleeps == []
    assert enricher.enrich_tracked.await_count == 1


@pytest.mark.asyncio
async def test_does_not_wait_when_no_credential_is_configured_at_all(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    _settings(monkeypatch, INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS=6)
    enricher = _enricher(NoAvailableCredentialError("EXTRACTION"))

    with pytest.raises(NoAvailableCredentialError):
        await _enrich(enricher)

    assert sleeps == []
    assert enricher.enrich_tracked.await_count == 1


@pytest.mark.asyncio
async def test_waits_when_credentials_are_suspended_from_an_earlier_chunk(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    """The next chunk starts after every key was put on cooldown: no failure of its own,
    only the stored suspension reasons."""

    _settings(
        monkeypatch,
        INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS=35.0,
        INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS=6,
    )
    suspended = NoAvailableCredentialError("EXTRACTION", suspension_reasons=("LLM_UNAVAILABLE",))
    enriched = object()
    enricher = _enricher(suspended, enriched)

    assert await _enrich(enricher) is enriched

    assert sleeps == [35.0]
