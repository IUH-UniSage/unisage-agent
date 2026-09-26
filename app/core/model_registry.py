"""In-memory snapshot of `backend-java`'s model registry (plan.md "Internal API
contract" endpoint #1, "Cutover khỏi cấu hình `.env` tĩnh").

Loaded once at FastAPI startup (`app.main.lifespan`) and once per Celery worker
process (`app.worker.celery_app`'s `worker_process_init` handler) — Task 4 does
not wire hot-reload/polling (that's Task 7), so within a process this snapshot
never changes after the initial load.

Every credential's `api_key` is plaintext, straight off the wire. This module
never logs the raw snapshot or a credential — `CredentialConfig.api_key` is
`repr=False` so neither `repr()` nor the default `str()` (which falls back to
`repr()` for a dataclass with no explicit `__str__`) ever prints it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.config import settings
from app.integrations.backend_java_client import BackendJavaClient

logger = logging.getLogger(__name__)


class ModelRegistryError(Exception):
    """Raised when the registry snapshot can't be loaded, or is missing something startup
    requires."""


@dataclass(frozen=True)
class CredentialConfig:
    """One ACTIVE credential row for one purpose — mirrors plan.md's snapshot JSON
    `CredentialEntry` shape exactly (field names/types match `InternalModelRegistrySnapshotResponse`
    on the Java side) so whoever wires Task 5 (`get_graph_models()`) and Task 6/9's health-report
    call has everything they need already named sensibly: `provider`/`model_name`/`api_base_url`/
    `api_key`/`priority` per purpose, and `revision` to attach to health reports.
    """

    id: str
    revision: int
    source_type: str
    provider: str | None
    model_name: str | None
    api_base_url: str | None
    priority: int | None
    max_rpm: int | None
    api_key: str = field(repr=False)


@dataclass(frozen=True)
class EmbeddingIndexIdentity:
    """Identity of the vector space currently in the configured Qdrant collection — plan.md
    "Embedding identity guard". `None` on the snapshot means no identity has been established yet.
    """

    provider: str | None
    model_name: str | None
    dimension: int | None
    fingerprint: tuple[tuple[float, ...], ...] | None


@dataclass(frozen=True)
class ModelRegistrySnapshot:
    """The full registry snapshot for one process's lifetime. `version` is the registry's own
    monotonic counter (plan.md "Hot-reload consistency") — carried here, alongside each
    credential's own `revision`, so a later task can attach `credentialRevision`/`snapshotVersion`
    to a health report without having to re-fetch anything.
    """

    version: int
    generated_at: datetime
    purposes: dict[str, tuple[CredentialConfig, ...]]
    embedding_index_identity: EmbeddingIndexIdentity | None

    def credentials_for(self, purpose: str) -> tuple[CredentialConfig, ...]:
        return self.purposes.get(purpose, ())


def _parse_credential(raw: dict[str, Any]) -> CredentialConfig:
    return CredentialConfig(
        id=str(raw["id"]),
        revision=int(raw["revision"]),
        source_type=str(raw["sourceType"]),
        provider=raw.get("provider"),
        model_name=raw.get("modelName"),
        api_base_url=raw.get("apiBaseUrl"),
        priority=raw.get("priority"),
        max_rpm=raw.get("maxRpm"),
        api_key=raw.get("apiKey") or "",
    )


def _parse_identity(raw: dict[str, Any] | None) -> EmbeddingIndexIdentity | None:
    if raw is None:
        return None
    fingerprint_raw = raw.get("fingerprint")
    fingerprint = (
        tuple(tuple(vector) for vector in fingerprint_raw) if fingerprint_raw is not None else None
    )
    return EmbeddingIndexIdentity(
        provider=raw.get("provider"),
        model_name=raw.get("modelName"),
        dimension=raw.get("dimension"),
        fingerprint=fingerprint,
    )


def parse_snapshot(payload: dict[str, Any]) -> ModelRegistrySnapshot:
    """Pure parse of the raw `GET /snapshot` JSON — no I/O, no startup-gate check, so it's cheap
    to unit test against a hand-built payload."""

    purposes_raw = payload.get("purposes") or {}
    purposes: dict[str, tuple[CredentialConfig, ...]] = {
        purpose: tuple(_parse_credential(c) for c in credentials)
        for purpose, credentials in purposes_raw.items()
    }
    return ModelRegistrySnapshot(
        version=int(payload["version"]),
        generated_at=datetime.fromisoformat(str(payload["generatedAt"]).replace("Z", "+00:00")),
        purposes=purposes,
        embedding_index_identity=_parse_identity(payload.get("embeddingIndexIdentity")),
    )


# Process-local cache — one load per process lifetime (Task 4 scope; Task 7 adds hot-reload).
_current_snapshot: ModelRegistrySnapshot | None = None


def get_current_snapshot() -> ModelRegistrySnapshot | None:
    """`None` means either the registry was never loaded (feature flag off) or this is called
    before `init_model_registry()` has run."""

    return _current_snapshot


def active_credentials_for(purpose: str) -> tuple[CredentialConfig, ...]:
    """ACTIVE credentials for one purpose from the current snapshot - empty tuple if no
    snapshot has been loaded yet (registry disabled, or called before startup's
    `init_model_registry()` ran)."""

    snapshot = get_current_snapshot()
    return snapshot.credentials_for(purpose) if snapshot else ()


def require_top_priority_credential(purpose: str) -> CredentialConfig:
    """The highest-priority (lowest `priority` number, `None` sorts last) ACTIVE credential for
    `purpose`, or raise `ModelRegistryError` if none exists.

    Never a `.env` fallback (plan.md "Cutover khỏi cấu hình `.env` tĩnh") - every provider call
    site (`get_graph_models()`, `OpenAIEmbedder`, `MultiRepresentationEnricher`) shares this one
    function instead of each re-implementing "pick a credential or fail loudly". No auto-failover
    here (Task 5's explicit scope for CHAT; EMBEDDING never gets one at all per plan.md - the
    unique partial index guarantees at most one ACTIVE EMBEDDING row exists, so "top priority"
    picking among one row is a no-op there) - just the single top-priority pick.
    """

    credentials = active_credentials_for(purpose)
    if not credentials:
        raise ModelRegistryError(
            f"Model registry has no ACTIVE {purpose} credential - refusing to fall back to "
            "any static .env credential."
        )
    return min(
        credentials, key=lambda credential: (credential.priority is None, credential.priority)
    )


def set_current_snapshot(snapshot: ModelRegistrySnapshot) -> None:
    """The hot-reload swap (plan.md "Hot-reload consistency", Task 8) — a single reference
    reassignment of the module-level global, nothing more.

    This is the whole trick: `ModelRegistrySnapshot` is `frozen`, and CPython's GIL makes a
    single name rebind atomic, so a request that already called `get_current_snapshot()` and
    is holding the old object keeps seeing consistent (if stale) data for the rest of its
    lifetime — it never observes a half-updated snapshot, and it is never mutated out from
    under it. The caller (`app.core.registry_subscriber`) is responsible for only calling this
    with a snapshot whose `version` is newer than the current one and for serializing calls
    (its own lock) so two concurrent reloads don't race pointlessly; this function itself does
    no version check and no locking — it is deliberately just the swap.
    """

    global _current_snapshot
    _current_snapshot = snapshot


async def init_model_registry(
    client: BackendJavaClient | None = None,
) -> ModelRegistrySnapshot | None:
    """Load the registry snapshot once — call this from FastAPI's lifespan and from Celery's
    `worker_process_init` (see `app.worker.celery_app`), never from a request/task handler.

    There is no `.env`-based credential path left anywhere in this codebase (plan.md "Cutover
    khỏi cấu hình `.env` tĩnh") — `MODEL_REGISTRY_ENABLED` only controls whether THIS function
    fetches a snapshot at all. Returns `None` and does nothing when it's `false` (meant for a
    process that's deliberately started without one, e.g. most unit tests). When it's `true`
    (the default), this raises `ModelRegistryError` — and the caller is expected to let that
    fail startup — if the snapshot has no ACTIVE CHAT credential.
    """

    global _current_snapshot

    if not settings.MODEL_REGISTRY_ENABLED:
        logger.info("MODEL_REGISTRY_ENABLED=false - staying on static .env model credentials")
        return None

    owned_client = client is None
    client = client or BackendJavaClient()
    try:
        payload = await client.get_model_registry_snapshot()
    finally:
        del owned_client  # BackendJavaClient holds no persistent connection to close between calls.

    snapshot = parse_snapshot(payload)
    if not snapshot.credentials_for("CHAT"):
        raise ModelRegistryError(
            "Model registry snapshot has no ACTIVE CHAT credential - refusing to start. "
            "Set MODEL_REGISTRY_ENABLED=false to fall back to static .env credentials instead."
        )

    _current_snapshot = snapshot
    logger.info(
        "Loaded model registry snapshot version=%s chat=%d embedding=%d extraction=%d",
        snapshot.version,
        len(snapshot.credentials_for("CHAT")),
        len(snapshot.credentials_for("EMBEDDING")),
        len(snapshot.credentials_for("EXTRACTION")),
    )
    return snapshot
