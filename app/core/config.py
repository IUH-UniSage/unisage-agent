from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_INTERNAL_SECRET = "unisage-internal-secret-key-2026"
_MIN_INTERNAL_SECRET_LENGTH = 32
_DEV_ONLY_HOSTS = {"host.docker.internal", "localhost", "127.0.0.1"}


class Settings(BaseSettings):
    """Env vars, grouped by prefix - see `docs/guide/cau-hinh.md`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- APP_: process-level (name, environment, debug logging, the
    # Gateway-shared secret that gates every inbound request) ---
    APP_NAME: str = "UniSage AI Agent Service"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True

    # Internal service-to-service auth: must match the shared secret the API
    # Gateway sends as `X-Internal-Secret` on every proxied request (see
    # api-gateway's `python-ai-agent-route` filters in application.yml).
    # Default matches that route's own fallback so local dev works
    # out-of-the-box; override both sides together in real deployments.
    APP_INTERNAL_SECRET_KEY: str = "unisage-internal-secret-key-2026"

    # --- DB_: this service's own Postgres database ---
    # Database: same physical Postgres server as backend-java (see
    # backend-java/docker-compose.yml's `unisage-db` service, port 5433 on
    # host), but a separate database (`unisage_agent_db`, not Java's
    # `assistant_DB`) - per SPEC-ingestion-resume.md, no shared
    # schema/tables, no cross-service FK.
    DB_URL: str = "postgresql+asyncpg://postgres:123456@localhost:5433/unisage_agent_db"

    # --- OPENAI_: the LLM/embedding provider. Shared by ingestion (chunk
    # enrichment, embedding) and chat (classification/HyDE/generation) - kept
    # under its own vendor prefix rather than forced into CHAT_/INGEST_. ---
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # --- MINIO_: object storage for ingested source files ---
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "unisage-documents"
    MINIO_SECURE: bool = False

    # --- Single-var external systems: naming the system is enough on its
    # own, a prefix group of one adds nothing ---
    # DB 0 only - registry/circuit-breaker/lock/event use (key prefix `mr:`).
    # Celery's own broker/backend use their own DBs below - see plan.md's "Hot-reload
    # consistency": the three were sharing this one DB, which meant a Celery `purge`
    # or the registry's key sweep could clobber each other's keys.
    REDIS_URL: str = "redis://localhost:6379/0"
    BACKEND_JAVA_BASE_URL: str = "http://localhost:8401/api/v1"
    # Set true only when Python <-> Java is actually TLS/mTLS or an encrypted private network.
    INTERNAL_NETWORK_ENCRYPTED: bool = False
    # Rollout flag (plan.md "Cutover khỏi cấu hình .env tĩnh"): false keeps the old
    # OPENAI_*-from-.env path alive; true means the model registry snapshot from Java is the
    # only source of provider credentials, and startup fails loudly if it has no ACTIVE CHAT
    # credential. Flip once the registry is seeded and stable in an environment - never both
    # at once, there is no "read registry, fall back to .env" middle state.
    MODEL_REGISTRY_ENABLED: bool = False

    # --- CELERY_: Celery's own broker/result-backend, isolated from REDIS_URL's
    # DB 0 (see above) so a broker purge/flush never touches registry state and
    # vice versa. DB 1/2 by convention, not enforced - point these at whatever DB
    # you like, just not DB 0. ---
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    # Prefix for every Celery queue name this service declares. The integration
    # harness (tests/e2e/) sets a fresh prefix per run (e.g. `it-<uuid8>`) so its
    # fixture can `celery purge -Q <prefix>-<queue>` without ever touching another
    # run's or another service's queue.
    CELERY_QUEUE_PREFIX: str = "unisage"
    # Redis pub/sub channel the Java side publishes `{"version": N}` to after a
    # registry-affecting commit (plan.md "Hot-reload consistency" - "Publish after
    # commit"). Pub/sub isn't namespaced by DB, so the channel name itself is what
    # separates one harness run's Java from another's.
    MODEL_REGISTRY_CHANNEL: str = "model-registry:updates"
    # How often Celery Beat's own heartbeat task runs - the one thing this task's
    # integration harness needs Beat to visibly do before Task 8 gives it a real
    # verify-poll schedule to run. The harness's integration profile overrides this
    # to a couple seconds so `test_model_registry_smoke.py` doesn't wait 15s+ for
    # a tick.
    CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS: int = 15

    # --- QDRANT_: the vector store ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "unisage_chunks"

    # --- INGEST_: only ever read at document-ingestion time (chunking,
    # enrichment) - never during a chat turn ---
    INGEST_MULTI_REP_LLM_MODEL: str = "gpt-4o-mini"
    INGEST_MULTI_REP_QUESTION_COUNT: int = 3
    INGEST_SEMANTIC_MAX_TOKEN_FACTOR: float = 1.5
    INGEST_TABLE_CHUNK_MAX_TOKENS: int = 800
    INGEST_CHUNKING_VERSION: str = "2026-09-structural-v2"

    # --- CHAT_: read on every chat turn ---
    CHAT_CLARIFICATION_MAX_RETRY: int = 2
    CHAT_RETRIEVAL_MAX_CHUNKS: int = 8
    CHAT_RERANK_SCORE_THRESHOLD: float = 0.70
    CHAT_HISTORY_MESSAGE_LIMIT: int = 15
    # Max sub-queries the decomposer may split one comparison question into.
    CHAT_MAX_SUB_QUERIES: int = Field(default=3, ge=2)

    # GenerationSynthesisNode's JSON-repair follow-up call (see
    # generation_synthesis.py::_repair_missing_ask_form) - a cheap regex
    # heuristic fires a second LLM call when a response reads like it forgot
    # the mandatory ```json ask_user_form``` block. The heuristic has a known
    # false-positive gap (a closing offer phrased "..., nếu bạn cần..." -
    # condition trailing, not leading - isn't recognized as non-committal),
    # which can make the repair call hallucinate a field nobody asked about.
    # Off (false) skips the repair call entirely: a real missed form goes
    # unfixed, but no spurious one is ever hallucinated. Defaults on to keep
    # existing behavior; flip off in .env if the false positives outweigh
    # the (rare) real misses it exists to catch.
    CHAT_ALLOW_REPAIR_JSON: bool = True

    @model_validator(mode="after")
    def _validate_production_safety(self) -> "Settings":
        if self.APP_ENV != "production":
            return self

        if (
            not self.APP_INTERNAL_SECRET_KEY
            or self.APP_INTERNAL_SECRET_KEY == _DEFAULT_INTERNAL_SECRET
            or len(self.APP_INTERNAL_SECRET_KEY) < _MIN_INTERNAL_SECRET_LENGTH
        ):
            raise ValueError(
                f"APP_INTERNAL_SECRET_KEY must be a non-default value with at least "
                f"{_MIN_INTERNAL_SECRET_LENGTH} characters when APP_ENV=production"
            )

        parsed = urlsplit(self.BACKEND_JAVA_BASE_URL)
        if parsed.scheme == "http" and not self.INTERNAL_NETWORK_ENCRYPTED:
            raise ValueError(
                "BACKEND_JAVA_BASE_URL is http:// but INTERNAL_NETWORK_ENCRYPTED is not "
                "true; production requires TLS/mTLS or an encrypted private network"
            )
        if parsed.hostname in _DEV_ONLY_HOSTS:
            raise ValueError(
                f"BACKEND_JAVA_BASE_URL host '{parsed.hostname}' is dev-only; "
                "production must point at a real internal service name"
            )
        return self


settings = Settings()
