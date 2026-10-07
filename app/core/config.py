from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

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

    # --- MINIO_: object storage for ingested source files ---
    MINIO_ENDPOINT: str = "localhost:9100"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "unisage-documents"
    MINIO_SECURE: bool = False

    # --- Single-var external systems: naming the system is enough on its
    # own, a prefix group of one adds nothing ---
    # DB 0 only - registry/circuit-breaker/lock/event use (key prefix `mr:`).
    # Celery's own broker/backend use their own DBs below - the three used to share
    # this one DB, which meant a Celery `purge` or the registry's key sweep could
    # clobber each other's keys.
    REDIS_URL: str = "redis://localhost:6379/0"
    BACKEND_JAVA_BASE_URL: str = "http://localhost:8401/api/v1"
    # Slack Incoming Webhook URL for operational alerts. Empty by default -
    # not every environment has Slack configured, and that's a normal,
    # expected state rather than a misconfiguration.
    SLACK_APIKEY_ALERT_WEBHOOK_URL: str = ""
    # Set true only when Python <-> Java is actually TLS/mTLS or an encrypted private network.
    INTERNAL_NETWORK_ENCRYPTED: bool = False
    # Rollout flag: the model registry snapshot
    # from Java is now the only source of provider credentials - there is no `.env` fallback
    # path left in the code to fall back to. Startup fails loudly if the snapshot has no
    # ACTIVE CHAT credential. Kept as a flag (rather than deleted outright) only so a process
    # can be started with the registry deliberately not loaded (e.g. a unit-test process, or
    # a deploy step that hasn't seeded credentials yet) - flipping it off does not resurrect
    # any static-credential behavior, it just means every provider call site raises.
    MODEL_REGISTRY_ENABLED: bool = True
    # Hot-reload poll fallback: every process
    # independently re-checks `/internal/model-registry/version` on this interval regardless of
    # whether Redis pub/sub is connected or a message was dropped, so it self-heals within this
    # many seconds no matter what. The integration harness may shorten this the same way it
    # shortens CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS.
    MODEL_REGISTRY_POLL_INTERVAL_SECONDS: float = 30.0

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
    # registry-affecting commit. Pub/sub isn't namespaced by DB, so the channel name
    # itself is what separates one harness run's Java from another's.
    MODEL_REGISTRY_CHANNEL: str = "model-registry:updates"
    # How often Celery Beat's own heartbeat task runs - the one thing this task's
    # integration harness needs Beat to visibly do before it has a real verify-poll
    # schedule to run. The harness's integration profile overrides this
    # to a couple seconds so `test_model_registry_smoke.py` doesn't wait 15s+ for
    # a tick.
    CELERY_BEAT_HEARTBEAT_INTERVAL_SECONDS: int = 15
    # Verify-before-active claim loop: how often Beat wakes
    # it up on its own, independent of the verification-requested pub/sub nudge below - this is
    # the backstop that guarantees a queued job eventually gets claimed even if every publish is
    # missed. The integration harness may shorten this the same way it shortens the heartbeat.
    MODEL_REGISTRY_VERIFICATION_INTERVAL_SECONDS: float = 15.0
    # Channel backend-java publishes to right after committing a new/superseded verification job -
    # distinct from
    # MODEL_REGISTRY_CHANNEL (config/version changes), since this one only ever means "there may
    # be a job to claim", never carries a version to compare.
    MODEL_REGISTRY_VERIFICATION_CHANNEL: str = "model-registry:verification-requested"
    # Max jobs claimed per run of the verify loop - small on purpose, since each claimed job
    # makes one live provider call (up to 15s) sequentially before the next.
    MODEL_REGISTRY_VERIFICATION_CLAIM_LIMIT: int = 5

    # --- QDRANT_: the vector store ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "unisage_chunks"

    # --- TAVILY_: web search API used by WebSearchNode when retrieval finds nothing for a
    # sub-query (see app/graph/nodes/web_search.py). Empty key = web search is skipped. ---
    TAVILY_API_KEY: str = ""
    TAVILY_BASE_URL: str = "https://api.tavily.com"
    # Comma-separated in .env. Only these sites (and their subdomains) are searched - the
    # answer must come from the university's own pages, never a forum or another school.
    TAVILY_INCLUDE_DOMAINS: Annotated[list[str], NoDecode] = ["iuh.edu.vn"]
    # `basic` costs 1 credit per search, `advanced` 2 but returns longer, more relevant snippets.
    TAVILY_SEARCH_DEPTH: Literal["basic", "advanced"] = "basic"
    # Deadline for the whole search call. Tavily usually answers in ~3s but has spikes
    # past 10s; a timeout only drops web results, the turn still ends in the ticket fallback.
    TAVILY_TIMEOUT_SECONDS: float = 15.0

    # --- INGEST_: only ever read at document-ingestion time (chunking,
    # enrichment) - never during a chat turn ---
    INGEST_MULTI_REP_QUESTION_COUNT: int = 3
    INGEST_SEMANTIC_MAX_TOKEN_FACTOR: float = 1.5
    INGEST_TABLE_CHUNK_MAX_TOKENS: int = 800
    INGEST_CHUNKING_VERSION: str = "2026-09-structural-v2"
    # Minimum seconds between the START of two consecutive extraction calls in one embed job.
    # Free-tier keys allow ~15 requests/minute, but a sequential job easily goes faster than
    # that; 0 disables the pause. A call that already took longer than this adds no extra wait.
    INGEST_EXTRACTION_MIN_INTERVAL_SECONDS: float = Field(default=0.0, ge=0)
    # When every EXTRACTION credential is cooling down, wait this long before retrying the
    # same chunk. A bit over the router's 30s default cooldown so the first key is usable again.
    INGEST_EXTRACTION_CREDENTIAL_WAIT_SECONDS: float = Field(default=35.0, ge=0)
    # How many such waits one chunk gets before the embed job is failed. 0 = never wait.
    INGEST_EXTRACTION_MAX_CREDENTIAL_WAITS: int = Field(default=6, ge=0)

    # --- CHAT_: read on every chat turn ---
    CHAT_CLARIFICATION_MAX_RETRY: int = 2
    # Thinking for the short auxiliary calls (classification, query transformation, LLM
    # rerank); generation keeps the model default. False = the model's lowest level.
    CHAT_AUX_THINKING: bool | Literal["minimal", "low", "medium", "high"] = False
    # Candidates fetched per turn (split across sub-queries) for rerank to judge.
    # CONTEXT_MAX_CHUNKS separately caps what reaches the generation prompt, so recall
    # can grow without bloating the prompt - also when the LLM rerank fails open.
    CHAT_RETRIEVAL_MAX_CHUNKS: int = Field(default=16, ge=1)
    CHAT_CONTEXT_MAX_CHUNKS: int = Field(default=8, ge=1)
    CHAT_RERANK_SCORE_THRESHOLD: float = 0.70
    # Max sub-queries the decomposer may split one comparison question into.
    CHAT_MAX_SUB_QUERIES: int = Field(default=3, ge=2)
    # LLMRerankNode: one call to the RERANK model (EXTRACTION if none) per turn checks
    # which reranked chunks actually answer which sub-query - the score threshold alone
    # lets a chunk through on shared keywords. A sub-query left with none goes to
    # WebSearchNode. Each chunk is shown to the model cut to SNIPPET_CHARS.
    CHAT_LLM_RERANK_ENABLED: bool = True
    CHAT_LLM_RERANK_SNIPPET_CHARS: int = Field(default=800, ge=100)
    # When the LLM rerank keeps nothing for any sub-query, each sub-query whose best
    # chunk reached RESCUE_MIN_SCORE keeps its top RESCUE_KEEP by score: a small model
    # drops e.g. a fill-in form full of dot leaders whose "Hồ sơ đính kèm" line is the
    # answer, and the turn would end in TicketFallback with the right document retrieved.
    CHAT_LLM_RERANK_RESCUE_MIN_SCORE: float = Field(default=0.75, ge=0, le=1)
    CHAT_LLM_RERANK_RESCUE_KEEP: int = Field(default=2, ge=0)
    # WebSearchNode: searches the web (TAVILY_*) for each sub-query rerank left with no
    # chunk, before giving up to TicketFallbackNode. Per-turn cap and per-result char cap
    # bound how much web text reaches the system prompt (~3000 chars at the defaults),
    # however long the question or however many sub-queries it split into.
    CHAT_WEB_SEARCH_ENABLED: bool = False
    # Tavily bills per search, not per result, so asking for more candidates is free;
    # MIN_SCORE and PER_TURN still bound what reaches the prompt.
    CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB: int = Field(default=5, ge=1, le=20)
    CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN: int = Field(default=2, ge=1)
    # Searches per turn: a message with several tasks can leave many sub-queries without
    # chunks, but only PER_TURN pages reach the prompt, so the rest would be paid-for
    # searches thrown away. The worst-missed sub-queries are searched first.
    CHAT_WEB_SEARCH_MAX_QUERIES: int = Field(default=2, ge=1)
    CHAT_WEB_SEARCH_MIN_SCORE: float = Field(default=0.5, ge=0.0, le=1.0)
    CHAT_WEB_SEARCH_RESULT_MAX_CHARS: int = Field(default=1500, ge=100)

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

    # --- Cost tracking & budget ---
    # Must match backend-java's app.timezone: DAILY/MONTHLY budget periodKeys are cut
    # in this zone.
    APP_TIMEZONE: str = "Asia/Ho_Chi_Minh"
    # Used by app/core/usage/cost_calculator.py when backend-java has no price
    # for a model - a conservative non-zero placeholder so budget reservation
    # never silently estimates $0 for an unpriced model.
    BUDGET_RESERVATION_FALLBACK_USD: float = 0.05
    # Request-level Chat reservation = primary model's estimate x this, to cover the
    # 2-3 secondary LLM calls (classification, query transformation) the request-level
    # reservation is taken before any of them run.
    BUDGET_RESERVATION_MULTIPLIER_CHAT: float = 1.5
    # TTL past which a reservation hash is considered stale/abandoned (process crash) -
    # release_expired_reservations sweeps these.
    BUDGET_RESERVATION_TTL_SECONDS: int = 600
    # How often BudgetSnapshot re-polls GET /internal/budgets/snapshot, independent of
    # the config_version pub/sub nudge (same "self-heal on a timer" posture as
    # MODEL_REGISTRY_POLL_INTERVAL_SECONDS).
    BUDGET_SNAPSHOT_REFRESH_SECONDS: float = 60.0
    # How often PricingSnapshot re-polls GET /internal/model-pricing/snapshot. A price an SA
    # edits applies to new calls within this window.
    MODEL_PRICING_SNAPSHOT_REFRESH_SECONDS: float = 60.0
    # How often Beat runs drain_usage_outbox.
    USAGE_OUTBOX_DRAIN_INTERVAL_SECONDS: float = 5.0
    # Upper-bound output tokens assumed for the request-level Chat reservation
    # estimate - deliberately generous (a real response rarely reaches this),
    # since under-reserving would let a request through that a THROTTLE/BLOCK
    # budget should have caught.
    BUDGET_ESTIMATE_MAX_OUTPUT_TOKENS: int = 2000

    @field_validator("TAVILY_INCLUDE_DOMAINS", mode="before")
    @classmethod
    def _split_domains(cls, value: object) -> object:
        if isinstance(value, str):
            return [domain.strip() for domain in value.split(",") if domain.strip()]
        return value

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
