from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Covers clock skew between agent workers sharing the clarification lease.
CLARIFICATION_LEASE_MARGIN_SECONDS = 60.0
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
    # ANSI colors on console log lines (only when the stream is a terminal).
    LOG_COLOR: bool = True

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
    # Base collection of this environment. The live one starts as this and moves to
    # `<base>__<timestamp>` after an embedding re-index; the registry snapshot names it.
    QDRANT_COLLECTION: str = "unisage_chunks"
    # --- EMBEDDING_REINDEX_: re-embedding the live collection into a new model's collection
    # (changes/17-10-2026-Embedding-Reindex in unisage-backend). Beat polls for a queued run;
    # each batch of points is one embedding call (3 texts per point). ---
    EMBEDDING_REINDEX_POLL_INTERVAL_SECONDS: float = 15.0
    EMBEDDING_REINDEX_BATCH_SIZE: int = 32
    EMBEDDING_REINDEX_MAX_ATTEMPTS: int = 3

    # --- TAVILY_: web search API used by WebSearchNode when retrieval finds nothing for a
    # sub-query (see app/graph/nodes/web_search.py). Empty key = web search is skipped. ---
    TAVILY_API_KEY: str = ""
    TAVILY_BASE_URL: str = "https://api.tavily.com"
    # Comma-separated in .env. Only these sites (and their subdomains) are searched - the
    # answer must come from the university's own pages, never a forum or another school.
    TAVILY_INCLUDE_DOMAINS: Annotated[list[str], NoDecode] = ["iuh.edu.vn"]
    # `basic` costs 1 credit per search, `advanced` 2 but returns longer, more relevant snippets:
    # up to TAVILY_CHUNKS_PER_SOURCE passages of each page that match the query, picked by
    # Tavily, instead of one short excerpt.
    TAVILY_SEARCH_DEPTH: Literal["basic", "advanced"] = "basic"
    TAVILY_CHUNKS_PER_SOURCE: int = Field(default=3, ge=1, le=3)
    # Deadline for the whole search call. Tavily usually answers in ~3s but has spikes
    # past 10s; a timeout only drops web results, the turn still ends in the ticket fallback.
    TAVILY_TIMEOUT_SECONDS: float = 15.0

    # --- INGEST_: only ever read at document-ingestion time (chunking,
    # enrichment) - never during a chat turn ---
    INGEST_MULTI_REP_QUESTION_COUNT: int = 3
    INGEST_SEMANTIC_MAX_TOKEN_FACTOR: float = 1.5
    INGEST_TABLE_CHUNK_MAX_TOKENS: int = 800
    # A PDF heading candidate longer than this stays body text - it is prefixed into every
    # chunk under it, so an over-long one (a form's fill-in line) would overflow chunk_size.
    INGEST_MAX_HEADING_CHARS: int = 200
    INGEST_CHUNKING_VERSION: str = "2026-09-structural-v2"
    # Tesseract language(s) for scanned PDF pages (pymupdf4llm OCRs pages with no text layer).
    # Vietnamese needs the `vie` traineddata; a language whose data is missing is dropped (with a
    # warning) rather than failing the ingest, so "vie+eng" falls back to "eng" on such a machine.
    INGEST_OCR_LANGUAGE: str = "vie+eng"
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
    # Let the LLM answer calculations Python does not compute (target questions,
    # formulas from documents, follow-ups). Off = only the 3 built-in formulas are
    # calculated (kill switch - there is no pre-release eval; Đúng/Sai feedback instead).
    CHAT_CALC_LLM_ENABLED: bool = True
    # Timeout of that LLM call (not streamed: the whole answer is parsed before it is shown).
    CHAT_CALC_LLM_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    # Hard deadline for everything a turn does after claiming a clarification panel
    # (start_turn → graph → finalize). Past it the task is cancelled before any more writes.
    CHAT_CLAIMED_TURN_DEADLINE_SECONDS: float = Field(default=150.0, gt=0)
    # How long a claimed panel stays PROCESSING before another request may treat it as
    # abandoned. Must exceed the deadline by CLARIFICATION_LEASE_MARGIN_SECONDS so a claimed
    # turn is always dead before its lease can be taken over.
    CHAT_CLARIFICATION_LEASE_SECONDS: float = Field(default=210.0, gt=0)
    # Thinking for the short auxiliary calls (classification, query transformation, LLM
    # rerank). False = the model's lowest level.
    CHAT_AUX_THINKING: bool | Literal["minimal", "low", "medium", "high"] = False
    # Thinking for the answer itself (generation, ticket fallback). False = the model's lowest
    # level (fastest first token); None = the model's own default (gpt-5-mini: medium).
    CHAT_GENERATION_THINKING: bool | Literal["minimal", "low", "medium", "high"] | None = False
    # A streamed answer with no visible text after this long fails over to the next
    # credential; once text has started streaming it is never cut. 0 = no limit.
    CHAT_FIRST_TOKEN_TIMEOUT_SECONDS: float = Field(default=20.0, ge=0)
    # Whole-call limit for the non-streamed calls (classification, query transformation,
    # LLM rerank) before failing over. 0 = no limit.
    CHAT_AUX_CALL_TIMEOUT_SECONDS: float = Field(default=30.0, ge=0)
    # Classification also writes each advisory task's HyDE text / sub-queries, so
    # QueryTransformationNode only calls its own LLM when that output is missing.
    CHAT_CLASSIFY_WITH_RETRIEVAL: bool = True
    # Candidates fetched per turn (split across sub-queries) for rerank to judge.
    # CONTEXT_MAX_CHUNKS separately caps what reaches the generation prompt, so recall
    # can grow without bloating the prompt - also when the LLM rerank fails open.
    CHAT_RETRIEVAL_MAX_CHUNKS: int = Field(default=16, ge=1)
    CHAT_CONTEXT_MAX_CHUNKS: int = Field(default=8, ge=1)
    # Cosine cutoff of the score rerank when NO LLM rerank will run (no RERANK/EXTRACTION
    # model, or CHAT_LLM_RERANK_ENABLED=false): the pre-hybrid behaviour.
    CHAT_RERANK_SCORE_THRESHOLD: float = 0.70
    # Hybrid retrieval (docs/specs/SPEC-hybrid-retrieval.md): every sub-query also runs a
    # BM25 sparse query (`bm25` vector, built from the standalone question); the dense
    # lists merged by best cosine and the BM25 list are fused by RRF. Off, or a collection
    # without the `bm25` vector, means the dense-only search of before.
    RETRIEVAL_HYBRID_ENABLED: bool = True
    # With an LLM rerank ahead, the score rerank only applies this low floor to the dense
    # cosine; the LLM decides what answers the question (decision (a), 20/10/2026).
    CHAT_RETRIEVAL_DENSE_FLOOR: float = Field(default=0.45, ge=0.0, le=1.0)
    # `RetrievedChunk.score` of a chunk found by BM25 only (no dense cosine known): what
    # the floor, the rescue rule and web search's ordering see for it.
    CHAT_RETRIEVAL_LEXICAL_HIT_SCORE: float = Field(default=0.70, ge=0.0, le=1.0)
    # RRF: score = sum of 1/(k + rank) over the two lists. Measured on the eval corpus
    # (21/10/2026): with the textbook k=60 a chunk at BM25 rank 1 but outside the dense
    # top-32 loses to every chunk present in both lists, even at rank 30, and falls out
    # of the top-16; k=1 puts it at rank 2-3 while the dense top-8 stays intact.
    CHAT_RETRIEVAL_RRF_K: int = Field(default=1, ge=1)
    # Each list fetches limit x this before fusion, so a chunk strong in one list but
    # absent from the other still competes.
    CHAT_RETRIEVAL_HYBRID_CANDIDATES_MULTIPLIER: int = Field(default=2, ge=1)
    # Max sub-queries the decomposer may split one comparison question into.
    CHAT_MAX_SUB_QUERIES: int = Field(default=3, ge=2)
    # LLMRerankNode: one call to the RERANK model (EXTRACTION if none) per turn checks
    # which reranked chunks actually answer which sub-query - the score threshold alone
    # lets a chunk through on shared keywords. A sub-query left with none goes to
    # WebSearchNode. Each chunk is shown to the model whole.
    CHAT_LLM_RERANK_ENABLED: bool = True
    # When the LLM rerank keeps nothing for any sub-query, each sub-query whose best
    # chunk reached RESCUE_MIN_SCORE keeps its top RESCUE_KEEP by score: a small model
    # drops e.g. a fill-in form full of dot leaders whose "Hồ sơ đính kèm" line is the
    # answer, and the turn would end in TicketFallback with the right document retrieved.
    CHAT_LLM_RERANK_RESCUE_MIN_SCORE: float = Field(default=0.75, ge=0, le=1)
    CHAT_LLM_RERANK_RESCUE_KEEP: int = Field(default=2, ge=0)
    # Cross-encoder rerank (docs/specs/SPEC-cross-encoder-rerank.md): a local
    # bge-reranker-v2-m3 service (`task reranker:serve`) scores (question, chunk) pairs.
    # `llm`: the LLM rerank above, as before (the service is never called).
    # `cross_encoder`: CrossEncoderRerankNode keeps each sub-query's top CROSS_ENCODER_KEEP
    # by cross-encoder score and the LLM rerank is skipped.
    # `cross_encoder_then_llm`: the cross-encoder cuts to KEEP, the LLM rerank judges those.
    # `none`: no rerank model at all - the retrieval order (RRF for hybrid) above the dense
    # floor, cut to CONTEXT_MAX_CHUNKS; the benchmark's "no rerank" baseline, measurable e2e.
    # A service error or timeout falls over to the LLM rerank with an AI-admin warning.
    # `auto`: the `none` path, plus the LLM rerank only for a turn whose evidence signals
    # (app/graph/evidence.py) call a sub-query weak - the gate "no document answers this"
    # kept for the risky tail at a fraction of the LLM calls.
    CHAT_RERANK_MODE: Literal["llm", "cross_encoder", "cross_encoder_then_llm", "none", "auto"] = (
        "llm"
    )
    CROSS_ENCODER_URL: str = "http://127.0.0.1:8403"
    CROSS_ENCODER_TIMEOUT_SECONDS: float = Field(default=2.0, gt=0)
    CHAT_CROSS_ENCODER_KEEP: int = Field(default=8, ge=1)
    # Evidence sufficiency (app/graph/evidence.py), logged every turn and the trigger of
    # `auto`: a sub-query is `weak` when its best dense cosine is below MIN_COSINE and, when
    # a cross-encoder score exists, that score is below MIN_CE too. 0,60 from the threshold
    # study (139 câu demo, 26/10/2026): flags ~10 % of turns, catches 3 of the 5 questions
    # whose document is not in the top-8 (report Phase 6); re-run `task benchmark:evidence`
    # on a new corpus before changing it.
    CHAT_EVIDENCE_MIN_COSINE: float = Field(default=0.60, ge=0.0, le=1.0)
    CHAT_EVIDENCE_MIN_CE: float = 1.0
    # WebSearchNode: searches the web (TAVILY_*) for each sub-query rerank left with no
    # chunk, before giving up to TicketFallbackNode. Per-turn cap and per-result char cap
    # bound how much web text reaches the system prompt (~3000 chars at the defaults),
    # however long the question or however many sub-queries it split into.
    CHAT_WEB_SEARCH_ENABLED: bool = False
    # Fallback web search policy (app/graph/web_policy.py): besides sub-queries the LLM
    # rerank emptied or retrieval found nothing for, also search those whose evidence is
    # only `weak` (app/graph/evidence.py). Off by default: measure the fallback on its own.
    CHAT_WEB_SEARCH_ON_WEAK_EVIDENCE: bool = False
    # `speculative`: for a task whose topic is in SPECULATIVE_TOPICS (fees, admission,
    # notices, contacts - what changes every year and is posted on the website first),
    # the search starts with the advisory branch, alongside query transformation, retrieval
    # and rerank. `fallback`: searched only after rerank left a sub-query empty.
    CHAT_WEB_SEARCH_MODE: Literal["fallback", "speculative"] = "speculative"
    # With a speculative search, its pages are judged for EVERY sub-query alongside
    # retrieval and rerank, and generation gets them next to the kept chunks: a kept chunk
    # can be related yet stale (the 2022 exam results kept for "the latest exam date",
    # 14-10-2026). Off: judged only when a sub-query is left without a chunk.
    CHAT_WEB_JUDGE_WITH_CHUNKS: bool = True
    CHAT_WEB_SEARCH_SPECULATIVE_TOPICS: Annotated[list[str], NoDecode] = [
        "hoc_phi",
        "tuyen_sinh",
        "lich_thong_bao",
        "lien_he_co_so",
    ]
    # Tavily bills per search, not per result, so asking for more candidates is free; every
    # page left after the junk filters goes to the web judge, and ANSWERS_PER_SUB /
    # MAX_SOURCES bound what reaches the prompt.
    CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB: int = Field(default=10, ge=1, le=20)
    # Pages with the answer kept per sub-query - the web judge picks them, best first (not
    # Tavily's score order).
    CHAT_WEB_SEARCH_MAX_ANSWERS_PER_SUB: int = Field(default=2, ge=1)
    # Official pages the web judge kept although their snippet lacks the answer (a scanned
    # fee decision): at most this many are linked by the ticket fallback.
    CHAT_WEB_SEARCH_MAX_SOURCES: int = Field(default=3, ge=0)
    # Searches per turn: a MULTI task searches each sub-query with web slots on its own (up
    # to CHAT_MAX_SUB_QUERIES), each one billed by Tavily. The first ones are searched.
    CHAT_WEB_SEARCH_MAX_QUERIES: int = Field(default=3, ge=1)
    # Tavily's score depends on how the query is worded far more than on relevance (a
    # question-form query scores the right page 0.08, a keyword query scores an off-topic
    # page 0.90), so it only drops junk (SCORE_FLOOR); the web judge decides relevance.
    CHAT_WEB_SEARCH_SCORE_FLOOR: float = Field(default=0.05, ge=0.0, le=1.0)
    # Only used when the judge can't run (no RERANK/EXTRACTION model, or it failed):
    # pages at or above this score are used as they are, the rule before the judge existed.
    CHAT_WEB_SEARCH_MIN_SCORE: float = Field(default=0.5, ge=0.0, le=1.0)
    CHAT_WEB_SEARCH_RESULT_MAX_CHARS: int = Field(default=1500, ge=100)
    # Redis cache of raw search results, keyed by the query and Tavily's parameters; its
    # TTL depends on the topic (app/rag/web_query.py). Its own Redis instance (`redis-cache`
    # in .devcontainer/docker-compose.yml: maxmemory + volatile-lru), since eviction covers a
    # whole instance and would otherwise reach the budget/Celery keys on REDIS_URL that also
    # carry a TTL. Empty URL = no cache.
    WEB_SEARCH_CACHE_REDIS_URL: str = "redis://localhost:6380/0"

    # Retrieval cache (docs/specs/SPEC-retrieval-cache.md): two exact-match layers that keep
    # the chunk ids a turn's retrieval + rerank ended with, keyed by the student's question
    # (L1, no-history turns) or by the standalone questions the classifier wrote (L2). Same
    # `redis-cache` instance as the web search cache, its own logical DB (2 dev, 3 eval).
    # Empty URL = no cache; tests and eval run with it empty.
    RETRIEVAL_CACHE_REDIS_URL: str = "redis://localhost:6380/2"
    # Absolute TTL (no refresh on hit), +/-10% jitter at write. Event-driven invalidation
    # (document reverse index, per-department generation) is the real staleness guard;
    # this is the backstop.
    RETRIEVAL_CACHE_TTL_SECONDS: int = Field(default=21600, ge=60)
    # Socket + connect timeout of every cache call: a lookup must never cost more than this.
    RETRIEVAL_CACHE_TIMEOUT_MS: int = Field(default=50, ge=5)
    # Bump by hand when the key or value layout changes (the config/prompt fingerprint
    # already changes the key on its own).
    RETRIEVAL_CACHE_KEY_VERSION: str = "1"
    # Write an entry only when the same key missed twice within the TTL: one-off questions
    # (most follow-ups) then never take space; a repeated one is cached from its 3rd sighting.
    RETRIEVAL_CACHE_WRITE_ON_SECOND_SIGHTING: bool = True

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
    # Grounding check after generation (app/graph/grounding.py): numbers in the answer that
    # no cited source / shown source / the question contains get a caution line appended
    # (the answer is already streamed) and an AI-admin warning; `[n]` outside the sources
    # and answers with figures but no citation are reported to AI admins.
    CHAT_GROUNDING_CHECK: bool = True

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

    @field_validator("TAVILY_INCLUDE_DOMAINS", "CHAT_WEB_SEARCH_SPECULATIVE_TOPICS", mode="before")
    @classmethod
    def _split_domains(cls, value: object) -> object:
        if isinstance(value, str):
            return [domain.strip() for domain in value.split(",") if domain.strip()]
        return value

    @model_validator(mode="after")
    def _lease_outlives_claimed_turn(self) -> "Settings":
        margin = CLARIFICATION_LEASE_MARGIN_SECONDS
        if self.CHAT_CLARIFICATION_LEASE_SECONDS < self.CHAT_CLAIMED_TURN_DEADLINE_SECONDS + margin:
            raise ValueError(
                "CHAT_CLARIFICATION_LEASE_SECONDS must be at least "
                f"CHAT_CLAIMED_TURN_DEADLINE_SECONDS + {margin:g}"
            )
        return self

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
