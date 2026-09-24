from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "UniSage AI Agent Service"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # Internal service-to-service auth: must match the shared secret the API
    # Gateway sends as `X-Internal-Secret` on every proxied request (see
    # api-gateway's `python-ai-agent-route` filters in application.yml).
    # Default matches that route's own fallback so local dev works
    # out-of-the-box; override both sides together in real deployments.
    INTERNAL_SECRET_KEY: str = "unisage-internal-secret-key-2026"

    # Database: same physical Postgres server as backend-java (see
    # backend-java/docker-compose.yml's `unisage-db` service, port 5433 on
    # host), but a separate database (`unisage_agent_db`, not Java's
    # `assistant_DB`) - per SPEC-ingestion-resume.md, no shared
    # schema/tables, no cross-service FK.
    DATABASE_URL: str = "postgresql+asyncpg://postgres:123456@localhost:5433/unisage_agent_db"

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    MULTI_REP_LLM_MODEL: str = "gpt-4o-mini"
    MULTI_REP_QUESTION_COUNT: int = 3

    SEMANTIC_MAX_TOKEN_FACTOR: float = 1.5
    TABLE_CHUNK_MAX_TOKENS: int = 800

    CHUNKING_VERSION: str = "2026-09-structural-v2"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "unisage-documents"
    MINIO_SECURE: bool = False

    # Redis / Celery
    REDIS_URL: str = "redis://localhost:6379/0"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "unisage_chunks"

    BACKEND_JAVA_BASE_URL: str = "http://localhost:8401/api/v1"

    CLARIFICATION_MAX_RETRY: int = 2

    # Retrieval / rerank (nodes 08/09).
    RETRIEVAL_MAX_CHUNKS: int = 8
    RERANK_SCORE_THRESHOLD: float = 0.70
    HISTORY_MESSAGE_LIMIT: int = 15

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
    ALLOW_REPAIR_JSON: bool = True


settings = Settings()
