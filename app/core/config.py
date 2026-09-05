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

    # backend-java integration (see tasks/plan.md "Auth" section): Python
    # forwards the caller's original `Authorization` header as-is on every
    # call - Java's GatewayHeaderFilter re-verifies it. Every call also
    # carries `X-Internal-Secret` (INTERNAL_SECRET_KEY below), which Java's
    # InternalSecretFilter requires on its Python-only endpoints (e.g.
    # PATCH /messages/{id}) and uses to decide whether to trust a forwarded
    # X-Forwarded-For for guest ownership checks. Must include Java's
    # `server.servlet.context-path` (`/api/v1` by default in backend-java's
    # application.properties) since this client's paths are context-relative.
    BACKEND_JAVA_BASE_URL: str = "http://localhost:8401/api/v1"

    # Missing-metadata clarification guard (node 02) - see
    # missing_metadata_clarification_design.md section 5. Number of mismatched
    # replies tolerated before the pending clarification is discarded and the
    # flow falls back to a safe, branch-covering answer.
    CLARIFICATION_MAX_RETRY: int = 2

    # Retrieval / rerank (nodes 10/11).
    RETRIEVAL_MAX_CHUNKS: int = 8
    RERANK_SCORE_THRESHOLD: float = 0.70


settings = Settings()
