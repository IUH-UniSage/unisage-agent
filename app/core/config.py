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

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/unisage_db"

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


settings = Settings()
