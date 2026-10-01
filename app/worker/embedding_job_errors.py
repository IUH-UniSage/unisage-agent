"""How an `embed_chunks` ingest job (`app.worker.celery_app`) reports failure: the
terminal exception Celery stores, which failures abort the whole job, and the
client-facing reason for a single failed chunk.

Kept out of `app.worker.tasks.ingestion` so the API (`GET /ingestion/jobs/{id}`) can read a stored
failure without depending on the task module's internals.
"""

from typing import Any

from app.core.budget.tracker import RequestBudgetRejectedError
from app.core.errors.error_codes import ErrorCode
from app.core.errors.llm_failure import describe_llm_failure, is_model_failure
from app.core.errors.provider_errors import EmbeddingProviderError
from app.core.llm.provider_models import UnsupportedProviderError
from app.core.registry.errors import NoAvailableCredentialError, NoBudgetAvailableError
from app.core.registry.model_registry import ModelRegistryError


class IngestionJobFailedError(Exception):
    """Terminal failure of an `embed_chunks` job, raised so Celery records the task FAILED.

    `args` are `(message, error_code)` - `message` is already the friendly, client-facing
    reason (see `describe_llm_failure`). Celery's result backend stores the exception as
    its type + args and rebuilds it on read (this module is imported by the API process,
    so the type resolves), which is how `GET /ingestion/jobs/{id}` (`_read_task_progress`)
    shows the same specific reason the live WebSocket frame did.
    """

    def __init__(self, message: str, error_code: int = ErrorCode.EMBEDDING_JOB_FAILED.code) -> None:
        super().__init__(message, error_code)
        self.message = message
        self.error_code = error_code


# Failures that mean the EMBEDDING/EXTRACTION model is unusable for the whole job (not just
# for one chunk) - see `app.worker.tasks.ingestion._run_embed_chunks`.
JOB_FATAL_ERRORS: tuple[type[Exception], ...] = (
    EmbeddingProviderError,
    NoAvailableCredentialError,
    NoBudgetAvailableError,
    RequestBudgetRejectedError,
    ModelRegistryError,
    UnsupportedProviderError,
)


def chunk_failure_reason(exc: Exception) -> str:
    """A client-safe explanation of one chunk's failure (no provider text)."""

    if is_model_failure(exc):
        return describe_llm_failure(exc, purpose="EXTRACTION").message
    if type(exc).__module__.startswith("qdrant_client"):
        return ErrorCode.VECTOR_STORE_ERROR.message
    return f"Lỗi nội bộ khi xử lý đoạn ({type(exc).__name__})."


def partial_failure_message(results: list[dict[str, Any]], total: int) -> str:
    """The "N/M đoạn nạp liệu thất bại." line plus the first failed chunk's reason, so the client
    learns why, not just how many."""

    failed = [r for r in results if r.get("status") == "FAILED"]
    message = f"{len(failed)}/{total} đoạn nạp liệu thất bại."
    first_reason = next((r.get("reason") for r in failed if r.get("reason")), None)
    if first_reason:
        message += f" Lỗi đầu tiên (đoạn #{failed[0]['chunk_index']}): {first_reason}"
    return message
