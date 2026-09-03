from functools import lru_cache

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.error_codes import ErrorCode
from app.core.exceptions import UniSageException


class ObjectNotFoundException(UniSageException):
    """Exception raised when the requested object does not exist in MinIO."""

    def __init__(self, object_key: str):
        super().__init__(
            message=f"Object '{object_key}' was not found in storage.",
            error_code=ErrorCode.OBJECT_NOT_FOUND,
            status_code=404,
        )


@lru_cache(maxsize=1)
def _get_client() -> Minio:
    """Build the MinIO client once and reuse it across requests."""

    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def get_object_bytes(object_key: str) -> bytes:
    """Fetch an object's raw bytes from the configured MinIO bucket."""

    client = _get_client()
    try:
        response = client.get_object(settings.MINIO_BUCKET, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
    except S3Error as exc:
        if exc.code == "NoSuchKey":
            raise ObjectNotFoundException(object_key) from exc
        raise
