from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error

from app.rag.ingestion.minio_client import ObjectNotFoundException, get_object_bytes


def _s3_error(code: str) -> S3Error:
    return S3Error(
        code=code,
        message="mocked",
        resource="resource",
        request_id="request-id",
        host_id="host-id",
        response=MagicMock(),
    )


@patch("app.rag.ingestion.minio_client._get_client")
def test_get_object_bytes_returns_content(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.read.return_value = b"hello world"
    mock_client.get_object.return_value = mock_response
    mock_get_client.return_value = mock_client

    content = get_object_bytes("docs/handbook.txt")

    assert content == b"hello world"
    mock_response.close.assert_called_once()
    mock_response.release_conn.assert_called_once()


@patch("app.rag.ingestion.minio_client._get_client")
def test_get_object_bytes_raises_typed_error_when_missing(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.get_object.side_effect = _s3_error("NoSuchKey")
    mock_get_client.return_value = mock_client

    with pytest.raises(ObjectNotFoundException):
        get_object_bytes("docs/missing.txt")


@patch("app.rag.ingestion.minio_client._get_client")
def test_get_object_bytes_reraises_other_s3_errors(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.get_object.side_effect = _s3_error("InternalError")
    mock_get_client.return_value = mock_client

    with pytest.raises(S3Error):
        get_object_bytes("docs/handbook.txt")
