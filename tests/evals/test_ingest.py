import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from evals.crawl.download import DownloadState, write_csv
from evals.ingest import (
    ApiError,
    Labels,
    Run,
    UniSageClient,
    document_fields,
    is_quota,
    link_rows,
    pick_rows,
    target_for,
)

LABELS = Labels({"PHONG_DAO_TAO": "dept-uuid"}, {3: "level-3-uuid"})


def _row(**extra: str) -> dict[str, str]:
    return {
        "file_id": "a1",
        "department_id": "PHONG_DAO_TAO",
        "is_public": "false",
        "access_level": "3",
        "title": "Quyết định 1035/QĐ-ĐHCN về học phí",
        "local_path": "files/pdt/a1.pdf",
        "selected": "true",
        "ingest_status": "",
        "document_id": "",
        "source_url": f"https://iuh.edu.vn/{extra.get('file_id', 'a1')}.pdf",
        "sha256": extra.get("file_id", "a1"),
        "unit": "Phòng Đào tạo",
        **extra,
    }


def _ok(data: Any) -> httpx.Response:
    return httpx.Response(200, json={"code": 1000, "message": "ok", "data": data})


class FakeStack:
    """Gateway stand-in recording every call."""

    def __init__(
        self,
        *,
        empty_text: bool = False,
        task_states: list[str] | None = None,
        failure_message: str = "boom",
        job_errors: int = 0,
    ) -> None:
        self.failure_message = failure_message
        self.job_errors = job_errors
        self.calls: list[tuple[str, str]] = []
        self.empty_text = empty_text
        self.task_states = task_states or ["SUCCESS"]
        self.uploads: list[dict[str, Any]] = []
        self.embed_bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        if path == "/api/v1/master/documents" and request.method == "POST":
            self.uploads.append({"body": request.content})
            return _ok({"id": "doc-1", "sourceUrl": "uuid_Quyết định.pdf"})
        if path == "/api/v1/master/documents/doc-1":
            return _ok({"id": "doc-1", "sourceUrl": "uuid_Quyết định.pdf"})
        if path == "/api/v1/ai/ingestion/chunking":
            if self.empty_text:
                return httpx.Response(422, json={"code": 4221, "message": "empty", "data": None})
            return _ok({"chunks": [{"chunk_index": 0, "content": "x", "region_type": "text"}]})
        if path == "/api/v1/ai/ingestion/embedding":
            self.embed_bodies.append(json.loads(request.content))
            return httpx.Response(
                202, json={"code": 1000, "message": "ok", "data": {"task_id": "t"}}
            )
        if path == "/api/v1/ai/ingestion/jobs/doc-1":
            if self.job_errors:
                self.job_errors -= 1
                return httpx.Response(500, text="Internal Server Error")
            state = self.task_states.pop(0) if len(self.task_states) > 1 else self.task_states[0]
            return _ok(
                {
                    "task_state": state,
                    "task_message": self.failure_message if state == "FAILURE" else None,
                }
            )
        return httpx.Response(404, json={"code": 404, "message": path})


def _run(tmp_path: Path, stack: FakeStack, rows: list[dict[str, str]], **run: Any) -> DownloadState:
    (tmp_path / "files" / "pdt").mkdir(parents=True)
    (tmp_path / "files" / "pdt" / "a1.pdf").write_bytes(b"%PDF")
    write_csv(tmp_path / "manifest.csv", list(rows[0]), rows)
    state = DownloadState(tmp_path)
    client = UniSageClient(httpx.Client(base_url="http://gw", transport=httpx.MockTransport(stack)))
    Run(client, state, LABELS, poll_seconds=0, **run).ingest(pick_rows(state, None, None))
    return state


def test_ingest_uploads_chunks_embeds_and_marks_done(tmp_path: Path) -> None:
    stack = FakeStack()
    state = _run(tmp_path, stack, [_row()])

    row = state.manifest[0]
    assert row["document_id"] == "doc-1"
    assert row["ingest_status"] == "done"
    upload = stack.uploads[0]["body"].decode("utf-8", "replace")
    assert 'filename="Quyết định 1035-QĐ-ĐHCN về học phí.pdf"' in upload
    assert "level-3-uuid" in upload
    embed = stack.embed_bodies[0]
    assert embed["department_id"] == "dept-uuid"
    assert embed["access_level"] == 3
    assert embed["is_public"] is False
    assert embed["object_key"] == "uuid_Quyết định.pdf"


def test_ingest_marks_empty_text_without_embedding(tmp_path: Path) -> None:
    stack = FakeStack(empty_text=True)
    state = _run(tmp_path, stack, [_row()])

    assert state.manifest[0]["ingest_status"] == "empty_text"
    assert stack.embed_bodies == []


def test_ingest_resumes_an_uploaded_row_without_uploading_again(tmp_path: Path) -> None:
    stack = FakeStack()
    state = _run(tmp_path, stack, [_row(document_id="doc-1", ingest_status="error:timeout")])

    assert stack.uploads == []
    assert state.manifest[0]["ingest_status"] == "done"


def test_ingest_waits_for_the_task_and_records_failure(tmp_path: Path) -> None:
    stack = FakeStack(task_states=["STARTED", "PROGRESS", "FAILURE"])
    state = _run(tmp_path, stack, [_row()])

    assert state.manifest[0]["ingest_status"] == "error:FAILURE boom"
    polls = [call for call in stack.calls if call[1].endswith("/jobs/doc-1")]
    assert len(polls) == 3


def test_upload_is_saved_before_chunking_so_a_stopped_run_never_uploads_twice(
    tmp_path: Path,
) -> None:
    class StoppedDuringChunking(FakeStack):
        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/ai/ingestion/chunking":
                raise KeyboardInterrupt
            return super().__call__(request)

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, StoppedDuringChunking(), [_row()])

    assert DownloadState(tmp_path).manifest[0]["document_id"] == "doc-1"


def test_ingest_keeps_polling_through_a_transient_job_500(tmp_path: Path) -> None:
    stack = FakeStack(task_states=["PROGRESS", "SUCCESS"], job_errors=2)
    state = _run(tmp_path, stack, [_row()])

    assert state.manifest[0]["ingest_status"] == "done"
    assert len(stack.embed_bodies) == 1


def test_public_row_has_no_access_level_field() -> None:
    target = target_for(_row(is_public="true", access_level=""), LABELS)
    fields = document_fields(target, LABELS)
    assert target.access_level == 0
    assert "minAccessLevelId" not in fields
    assert fields["isPublic"] == "true"


def test_target_rejects_unknown_department() -> None:
    with pytest.raises(ValueError, match="KHOA_X"):
        target_for(_row(department_id="KHOA_X"), LABELS)


def test_pick_rows_skips_finished_and_filters_by_only(tmp_path: Path) -> None:
    rows = [
        _row(file_id="a1"),
        _row(file_id="b2", ingest_status="done"),
        _row(file_id="c3", ingest_status="empty_text"),
        _row(file_id="d4"),
    ]
    write_csv(tmp_path / "manifest.csv", list(rows[0]), rows)
    only = tmp_path / "only.csv"
    only.write_text("﻿file_id\nd4\nb2\n", encoding="utf-8")

    state = DownloadState(tmp_path)
    assert [row["file_id"] for row in pick_rows(state, None, None)] == ["a1", "d4"]
    assert [row["file_id"] for row in pick_rows(state, only, None)] == ["d4"]


def test_api_error_names_the_call_and_keeps_a_spring_error_body() -> None:
    def spring_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500, json={"timestamp": "t", "status": 500, "error": "Internal Server Error"}
        )

    http = httpx.Client(base_url="http://gw", transport=httpx.MockTransport(spring_500))
    with pytest.raises(ApiError) as caught:
        UniSageClient(http).create_document({}, "a.pdf", b"x")

    message = str(caught.value)
    assert message.startswith("POST /api/v1/master/documents -> 500")
    assert "Internal Server Error" in message


def test_link_rows_matches_hand_ingested_documents_by_title() -> None:
    documents = [
        {"id": "doc-a", "title": "Quy chế học vụ"},
        {"id": "doc-b", "title": "Hai bản"},
        {"id": "doc-c", "title": "Hai bản"},
        {"id": "doc-d", "title": "Chưa embedding"},
    ]
    jobs = {"doc-a": {"task_state": "SUCCESS"}}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/master/documents":
            return _ok({"data": documents, "totalPages": 1})
        document_id = path.rsplit("/", 1)[-1]
        if document_id in jobs:
            return _ok(jobs[document_id])
        return httpx.Response(404, json={"code": 4040, "message": "no job"})

    client = UniSageClient(
        httpx.Client(base_url="http://gw", transport=httpx.MockTransport(handler))
    )
    rows = [
        _row(file_id="a1", title="Quy chế học vụ"),
        _row(file_id="b2", title="Hai bản"),
        _row(file_id="c3", title="Chưa embedding"),
        _row(file_id="d4", title="Không có trên web"),
    ]

    unmatched = link_rows(client, rows)

    assert (rows[0]["document_id"], rows[0]["ingest_status"]) == ("doc-a", "done")
    assert rows[1]["document_id"] == ""
    assert (rows[2]["document_id"], rows[2]["ingest_status"]) == ("doc-d", "uploaded")
    assert unmatched == ["Hai bản (2 matches)", "Không có trên web (0 matches)"]


QUOTA_MESSAGE = "1/1 đoạn nạp liệu thất bại. Lỗi đầu tiên (đoạn #0): đã hết hạn mức/credit."


def test_quota_stops_starting_new_rows_and_marks_the_row_for_later(tmp_path: Path) -> None:
    stack = FakeStack(task_states=["FAILURE", "SUCCESS"], failure_message=QUOTA_MESSAGE)
    state = _run(tmp_path, stack, [_row(), _row(file_id="b2", sha256="b2")], max_inflight=1)

    statuses = {row["file_id"]: row["ingest_status"] for row in state.manifest}
    assert statuses == {"a1": "quota", "b2": ""}
    assert len(stack.uploads) == 1
    assert pick_rows(state, None, None)[0]["file_id"] == "a1"


def test_quota_waits_then_retries_and_finishes(tmp_path: Path) -> None:
    stack = FakeStack(task_states=["FAILURE", "SUCCESS"], failure_message=QUOTA_MESSAGE)
    slept: list[float] = []
    state = _run(
        tmp_path,
        stack,
        [_row(), _row(file_id="b2", sha256="b2")],
        max_inflight=1,
        quota_wait=60,
        sleep=slept.append,
    )

    assert [row["ingest_status"] for row in state.manifest] == ["done", "done"]
    assert 60 in slept
    assert len(stack.uploads) == 2  # a1 is re-chunked from its existing document, not re-uploaded


def test_is_quota_reads_the_code_or_the_message() -> None:
    assert is_quota(5010, "")
    assert is_quota(None, "Đã đạt giới hạn ngân sách sử dụng mô hình AI")
    assert not is_quota(5008, "API key không hợp lệ")


def test_a_5xx_on_the_job_status_is_retried_before_failing(tmp_path: Path) -> None:
    stack = FakeStack()
    calls = {"n": 0}
    real = stack.__call__

    def flaky(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs/doc-1") and calls["n"] < 2:
            calls["n"] += 1
            return httpx.Response(500, text="Internal Server Error")
        return real(request)

    state = _run(tmp_path, flaky, [_row()])  # type: ignore[arg-type]

    assert state.manifest[0]["ingest_status"] == "done"
