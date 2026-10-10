"""Ingest manifest rows through the real upload flow (Task 4 / Task 7, spec §12.1).

Same calls as the web ingest wizard, through the API Gateway with an admin's
token, so every document gets a `documents` row (editable in the admin page),
its file in MinIO, and chunks in Qdrant that normal chat retrieves:

1. backend `POST /documents` (multipart): title = manifest `title`, file name
   `<title>.pdf`, department UUID looked up from the department code, private
   documents get the `minAccessLevelId` of their `access_level`;
2. agent `POST /ingestion/chunking` (markdown_aware, the wizard's defaults)
   with `object_key = document.sourceUrl`;
3. agent `POST /ingestion/embedding` - a Celery task;
4. poll `GET /ingestion/jobs/{document_id}` until the task ends.

Progress is saved in the manifest after every step (`document_id`,
`ingest_status` in uploaded / embedding / done / empty_text / error:<reason>),
so a re-run skips `done` rows and resumes the others without re-uploading.
Runs only against an eval environment (`APP_ENV=eval`, see `task env:show`).

Usage (from unisage-agent, eval stack running):
    task env:show env=eval
    APP_ENV=eval UNISAGE_ADMIN_PASSWORD=... python -m evals.ingest \\
        --dataset ../unisage-gateway/dataset/official \\
        --only ../unisage-gateway/dataset/demo/demo_manifest.csv --limit 1
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from evals.crawl.download import DownloadState
from evals.titles import upload_name

SUCCESS_CODE = 1000
EMPTY_TEXT_CODE = 4221
CHUNKING = {"strategy": "markdown_aware", "params": {"chunk_size": 800, "overlap": 120}}
FINISHED = {"done", "empty_text"}
TERMINAL_TASK_STATES = {"SUCCESS", "FAILURE", "PARTIAL_FAILURE", "REVOKED"}


class ApiError(Exception):
    def __init__(self, status: int, code: int | None, message: str) -> None:
        super().__init__(f"{status} {code}: {message}")
        self.code = code


def _data(response: httpx.Response) -> Any:
    try:
        body = response.json()
    except ValueError:
        raise ApiError(response.status_code, None, response.text[:200]) from None
    if response.is_error or body.get("code") != SUCCESS_CODE:
        raise ApiError(response.status_code, body.get("code"), str(body.get("message"))[:200])
    return body.get("data")


@dataclass
class UniSageClient:
    """Thin wrapper over the gateway: `/api/v1/master/**` (backend-java) and
    `/api/v1/ai/**` (agent), with one admin bearer token."""

    http: httpx.Client
    token: str = ""

    def login(self, code: str, password: str) -> None:
        data = _data(
            self.http.post("/api/v1/master/auth/login", json={"code": code, "password": password})
        )
        self.token = data["accessToken"]
        self.http.headers["Authorization"] = f"Bearer {self.token}"

    def department_ids(self) -> dict[str, str]:
        """Department code (`departments.name`) -> UUID, over the whole tree."""

        found: dict[str, str] = {}
        pending = _data(self.http.get("/api/v1/master/departments/roots"))
        while pending:
            department = pending.pop()
            found[department["name"]] = department["id"]
            children = self.http.get(f"/api/v1/master/departments/{department['id']}/children")
            pending.extend(_data(children))
        return found

    def access_level_ids(self) -> dict[int, str]:
        levels = _data(self.http.get("/api/v1/master/access-levels"))
        return {int(level["level"]): level["id"] for level in levels}

    def create_document(self, fields: dict[str, str], file_name: str, content: bytes) -> Any:
        files = {"file": (file_name, content, "application/pdf")}
        return _data(self.http.post("/api/v1/master/documents", data=fields, files=files))

    def get_document(self, document_id: str) -> Any:
        return _data(self.http.get(f"/api/v1/master/documents/{document_id}"))

    def chunk(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        return list(_data(self.http.post("/api/v1/ai/ingestion/chunking", json=body))["chunks"])

    def embed(self, body: dict[str, Any]) -> None:
        _data(self.http.post("/api/v1/ai/ingestion/embedding", json=body))

    def job(self, document_id: str) -> Any:
        return _data(self.http.get(f"/api/v1/ai/ingestion/jobs/{document_id}"))


@dataclass
class Labels:
    departments: dict[str, str]
    access_levels: dict[int, str]


@dataclass
class Target:
    """What the agent needs for one row once its document exists."""

    row: dict[str, str]
    department_uuid: str
    access_level: int
    is_public: bool
    object_key: str = ""


def target_for(row: dict[str, str], labels: Labels) -> Target:
    code = row["department_id"]
    if code not in labels.departments:
        raise ValueError(f"department {code} not in the backend")
    is_public = row["is_public"] == "true"
    level = 0 if is_public else int(row["access_level"])
    if not is_public and level not in labels.access_levels:
        raise ValueError(f"access level {level} not in the backend")
    return Target(row, labels.departments[code], level, is_public)


def document_fields(target: Target, labels: Labels) -> dict[str, str]:
    fields = {
        "title": target.row["title"],
        "fileType": "PDF",
        "isPublic": str(target.is_public).lower(),
        "docPackageId": target.department_uuid,
    }
    if not target.is_public:
        fields["minAccessLevelId"] = labels.access_levels[target.access_level]
    return fields


def start(client: UniSageClient, dataset: Path, target: Target, labels: Labels) -> None:
    """Upload (unless already uploaded), chunk and dispatch embedding for one row.
    Leaves `ingest_status` at `embedding`, or `empty_text`."""

    row = target.row
    if row.get("document_id"):
        document = client.get_document(row["document_id"])
    else:
        content = (dataset / row["local_path"]).read_bytes()
        document = client.create_document(
            document_fields(target, labels), upload_name(row["title"]), content
        )
        row["document_id"] = document["id"]
        row["ingest_status"] = "uploaded"
    target.object_key = document["sourceUrl"]

    common = {
        "document_id": row["document_id"],
        "department_id": target.department_uuid,
        "object_key": target.object_key,
    }
    try:
        chunks = client.chunk({**common, **CHUNKING})
    except ApiError as error:
        if error.code == EMPTY_TEXT_CODE:
            row["ingest_status"] = "empty_text"
            return
        raise
    client.embed(
        {
            **common,
            "access_level": target.access_level,
            "is_public": target.is_public,
            "chunks": chunks,
        }
    )
    row["ingest_status"] = "embedding"


def poll(client: UniSageClient, row: dict[str, str]) -> bool:
    """True once the row's embedding task has ended (status set accordingly)."""

    job = client.job(row["document_id"])
    state = job.get("task_state")
    if state not in TERMINAL_TASK_STATES:
        return False
    if state == "SUCCESS":
        row["ingest_status"] = "done"
    else:
        row["ingest_status"] = f"error:{state} {job.get('task_message') or ''}".strip()
    return True


@dataclass
class Run:
    client: UniSageClient
    state: DownloadState
    labels: Labels
    max_inflight: int = 4
    poll_seconds: float = 5.0
    inflight: list[dict[str, str]] = field(default_factory=list)

    def _fail(self, row: dict[str, str], error: Exception) -> None:
        row["ingest_status"] = f"error:{str(error)[:200]}"
        print(f"  ! {row['file_id']}: {row['ingest_status']}", file=sys.stderr)

    def _drain(self, until: int) -> None:
        while len(self.inflight) > until:
            for row in list(self.inflight):
                try:
                    finished = poll(self.client, row)
                except (ApiError, httpx.HTTPError) as error:
                    self._fail(row, error)
                    finished = True
                if finished:
                    self.inflight.remove(row)
                    print(f"  {row['file_id']}: {row['ingest_status']}")
                    self.state.save()
            if len(self.inflight) > until:
                time.sleep(self.poll_seconds)

    def ingest(self, rows: list[dict[str, str]]) -> None:
        for index, row in enumerate(rows, 1):
            if row.get("ingest_status") == "embedding":
                self.inflight.append(row)
            else:
                print(f"[{index}/{len(rows)}] {row['title'][:80]}")
                try:
                    start(
                        self.client, self.state.dataset, target_for(row, self.labels), self.labels
                    )
                except (ApiError, httpx.HTTPError, ValueError, OSError) as error:
                    self._fail(row, error)
                self.state.save()
                if row["ingest_status"] == "embedding":
                    self.inflight.append(row)
            self._drain(until=self.max_inflight - 1)
        self._drain(until=0)


def pick_rows(state: DownloadState, only: Path | None, limit: int | None) -> list[dict[str, str]]:
    wanted: set[str] | None = None
    if only is not None:
        with only.open(newline="", encoding="utf-8-sig") as handle:
            wanted = {row["file_id"] for row in csv.DictReader(handle)}
    rows = [
        row
        for row in state.manifest
        if row.get("selected") == "true"
        and row.get("ingest_status") not in FINISHED
        and (wanted is None or row["file_id"] in wanted)
    ]
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--only", type=Path, help="CSV whose file_id column limits the rows")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--gateway", default=os.getenv("UNISAGE_GATEWAY_URL", "http://localhost:8400")
    )
    parser.add_argument("--code", default=os.getenv("UNISAGE_ADMIN_CODE", "SA-001"))
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="list the rows, call nothing")
    args = parser.parse_args()

    if os.getenv("APP_ENV") != "eval":
        sys.exit("APP_ENV is not 'eval': run through `task ... env=eval` or set APP_ENV=eval")

    if not (args.dataset / "manifest.csv").is_file():
        sys.exit(
            f"no manifest.csv in {args.dataset}: set EVAL_DATASET_DIR in .env/.env.eval or pass "
            "dataset=<gateway repo>/dataset/official to task"
        )
    if args.only is not None and not args.only.is_file():
        sys.exit(f"--only list not found: {args.only}")
    state = DownloadState(args.dataset)
    rows = pick_rows(state, args.only, args.limit)
    missing = [row["file_id"] for row in rows if not row.get("title")]
    if missing:
        sys.exit(f"rows without a title (run evals.titles first): {missing[:10]}")
    no_pdf = [row["local_path"] for row in rows if not (args.dataset / row["local_path"]).is_file()]
    if no_pdf:
        sys.exit(
            f"{len(no_pdf)} PDF(s) missing under {args.dataset}, e.g. {no_pdf[0]}. Download: "
            "hf download hgjyhm/unisage-iuh-dataset --repo-type dataset "
            '--include "official/files/**" --local-dir <gateway repo>/dataset'
        )
    print(f"{len(rows)} rows to ingest via {args.gateway}")
    if args.dry_run:
        for row in rows:
            print(
                f"  {row['file_id']} {row['department_id']} public={row['is_public']} "
                f"level={row['access_level'] or '-'} {upload_name(row['title'])}"
            )
        return

    password = os.getenv("UNISAGE_ADMIN_PASSWORD")
    if not password:
        sys.exit("set UNISAGE_ADMIN_PASSWORD (the admin's password in the eval backend)")
    with httpx.Client(base_url=args.gateway, timeout=120) as http:
        client = UniSageClient(http)
        client.login(args.code, password)
        labels = Labels(client.department_ids(), client.access_level_ids())
        Run(client, state, labels, max_inflight=args.max_inflight).ingest(rows)

    done = [row for row in rows if row.get("ingest_status") == "done"]
    errors = [row for row in rows if str(row.get("ingest_status", "")).startswith("error")]
    print(f"done {len(done)}/{len(rows)}, errors {len(errors)}")


if __name__ == "__main__":
    main()
