"""Celery secret-safety tests — plan.md "Secret redaction":

  "Celery: verifier task ignore_result=True (+ store_errors_even_if_ignored=False),
  không nhận argument chứa credential — task tự gọi claim, claim response chỉ
  sống trong biến cục bộ, không bao giờ đi qua broker hay result backend. Task
  khác của feature này cũng không nhận credential làm argument."

Celery persists a task's call arguments to the broker (as the task message)
and, unless `ignore_result=True`, its return value to the result backend — a
credential passed as an argument or returned in a result would sit there in
plaintext for as long as the message/result lives. Two checks, scoped to every
task this service currently registers:

  1. No task parameter name looks like it's meant to carry a raw secret
     (`api_key`, `apiKey`, `secret`, `credential`, `token`, `password`, ...).
     This is a naming-convention check, not a values/dataflow check — it
     can't prove a param typed `dict`/`Any` never *contains* a credential
     under some other key, only that no parameter is *named* like one.
  2. Every registered task has `ignore_result=True` and
     `store_errors_even_if_ignored=False`, **except** `embed_chunks`, which
     is a documented, pre-existing exception: its return value is the
     ingestion client's reconciliation-sweep mechanism (see
     `app/worker/celery_app.py`'s own docstring — `result_expires` is set to
     7 days specifically so that sweep can read a terminal state), and it
     carries no credential (verified by check 1) — only chunk-processing
     status. Any *new* task this feature adds (verifier, health-report, ...)
     must not be added to this exemption list; it must instead set
     `ignore_result=True` like `beat_heartbeat` already does.
"""

from __future__ import annotations

import inspect

import app.worker.tasks  # noqa: F401 - registers every task on celery_app
from app.worker.celery_app import celery_app

# plan.md "Secret redaction" — see module docstring for why this is
# grandfathered rather than fixed by setting ignore_result=True.
_RESULT_RETENTION_EXEMPT_TASKS = {"embed_chunks"}

_SECRET_LOOKING_PARAM_SUBSTRINGS = (
    "api_key",
    "apikey",
    "secret",
    "credential",
    "token",
    "password",
    "passwd",
)


def _our_tasks() -> dict[str, object]:
    return {
        name: task
        for name, task in celery_app.tasks.items()
        if not name.startswith("celery.")  # exclude Celery's own builtin tasks
    }


def test_at_least_one_task_is_registered() -> None:
    """Guards against an import-order regression silently emptying
    `celery_app.tasks` and every check below passing vacuously."""

    assert _our_tasks(), "expected at least one non-builtin task to be registered"


def test_no_task_parameter_looks_like_a_raw_secret() -> None:
    violations: list[str] = []
    for name, task in _our_tasks().items():
        signature = inspect.signature(task.run)
        for param_name in signature.parameters:
            lowered = param_name.lower()
            if any(needle in lowered for needle in _SECRET_LOOKING_PARAM_SUBSTRINGS):
                violations.append(f"task {name!r} has a secret-looking parameter {param_name!r}")

    assert not violations, "\n".join(violations)


def test_non_exempt_tasks_ignore_their_result() -> None:
    violations: list[str] = []
    for name, task in _our_tasks().items():
        if name in _RESULT_RETENTION_EXEMPT_TASKS:
            continue
        if not getattr(task, "ignore_result", False):
            violations.append(
                f"task {name!r} does not set ignore_result=True — its return value "
                "will be persisted to the Celery result backend"
            )
        if getattr(task, "store_errors_even_if_ignored", False):
            violations.append(
                f"task {name!r} sets store_errors_even_if_ignored=True, which persists "
                "a failed task's exception (and traceback) to the result backend even "
                "though ignore_result=True"
            )

    assert not violations, "\n".join(violations)


def test_exemption_list_only_contains_currently_registered_tasks() -> None:
    """Guards against the exemption list silently going stale (a task renamed
    or removed while still listed here would make the exemption meaningless)."""

    task_names = set(_our_tasks())
    stale = _RESULT_RETENTION_EXEMPT_TASKS - task_names
    assert not stale, f"exempted task(s) no longer registered: {stale}"
