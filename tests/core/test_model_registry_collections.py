from collections.abc import Iterator

import pytest

from app.core.config import settings
from app.core.registry import model_registry
from app.core.registry.model_registry import (
    active_collection,
    index_collections,
    open_reindex,
    parse_snapshot,
)


def _payload(**extra: object) -> dict[str, object]:
    return {
        "version": 3,
        "generatedAt": "2026-10-17T03:00:00Z",
        "purposes": {},
        "embeddingIndexIdentity": None,
        **extra,
    }


@pytest.fixture(autouse=True)
def _no_snapshot() -> Iterator[None]:
    previous = model_registry._current_snapshot
    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = previous


def test_without_a_snapshot_the_base_collection_is_live() -> None:
    assert active_collection() == settings.QDRANT_COLLECTION
    assert open_reindex() is None
    assert index_collections() == (settings.QDRANT_COLLECTION,)


def test_a_backend_that_predates_reindex_keeps_the_base_collection() -> None:
    model_registry.set_current_snapshot(parse_snapshot(_payload()))

    assert active_collection() == settings.QDRANT_COLLECTION
    assert index_collections() == (settings.QDRANT_COLLECTION,)


def test_the_snapshot_names_the_live_collection_after_a_cutover() -> None:
    model_registry.set_current_snapshot(
        parse_snapshot(_payload(embeddingCollection="unisage_chunks__20261017030000"))
    )

    assert active_collection() == "unisage_chunks__20261017030000"


def test_an_open_reindex_adds_its_target_to_the_collections_edits_reach() -> None:
    model_registry.set_current_snapshot(
        parse_snapshot(
            _payload(
                embeddingCollection="unisage_chunks",
                embeddingReindex={
                    "runId": "run-1",
                    "status": "QUEUED",
                    "targetCollection": "unisage_chunks__20261017030000",
                },
            )
        )
    )

    run = open_reindex()
    assert run is not None and run.run_id == "run-1" and run.status == "QUEUED"
    assert index_collections() == ("unisage_chunks", "unisage_chunks__20261017030000")
