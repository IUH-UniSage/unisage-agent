"""`streaming_graph._make_failover_applier` - the glue that lets a credential
failing in one node (e.g. MessageClassificationNode) spare every LATER node
in the same request from independently rediscovering the identical failure.

Each node's own failover mechanism (retry with the next credential) is
already covered per-node in `tests/graph/test_message_classification_node.py`,
`tests/graph/test_query_transformation_node.py`, and
`tests/api/test_chat_stream_errors.py`. This file covers the piece specific
to `streaming_graph.py`: that firing the callback actually updates the
shared `GraphModels` instance every node reads from.
"""

from pydantic_ai.models.function import FunctionModel

from app.core.registry.model_registry import CredentialConfig
from app.graph.streaming_graph import _make_failover_applier
from app.graph.streaming_state import GraphModels


def _credential(credential_id: str) -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider="google",
        model_name="gemini-2.5-flash",
        api_base_url="https://generativelanguage.googleapis.com",
        priority=1,
        max_rpm=60,
        api_key="key",
    )


def test_failover_applier_updates_all_three_model_fields_and_the_credential() -> None:
    original_model = FunctionModel(lambda messages, info: None, model_name="dead")  # type: ignore[arg-type]
    fallback_model = FunctionModel(lambda messages, info: None, model_name="fallback")  # type: ignore[arg-type]

    models = GraphModels(
        classification=original_model,
        query_transformation=original_model,
        generation=original_model,
        retrieval=None,  # type: ignore[arg-type] - unused by this test
    )
    fallback_credential = _credential("cred-fallback")

    apply = _make_failover_applier(models)
    apply(fallback_credential, fallback_model)

    # Every node's model field switches together - a node running after this
    # point starts directly with the known-good credential instead of
    # rediscovering the same failure the earlier node already hit.
    assert models.classification is fallback_model
    assert models.query_transformation is fallback_model
    assert models.generation is fallback_model
    assert models.generation_credential is fallback_credential


def test_graph_models_is_mutable_so_the_applier_can_update_it_in_place() -> None:
    """Regression guard: `GraphModels` must stay a plain (non-frozen)
    dataclass - freezing it again would make `_make_failover_applier`'s
    in-place mutation raise `dataclasses.FrozenInstanceError` the first time
    any node actually fails over."""

    models = GraphModels(
        classification="model-a",
        query_transformation="model-a",
        generation="model-a",
        retrieval=None,  # type: ignore[arg-type]
    )

    models.generation = "model-b"  # must not raise

    assert models.generation == "model-b"
