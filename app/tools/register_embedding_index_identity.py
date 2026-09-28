"""Bootstrap CLI — registers the embedding identity of an EXISTING Qdrant collection before the
model registry is trusted for embedding (plan.md "Embedding identity guard" — a mandatory
rollout step).

Run once, by hand, with the OLD `.env` embedding configuration, BEFORE flipping
`MODEL_REGISTRY_ENABLED=true` for embedding:

    python -m app.tools.register_embedding_index_identity

It measures the fingerprint of the credential that actually produced the vectors already sitting
in `settings.QDRANT_COLLECTION`, reads that collection's configured vector dimension, and calls
`PUT /internal/model-registry/embedding-index/{collection}/identity` (only-if-absent — a second
run, or a run after `app.core.registry.embedding_identity`'s own first-upsert bootstrap already
won the race, gets a 409 and this tool reads it back and confirms it matches rather than treating
that as an error).

Credential source — deliberately NOT `app.core.config.Settings`
-----------------------------------------------------------------
An earlier cutover session removed every `OPENAI_*` field from `Settings` (the model registry is
now the only credential source for the rest of the app). This tool is the one deliberate
exception: it exists specifically to run BEFORE the registry can be trusted for embedding, using
whatever credential produced the vectors that are already in Qdrant — which is usually still
described by the OLD `.env` file's `OPENAI_API_KEY`/`OPENAI_EMBEDDING_MODEL`, not by any registry
row yet. Reading `Settings` here would be circular (this tool exists because the registry isn't
trusted yet), so it reads the environment directly (`OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL`,
`OPENAI_API_BASE`) with CLI flags as the override — whichever is more convenient for a one-off,
by-hand run. Nothing here is read by any other module; this is a standalone script, not a config
source for the running services.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from openai import OpenAI

from app.core.config import settings
from app.core.llm.embedding_probe import (
    EmbeddingFingerprint,
    fingerprints_match,
    measure_fingerprint,
    unflatten_fingerprint,
)
from app.core.llm.http_client import ProviderConnectionInfo, build_provider_http_client_sync
from app.core.observability.logging_config import configure_logging
from app.integrations.backend_java_client import BackendJavaClient, BackendJavaHTTPError
from app.rag.vectorstore import qdrant_store

_DEFAULT_MODEL_NAME = "text-embedding-3-small"
_DEFAULT_API_BASE_URL = "https://api.openai.com/v1"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register the embedding identity of the existing Qdrant collection, using the OLD "
            "(pre-registry) embedding credential. Run once, before enabling the model registry "
            "for embedding."
        )
    )
    parser.add_argument(
        "--collection",
        default=settings.QDRANT_COLLECTION,
        help="Qdrant collection to register (default: the configured QDRANT_COLLECTION).",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Provider name to record as the collection's identity (default: openai).",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", _DEFAULT_MODEL_NAME),
        help=(
            "Embedding model that produced the vectors already in the collection "
            "(default: $OPENAI_EMBEDDING_MODEL, falling back to "
            f"'{_DEFAULT_MODEL_NAME}')."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key for the credential above (default: $OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("OPENAI_API_BASE", _DEFAULT_API_BASE_URL),
        help=(
            f"API base URL (default: $OPENAI_API_BASE, falling back to '{_DEFAULT_API_BASE_URL}')."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.api_key:
        print(
            "No API key given - pass --api-key or set $OPENAI_API_KEY to the credential that "
            "produced the vectors already in the collection.",
            file=sys.stderr,
        )
        return 1

    qdrant_client = qdrant_store.get_client()
    if not qdrant_client.collection_exists(args.collection):
        print(
            f"Collection '{args.collection}' does not exist yet - nothing to register an "
            "identity against. This tool is only for an EXISTING collection that already has "
            "vectors from a pre-registry credential; an empty collection establishes its own "
            "identity on the first ingest batch instead "
            "(see app.core.registry.embedding_identity).",
            file=sys.stderr,
        )
        return 1

    dimension = qdrant_store.get_collection_dimension(qdrant_client)
    if dimension is None:
        print(
            f"Could not read the configured vector dimension for '{args.collection}'.",
            file=sys.stderr,
        )
        return 1

    openai_client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
        http_client=build_provider_http_client_sync(
            ProviderConnectionInfo(api_base_url=args.api_base_url)
        ),
    )

    def _embed_probe(texts: list[str]) -> list[list[float]]:
        response = openai_client.embeddings.create(model=args.model_name, input=texts)
        return [item.embedding for item in response.data]

    fingerprint = measure_fingerprint(_embed_probe)
    if fingerprint.dimension != dimension:
        print(
            f"Measured fingerprint dimension ({fingerprint.dimension}) does not match the "
            f"collection's configured vector dimension ({dimension}) - refusing to register a "
            "self-contradictory identity. Check --model-name/--api-base-url.",
            file=sys.stderr,
        )
        return 1

    backend_client = BackendJavaClient()
    try:
        asyncio.run(
            backend_client.put_embedding_index_identity(
                collection=args.collection,
                provider=args.provider,
                model_name=args.model_name,
                model_source_ref=None,
                api_base_url=args.api_base_url,
                dimension=fingerprint.dimension,
                fingerprint=fingerprint.flattened(),
                established_by="bootstrap-cli",
            )
        )
    except BackendJavaHTTPError as exc:
        if exc.status_code != 409:
            print(f"Failed to register embedding identity: {exc}", file=sys.stderr)
            return 1
        return _reconcile_after_already_registered(
            backend_client, args.collection, args.provider, args.model_name, fingerprint
        )

    print(
        f"Registered embedding identity for '{args.collection}': provider={args.provider} "
        f"model={args.model_name} dimension={fingerprint.dimension}"
    )
    return 0


def _reconcile_after_already_registered(
    backend_client: BackendJavaClient,
    collection: str,
    provider: str,
    model_name: str,
    fingerprint: EmbeddingFingerprint,
) -> int:
    """A 409 means an identity is already registered - plan.md's only-if-absent contract means
    this tool never overwrites it. Read it back and confirm it's the SAME identity this run would
    have registered, rather than silently treating "already registered" as success without
    checking."""

    existing = asyncio.run(backend_client.get_embedding_index_identity(collection=collection))
    if existing is None:
        print(
            "PUT returned 409 (already registered) but a follow-up GET found no identity - "
            "this should not happen; investigate before retrying.",
            file=sys.stderr,
        )
        return 1

    if existing.get("provider") != provider or existing.get("modelName") != model_name:
        print(
            f"An embedding identity is already registered for '{collection}', but it does NOT "
            f"match this run: registered provider={existing.get('provider')} "
            f"model={existing.get('modelName')}, this run provider={provider} model={model_name}. "
            "Not overwriting - resolve manually.",
            file=sys.stderr,
        )
        return 1

    registered_dimension = existing.get("dimension")
    registered_fingerprint_raw = existing.get("fingerprint")
    if (
        registered_dimension != fingerprint.dimension
        or registered_fingerprint_raw is None
        or not fingerprints_match(
            fingerprint,
            unflatten_fingerprint(registered_fingerprint_raw, int(registered_dimension)),
        )
    ):
        print(
            f"An embedding identity is already registered for '{collection}' with the same "
            "provider/model, but its measured fingerprint or dimension does not match this run. "
            "Not overwriting - resolve manually.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Embedding identity for '{collection}' is already registered and matches this run "
        "(provider/model/dimension/fingerprint) - nothing to do."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
