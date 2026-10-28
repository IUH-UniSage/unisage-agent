"""Rerank benchmark (docs/specs/SPEC-cross-encoder-rerank.md): the LLM rerank against the
cross-encoder service on the candidates an eval run retrieved.

    task benchmark:run env=eval -- hybrid-on -a            # accuracy on runs/hybrid-on
    task benchmark:run env=eval -- hybrid-on -p -c          # latency + concurrency
    task benchmark:run env=eval -- hybrid-on --all --variant cross_encoder

Modules: `candidates` (what each question's rerank gets to see), `variants` (the three
rerank strategies behind one signature), `metrics`, `runner` (-a / -p / -c),
`resource_sampler` (CPU / RAM / VRAM thread), `service` (spawn and stop the reranker,
check the VRAM comes back), `report` (markdown + json), `cli`.
"""
