"""Offline RAG evaluation harness for UniSage.

Not part of the deployed service (`app/` only): these modules build the
evaluation dataset (crawl → label → ingest → questions) and score the chat
graph against it. See `unisage-backend/changes/30-09-2026-RAG-Evaluation/`.
"""
