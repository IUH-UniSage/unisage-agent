# Known Gaps: Ingestion Service

Tracked deliberately, not silently patched around. See `SPEC-ingestion.md` for
the full ingestion pipeline spec these gaps belong to.

## Java has no `.xlsx` in `AllowedFileType`

The `excel_row` chunking strategy is fully implemented and unit-tested
(`app/rag/chunking/excel_rows.py`), but `unisage-backend`'s upload whitelist
(`AllowedFileType`) does not yet include `.xlsx`. Until that's added on the
Java side, no `.xlsx` object can reach MinIO through the normal upload path,
so `excel_row` is unreachable end-to-end in practice even though the API and
chunker both work against a manually-placed `.xlsx` object.

## No caller updates Java's `DocStatus` after ingestion completes

Java's `Document.status` lifecycle (`PENDING -> COMPLETED`) has no caller in
this scope: per the spec, Python never calls back into `unisage-backend`, and
the embedding task's completion is only observable via the
`/ingestion/embedding/{task_id}/progress` WebSocket. A document that has
finished embedding will not have its Java-side status updated unless a future
change adds a webhook/callback (out of scope here, and explicitly listed as
never in `SPEC-ingestion.md`'s Boundaries: "never call back into the Java
backend from Python").

## `.doc` (legacy binary Word format) is not parsed

`extract_raw_text` (`app/rag/ingestion/parser.py`) and `split_regions`
(`app/rag/ingestion/table_aware_parser.py`) support `.txt`, `.pdf`, and
`.docx`, but raise `UnsupportedFileTypeException` for `.doc`. `pymupdf`/
`pymupdf4llm` only parse OOXML documents (`.docx`); the older binary `.doc`
format needs a different converter (e.g. LibreOffice headless, `antiword`)
that isn't part of this project's dependencies. If `.doc` uploads need to
work, this requires either adding such a converter or asking users to
re-save as `.docx` before upload.
