# Known Gaps

Tracked deliberately, not silently patched around.

## Ingestion

See `SPEC-ingestion.md` for the full ingestion pipeline spec these gaps
belong to.

### Java has no `.xlsx` in `AllowedFileType`

The `excel_row` chunking strategy is fully implemented and unit-tested
(`app/rag/chunking/excel_rows.py`), but `unisage-backend`'s upload whitelist
(`AllowedFileType`) does not yet include `.xlsx`. Until that's added on the
Java side, no `.xlsx` object can reach MinIO through the normal upload path,
so `excel_row` is unreachable end-to-end in practice even though the API and
chunker both work against a manually-placed `.xlsx` object.

### No caller updates Java's `DocStatus` after ingestion completes

Java's `Document.status` lifecycle (`PENDING -> COMPLETED`) has no caller in
this scope: per the spec, Python never calls back into `unisage-backend`, and
the embedding task's completion is only observable via the
`/ingestion/embedding/{task_id}/progress` WebSocket. A document that has
finished embedding will not have its Java-side status updated unless a future
change adds a webhook/callback (out of scope here, and explicitly listed as
never in `SPEC-ingestion.md`'s Boundaries: "never call back into the Java
backend from Python").

### A closed browser tab leaves a draft stuck in `EMBEDDING` forever

`document_process_logs.current_step` moves from `CHUNKED` to `EMBEDDING`
when `POST /ingestion/embedding` dispatches a Celery task, and back out
(row deleted) only via `DELETE /ingestion/jobs/{document_id}`, which the
frontend calls once it observes the task's WebSocket progress reach a
terminal state. The Celery worker itself has no DB session and never
writes this transition. If no client is ever open to observe the terminal
frame (tab closed and never reopened for that document), the row stays
`EMBEDDING` indefinitely with a `celery_task_id` whose Celery result will
eventually expire from the result backend — reopening the document later
shows a progress view stuck in the WebSocket's `error` fallback state
rather than resolving. See `SPEC-ingestion-resume.md`'s "Extension:
resuming into an in-flight embed" section for the full design and why a
worker-side or scheduled-sweep fix was left out of scope.

### Structural chunking metadata (2026-09) - known limitations

See `SPEC-ingestion.md`'s "Structural Chunking Metadata" section and
`changes/13-09-2026-Chunking-Structural-Metadata/plan.md` for the full design.
Deliberate, reviewed-and-accepted limitations, not bugs to silently patch:

- **DOCX heading detection** only recognizes Word's built-in `"Heading N"`
  paragraph styles (`paragraph.style.name` matching `^Heading (\d)$`). A
  document that uses a custom style name for its headings (e.g. a renamed or
  organization-specific style) will not have that heading picked up into
  `heading_path`.
- **PDF heading detection is font-size-based first, with a narrow ordinal-line
  fallback** (`pymupdf4llm`'s `IdentifyHeaders`: any font size larger than the
  document's most common ("body") font size becomes a `#`..`######` markdown
  heading; bold/underline/italic alone, at body font size, is never treated
  as a heading by `pymupdf4llm` itself). `_regions_from_pdf_pages` (2026-09-15,
  extended 2026-09-16) adds one narrow, document-pattern-specific fallback on
  top of that for a Roman-numeral (`I.`/`II.`/`III.`) or Arabic-numeral
  (`1.`/`2.`/`3.`) line at body font size - optionally wrapped in `**` (a
  fully-bold line) - that `pymupdf4llm` never marks up (`_looks_like_pdf_heading`):
  it is promoted to a heading ONLY if it carries an underline span
  (`<u>...</u>`), OR is entirely upper-case once its ordinal prefix is
  stripped (`_is_all_caps_title` - catches a bold-but-not-larger Roman-numeral
  section title), OR the very next non-blank line starts a table (`|`) or a
  BULLET list item (`-`) - deliberately NOT a numbered-list item, which would
  otherwise promote every item of an ordinary numbered list into its own
  heading. A plain numbered clause followed by more prose (e.g.
  `"1. Họ tên sinh viên: ..."`) matches none of these signals and stays
  ordinary text - this is NOT a general "every ordinal line is a heading"
  rule - see `_looks_like_pdf_heading`'s docstring.
  **Numbering-scheme-based level override** (`_heading_level_for_title`,
  2026-09-16): Vietnamese business documents commonly mix Roman-numeral
  top-level sections with Arabic-numeral subsections that `pymupdf4llm`
  renders at the EXACT SAME raw markdown heading level (it only looks at
  font size, with no notion of semantic nesting) - naively trusting that raw
  level let an Arabic subsection (`"1. Module..."`) pop a Roman section
  (`"II. Phân tích..."`) off `heading_stack` as if they were siblings,
  permanently losing the Roman section from every subsequent chunk's
  `heading_path` (confirmed as a real bug against `BAS.pdf`). Every heading -
  both `#`-detected and promoted - now has its nesting level overridden by
  its own numbering scheme (Roman title -> shallower, Arabic title -> one
  level deeper), independent of the raw markdown level, so Roman sections
  stay on the stack while Arabic subsections correctly nest under them. Known
  false-positive risk of the Roman-numeral regex (`^[IVXLCDM]+\.`): a single
  capital letter that is also a valid Roman numeral (I, V, X, L, C, D, M) as
  a *lettered* list marker (`"A. ... B. ... C. ..."`) would be misread as a
  Roman-numeral heading if it also matches one of the 2 promotion signals -
  not yet seen in a real document, but a known theoretical gap.
  **`_heading_level_for_title` is 2 hardcoded numbering buckets (Roman -> 2,
  Arabic -> 3), not a general N-level numbering-scheme detector** - it was
  designed and validated against the 2 real PDFs on hand, both of which mix
  exactly 2 numbering tiers (Roman sections, Arabic subsections). A document
  with a 3rd numbering tier - e.g. `Tiêu đề > A./B./C./D. > I./II./III. >
  1./2./3.` (lettered top sections, Roman sub-sections, Arabic sub-
  subsections) - would break the same way the original bug did, one tier up:
  a lettered heading (`"A."`, `"B."`) matches neither the Roman nor the
  Arabic regex, so it falls through to `fallback_level` (pymupdf4llm's raw
  `#`-count, or the flat promoted-heading constant) instead of getting its
  own dedicated level - if that raw level collides with the Roman tier's
  raw level (plausible, since `pymupdf4llm` only looks at font size), the
  lettered section gets popped off `heading_stack` by a Roman sub-heading
  exactly like "II." was popped by "1." before this fix. It gets worse for
  `"C."`/`"D."` specifically: single-letter Roman numerals (I, V, X, L, C, D,
  M) mean a *lettered* heading using one of those letters is itself
  misclassified as level-2 Roman by the regex, colliding directly with real
  Roman sub-headings at the same level. Fixing this properly needs a
  different design - assigning each heading's level dynamically based on
  the order distinct numbering styles are first encountered in the document
  (not a fixed lookup table) - which has NOT been built, since no real
  document with a 3rd numbering tier has been seen yet; building it against
  a hypothetical case risks the same "invented structure that isn't
  evidenced" problem this heuristic work has otherwise avoided. If such a
  document turns up, this is the first place to revisit.
  `heading_path` entries also have markdown emphasis markers (`**`/`__`) and
  HTML tags stripped (`_clean_heading_title`), and a plain-text line repeated
  verbatim across 2+ pages (a running header/footer) is dropped before
  heading/region processing (`_find_pdf_page_boilerplate`) so it can't become
  its own tiny chunk once a heading right after it gets promoted. Confirmed
  against 2 real PDFs: `DuongHoangHuy_DeCuongTTDN.pdf` (2026-09-15: an
  underlined, table-introducing numbered line correctly promoted; genuinely
  plain numbered clauses correctly left alone; repeated letterhead lines
  correctly dropped) and `BAS.pdf` (2026-09-15: a table row with a genuinely
  empty leading/trailing cell - see the `_markdown_row_cells` fix below;
  2026-09-16: 2 Roman-numeral top sections were vanishing from
  `heading_path` due to the level-collision bug above, and 3 bold-but-
  not-larger section titles - one Arabic sibling subsection, two Roman-
  numeral all-caps titles - were staying unpromoted as plain text). This
  remains a heuristic, not a structural guarantee: an ordinal line followed
  by ordinary mixed-case prose that a human would still consider a heading
  (no underline, not all-caps, no table/bullet-list right after it) is still
  missed - a known, accepted residual gap, not a bug to keep chasing further.
- **`_markdown_row_cells` (PDF pipe-table parsing) fix (2026-09-15)**: the
  original implementation used `line.strip("|")`, which strips an
  *unbounded run* of `|` characters from each end - a row with a genuinely
  empty leading or trailing cell renders as a double pipe at that edge
  (`"||content||"`: empty cell + delimiter), which `strip("|")` collapsed
  into one, silently dropping that cell and desyncing the row's cell count
  from the header's. This tripped `TableRowChunker`'s `TableStructureError`
  self-check (correctly - it caught the corruption, but did so by crashing
  the whole request instead of parsing the table) on a real PDF
  (`BAS.pdf`). Fixed to strip exactly the single leading/trailing delimiter
  pipe, preserving any inner empty cells.
- **PDF `header_confidence = 0.6`** for every table PDF's chunker infers a
  header row for is a fixed constant, not calibrated against any real data
  (`pymupdf4llm` gives no structural header signal for PDF tables, unlike
  HTML `<th>`/DOCX `<w:tblHeader>`). Calibrating this against real
  labeled data is future work.
- **The "don't cut mid-sentence" check remains a weak heuristic** (a
  warning log, not a hard validation rule) for `RecursiveChunker`/
  `TokenBasedChunker`/`SemanticChunker` - only the table row-atomicity
  guarantee (`TableRowChunker`) is a hard invariant.
- **Citation stays LLM-generated free text**, not a structured
  `citation_index -> {chunk_id, source, page}` mapping. Confirmed by reading
  `app/api/v1/chat.py` (2026-09-14): `POST /chat/stream` streams raw LLM
  tokens over SSE and stores the response verbatim in Java - there is no
  code path anywhere that parses `[1]`/`[2]` into a structured object. This
  was a deliberate scope decision (a structured citation API would be a
  cross-repo change touching Java + this service + `unisage-web`'s SSE
  consumer), not a gap discovered after the fact. Consequence: whether a
  page number actually appears correctly in a given answer depends entirely
  on the LLM following `citation_rules.yaml` - there is no server-side
  guarantee or validation of citation accuracy.
- **XLSX `header_source = EXPLICIT`/`confidence = 1.0`** encodes a
  pre-existing PRODUCT CONTRACT ("the first row of an uploaded spreadsheet is
  always its header"), not a verified structural signal the way HTML `<th>`
  or DOCX `<w:tblHeader>` are. If a user uploads an `.xlsx` file whose first
  row is not actually a header, the system will still treat it as one -
  this is unchanged prior behavior, not a new limitation introduced here.
- **A PDF table spanning multiple pages gets a different `table_id` per
  page** (`table_id = f"table-{block_index}"`, and `block_index` changes
  across a page boundary), even though it is logically one table. Merging
  those into a single citable table would need a `table_group_id`
  distinct from `block_index` - left for a future phase, does not block
  citation-by-page (the primary goal here).
- **A PDF table that ends mid-page is merged with the next page's table when
  that table repeats its header** (2026-09-24, accepted as a provisional
  rule). `decide_merge` (`app/rag/ingestion/table_merger.py`) used to block a
  merge whenever the first table did not reach the bottom of its page. Real
  documents often break a table early and continue it at the top of the next
  page under a repeated header (`Quyet Dinh 1035 - Hoc phi 2025-2026.pdf`
  pages 3-4, `Thong Bao 867 - Tuyen sinh Dai hoc Chinh quy 2026.pdf` pages
  2-3-4). Split, each part got its own hierarchy, so rows after the break lost
  their ancestors (`5 Đại học liên thông` lost `A ĐỐI VỚI TRỤ SỞ CHÍNH`, and
  `B PHÂN HIỆU…` was wrongly nested under `5 > 5.2`). Page position now only
  counts as a missing signal, not a blocker, when all of these hold: the
  second table starts at the top of the next page, its header matches
  (`MERGE_SCORING.early_break_min_header`, 0.9; a column also matches when
  one name is a prefix of the other, e.g. `… (Tiếp theo)`), and the column
  borders line up (`early_break_min_bounds`, 0.8). The existing blockers
  still apply: nothing but page furniture between the tables, the same
  `heading_path`, and the next page. **Residual risk:** two genuinely
  different tables that share a header, sit on consecutive pages with
  nothing between them, and have the second starting at the top of its page
  are merged into one table. Their rows and cells stay correct, but they
  share one `table_id` and one continuous row numbering, and the second
  table's rows can pick up ancestors from the first. The geometry alone
  cannot tell this case apart from a table broken early. Pinned by
  `test_two_short_tables_with_the_same_header_on_consecutive_pages_are_merged`.
  **Alternative if this bites:** keep the tables separate and instead seed
  the second table's hierarchy inference with the first table's final
  ancestor stack when their headers and `heading_path` match. That fixes the
  ancestors without merging `table_id`/row numbering, at the cost of a
  cross-table hand-off in `infer_hierarchy`.
- **Draft re-chunk race condition**: if a user re-chunks the same document
  (overwriting its draft, `upsert_chunking_draft` is "last write wins", no
  locking) while another tab/request is mid-`POST /ingestion/embedding` for
  the same `document_id` (past the canonical-draft read, not yet dispatched
  to Celery), the two requests can observe two different draft versions with
  no defined "winner". Accepted for this phase deliberately (no
  `draft_version`/optimistic-lock check was added, to avoid scope creep
  beyond structural metadata) - if this becomes a real problem with multiple
  concurrent editors per document, add draft versioning as a follow-up.
- **`POST /ingestion/embedding` now rejects (`HTTP 409`) any draft whose
  `chunking_version == "legacy"`**, or where any canonical chunk still has
  `source_type`/`block_index` as `None` (pre-migration data). This only
  blocks creating NEW embeddings from an old draft - chunks already upserted
  into Qdrant from before this change are untouched; no backfill/migration
  of existing Qdrant points was done or is planned (confirmed acceptable:
  existing Qdrant data is not a concern for this project).

### `.doc` (legacy binary Word format) is not parsed

`extract_raw_text` (`app/rag/ingestion/parser.py`) and `split_regions`
(`app/rag/ingestion/table_aware_parser.py`) support `.txt`, `.pdf`, and
`.docx`, but raise `UnsupportedFileTypeException` for `.doc`. `pymupdf`/
`pymupdf4llm` only parse OOXML documents (`.docx`); the older binary `.doc`
format needs a different converter (e.g. LibreOffice headless, `antiword`)
that isn't part of this project's dependencies. If `.doc` uploads need to
work, this requires either adding such a converter or asking users to
re-save as `.docx` before upload.

## Graph / Retrieval

See `RAG_Graph/KLTN/nodes/` for the full node-by-node design these gaps
belong to, and `backend-java`'s `changes/24-09-2026-GraphFlowAlignment/`
(`plan.md`'s Architecture Decisions, `todo.md`) for the implementation
decisions behind each one.

### No hybrid search, RRF, or cross-encoder rerank (2026-09)

The design (`RAG_Graph/KLTN/nodes/08`, `09`) describes `RetrievalFilteringNode`
as pre-filter + dense + BM25 + RRF fusion, and `PostRetrievalRerankNode` as a
`bge-reranker-base` cross-encoder pass + a 0.70 threshold + context
compression. The code does pre-filter + dense search across 3 named vectors
(`content`/`summary`/`questions`, keeping each chunk's best score) + a
threshold on the raw cosine score. This is a deliberate decision, not an
oversight:

- **BM25 / sparse search:** its main value is exact matching on codes/numbers
  (`QĐ-45/2023`, `INT1001`), while students mostly ask by meaning. The
  runtime cost is near zero, but it needs a sparse vector added to the Qdrant
  collection - i.e. an ingestion change plus **re-indexing every document**.
  No evidence yet that dense search actually misses this kind of question, so
  that cost isn't justified.
- **RRF:** RRF scores are rank-based (`1/(60 + rank)`, in the 0.01-0.03
  range), not on the same scale as the 0.70 threshold. Switching node 08 to
  RRF without a cross-encoder re-scoring at node 09 would put every chunk
  under the threshold and send every question to TicketFallback. RRF only
  makes sense paired with a reranker. The MULTI flow's sub-queries are
  therefore merged by quota + best score, not RRF.
- **Cross-encoder rerank:** the most expensive piece (per-question API cost,
  or RAM/CPU/GPU to self-host). Its benefit is reduced here because (1) the
  pre-generated `questions` vector already closely matches how students
  phrase things, and (2) each turn only retrieves 8 chunks. Also, the
  design's `bge-reranker-base` is weak on Vietnamese; if this is built,
  `bge-reranker-v2-m3` (self-hosted via HuggingFace TEI) or a multilingual
  API (Cohere `rerank-v3.5`, Jina, Voyage) would be the better pick.
- **Context compression** (`agents/reranker_compressor.yaml`): the template is
  loaded but unused by any node, since it only makes sense once reranking
  exists.

**Residual risk:** `CHAT_RERANK_SCORE_THRESHOLD = 0.70` is applied to
`text-embedding-3-small`'s raw cosine score, while the design set this
threshold for a cross-encoder score. A genuinely relevant chunk pair can
score under 0.70 on cosine alone (or an irrelevant pair over it), causing a
wrong TicketFallback (or the reverse). This needs measuring against a
30-50 question set with labeled correct chunks to retune the threshold.

**When to revisit:** the evaluation set shows misses on code/number-heavy
questions → add BM25; correct chunks are retrieved but ranked wrong, or the
cosine threshold can't separate right from wrong → add reranking (and only
then RRF).

### Prompt injection is logged, not blocked (2026-10)

Two layers exist, neither of which refuses a request:

- **Prompt framing:** `<academic_context>` and `<websearch>` are declared as data, never
  instructions, and any context-frame tag inside retrieved text is escaped
  (`app/rag/prompting/builder.py::_neutralize_frame_tags`) so a document or web page can't close
  its frame early.
- **Detection:** `detect_prompt_injection` (`app/core/security/sanitizer.py`) matches a handful of
  named Vietnamese/English patterns on diacritic-folded text. `POST /chat/stream` logs one WARNING
  per match with only `pattern`, `role`, `conversation_id` and `message_length`; the turn runs
  unchanged.

Deliberately log-only: regex on Vietnamese is coarse, and blocking on it would refuse real questions
("bỏ qua môn này có sao không") without any measurement of the false-positive rate. The patterns
also only see the user's message - text inside ingested documents is protected by the framing
alone.

**When to revisit:** after 1-2 weeks of production logs, compare match counts against a sample of
matched conversations. If matches are mostly real attacks, consider refusing with a static
template (like OffTopic); if attacks get through without a match, consider an LLM classifier
instead of more regex.

### `CalculationNode` is a placeholder

> **Đang xử lý ở UNISAGE-99** (`SPEC-calculation-node.md`). Mục này vẫn đúng với code hiện tại trên `main`; sẽ được gỡ hoặc
> viết lại ở T20 của `changes/09-10-2026-calculation-flow/todo.md`.

No extractor, no Calculator Tool, no data source for scores/tuition figures.
`app/graph/nodes/calculation.py` streams a static "under development" message
and never calls an LLM or tool.

### Resuming a clarification round in the MULTI flow re-runs every origin task

> **Đang xử lý ở UNISAGE-99** (`SPEC-clarification-panel.md` + `SPEC-calculation-node.md` §5). Mục này vẫn đúng với code hiện tại trên `main`; sẽ được gỡ hoặc
> viết lại ở T20 của `changes/09-10-2026-calculation-flow/todo.md`.

`PendingClarification.origin_tasks` lets a resume turn re-run exactly the
advisory tasks that were running when the round started, at their own modes
(see AD6/AD15). But it always re-runs **all** of them - `pending_sub_query_id`
(which `SQk` the `ask_user_form` actually targeted) is recorded and
round-tripped through the database, but nothing yet uses it to resume only
the one sub-query that was missing information. Splitting a form reply across
several tasks/sub-queries this precisely is future work.

### A turn needing clarification from two branches at once isn't supported

> **Đang xử lý ở UNISAGE-99** (`SPEC-clarification-panel.md`). Mục này vẫn đúng với code hiện tại trên `main`; sẽ được gỡ hoặc
> viết lại ở T20 của `changes/09-10-2026-calculation-flow/todo.md`.

`PendingClarification` has a single `origin_node`. Today only the advisory
branch (node 06 onward) can raise a clarification form - `CalculationNode` is
still a placeholder and never does. Once CalculationNode does real work, a
turn where both the calculation and advisory branches need to ask something
in the same turn has no way to hold two clarification rounds in parallel.

### At most 3 tasks per message

A message with more than 3 genuinely different-intent questions only gets
the first 3 answered (`MessageClassificationNode`'s `MAX_TASKS`); the rest
are silently dropped (logged, not surfaced to the user). This is a
deliberate cap on LLM calls/retrieval fan-out per turn (see AD2), not a
parsing bug.

### A decomposer failure on a merged MULTI task loses per-question coverage

Since AD15 (2026-09-25), `MessageClassificationNode` no longer splits 2+
questions under `academic_advisory` into separate tasks - they are always one
task with `routing_mode = MULTI`, and the decomposer (node 06) is the only
place that breaks it into sub-queries. If the decomposer returns unusable
output (empty, malformed JSON, or only 1 sub-query), `_transform_task` falls
back to running HyDE once over the **whole merged task query** (both/all
original questions concatenated) as if it were a single question. Before
AD15, each question was already its own task, so a HyDE fallback for one
question never affected the other. Now, a decomposer failure on a
multi-question task risks HyDE producing a hypothetical document skewed
toward only one of the questions, silently under-serving the other(s) in
retrieval. No evaluation data yet on how often the decomposer actually fails
on genuinely independent (non-comparison) questions - if this turns out to
be common, the HyDE fallback would need its own multi-question handling
instead of treating the merged text as one question.

### Calculation results don't reach `GenerationSynthesisNode`'s prompt

> **Đang xử lý ở UNISAGE-99** (`SPEC-calculation-node.md` §3). Mục này vẫn đúng với code hiện tại trên `main`; sẽ được gỡ hoặc
> viết lại ở T20 của `changes/09-10-2026-calculation-flow/todo.md`.

While `CalculationNode` is a placeholder, its static message is appended
after the advisory answer deterministically (string concatenation outside
the LLM, see AD14) rather than being reasoned about together with the
advisory answer. Once CalculationNode does real work, its result should flow
into node 10's prompt so the model can phrase both parts as one coherent
answer instead of two concatenated pieces.
