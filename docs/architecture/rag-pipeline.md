# RAG Pipeline Architecture

How a chat turn and a document ingestion actually run today. Product rules are in
[`docs/product/PRODUCT.md`](../product/PRODUCT.md); deliberate gaps against the original
design (BM25, RRF, cross-encoder, calculation) are in
[`docs/specs/known-gaps.md`](../specs/known-gaps.md).

## Service Boundary

- The API Gateway authenticates the user and injects 5 trusted headers (`X-User-Id`,
  `X-User-Role`, `X-User-Code`, `X-User-Department-Access`, `X-User-Permissions`) plus
  `X-Internal-Secret`. This service never decodes a JWT; no `X-User-Id` means a guest.
- `unisage-backend` (Java) owns conversations and messages. Each turn this service reads
  history from Java, posts the USER message and an ASSISTANT placeholder, then patches the
  final answer, citations and metadata (panel projection, calculation summary) back. Staff-only
  calculation traces go to Java's `/internal/calculation-traces`. It never stores conversation
  content itself.
- Document access comes only from `department_access` in those headers, applied as a Qdrant
  pre-filter before any search. Facts a student states in the conversation never widen it.

## The Orchestrator

`app/graph/streaming_graph.py::run_graph` is **one async function** that branches with
plain `if` statements. It is **not** a `pydantic_graph.Graph`: `Graph.run()` returns a
single final output and can't stream tokens, which this endpoint needs (SSE). Node names
(`01_…` to `11_…`) are labels for `GraphTrace` logs and for matching the original design;
each node's logic lives in `app/graph/nodes/`.

`POST /api/v1/chat/stream` (`app/api/v1/chat.py`) is a thin controller: it reads the HTTP
request and returns the stream `ChatStreamService` (`app/services/chat_stream_service.py`)
produces. The service checks the body size and the clarification panel gate
(`app/services/clarification_service.py`), sanitizes the message, logs suspected prompt
injection (log-only), does the Java calls above, then runs `run_graph` as a background task
whose tokens the SSE response reads from a queue, so the answer is still persisted if the
client disconnects.

## Chat Flow

```mermaid
flowchart TD
    Start(["Request"]) --> Gate{"Panel gate (service, before Java)<br/>conversation_clarification_states"}
    Gate -->|"message while OPEN / PROCESSING"| R409["409 4092 / 4093"]
    Gate -->|cancel| Cancel["claim → PATCH projection → clarification_closed<br/>(no message, no quota, no LLM)"]
    Gate -->|submit| Claim["validate answers → claim OPEN→PROCESSING<br/>→ start_turn(summary + clarification_answers)"]
    Gate -->|"message, no round"| Turn["start_turn"]
    Claim --> Resume["02 ClarificationResume<br/>only the round's tasks; no classification"]
    Turn --> N01{"01 Greeting<br/>first turn + pure greeting?"}
    N01 -->|yes| Greeting["Static greeting template"]
    N01 -->|no| N03["03 MessageClassification (LLM)<br/>up to 3 tasks"]
    N03 --> N04{"04 IntentRouting"}
    N04 -->|social| Social["Static social template"]
    N04 -->|off-topic| N05["05 OffTopicReject<br/>static template"]
    N04 -->|calculation| N07["07 Calculation (concurrent)<br/>rule router → extractor LLM → built-in formula<br/>or Qdrant formula + 7 checks → Python computes"]
    N04 -->|advisory| N06["06 QueryTransformation (LLM)<br/>SINGLE: HyDE · MULTI: decompose into sub-queries"]
    Resume --> N07
    Resume --> N06

    N06 --> N08["08 RetrievalFiltering<br/>access pre-filter + dense search on<br/>content / summary / questions vectors"]
    N08 --> N09["09 PostRetrievalRerank<br/>cosine score threshold"]
    N09 --> Q9a{"CHAT_LLM_RERANK_ENABLED<br/>and context found<br/>and rerank model available?"}
    Q9a -->|yes| N09a["09a LLMRerank (EXTRACTION model)<br/>keep chunks that answer each sub-query"]
    Q9a -->|no| Q9b
    N09a --> Q9b{"CHAT_WEB_SEARCH_ENABLED<br/>and some sub-query has no chunk?"}
    Q9b -->|yes| N09b["09b WebSearch (Tavily)<br/>school domains only"]
    Q9b -->|no| Barrier
    N09b --> Barrier{{"Barrier: await 07,<br/>stream calculation blocks (Python)"}}
    N07 --> Barrier
    Barrier --> Q10{"any chunk or web result?"}
    Q10 -->|no| N11["11 TicketFallback (LLM)<br/>suggest a support ticket"]
    Q10 -->|yes| N10["10 GenerationSynthesis (LLM)<br/>answer with [n] citations; ask_user_form<br/>blocks filtered out of the stream"]
    N07 -->|"calculation only"| Note["checked note (chat_calculation.yaml, not streamed)<br/>or fixed lead when only input is missing"]
    N10 --> Panel["One panel: calculation params + advisory asks<br/>state OPEN → PATCH metadata → event: clarification"]
    Note --> Panel
```

Notes:

- **09a and 09b are optional.** 09a runs only when `CHAT_LLM_RERANK_ENABLED` is on, 09 found
  context, and an EXTRACTION model is available; if the model is unavailable or fails, the
  turn continues with 09's result and an admin warning. 09b runs only when
  `CHAT_WEB_SEARCH_ENABLED` is on and at least one sub-query was left with no chunk.
- **Rerank is a threshold, not a cross-encoder.** 09 sorts by the best cosine score across the
  3 named vectors and drops chunks under `CHAT_RERANK_SCORE_THRESHOLD`.
- **Citations** are built from the chunks and web pages the model cited by `[n]`
  (`app/rag/prompting/citations.py`); the model never supplies a source itself.
- **Prompt framing:** retrieved text sits inside `<academic_context>` / `<websearch>`, declared
  as data, with any frame tag inside the text escaped.
- **Clarification panel** (`docs/specs/SPEC-clarification-panel.md`, `contracts/chat-sse.md`):
  10's `ask_user_form` blocks are removed from the stream by `FenceRedactor` and become choice
  questions; calculation tasks missing parameters add questions built from `ParamSpec`. One
  panel per turn, one tab per question (every question asked; no chain limit). State machine `none → OPEN → PROCESSING →
  none | OPEN` in `conversation_clarification_states`, fenced by `claim_token`, bounded by a
  lease that outlives the claimed turn's hard deadline. Java's `messages.metadata` is only a
  projection; the event is sent after both are written.
- **Calculation** (`docs/specs/SPEC-calculation-node.md`, `app/calculation/`): the LLM never
  computes. Built-in formulas (GPA, course score, grade conversion) live in `formulas.py`;
  regulation formulas pass 7 fail-closed checks and are labelled "Kết quả tham khảo theo quy
  chế" (kill switch `CHAT_CALC_RETRIEVED_FORMULA_ENABLED`). Node 10 only gets the titles of
  what was computed, never a number.

## Ingestion Flow

```mermaid
flowchart LR
    Upload["File in MinIO<br/>(uploaded via unisage-backend)"] --> Preview["POST /ingestion/preview<br/>parse to raw text"]
    Preview --> Chunking["POST /ingestion/chunking<br/>chosen strategy → Postgres draft"]
    Chunking --> Embed["POST /ingestion/embedding<br/>Celery task"]
    Embed --> Enrich["summary + sample questions<br/>per chunk (EXTRACTION model)"]
    Enrich --> Qdrant["3 named vectors + permission payload<br/>into Qdrant"]
    Embed -. progress .-> WS["WS /ingestion/events"]
```

Supported inputs: PDF, DOCX, TXT, XLSX (`.doc` is not). Chunking strategies: recursive,
token-based, semantic, markdown-aware, table-row (row-atomic tables), Excel-row. Details and
invariants are in [`docs/specs/SPEC-ingestion.md`](../specs/SPEC-ingestion.md).
