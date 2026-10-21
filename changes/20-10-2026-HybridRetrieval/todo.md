# Todo: Hybrid retrieval BM25 + dense

Spec: `docs/specs/SPEC-hybrid-retrieval.md`. Plan: `plan.md`.

## Task 1: `lexical.py` thuần

**Description:** `app/rag/retrieval/lexical.py`: `tokenize(text) -> list[str]` (NFC, casefold,
tách theo ký tự không phải chữ/số, bỏ token 1 ký tự không phải số), `sparse_vector(text)
-> (indices, values)` (CRC32 u32, tf), `rrf_merge(ranked_lists, k=60) -> list[(id, rrf)]`.

**Acceptance criteria:**

- [x] "Học phần 2105618 (mã) - K18" → tokens gồm "2105618", "k18", "học", "phần"; không có "(" hay "-".
- [x] Cùng text hai lần → cùng indices; NFD/NFC cho cùng kết quả.
- [x] `rrf_merge`: chunk có mặt ở nhiều danh sách lên trên; tie-break ổn định theo id.

**Verification:**

- [x] `uv run pytest tests/rag/test_lexical.py -q`; ruff; mypy

**Dependencies:** None

**Files likely touched:** `app/rag/retrieval/lexical.py`, `tests/rag/test_lexical.py`

**Estimated scope:** S

## Task 2: `qdrant_store` sparse

**Description:** `ensure_collection` tạo `sparse_vectors_config={"bm25": SparseVectorParams(modifier=IDF)}`;
`ChunkPoint.bm25: tuple[list[int], list[float]] | None`; `upsert_chunk`/`upsert_points` ghi
`"bm25": SparseVector`; `has_sparse_vector(client, collection)` (cache theo tên);
`add_sparse_vector_config(client, collection)`; `iter_points_for_backfill`, `set_sparse_vectors(batch)`.

**Acceptance criteria:**

- [x] Fake client ghi lại `create_collection` có sparse config; `upsert` có `bm25` khi point có.
- [x] `has_sparse_vector` đọc `config.params.sparse_vectors`, cache, trả False khi thiếu.
- [x] Backfill helpers gọi `update_vectors` với `SparseVector` đúng point id.

**Verification:**

- [x] `uv run pytest tests/test_qdrant_store.py -q`

**Dependencies:** Task 1

**Files likely touched:** `app/rag/vectorstore/qdrant_store.py`, `tests/test_qdrant_store.py`

**Estimated scope:** M

## Task 3: CLI backfill tại chỗ (dev/eval, dự phòng) + xác minh trên eval

**Description:** Lối tắt không cutover cho dev và eval, và là dự phòng nếu rebuild (Task 9)
chưa sẵn. `app/tools/bm25_backfill.py`: ~~`add_sparse_vector_config` rồi `update_vectors`~~
**(21/10: Qdrant không cho thêm vector vào collection có sẵn)** → xây lại cùng tên qua
`<name>__bm25tmp` bằng `copy_points_with_bm25` (scroll `with_vectors=True`, payload content +
heading_path, lô 256, in tiến độ); `--dry-run`; `--collection` (mặc định collection đang dùng).
`taskfiles/db.yml`: `db:bm25-backfill`.

**Acceptance criteria:**

- [x] Chạy trên `unisage_eval_chunks`: Qdrant **không** chấp nhận `update_collection(sparse_vectors_config)` trên collection có dữ liệu ("Not existing vector name", 21/10) → ghi vào plan/spec; dev/eval đi đường xây lại cùng tên, production đi rebuild (Task 9).
- [x] Sau chạy: 2.322/2.323 point có `bm25` (1 point không có text); lần 2 "already present - nothing to do". — 21/10

**Verification:**

- [x] `task db:bm25-backfill env=eval -- --dry-run` rồi không dry-run (16,3 s); kiểm tra bằng `collections/<name>` info và scroll `with_vectors=["bm25"]` — 21/10

**Dependencies:** Task 2

**Files likely touched:** `app/tools/bm25_backfill.py`, `taskfiles/db.yml`

**Estimated scope:** S

## Checkpoint: Lưu trữ

- [x] 100 % point eval có `bm25`; test store xanh; ruff, mypy sạch — 21/10

## Task 9: Re-index "rebuild" (đường production, 3 repo)

**Description:** Cho phép một lượt re-index blue-green nhắm vào chính model đang live để dựng
collection mới có `bm25` mà không embed lại.

- 9a backend (`unisage-backend`): `EmbeddingReindexRequest.rebuild` (mặc định false); `start()`
  với `rebuild=true` chấp nhận model đang ACTIVE/live, bỏ qua kiểm tra identity, vẫn chặn
  `EMBEDDING_REINDEX_IN_PROGRESS`; cột `rebuild` trên `embedding_reindex_runs` (migration);
  `InternalEmbeddingReindexClaimResponse.rebuild`; cutover như cũ. Test: start rebuild khi
  identity trùng → 201; start thường khi identity trùng → vẫn NOT_NEEDED.
- 9b agent (`app/worker/embedding_reindex.py`): `_Claim.rebuild`; khi true, `scroll_points(...,
  with_vectors=True)`, upsert sang target với 3 vector dense copy + `bm25` tính từ payload,
  không gọi `embedder`; `ensure_collection(target)` đã có sparse config (Task 2); report tiến
  độ như cũ. Test: rebuild không gọi `embed_tracked`, point target có đủ 4 vector.
- 9c web (`unisage-web`): dialog re-index thêm lựa chọn "Xây lại chỉ mục với model hiện tại
  (thêm BM25)"; `embeddingReindexRunSchema.rebuild`; panel hiển thị nhãn "rebuild".

**Acceptance criteria:**

- [x] Trên eval: rebuild → `unisage_eval_chunks__20261021030324` 2.323 point, 2.322 với `bm25`; cutover; retrieval OK (câu mã hạng 2) — 21/10
- [x] Log worker không có lượt gọi embedding; 2.323 point trong 10 s (started 03:03:38 → finished 03:03:48) — 21/10
- [x] Retrieval cache: `coll` nằm trong khoá (test fingerprint); sau cutover `stats` báo 0 entry — 21/10

**Verification:**

- [x] Backend: `./mvnw test -Dtest=EmbeddingReindex*` (23 xanh); agent: `uv run pytest tests/worker/test_embedding_reindex.py -q` (12 xanh); web: `pnpm test` chat-models (21 xanh) — 21/10
- [x] Chạy tay trên eval, `task db:retrieval-cache env=eval -- stats` chạy được, 0 entry sau cutover — 21/10

**Dependencies:** Task 2 (sparse config + helper), Task 1

**Files likely touched:** backend: `EmbeddingReindexRequest.java`, `EmbeddingReindexServiceImpl.java`,
`EmbeddingReindexRun.java`, migration `V38__reindex_rebuild.sql`, internal claim DTO; agent:
`app/worker/embedding_reindex.py`, `tests/worker/test_embedding_reindex.py`; web:
`embedding-reindex-dialog.tsx`, `embedding-index-schemas.ts`, `embedding-index-api.ts`

**Estimated scope:** L (chia 3 PR theo repo)

## Task 4: Truy vấn hybrid

**Description:** `search_chunks_batch(..., sparse_queries=None)`: mỗi truy vấn thêm một
`QueryRequest(query=SparseVector, using="bm25", limit=limit*mult, filter=...)`; ~~trộn 4 danh sách~~
**(21/10)** gộp 3 dense theo max cosine rồi `rrf_merge` 2 danh sách với k = 1; lấy `limit` đầu; `score` = max cosine dense nếu có, else
`CHAT_RETRIEVAL_LEXICAL_HIT_SCORE`. `RetrievalService.retrieve_many(_async)(queries, *,
lexical_queries=None, security, limit)`; `lexical_queries` None hoặc cờ tắt hoặc
`not has_sparse_vector` → đường cũ.

**Acceptance criteria:**

- [x] Fake client nhận 4 request/query, filter giống nhau trên cả 4; tắt cờ → 3 request. — 21/10
- [x] Chunk chỉ có trong danh sách sparse vẫn vào kết quả với score lexical; chunk dense giữ cosine. — 21/10
- [x] Thứ tự theo RRF; `limit` được tôn trọng. — 21/10

**Verification:**

- [x] `uv run pytest tests/test_qdrant_store_hybrid.py tests/rag -q` — 21/10

**Dependencies:** Task 2

**Files likely touched:** `app/rag/vectorstore/qdrant_store.py`, `app/rag/retrieval/service.py`,
`app/core/config.py`, `tests/test_qdrant_store.py`

**Estimated scope:** M

## Task 5: Ingest và re-index ghi sparse

**Description:** `embed_chunks`: `ChunkPoint.bm25 = sparse_vector(content + heading_path)`.
`embedding_reindex`: khi copy point sang collection mới, tính sparse từ payload; `ensure_collection`
của collection đích đã có sparse config (Task 2).

**Acceptance criteria:**

- [x] Test ingestion: `ChunkPoint` nhận `bm25` không rỗng cho chunk có text.
- [x] Test re-index: point upsert sang collection mới có `bm25`.

**Verification:**

- [x] `uv run pytest tests/test_ingestion_embedding.py tests/worker/test_embedding_reindex.py -q`

**Dependencies:** Task 2

**Files likely touched:** `app/worker/tasks/ingestion.py`, `app/worker/embedding_reindex.py`,
`tests/test_ingestion_embedding.py`, `tests/worker/test_embedding_reindex.py`

**Estimated scope:** S

## Task 6: Graph: lexical query, sàn, cờ, cache

**Description:** `retrieve_chunks` truyền `[sq.question for sq in sub_queries]` làm lexical
queries. `rerank()`/`rerank_chunks` áp `CHAT_RETRIEVAL_DENSE_FLOOR` khi LLM rerank sẽ chạy
(`CHAT_LLM_RERANK_ENABLED` và `models.rerank` có), ngược lại `CHAT_RERANK_SCORE_THRESHOLD`.
`config_fingerprint` của retrieval cache thêm `RETRIEVAL_HYBRID_ENABLED`, sàn, lexical score, RRF k,
multiplier.

**Acceptance criteria:**

- [x] Wiring test: fake retrieval nhận `lexical_queries` = câu độc lập.
- [x] Có LLM rerank: chunk cosine 0,5 đi tới LLM rerank; không có LLM rerank: bị cắt ở 0,70 như cũ.
- [x] Cache: bật/tắt hybrid → khoá khác (test fingerprint).

**Verification:**

- [x] `uv run pytest tests/graph tests/test_retrieval_cache.py -q`

**Dependencies:** Task 4, Task 5

**Files likely touched:** `app/graph/nodes/retrieval_filtering.py`, `app/graph/nodes/post_retrieval_rerank.py`,
`app/rag/reranking/cross_encoder.py`, `app/graph/streaming_graph.py`, `app/integrations/retrieval_cache.py`

**Estimated scope:** M

## Checkpoint: Truy vấn

- [x] Test graph xanh; chạy tay eval stack: "học phần <mã có trong kho>" có chunk trong 16 ứng viên (hạng 2–3; dense-only không có); canary quyền; log `Retrieval timing` — 21/10

## Task 7: Eval trước/sau + lát cắt mã/tên

**Description:** (1) `task eval:run env=eval -- --set demo --out runs/hybrid-baseline` với cờ tắt
(21/10: chạy `--category normal --category calculation --variant exact --web off`, cache tắt
bằng `RETRIEVAL_CACHE_REDIS_URL=`; cần `departments.json` trong thư mục dataset — file bị
gitignore, dựng lại từ bảng `departments` của DB backend eval khi máy mới).
(2) Chọn ≥ 30 câu official có mã học phần / "Quyết định" / tên học phần → `expected_doc_ids` →
`evals/dataset/code_slice.csv`; `task eval:ingest env=eval -- --only ...`. (3) Backfill, bật cờ,
chạy `runs/hybrid-on` cho demo và lát cắt; (4) ghi recall@5, leak, fallback, timing vào plan.

**Acceptance criteria:**

- [x] Có 2 run so sánh được trên cùng corpus (`runs/hybrid-baseline`, `runs/hybrid-on`); bảng kết quả trong plan.md — 21/10
- [x] Quyết định: giữ bật mặc định (tiêu chí 4, 6 đạt; 5 chờ lát cắt mã) — 21/10

**Verification:**

- [x] Thủ công theo lệnh trên (demo normal+calculation; lát cắt mã chỉ có 6 câu đủ điều kiện, xem plan) — 21/10

**Dependencies:** Task 3, Task 6

**Files likely touched:** `evals/dataset/code_slice.csv`, `changes/20-10-2026-HybridRetrieval/plan.md`

**Estimated scope:** M (thời gian chạy)

## Task 8: Tài liệu

**Description:** `docs/guide/cau-hinh.md` (5 biến mới, lệnh backfill), `.env.example`, `.env.eval.example`,
`dev-onboard.txt` mục mới, cập nhật docstring `cross_encoder.py` (giờ là sàn, không phải reranker).

**Acceptance criteria:**

- [x] Biến và lệnh khớp code; spec khớp code — 21/10

**Dependencies:** Task 6

**Estimated scope:** S

## Checkpoint: Hoàn tất

- [~] `task test` 2.357 xanh (1 flaky `test_judge` có sẵn, `tests/e2e` lỗi import có sẵn); ruff/mypy sạch trên file đổi, lỗi còn lại ở file khác có từ trước — 21/10
- [~] Eval đạt tiêu chí 1, 2, 3, 4, 6; tiêu chí 5 (lát cắt mã) thiếu dữ liệu; PR đã mở: agent #26 (draft), backend #31, web #46 — chưa merge — 21/10

## Phase sau (không làm đợt này)

- [ ] Cross-encoder `bge-reranker-v2-m3` trên máy deploy RTX 3060: theo bảng trong plan.md
