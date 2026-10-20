# Todo: Hybrid retrieval BM25 + dense

Spec: `docs/specs/SPEC-hybrid-retrieval.md`. Plan: `plan.md`.

## Task 1: `lexical.py` thuần

**Description:** `app/rag/retrieval/lexical.py`: `tokenize(text) -> list[str]` (NFC, casefold,
tách theo ký tự không phải chữ/số, bỏ token 1 ký tự không phải số), `sparse_vector(text)
-> (indices, values)` (CRC32 u32, tf), `rrf_merge(ranked_lists, k=60) -> list[(id, rrf)]`.

**Acceptance criteria:**

- [ ] "Học phần 2105618 (mã) - K18" → tokens gồm "2105618", "k18", "học", "phần"; không có "(" hay "-".
- [ ] Cùng text hai lần → cùng indices; NFD/NFC cho cùng kết quả.
- [ ] `rrf_merge`: chunk có mặt ở nhiều danh sách lên trên; tie-break ổn định theo id.

**Verification:**

- [ ] `uv run pytest tests/rag/test_lexical.py -q`; ruff; mypy

**Dependencies:** None

**Files likely touched:** `app/rag/retrieval/lexical.py`, `tests/rag/test_lexical.py`

**Estimated scope:** S

## Task 2: `qdrant_store` sparse

**Description:** `ensure_collection` tạo `sparse_vectors_config={"bm25": SparseVectorParams(modifier=IDF)}`;
`ChunkPoint.bm25: tuple[list[int], list[float]] | None`; `upsert_chunk`/`upsert_points` ghi
`"bm25": SparseVector`; `has_sparse_vector(client, collection)` (cache theo tên);
`add_sparse_vector_config(client, collection)`; `iter_points_for_backfill`, `set_sparse_vectors(batch)`.

**Acceptance criteria:**

- [ ] Fake client ghi lại `create_collection` có sparse config; `upsert` có `bm25` khi point có.
- [ ] `has_sparse_vector` đọc `config.params.sparse_vectors`, cache, trả False khi thiếu.
- [ ] Backfill helpers gọi `update_vectors` với `SparseVector` đúng point id.

**Verification:**

- [ ] `uv run pytest tests/test_qdrant_store.py -q`

**Dependencies:** Task 1

**Files likely touched:** `app/rag/vectorstore/qdrant_store.py`, `tests/test_qdrant_store.py`

**Estimated scope:** M

## Task 3: CLI backfill tại chỗ (dev/eval, dự phòng) + xác minh trên eval

**Description:** Lối tắt không cutover cho dev và eval, và là dự phòng nếu rebuild (Task 9)
chưa sẵn. `app/tools/bm25_backfill.py`: `add_sparse_vector_config` rồi scroll mọi point
(payload content + heading_path), tính sparse, `update_vectors` lô 256, in tiến độ; `--dry-run`;
`--collection` (mặc định collection đang dùng). `taskfiles/db.yml`: `db:bm25-backfill`.

**Acceptance criteria:**

- [ ] Chạy trên `unisage_eval_chunks`: Qdrant chấp nhận `update_collection(sparse_vectors_config)` trên collection có dữ liệu (ghi kết quả vào plan). Không chấp nhận → ghi lại, chuyển kế hoạch sang re-index.
- [ ] Sau chạy: `count` point có vector `bm25` = tổng point; chạy lần 2 không lỗi.

**Verification:**

- [ ] `task db:bm25-backfill env=eval -- --dry-run` rồi không dry-run; kiểm tra bằng `collections/<name>` info và scroll `with_vectors=["bm25"]`

**Dependencies:** Task 2

**Files likely touched:** `app/tools/bm25_backfill.py`, `taskfiles/db.yml`

**Estimated scope:** S

## Checkpoint: Lưu trữ

- [ ] 100 % point eval có `bm25`; test store xanh; ruff, mypy sạch

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

- [ ] Trên eval: rebuild → collection `<base>__<ts>` có 100 % point với `bm25`; cutover; chat OK.
- [ ] Log worker không có lượt gọi embedding; thời gian rebuild cho 2.323 point dưới 2 phút.
- [ ] Retrieval cache tự miss sau cutover (`coll` trong khoá đổi).

**Verification:**

- [ ] Backend: `./mvnw test -Dtest=EmbeddingReindex*`; agent: `uv run pytest tests/worker/test_embedding_reindex.py -q`; web: `pnpm test`.
- [ ] Chạy tay trên eval, `task db:retrieval-cache env=eval -- stats` thấy miss sau cutover.

**Dependencies:** Task 2 (sparse config + helper), Task 1

**Files likely touched:** backend: `EmbeddingReindexRequest.java`, `EmbeddingReindexServiceImpl.java`,
`EmbeddingReindexRun.java`, migration `V38__reindex_rebuild.sql`, internal claim DTO; agent:
`app/worker/embedding_reindex.py`, `tests/worker/test_embedding_reindex.py`; web:
`embedding-reindex-dialog.tsx`, `embedding-index-schemas.ts`, `embedding-index-api.ts`

**Estimated scope:** L (chia 3 PR theo repo)

## Task 4: Truy vấn hybrid

**Description:** `search_chunks_batch(..., sparse_queries=None)`: mỗi truy vấn thêm một
`QueryRequest(query=SparseVector, using="bm25", limit=limit*mult, filter=...)`; trộn 4 danh sách
bằng `rrf_merge`; lấy `limit` đầu; `score` = max cosine dense nếu có, else
`CHAT_RETRIEVAL_LEXICAL_HIT_SCORE`. `RetrievalService.retrieve_many(_async)(queries, *,
lexical_queries=None, security, limit)`; `lexical_queries` None hoặc cờ tắt hoặc
`not has_sparse_vector` → đường cũ.

**Acceptance criteria:**

- [ ] Fake client nhận 4 request/query, filter giống nhau trên cả 4; tắt cờ → 3 request.
- [ ] Chunk chỉ có trong danh sách sparse vẫn vào kết quả với score lexical; chunk dense giữ cosine.
- [ ] Thứ tự theo RRF; `limit` được tôn trọng.

**Verification:**

- [ ] `uv run pytest tests/test_qdrant_store.py tests/rag -q`

**Dependencies:** Task 2

**Files likely touched:** `app/rag/vectorstore/qdrant_store.py`, `app/rag/retrieval/service.py`,
`app/core/config.py`, `tests/test_qdrant_store.py`

**Estimated scope:** M

## Task 5: Ingest và re-index ghi sparse

**Description:** `embed_chunks`: `ChunkPoint.bm25 = sparse_vector(content + heading_path)`.
`embedding_reindex`: khi copy point sang collection mới, tính sparse từ payload; `ensure_collection`
của collection đích đã có sparse config (Task 2).

**Acceptance criteria:**

- [ ] Test ingestion: `ChunkPoint` nhận `bm25` không rỗng cho chunk có text.
- [ ] Test re-index: point upsert sang collection mới có `bm25`.

**Verification:**

- [ ] `uv run pytest tests/test_ingestion_embedding.py tests/worker/test_embedding_reindex.py -q`

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

- [ ] Wiring test: fake retrieval nhận `lexical_queries` = câu độc lập.
- [ ] Có LLM rerank: chunk cosine 0,5 đi tới LLM rerank; không có LLM rerank: bị cắt ở 0,70 như cũ.
- [ ] Cache: bật/tắt hybrid → khoá khác (test fingerprint).

**Verification:**

- [ ] `uv run pytest tests/graph tests/test_retrieval_cache.py -q`

**Dependencies:** Task 4, Task 5

**Files likely touched:** `app/graph/nodes/retrieval_filtering.py`, `app/graph/nodes/post_retrieval_rerank.py`,
`app/rag/reranking/cross_encoder.py`, `app/graph/streaming_graph.py`, `app/integrations/retrieval_cache.py`

**Estimated scope:** M

## Checkpoint: Truy vấn

- [ ] Test graph xanh; chạy tay eval stack: "học phần <mã có trong kho>" có chunk trong 16 ứng viên; canary quyền; log `Retrieval timing`

## Task 7: Eval trước/sau + lát cắt mã/tên

**Description:** (1) `task eval:run env=eval -- --set demo --out runs/hybrid-baseline` với cờ tắt.
(2) Chọn ≥ 30 câu official có mã học phần / "Quyết định" / tên học phần → `expected_doc_ids` →
`evals/dataset/code_slice.csv`; `task eval:ingest env=eval -- --only ...`. (3) Backfill, bật cờ,
chạy `runs/hybrid-on` cho demo và lát cắt; (4) ghi recall@5, leak, fallback, timing vào plan.

**Acceptance criteria:**

- [ ] Có 2 run so sánh được trên cùng corpus; bảng kết quả trong plan.md.
- [ ] Quyết định bật mặc định hay không dựa trên tiêu chí 4–6 của spec.

**Verification:**

- [ ] Thủ công theo lệnh trên

**Dependencies:** Task 3, Task 6

**Files likely touched:** `evals/dataset/code_slice.csv`, `changes/20-10-2026-HybridRetrieval/plan.md`

**Estimated scope:** M (thời gian chạy)

## Task 8: Tài liệu

**Description:** `docs/guide/cau-hinh.md` (5 biến mới, lệnh backfill), `.env.example`, `.env.eval.example`,
`dev-onboard.txt` mục mới, cập nhật docstring `cross_encoder.py` (giờ là sàn, không phải reranker).

**Acceptance criteria:**

- [ ] Biến và lệnh khớp code; spec khớp code

**Dependencies:** Task 6

**Estimated scope:** S

## Checkpoint: Hoàn tất

- [ ] `task test`, `uv run ruff check app tests`, `uv run ruff format --check app tests`, `uv run mypy app tests`
- [ ] Eval đạt tiêu chí; PR lên `main`

## Phase sau (không làm đợt này)

- [ ] Cross-encoder `bge-reranker-v2-m3` trên máy deploy RTX 3060: theo bảng trong plan.md
