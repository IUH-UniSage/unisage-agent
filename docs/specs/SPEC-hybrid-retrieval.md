# Spec: Hybrid retrieval BM25 + dense trong Qdrant (unisage-agent)

> Nhánh: `feature/huydh-hybrid-retrieval` (đổi tên theo số ticket khi có). Backend và web có
> phần việc riêng cho đường re-index rebuild (plan Task 9).
> Spec này **chỉ thêm**: tắt cờ `RETRIEVAL_HYBRID_ENABLED` hoặc collection chưa có sparse
> vector thì retrieval chạy y như hôm nay (3 vector dense, max điểm theo chunk).
>
> Buổi phỏng vấn chốt ý định: 20/10/2026. Phase cross-encoder ở cuối là việc sau, ghi để
> không phải nhớ lại.

## Ý định đã chốt

- **Outcome:** câu hỏi mang token chính xác (mã học phần "2105618", "Quyết định 1035", tên
  học phần, tên hội thảo, "K18") kéo được đúng chunk vào danh sách đưa cho rerank, điều mà
  embedding dense làm kém vì coi mã số gần như vô nghĩa.
- **Người hưởng lợi:** sinh viên hỏi theo mã và tên cụ thể. Chưa có ca lỗi thật trong tay;
  làm vì độ chính xác, độ trễ không phải ràng buộc.
- **Thành công (đo bằng bộ eval, `evals/run`):** (1) recall@k tài liệu kỳ vọng trên bộ
  demo không giảm; (2) trên lát cắt mới gồm câu có mã/tên chính xác (ingest thêm từ bộ
  official) recall@k tăng rõ; (3) không tăng số ca rò rỉ tài liệu sai quyền
  (`leaked_documents`); (4) tỷ lệ ticket fallback trên bộ demo không tăng.
- **Gate:** LLM rerank quyết định "có ngữ cảnh" (giữ ≥ 1 chunk) hay rơi xuống ticket
  fallback; trước nó chỉ còn một sàn dense thấp. Ngưỡng cosine 0,70 chỉ dùng khi LLM
  rerank không có (không có model RERANK/EXTRACTION) để giữ hành vi cũ.
- **Ngoài phạm vi:** cross-encoder (phase sau), đổi model embedding, SPLADE/learned
  sparse, thay đổi ở backend-java hay web.

## Assumptions (sửa nếu sai)

1. Qdrant 1.15.4 và `qdrant-client` hiện có hỗ trợ sparse vector trên cùng collection,
   `update_collection(sparse_vectors_config=...)` để **thêm** sparse vector vào collection
   đang chạy, và `update_vectors` để điền sparse cho point đã có mà không đụng vector dense.
   Đã kiểm tra chữ ký client ngày 20/10; nếu Qdrant từ chối thêm sparse vào collection cũ
   thì đi đường re-index blue-green (UNISAGE-103) sang collection mới.
2. BM25 tự tính ở phía agent: tokenizer của mình + term frequency, IDF do Qdrant tính
   (`Modifier.IDF`). Không thêm dependency model (fastembed, rank-bm25).
3. Tiếng Việt tách theo âm tiết là đủ cho mục tiêu này: thứ cần khớp là mã, số, tên
   riêng; không cần tách từ ghép.
4. Dense vẫn embed HyDE text như hôm nay; sparse lấy từ **câu hỏi độc lập** (dòng 1 của
   HyDE, `SubQuery.question`) vì đó là nơi mã và tên xuất hiện nguyên văn, HyDE passage
   thì diễn giải và dễ bỏ mã.
5. Điểm `RetrievedChunk.score` vẫn là cosine dense (các chỗ sau đang hiểu nó là cosine:
   sàn, rescue của LLM rerank ≥ 0,75, `best_scores` xếp sub-query cho web search, cache).
   Fusion chỉ quyết định **thứ tự và tập** ứng viên, không thay điểm.

## Objective

### Hiện trạng

`search_chunks_batch` gửi mỗi truy vấn tới 3 vector dense (content, summary, questions),
lấy max cosine theo chunk, rồi `rerank()` sắp theo cosine và cắt ở 0,70; `has_valid_context`
rỗng thì ticket fallback. Không có lexical index. Với "học phần 2105618" dense chỉ đạt
0,595 và chunk chứa mã (nếu có) không lọt top 16.

### Thiết kế

1. **Sparse vector `bm25`** trên cùng collection, `SparseVectorParams(modifier=IDF)`.
   - Tokenizer (`app/rag/retrieval/lexical.py`): NFC → casefold → tách theo mọi ký tự không
     phải chữ/số → bỏ token 1 ký tự không phải số → giữ nguyên số và mã. Chỉ số = CRC32 của
     token (u32), giá trị = tần suất trong chunk. Cùng hàm cho chunk và câu hỏi.
   - Văn bản của chunk đưa vào BM25: `content` + `heading_path` (tên mục thường chứa mã học
     phần), không gồm summary/questions.
2. **Ingest** (`embed_chunks`): tính sparse từ chunk, đưa vào `ChunkPoint.bm25` và upsert
   cùng 3 vector dense. **Re-index** (`embedding_reindex`): tính sparse từ payload khi copy
   sang collection mới. `ensure_collection` tạo sparse config cho collection mới.
3. **Đưa `bm25` vào collection đang chạy**, hai đường:
   - **Re-index "rebuild"** (production): lượt re-index blue-green nhắm vào chính model đang
     live; agent copy 3 vector dense nguyên trạng (`with_vectors=True`), tính sparse từ
     payload, upsert sang collection mới, cutover nguyên tử. Không gọi OpenAI. Cần backend
     cho phép lượt `rebuild` (hiện `start()` chặn khi identity trùng) và web có lựa chọn
     tương ứng. Chi tiết: plan Task 9.
   - **Backfill tại chỗ** (`task db:bm25-backfill env=eval`, dev/eval và dự phòng): thêm
     sparse config bằng `update_collection`, scroll mọi point, tính sparse từ payload,
     `update_vectors` theo lô 256. Idempotent. Không cutover.
4. **Truy vấn** (`RetrievalService.retrieve_many[_async]` nhận thêm `lexical_queries`):
   mỗi sub-query gửi 3 dense + 1 sparse trong một `query_batch_points`; dense lấy
   `limit × 2`, sparse `limit × 2`; trộn bằng **RRF client-side** (k = 60) trên 4 danh sách;
   lấy `limit` chunk đầu theo RRF. `score` của chunk = max cosine dense nếu chunk có mặt
   trong bất kỳ danh sách dense nào, ngược lại (chỉ sparse) = `CHAT_RETRIEVAL_LEXICAL_HIT_SCORE`
   (mặc định 0,70, tức "đạt sàn cũ"). Filter quyền áp y hệt lên cả 4 truy vấn.
5. **Gate** (quyết định (a) ngày 20/10): `rerank()` áp **sàn** `CHAT_RETRIEVAL_DENSE_FLOOR`
   (0,45) thay cho 0,70 khi LLM rerank sẽ chạy; LLM rerank giữ ≥ 1 chunk → có ngữ cảnh.
   Khi `CHAT_LLM_RERANK_ENABLED=false` hoặc không có model RERANK/EXTRACTION, giữ ngưỡng
   `CHAT_RERANK_SCORE_THRESHOLD` 0,70 như cũ.
6. **Tự phát hiện:** collection không có vector `bm25` (kiểm tra một lần, cache theo tên
   collection) → chạy dense-only, log một dòng warning khi hybrid bật. Cờ
   `RETRIEVAL_HYBRID_ENABLED` (mặc định true) để A/B trong eval.
7. **Retrieval cache:** `config_fingerprint` thêm cờ hybrid, sàn, k của RRF và các hệ số;
   bật hybrid là khoá đổi, cache cũ tự miss.

### Acceptance

- Ingest một tài liệu có "mã học phần 2105618" → point có cả 3 dense lẫn `bm25`; hỏi
  "học phần 2105618" → chunk đó nằm trong 16 ứng viên đưa cho LLM rerank.
- Tắt cờ hybrid hoặc collection chưa backfill → kết quả bằng hôm nay (test so sánh thứ
  tự chunk với dense-only).
- Người dùng không có quyền với tài liệu → chunk không xuất hiện dù BM25 khớp mã (canary).
- Backfill chạy hai lần không lỗi, số point có `bm25` = tổng point.
- Re-index blue-green sang collection mới giữ `bm25`.
- Eval: tiêu chí ở "Thành công".

## Tech Stack

Python 3.12, FastAPI, Celery, Qdrant 1.15 + `qdrant-client` (sparse, `Modifier.IDF`,
`update_vectors`), pytest. Không thêm dependency.

## Commands

```
Test:         task test            |  task test:file -- tests/rag/test_lexical.py
Lint/format:  task code:check      |  uv run ruff check app tests && uv run ruff format --check app tests
Types:        uv run mypy app tests
Dev:          task be:dev env=eval
Backfill:     task db:bm25-backfill env=eval          # thêm sparse vào collection đang chạy
Eval:         task eval:run env=eval -- --set demo --out runs/hybrid-baseline   (trước)
              task eval:run env=eval -- --set demo --out runs/hybrid-on         (sau)
              task eval:ingest env=eval -- --only evals/dataset/code_slice.csv  (lát cắt mã/tên)
```

Thiếu dev deps trong venv: `uv run --with pytest --with pytest-asyncio --with aiosqlite --with docker pytest ...`.

## Project Structure

```
app/rag/retrieval/lexical.py          → tokenizer, sparse_vector(text) -> (indices, values), BM25_VECTOR = "bm25"
app/rag/vectorstore/qdrant_store.py   → ensure_collection (+sparse), ChunkPoint.bm25, upsert_chunk,
                                        search_chunks_batch (4 truy vấn + RRF), has_sparse_vector(),
                                        add_sparse_vector_config(), backfill helpers
app/rag/retrieval/service.py          → retrieve_many(_async)(queries, lexical_queries=...)
app/graph/nodes/retrieval_filtering.py → truyền SubQuery.question làm lexical query
app/graph/nodes/post_retrieval_rerank.py / app/rag/reranking/cross_encoder.py
                                      → sàn dense khi có LLM rerank, ngưỡng cũ khi không
app/worker/tasks/ingestion.py         → tính sparse khi upsert
app/worker/embedding_reindex.py       → copy sparse sang collection mới
app/tools/bm25_backfill.py            → CLI backfill; taskfiles/db.yml: db:bm25-backfill
app/core/config.py                    → RETRIEVAL_HYBRID_ENABLED, CHAT_RETRIEVAL_DENSE_FLOOR,
                                        CHAT_RETRIEVAL_LEXICAL_HIT_SCORE, CHAT_RETRIEVAL_RRF_K,
                                        CHAT_RETRIEVAL_HYBRID_CANDIDATES_MULTIPLIER
app/integrations/retrieval_cache.py   → config_fingerprint thêm các cờ trên
docs/guide/cau-hinh.md, .env.example  → biến mới
tests/rag/test_lexical.py             → tokenizer/sparse
tests/test_qdrant_store.py            → RRF, score dense/lexical, filter trên sparse, backfill
tests/graph/test_graph_wiring.py      → gate: sàn + LLM rerank; dense-only khi tắt cờ
tests/worker/test_embedding_reindex.py → sparse được copy
```

## Code Style

Giống `qdrant_store.py`: hàm thuần cho phần tính toán (tokenizer, RRF) để test không cần
Qdrant; docstring nói vì sao; type hint đầy đủ; ruff + mypy sạch.

```python
def sparse_vector(text: str) -> tuple[list[int], list[float]]:
    """BM25 term frequencies of `text` as a Qdrant sparse vector: index = CRC32 of the
    token, value = count. IDF is the collection's job (`Modifier.IDF`), so the same
    function serves a chunk at ingest and a question at query time."""
```

## Testing Strategy

- Unit: tokenizer (NFC, casefold, giữ "2105618", "K18", bỏ dấu câu, ổn định chỉ số); RRF
  (thứ tự, tie-break, chunk chỉ-sparse nhận điểm lexical); sàn/ngưỡng theo có hay không
  LLM rerank.
- Qdrant store: fake client ghi lại `query_batch_points` requests (4 request mỗi query,
  filter trên cả 4), `update_collection`/`update_vectors` của backfill.
- Graph wiring: hybrid bật/tắt, dense-only khi collection thiếu sparse, canary quyền.
- Re-index: point copy sang collection mới có `bm25`.
- Eval (không phải pytest): baseline và sau-hybrid trên demo + lát cắt mã/tên; ghi vào plan.

## Boundaries

- **Always:** filter quyền trên mọi truy vấn kể cả sparse; `score` giữ nghĩa cosine; test,
  ruff, mypy trước khi commit; backfill idempotent; eval baseline trước khi bật.
- **Ask first:** thêm dependency; đổi RECALL_K hay bộ câu hỏi eval; đổi tokenizer sang
  tách từ ghép; chạy re-index toàn bộ (tốn OpenAI); nới sàn dense dưới 0,45.
- **Never:** bỏ filter quyền khỏi truy vấn sparse; dùng điểm RRF làm `score`; xoá
  collection đang chạy; commit `.env`.

## Success Criteria

1. Cờ tắt hoặc collection chưa có sparse: mọi test cũ xanh, thứ tự chunk không đổi.
2. Collection eval sau backfill: 100 % point có `bm25`; backfill lần 2 là no-op.
3. "học phần <mã có trong kho>" (sau khi ingest lát cắt) → chunk chứa mã nằm trong 16 ứng
   viên; dense-only không có.
4. Eval demo: recall@5 (RECALL_K hiện tại) sau ≥ trước − 1 điểm (nhiễu); leak = 0; tỷ lệ
   ticket fallback không tăng.
5. Eval lát cắt mã/tên (≥ 30 câu, ≥ 30 tài liệu ingest thêm): recall@5 tăng ≥ 15 điểm
   tuyệt đối so với dense-only trên cùng corpus.
6. p95 thời gian `Retrieval timing` tăng dưới 100 ms so với trước (log sẵn có).

## Quyết định đã chốt (20/10/2026)

1. Chỉ BM25 hybrid đợt này; cross-encoder là phase sau (xem plan).
2. Gate = LLM rerank, sàn dense 0,45; ngưỡng 0,70 chỉ khi không có LLM rerank.
3. Giữ LLM rerank; nếu sau này cần thay thì ưu tiên reranker API (Cohere Rerank 3.5) qua
   registry trước khi nghĩ tới model local.
4. Lát cắt eval: ingest thêm ≥ 30 tài liệu official chứa mã/tên được hỏi để có ca đo thật.

## Open Questions

- Qdrant có chấp nhận `update_collection(sparse_vectors_config=...)` trên collection đang
  có dữ liệu không: xác minh ở Task 2 của plan trên collection eval; không được thì backfill
  đi đường re-index blue-green.
