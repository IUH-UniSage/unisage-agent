# Implementation Plan: Hybrid retrieval BM25 + dense

Spec: `docs/specs/SPEC-hybrid-retrieval.md` (duyệt ý định 20/10/2026). Tasks ở `todo.md`.

## Overview

Thêm sparse vector `bm25` vào collection Qdrant hiện có (backfill tại chỗ, không embed lại),
tính từ tokenizer riêng với IDF do Qdrant đảm nhiệm; mỗi sub-query gửi 3 dense + 1 sparse
rồi trộn RRF phía client; `score` của chunk vẫn là cosine dense để sàn, rescue, web search và
cache giữ nguyên nghĩa; gate chuyển sang LLM rerank với sàn dense thấp. Đo bằng bộ eval
trước và sau, cộng một lát cắt mới gồm câu có mã/tên chính xác.

## Architecture Decisions

- **Hai đường đưa `bm25` vào collection đang chạy** (chốt 20/10): (1) **re-index
  blue-green "rebuild"** là đường production: dựng collection mới có sparse, copy vector
  dense nguyên trạng từ collection cũ (không gọi OpenAI vì model không đổi), tính sparse từ
  payload, cutover nguyên tử bằng cơ chế UNISAGE-103; cần backend cho phép lượt re-index
  nhắm vào chính model đang live (hiện bị chặn bởi `EMBEDDING_REINDEX_NOT_NEEDED`). (2)
  **backfill cùng tên qua collection tạm** là lối tắt cho dev/eval và là dự phòng; không
  cutover, không tốn OpenAI. *Sửa 21/10:* Qdrant 1.15 không cho thêm vector vào collection
  đang có (`update_collection` → "Not existing vector name"), nên "tại chỗ" thành "xây lại
  cùng tên": copy sang `<name>__bm25tmp`, xoá/tạo lại gốc có sparse, copy về
  (`copy_points_with_bm25`, cũng là helper Task 9b dùng). Cả hai dùng chung tokenizer và
  helper tính sparse.
- **BM25 tự tính, IDF server-side** (`Modifier.IDF`): không dependency, cùng tokenizer cho
  ingest và truy vấn, chỉ số = CRC32 token. *Bổ sung 21/10 (đo trên corpus eval):* phía chunk
  dùng tf bão hoà BM25 (k1 1,2, b 0,75, avg 165 token) thay tf thô; phía câu hỏi bỏ stopword
  tiếng Việt và cho token có chữ số trọng số ×3. Không có ba điều này, chunk chứa mã đứng
  hạng 24 BM25 vì "có bao nhiêu tín chỉ" cộng idf vượt một mã.
- **RRF phía client** thay vì `FusionQuery` của Qdrant: cần giữ cosine dense làm `score`
  (các bước sau hiểu score là cosine), và code đã trộn client-side cho 3 vector dense.
  *Bổ sung 21/10:* trộn **2 danh sách** (dense gộp theo max cosine, BM25) chứ không 4 — 4
  danh sách cho dense 3 phiếu; và **k = 1** thay 60: lưới đo k ∈ {60, 20, 10, 5, 1} ×
  hệ số {1, 2} trên 5 câu có mã cho thấy k = 60 làm chunk hạng 1 BM25 rớt khỏi top-16 (mọi
  chunk có mặt ở cả hai danh sách thắng nó), k = 1 + hệ số 2 đưa nó lên hạng 2–3 và giữ
  nguyên top-8 dense.
- **Sparse từ câu hỏi độc lập, dense từ HyDE**: mã và tên nằm nguyên văn trong câu hỏi; HyDE
  passage diễn giải và dễ rớt mã.
- **Gate = LLM rerank + sàn dense 0,45**; 0,70 chỉ khi không có LLM rerank. Lý do: fusion
  làm ngưỡng cosine mất nghĩa như một gate, LLM rerank đã làm đúng việc đó cho 16 chunk.
- **Tự phát hiện sparse**: collection thiếu `bm25` → dense-only + warning; không làm hỏng
  môi trường chưa backfill.
- **Rebuild không gọi OpenAI**: vector dense copy nguyên trạng vì model không đổi; chỉ khi
  đổi model mới embed lại (đường re-index cũ, Task 5 đảm bảo nó cũng ghi sparse).
- **Eval có baseline**: chạy eval demo trước khi bật, lưu `runs/hybrid-baseline`; bật rồi
  chạy lại; thêm lát cắt mã/tên từ bộ official.

## Dependency Graph

```
T1 lexical.py (tokenizer, sparse_vector, RRF thuần)
    ├── T2 qdrant_store: sparse config, ChunkPoint.bm25, upsert, has_sparse, backfill helpers
    │       ├── T3 CLI db:bm25-backfill (dev/eval, dự phòng) + xác minh update_collection
    │       ├── T4 search_chunks_batch hybrid + RetrievalService lexical_queries
    │       │       └── T6 graph: truyền question, sàn/ngưỡng, cờ, cache fingerprint
    │       ├── T5 ingest + re-index ghi sparse
    │       └── T9 re-index "rebuild" (production): backend cho phép cùng model,
    │           agent copy dense + tính sparse, web có lựa chọn "Xây lại chỉ mục"
    └── T7 eval: baseline, lát cắt mã/tên (ingest thêm), chạy sau, ghi kết quả
T8 docs + .env.example (song song cuối)
Phase sau: cross-encoder (không làm đợt này)
```

## Task List

### Phase 1: Nền thuần

- [x] Task 1: `lexical.py`: tokenizer, `sparse_vector` (tf bão hoà), `sparse_query_vector` (stopword, trọng số mã), `rrf_merge` — 21/10

### Phase 2: Lưu trữ

- [x] Task 2: `qdrant_store`: sparse config, `ChunkPoint.bm25`, `has_sparse_vector`, `copy_points_with_bm25` — 21/10
- [x] Task 3: CLI `db:bm25-backfill` (xây lại cùng tên qua collection tạm), chạy trên eval 2.323 point/16 s; `update_collection` KHÔNG thêm được vector — 21/10

### Checkpoint: Lưu trữ

- [x] 100 % point eval có `bm25` (2.322/2.323 — 1 point không có text); backfill lần 2 no-op; test store xanh — 21/10

### Phase 2b: Re-index "rebuild" (đường production, 3 repo)

- [x] Task 9a (backend) — 21/10: `EmbeddingReindexRequest` thêm `rebuild: boolean`; `start()` với
      `rebuild=true` nhắm vào chính model đang live (bỏ `EMBEDDING_REINDEX_MODEL_INVALID` và
      `EMBEDDING_REINDEX_NOT_NEEDED` cho ca này), vẫn chặn khi có lượt đang mở; run ghi
      `rebuild`; claim response cho agent biết `rebuild`
- [x] Task 9b (agent) — 21/10: worker khi `rebuild`: scroll `with_vectors=True`, copy 3 vector dense
      nguyên trạng, tính `bm25` từ payload, upsert sang target; không gọi embedding; identity
      của target = identity nguồn; cutover như cũ
- [x] Task 9c (web) — 21/10: dialog re-index có lựa chọn "Xây lại chỉ mục với model hiện tại (thêm
      BM25)" gửi `rebuild: true`; panel hiển thị lượt rebuild

### Checkpoint: Rebuild

- [x] 21/10: rebuild trên eval (run e7818764, xếp thẳng vào DB vì không có đăng nhập) → `unisage_eval_chunks__20261021030324` 2.323 point, 2.322 có `bm25` (1 point không có text), worker không gọi embedding, 10 s; pointer chuyển, agent thấy collection mới, câu mã vẫn hạng 2; cache 0 entry sau cutover. Trên eval: chạy rebuild → collection mới có `bm25` đủ 100 % point, cutover xong, chat
      vẫn trả lời, retrieval cache tự miss (coll đổi); không có lượt gọi OpenAI trong log worker

### Phase 3: Truy vấn

- [x] Task 4: `search_chunks_batch` 4 truy vấn + RRF; `RetrievalService.retrieve_many(_async)` nhận `lexical_queries`
- [x] Task 5: ingest và re-index ghi sparse — 21/10
- [x] Task 6: graph: `SubQuery.question` làm lexical query; sàn dense khi có LLM rerank; cờ hybrid; cache fingerprint

### Checkpoint: Truy vấn

- [x] Test graph xanh; "học phần <mã>" có chunk trong 16 ứng viên (hạng 2–3 cho 2107555/8940390/2102435, dense-only: không có); canary quyền — 21/10
- [x] Chạy tay trên stack eval với `RETRIEVAL_HYBRID_ENABLED=true` (RetrievalService, SA) — 21/10

### Phase 4: Đo

- [~] Task 7: eval baseline (trước) và eval sau trên demo — **đã chạy 21/10/2026**; lát cắt mã/tên: **không đủ dữ liệu** (xem dưới)

  Lát cắt mã/tên (21/10): bộ câu hỏi official (575) chỉ có 25 câu chứa mã học phần / số quyết
  định, 22 câu trong đó dùng tài liệu đã có trong corpus demo, 3 câu còn lại không có PDF trên
  máy — không có nguồn cho "≥ 30 tài liệu, ≥ 30 câu" của tiêu chí 5. Đã chạy 6 câu đủ điều kiện
  (d0012, d0215, d0312, d0313, d0334, d0335; `runs/hybrid-slice-*`) hai lần mỗi cấu hình:
  recall@5 tắt 4/6 + 4/6, bật 4/6 + 3/6; câu đổi (d0335) có tài liệu đúng ở hạng 1 và 6 trong
  hai run bật, tức nhiễu HyDE/classifier giữa các lần chạy, không kết luận được. Muốn đo tiêu chí
  5 phải soạn thêm bộ câu hỏi có mã (≥ 30 câu, có đáp án và `expected_doc_ids`), là việc dữ liệu
  chưa làm. Bằng chứng trực tiếp hiện có vẫn là kiểm tra RetrievalService: 3 mã có trong kho,
  dense-only 0/3 trong 16 ứng viên, hybrid 3/3 ở hạng 2–3.

  Hai run cùng corpus eval (108 tài liệu, 2.323 point), `--set demo --variant exact --web off
  --category normal --category calculation`, cache tắt, concurrency 3, mỗi run ~5 phút:

  | Chỉ số | `runs/hybrid-baseline` (tắt) | `runs/hybrid-on` (bật) | Tiêu chí spec |
  |---|---|---|---|
  | recall@5 normal (116) | 105 | 104 | ≥ trước − 1 → đạt |
  | recall@5 calculation (38) | 16 | 23 | (+7, +18 điểm) |
  | leak | 0 | 0 | = 0 → đạt |
  | ticket fallback | 0 | 1 (d0333, lượt có failover RPM) | không tăng → lệch 1/154, xem ghi chú |
  | latency p50 / p95 (ms) | 5.289 / 12.834 | 5.345 / 9.453 | p95 +<100 ms → đạt |
  | pass calculation (LLM tính) | 20/38 | 18/38 | — (phụ thuộc model, run bật gặp 503 + RPM saturation) |

  Câu đổi recall: +11 (d0008, d0022, d0035 normal; d0244, d0308, d0311, d0325, d0327, d0331,
  d0332, d0335 calculation) / −5 (d0028, d0033, d0036, d0050 normal; d0309 calculation). Các câu
  mất đều là câu không có mã. Soi từng câu (21/10, từ `retrievals`/`context` trong results.jsonl):
  d0028, d0036, d0050 — tài liệu đúng CÓ trong 16 ứng viên của hybrid (hạng 1, 4, 6; baseline
  hạng 4, 2, 15) nhưng LLM rerank đẩy xuống hạng 6 hoặc loại; d0033, d0309 — run hybrid không
  chạy retrieval (classifier rẽ nhánh khác, d0309 baseline cũng 0 chunk sau rerank). Tức là
  nhiễu LLM rerank/classifier, không phải BM25 làm mất ứng viên. Không tinh chỉnh BM25 vì
  việc này; nếu muốn ổn định hơn thì là việc của reranker (phase cross-encoder). Quyết định: **giữ bật mặc định** (tiêu chí 4 đạt,
  tiêu chí 6 đạt, tiêu chí 5 chờ lát cắt mã). Cần `departments.json` (gitignore) trong thư mục
  dataset: dựng từ bảng `departments` của DB backend eval.

### Phase 5: Tài liệu

- [x] Task 8: `docs/guide/cau-hinh.md`, `.env.example`, `.env.eval.example`, `dev-onboard.txt` mục 14 (ngoài repo), docstring `cross_encoder.py`, spec khớp code — 21/10

### Checkpoint: Hoàn tất

- [~] `task test` (2.355 xanh, 1 flaky `test_judge` không liên quan, `tests/e2e` lỗi import có sẵn), ruff/mypy sạch cho file đổi (lỗi còn lại có sẵn ở file khác); eval đạt tiêu chí 4, 6; PR #26 (draft) — 21/10

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Qdrant không cho thêm sparse vào collection có dữ liệu | Med | Task 2/3 xác minh sớm trên eval; fallback re-index blue-green |
| BM25 kéo chunk khớp mã nhưng sai quyền | High | filter quyền trên cả 4 truy vấn; test canary; `leaked_documents` trong eval |
| Sàn 0,45 đưa rác vào LLM rerank, tăng token | Low | 16 ứng viên không đổi; đo chi phí rerank trước/sau |
| RRF đẩy chunk dense tốt xuống vì sparse khớp từ phổ thông | Med | sparse chỉ từ câu hỏi độc lập; IDF làm từ phổ thông gần 0; đo recall demo không giảm |
| Chunk chỉ-sparse nhận score giả 0,70 làm lệch `best_scores` web search | Low | đo tỷ lệ web search trên demo; hằng số cấu hình được |
| Lát cắt eval không đủ ca để kết luận | Med | ≥ 30 câu, ≥ 30 tài liệu; chọn câu có mã/tên trong bộ official |

## Phase sau: Cross-encoder (không làm đợt này, đã chốt 20/10/2026)

Máy deploy: Windows, RTX 3060 12 GB VRAM. Model: `BAAI/bge-reranker-v2-m3`, FP16.

**Giữ nguyên từ kế hoạch ban đầu:** model và FP16; không mua GPU; service riêng; warm-up
trước khi đo; đo P95 cấp request bằng chunk thật; reset thống kê bộ nhớ mỗi lượt.

**Điều chỉnh đã chốt:**

| Mục | Quyết định | Vì sao |
|---|---|---|
| Độ dài đầu vào | `max_length=1024` cho cả cặp query+passage (model fine-tune ở 1024). Kiểm tra độ dài thật sau tokenize và truncation trên chunk 800 token; so sánh với phương án rerank trên `summary` bằng eval chất lượng, không giả định | Chunk tới 800 token; cắt 512 mất phần chứa bảng, điều kiện |
| Tải mục tiêu | 2–5 QPS, 16 ứng viên/request; vẫn test burst và concurrency, không chỉ tải đều | Tải thật của một trường; pipeline chỉ đưa 16 chunk |
| Batch | Ma trận batch 4 → 8 → 16 → 32 ở 1024; ghi peak VRAM, throughput, P95; dừng khi gần hết VRAM hoặc P95 xấu đi | Không chốt trước mức activation |
| Serving | Service riêng. Baseline: FlagEmbedding + FastAPI, load model một lần, 1 GPU worker, chạy Windows native nếu CUDA tương thích, không thì WSL2. Chỉ chuyển sang Infinity/TEI sau khi xác minh runtime hỗ trợ đúng model và đo được lợi ích (TEI chưa liệt kê v2-m3) | Agent/worker không ôm model; tách service ≠ bắt buộc inference server |
| Nối vào hệ thống | Đăng ký purpose RERANK dạng self-hosted trong model registry; service sập → failover LLM rerank | Cơ chế sẵn có |
| Chất lượng | Cùng candidate set cho 3 cấu hình: LLM rerank / cross-encoder / cross-encoder rồi LLM rerank. Đo recall@k, NDCG@16, MRR@16, tỷ lệ fallback, P95, chi phí mỗi query, chất lượng câu trả lời cuối (học phí, điều kiện, ngoại lệ) | Tốc độ không phải câu hỏi |
| Ngưỡng | Nếu dùng điểm cross-encoder làm gate/fallback, hiệu chuẩn ngưỡng trên tập validation; điểm BGE không phải xác suất, không có ngưỡng chung | |
| Điều kiện production | P95 < 500 ms ở tải mục tiêu, không OOM, queue không tăng liên tục, chất lượng đạt ngưỡng eval | |

Lưu ý máy deploy: nếu generation cũng chạy local sau này thì chia 12 GB với reranker;
Celery Windows `--pool=solo` không ảnh hưởng vì rerank nằm ở chat API.

## Open Questions

- Số ticket cho nhánh.
- Danh sách ≥ 30 tài liệu official cho lát cắt mã/tên: chọn theo câu hỏi có `mã`, `Quyết
  định`, tên học phần trong `questions.jsonl`, lấy `expected_doc_ids` → `--only` CSV.
