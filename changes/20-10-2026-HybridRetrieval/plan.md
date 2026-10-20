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
  **backfill tại chỗ** (`update_collection` + `update_vectors`) là lối tắt cho dev/eval và là
  dự phòng; không cutover, không tốn OpenAI. Cả hai dùng chung tokenizer và helper tính sparse.
- **BM25 tự tính, IDF server-side** (`Modifier.IDF`): không dependency, cùng tokenizer cho
  ingest và truy vấn, chỉ số = CRC32 token.
- **RRF phía client** thay vì `FusionQuery` của Qdrant: cần giữ cosine dense làm `score`
  (các bước sau hiểu score là cosine), và code đã trộn client-side cho 3 vector dense.
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

- [ ] Task 1: `lexical.py`: tokenizer, `sparse_vector`, `rrf_merge`

### Phase 2: Lưu trữ

- [ ] Task 2: `qdrant_store`: sparse config, `ChunkPoint.bm25`, `has_sparse_vector`, helpers backfill
- [ ] Task 3: CLI `db:bm25-backfill`, chạy trên collection eval, xác minh `update_collection`

### Checkpoint: Lưu trữ

- [ ] 100 % point eval có `bm25`; backfill lần 2 no-op; test store xanh

### Phase 2b: Re-index "rebuild" (đường production, 3 repo)

- [ ] Task 9a (backend): `EmbeddingReindexRequest` thêm `rebuild: boolean`; `start()` với
      `rebuild=true` nhắm vào chính model đang live (bỏ `EMBEDDING_REINDEX_MODEL_INVALID` và
      `EMBEDDING_REINDEX_NOT_NEEDED` cho ca này), vẫn chặn khi có lượt đang mở; run ghi
      `rebuild`; claim response cho agent biết `rebuild`
- [ ] Task 9b (agent): worker khi `rebuild`: scroll `with_vectors=True`, copy 3 vector dense
      nguyên trạng, tính `bm25` từ payload, upsert sang target; không gọi embedding; identity
      của target = identity nguồn; cutover như cũ
- [ ] Task 9c (web): dialog re-index có lựa chọn "Xây lại chỉ mục với model hiện tại (thêm
      BM25)" gửi `rebuild: true`; panel hiển thị lượt rebuild

### Checkpoint: Rebuild

- [ ] Trên eval: chạy rebuild → collection mới có `bm25` đủ 100 % point, cutover xong, chat
      vẫn trả lời, retrieval cache tự miss (coll đổi); không có lượt gọi OpenAI trong log worker

### Phase 3: Truy vấn

- [ ] Task 4: `search_chunks_batch` 4 truy vấn + RRF; `RetrievalService.retrieve_many(_async)` nhận `lexical_queries`
- [ ] Task 5: ingest và re-index ghi sparse
- [ ] Task 6: graph: `SubQuery.question` làm lexical query; sàn dense khi có LLM rerank; cờ hybrid; cache fingerprint

### Checkpoint: Truy vấn

- [ ] Test graph xanh; "học phần <mã>" có chunk trong 16 ứng viên; canary quyền
- [ ] Chạy tay trên stack eval với `RETRIEVAL_HYBRID_ENABLED=true`

### Phase 4: Đo

- [ ] Task 7: eval baseline (trước), lát cắt mã/tên (chọn ≥ 30 tài liệu official, ingest), eval sau; ghi vào plan

### Phase 5: Tài liệu

- [ ] Task 8: `docs/guide/cau-hinh.md`, `.env.example`, `dev-onboard.txt` (ngoài repo), spec khớp code

### Checkpoint: Hoàn tất

- [ ] `task test`, ruff, mypy sạch; eval đạt tiêu chí; PR

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
