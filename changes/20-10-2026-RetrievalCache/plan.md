# Implementation Plan: Retrieval Cache hai tầng (UNISAGE-104)

Spec: `docs/specs/SPEC-retrieval-cache.md` (duyệt 20/10/2026). Nhánh
`feature/huydh-unisage-104-retrieved-chunks-cache`. Tasks ở `todo.md` cùng thư mục.

## Overview

Thêm một Redis cache (instance `redis-cache` 6380, DB 2 dev / 3 eval) lưu **ID chunk và
điểm** của kết quả retrieval đã qua rerank, hai tầng exact-match: L1 khoá theo câu hỏi gốc
(lượt không có lịch sử, bỏ qua classifier + embedding + Qdrant + LLM rerank), L2 khoá theo
danh sách câu độc lập mà classifier/HyDE viết ra (mọi lượt học vụ, bỏ qua embedding +
Qdrant + LLM rerank). Hit thì fetch lại chunk bằng `RetrievalService.fetch` có filter quyền;
thiếu chunk là miss. Ghi chỉ khi một khoá miss lần thứ hai trong TTL 6 giờ. Invalidation
theo sự kiện (reverse index tài liệu → khoá, generation theo phòng ban) với TTL là lưới
cuối. Fail-open, kill switch, bộ đếm hit/miss trong Redis và CLI `task db:retrieval-cache`.

## Architecture Decisions

- **Soi gương `app/integrations/web_search_cache.py`**: Protocol + lớp Redis, một kết nối
  ngắn mỗi lần gọi (request và background chạy trên event loop khác nhau), hàm thuần cho
  key/normalize để test không cần Redis, fail-open bằng `try/except` + log warning.
- **Chỉ lưu `chunk_id` + `score`**, rebuild `TurnRerankResult` bằng `turn_result()` có sẵn
  sau khi `RetrievalService.fetch` trả chunk (đã có filter quyền). Không có đường nào đưa
  nội dung từ cache vào prompt.
- **Khoá = SHA-256 của JSON canonical** gồm đầy đủ trường trong spec (q/qs, role, profile,
  scope, coll, emb, gen, cfg, date). `cfg` hash cả text template prompt liên quan, nên đổi
  prompt tự đổi khoá, không cần nhớ bump version.
- **Điều kiện "không có lịch sử"** cho L1 = `not graph_input.history and graph_input.resume
  is None`, tính từ input của graph, không dùng cờ `is_first_turn` của HTTP (cờ đó phục vụ
  greeting).
- **Ghi cache không chặn**: `asyncio.create_task` + `_finish_in_background` như các job web
  search; reverse index và dấu `seen` ghi trong cùng pipeline Redis.
- **Chính sách lần-thấy-thứ-hai** nằm trong lớp cache (`set_*` kiểm tra marker), graph
  không cần biết.
- **Stats trong Redis** (`INCR` theo ngày/tầng/outcome) thay vì thêm stack metrics; CLI
  `stats` đọc ra. Đây là cách đo tiêu chí thành công.
- **Eval/test tắt cache mặc định**: `RETRIEVAL_CACHE_REDIS_URL` rỗng trong `.env.eval` và
  trong `tests/conftest.py` (settings override), nên mọi test hiện có không đổi.
- **Cổng quyết định cho L2**: task đo độ ổn định câu độc lập chạy trước khi nối L2; dưới
  50 % thì cắt L2 khỏi v1 (ghi lại trong todo).

## Dependency Graph

```
T1 settings + schemas + normalize/key (thuần)
    │
    ├── T2 RedisRetrievalCache (get/set/seen/purge/bump/stats) + fake Redis
    │       │
    │       ├── T4 node helpers L2 (lookup/rebuild/store) ── T5 nối L2 vào graph 08
    │       │                                                     │
    │       ├── T6 hook invalidation (ingest, index-metadata, delete, chunk-delete)
    │       │                                                     │
    │       ├── T8 CLI + taskfile + docs                          │
    │       │                                                     │
    │       └── T9 kill switch + lock stampede                    │
    │                                                             │
    └── T3 đo độ ổn định câu độc lập (song song, cổng cho T4/T5)  │
                                                                  └── T7 nối L1 vào graph 03
```

## Task List

### Phase 1: Nền (thuần + lớp Redis)

- [x] Task 1: settings, schema bundle, `normalize_question`, `cache_key`, `describe_key`
- [x] Task 2: `RedisRetrievalCache` + fake Redis in-memory + chính sách lần-thấy-thứ-hai + stats

### Checkpoint: Nền

- [x] `task test:file -- tests/test_retrieval_cache.py` xanh; ruff + mypy sạch
- [x] Toàn bộ `task test` xanh với cache tắt

### Phase 2: L2 (retrieval bundle)

- [x] Task 3: đo độ ổn định câu độc lập trên bộ eval (cổng quyết định L2)
- [x] Task 4: helper graph L2: build key parts, lookup + rebuild `TurnRerankResult`, store
- [x] Task 5: nối L2 vào `streaming_graph` tại 08, trace `cached`, ghi nền

### Checkpoint: L2

- [x] Test graph: hit bỏ qua embed/search/LLM rerank; stale → miss + DEL; không ghi khi rỗng/lỗi rerank
- [x] (21/10, qua evals.run 3 lượt cùng câu: lượt 1 seen, lượt 2 ghi entry L1+L2, lượt 3 hit L1 1,4 s thay 8–9 s) Chạy tay `task be:dev env=eval` với URL cache bật: hỏi 3 lần cùng câu độc lập, log `miss_first`, `miss_second`, `hit`

### Phase 3: Invalidation

- [x] Task 6: hook ingest (purge + bump), `update_index_metadata`, `delete_index`, `delete_indexed_chunk`

### Checkpoint: Invalidation

- [x] Test: sửa quyền → purge đúng tài liệu; ingest → bump đúng scope; scope khác không đụng
- [x] Chạy tay: cache hit → "Chunk & embed lại" tài liệu (eval:ingest --rechunk) → entry + doc index xoá, gen public 0→1, lượt sau miss — 21/10

### Phase 4: L1 (turn bundle)

- [x] Task 7: nối L1 vào `streaming_graph` tại 03 với đủ điều kiện đọc/ghi

### Checkpoint: L1

- [x] Test: hit bỏ qua classifier; bypass khi có history/resume/metadata_updates/intent khác
- [x] Chạy tay: 3 hội thoại mới cùng câu → lần 3 hit L1, `retrievals=0`, `retrieval_cache_fetches=1` — 21/10

### Phase 5: Vận hành

- [x] Task 8: CLI `app/tools/retrieval_cache.py` + `task db:retrieval-cache` + docs + `.env.example`
- [x] Task 9: kill switch + lock stampede (SET NX, không chờ)

### Checkpoint: Hoàn tất

- [x] `task test`, `task code:lint`, `task code:format`, `uv run mypy app tests` sạch
- [x] `task db:retrieval-cache env=eval -- stats` in hit/miss theo ngày
- [x] Spec và `docs/guide/cau-hinh.md` khớp code; todo đánh dấu xong; sẵn sàng review

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| L2 hit rate thấp vì classifier viết câu độc lập không ổn định | Med | T3 đo trước; dưới 50 % cắt L2; L1 không phụ thuộc model |
| Hook invalidation sót một đường ghi Qdrant | Med | fetch-by-id với filter bắt chunk mất/đổi quyền; `stale` counter là chuông báo; test canary theo scope |
| Khoá thiếu trường → hit vào kết quả của cấu hình khác | High | `cfg` hash template + settings; test "đổi từng trường → khoá khác" |
| Redis chậm làm tăng độ trễ | Low | timeout 50 ms, fail-open, ghi nền |
| Eviction instance dùng chung với web cache | Low | ước tính < 30 MB; CLI `stats` in số entry/bộ nhớ |
| Bỏ qua classifier ở L1 làm mất `metadata_updates` | High | chỉ ghi L1 khi `metadata_updates` rỗng; `profile` + `role` trong khoá |
| Lượt L1 hit vẫn cần `web` slots cho web search | Med | bundle L1 lưu `ClassifiedTask` dump đầy đủ (kể cả field `exclude=True`) |

## Open Questions

- **Task 3 đã chạy** (20/10, `gpt-5-mini`, 50 câu eval có dấu x 3 lần, 48 câu học vụ):
  **23/48 = 47,9 % cho cùng khoá cả 3 lần; 38/48 = 79,2 % trùng ít nhất 2/3.** Lệch chủ yếu
  do dấu câu (ngoặc, hai chấm, ngoặc kép), model tự chèn "khóa K13 (nhập học 2016)", đồng
  nghĩa ("phải học trước"/"tiền quyết"), lỗi gõ của model ("vàphải"), và tách SINGLE/MULTI.
  Quyết định: **giữ L2**, vì (a) sát ngưỡng 50 % và một phần lệch là dấu câu đã được
  `normalize_question` bỏ hết sau lần đo này, (b) miss L2 chỉ tốn một GET Redis, không tốn
  LLM. Đo lại bằng `stats` thật sau 2 tuần; hit rate L2 dưới 5 % thì cắt. Đo thêm
  `--field question_user` (câu không dấu, viết tắt kiểu sinh viên gõ, 49 câu học vụ):
  **0/49 ổn định cả 3 lần, 14/49 = 28,6 % trùng 2/3.** Với đầu vào không dấu, classifier
  phải tự thêm dấu và viết lại nên gần như không lặp được. Kết luận: L2 chỉ có giá trị khi
  câu đầu vào đã sạch (có dấu, ít viết tắt); phần lớn lợi ích đến từ L1. Giữ L2 nhưng
  không kỳ vọng; cổng cắt: hit rate L2 trong `stats` < 5 % sau 2 tuần.
- **Kiểm tra tay** còn lại (checkpoint L2, Invalidation, L1): chạy với
  `RETRIEVAL_CACHE_REDIS_URL=redis://localhost:6380/3` trong `.env.eval`, xem log
  `retrieval_cache layer=... outcome=...` và `task db:retrieval-cache env=eval -- stats`.

## Ghi chú implement (20/10)

- Lock single-flight được `set()` giải phóng ngay sau khi ghi (hoặc sau khi chỉ đặt dấu
  `seen`), nếu không hai lượt tuần tự trong 10 giây sẽ không bao giờ ghi được.
- `embedding_identity` lấy `identity_key` của embedder thật; fake trong test cho "unknown".
- `store_l1` chỉ khi `metadata_updates` rỗng và mọi task là `academic_advisory`; lượt có
  task tính toán truyền `l1_pending=None` xuống.
- Hook invalidation nằm ở `app/services/retrieval_cache_invalidation.py`, no-op khi cache tắt.

## Known gaps / Phase sau (21/10)

- **Semantic cache cho L1**: chưa làm, có chủ đích. Khoảng trống, lý do, cổng mở lại (hit rate L1
  < 20 % sau 1–2 tuần log thật) và phác thảo cách làm ghi ở `docs/specs/SPEC-retrieval-cache.md`
  mục "Known gaps". Không bắt đầu trước khi có số đo hit rate exact-match thật.
