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

- [ ] Task 1: settings, schema bundle, `normalize_question`, `cache_key`, `describe_key`
- [ ] Task 2: `RedisRetrievalCache` + fake Redis in-memory + chính sách lần-thấy-thứ-hai + stats

### Checkpoint: Nền

- [ ] `task test:file -- tests/test_retrieval_cache.py` xanh; ruff + mypy sạch
- [ ] Toàn bộ `task test` xanh với cache tắt

### Phase 2: L2 (retrieval bundle)

- [ ] Task 3: đo độ ổn định câu độc lập trên bộ eval (cổng quyết định L2)
- [ ] Task 4: helper graph L2: build key parts, lookup + rebuild `TurnRerankResult`, store
- [ ] Task 5: nối L2 vào `streaming_graph` tại 08, trace `cached`, ghi nền

### Checkpoint: L2

- [ ] Test graph: hit bỏ qua embed/search/LLM rerank; stale → miss + DEL; không ghi khi rỗng/lỗi rerank
- [ ] Chạy tay `task be:dev env=eval` với URL cache bật: hỏi 3 lần cùng câu độc lập, log `miss_first`, `miss_second`, `hit`

### Phase 3: Invalidation

- [ ] Task 6: hook ingest (purge + bump), `update_index_metadata`, `delete_index`, `delete_indexed_chunk`

### Checkpoint: Invalidation

- [ ] Test: sửa quyền → purge đúng tài liệu; ingest → bump đúng scope; scope khác không đụng
- [ ] Chạy tay: cache hit → "Chunk & embed lại" tài liệu → lượt sau miss

### Phase 4: L1 (turn bundle)

- [ ] Task 7: nối L1 vào `streaming_graph` tại 03 với đủ điều kiện đọc/ghi

### Checkpoint: L1

- [ ] Test: hit bỏ qua classifier; bypass khi có history/resume/metadata_updates/intent khác
- [ ] Chạy tay: 3 hội thoại mới cùng câu → lần 3 không có node classifier trong trace

### Phase 5: Vận hành

- [ ] Task 8: CLI `app/tools/retrieval_cache.py` + `task db:retrieval-cache` + docs + `.env.example`
- [ ] Task 9: kill switch + lock stampede (SET NX, không chờ)

### Checkpoint: Hoàn tất

- [ ] `task test`, `task code:lint`, `task code:format`, `uv run mypy app tests` sạch
- [ ] `task db:retrieval-cache env=eval -- stats` in hit/miss theo ngày
- [ ] Spec và `docs/guide/cau-hinh.md` khớp code; todo đánh dấu xong; sẵn sàng review

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

- **Task 3 chưa chạy** (20/10): cần model CHAT thật. Lệnh: `task db:retrieval-cache-stability env=eval -- --limit 50 --runs 3`.
  Kết quả "fully stable" dưới 50 % → cắt L2 (xoá nhánh `lookup_l2`/`store_l2` trong
  `_run_advisory_flow`, giữ L1). Hiện L2 đang bật vì chi phí vận hành của nó bằng 0 khi miss.
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
