# Todo: Retrieval Cache hai tầng (UNISAGE-104)

Spec: `docs/specs/SPEC-retrieval-cache.md`. Plan: `plan.md` cùng thư mục.

## Task 1: Settings, schema bundle, normalize và key (thuần)

**Description:** Thêm `RETRIEVAL_CACHE_*` vào settings, model Pydantic cho bundle L1/L2 và
chunk ref, hàm `normalize_question`, `cache_key`, `describe_key`, `config_fingerprint`
(hash settings + template). Không có Redis ở task này.

**Acceptance criteria:**

- [x] `normalize_question`: NFC trước casefold; "Điều kiện tốt nghiệp là gì ?" và dạng NFD
      của nó cho cùng kết quả; giữ số và mã.
- [x] `cache_key("l2", parts)` ổn định giữa hai lần gọi, đổi bất kỳ trường nào (scope, coll,
      gen, cfg, qs, date) → khoá khác; prefix `rc:v1:l2:`.
- [x] `CachedRetrievalBundle`/`CachedTurnBundle` round-trip JSON; `schema_version = 1`.

**Verification:**

- [x] `uv run pytest tests/test_retrieval_cache.py -q`
- [x] `uv run ruff check app tests && uv run mypy app tests`

**Dependencies:** None

**Files likely touched:** `app/core/config.py`, `app/schemas/retrieval_cache.py`,
`app/integrations/retrieval_cache.py`, `tests/test_retrieval_cache.py`, `.env.example`

**Estimated scope:** M

## Task 2: RedisRetrievalCache, fake Redis, lần-thấy-thứ-hai, stats

**Description:** Lớp `RedisRetrievalCache(url)` với `get_l1/get_l2`, `set_l1/set_l2` (kiểm tra
marker `seen`, ghi bundle + khoá mô tả + reverse index trong một pipeline, TTL jitter),
`purge_document`, `bump_generation`, `read_generations`, `record(outcome)`, `is_disabled`.
Fake Redis in-memory trong `tests/fakes/fake_redis.py` (get/set/setex/delete/incr/mget/
sadd/smembers/expire/exists/scan_iter/pipeline tối thiểu). `default_cache()` theo settings.

**Acceptance criteria:**

- [x] Miss lần 1 chỉ tạo marker; miss lần 2 ghi bundle và xoá marker; lần 3 `get` trả bundle.
- [x] Tắt `RETRIEVAL_CACHE_WRITE_ON_SECOND_SIGHTING` → ghi ngay.
- [x] `purge_document(doc)` xoá mọi entry có doc trong reverse index và set đó; entry khác giữ nguyên.
- [x] `bump_generation("CNTT")` tăng `rc:v1:gen:dept:CNTT`; `read_generations` trả 0 cho khoá vắng.
- [x] Client ném lỗi → `get` trả None, `set` không raise, có log warning.

**Verification:**

- [x] `uv run pytest tests/test_retrieval_cache.py -q`
- [x] `uv run ruff check app tests && uv run mypy app tests`

**Dependencies:** Task 1

**Files likely touched:** `app/integrations/retrieval_cache.py`, `tests/fakes/fake_redis.py`,
`tests/test_retrieval_cache.py`

**Estimated scope:** M

## Checkpoint: Nền

- [x] `uv run pytest -q` toàn bộ xanh với cache tắt (mặc định test)
- [x] ruff + mypy sạch

## Task 3: Đo độ ổn định câu độc lập (cổng cho L2)

> Script và task đã có (`app/tools/retrieval_cache_stability.py`, `task db:retrieval-cache-stability`).
> Chưa chạy: cần gọi model CHAT thật (50 câu x 3 lần). Chạy xong ghi kết quả vào plan.md.

**Description:** Script `app/tools/retrieval_cache_stability.py`: đọc bộ câu hỏi eval, gọi
classifier N lần (mặc định 3) cho mỗi câu, chuẩn hoá câu độc lập, in tỷ lệ câu có N lần
ra cùng khoá, và top câu không ổn định. Thêm `task db:retrieval-cache-stability`.

**Acceptance criteria:**

- [ ] In: tổng câu, % ổn định hoàn toàn, % ổn định ≥ 2/3, 10 ví dụ lệch.
- [ ] Ghi kết quả vào `plan.md` mục Open Questions kèm quyết định giữ/cắt L2.

**Verification:**

- [ ] Chạy tay `task db:retrieval-cache-stability env=eval -- --limit 50`

**Dependencies:** Task 1

**Files likely touched:** `app/tools/retrieval_cache_stability.py`, `taskfiles/db.yml`

**Estimated scope:** S

## Task 4: Helper graph cho L2

**Description:** `app/graph/nodes/retrieval_cache.py`: `l2_key_parts(questions, security,
models, generations, date)`, `lookup_l2(cache, parts, retrieval_service, security)` → trả
`TurnRerankResult` hoặc None (stale → DEL + record), `store_l2(cache, parts, rerank_result)`
→ bundle từ `per_query`/`best_scores`/`chunks`.

**Acceptance criteria:**

- [x] Hit: `retrieval_service.fetch` được gọi đúng tập id, không gọi `retrieve_many_async`;
      `TurnRerankResult` rebuild có đúng `per_query`, `best_scores`, thứ tự `chunks`, score từ bundle.
- [x] Thiếu id khi fetch → trả None, entry bị xoá, `stale` tăng.
- [x] `store_l2` bỏ qua khi `chunks` rỗng.

**Verification:**

- [x] `uv run pytest tests/graph/test_retrieval_cache_node.py -q`

**Dependencies:** Task 2

**Files likely touched:** `app/graph/nodes/retrieval_cache.py`, `tests/graph/test_retrieval_cache_node.py`

**Estimated scope:** M

## Task 5: Nối L2 vào streaming_graph (node 08)

**Description:** Trong `_classified_turn` sau khi có `questions`: nếu cache bật và không
`resume`, `lookup_l2`; hit → bỏ `retrieve_chunks`, `rerank_chunks`, `llm_rerank`, trace
`08_RetrievalFilteringNode` với `cached=True`; miss → chạy như cũ rồi `store_l2` nền khi
LLM rerank không lỗi. `GraphTrace.node` nhận `cached: bool = False` và log.

**Acceptance criteria:**

- [x] Test graph với `FakeRetrievalService`: lần 3 cùng câu độc lập không gọi `retrieve_many_async`.
- [x] LLM rerank lỗi (warning `LLM_RERANK_FAILED`) → không ghi.
- [x] Cache tắt → không có lời gọi cache nào; mọi test wiring cũ xanh.

**Verification:**

- [x] `uv run pytest tests/graph -q`

**Dependencies:** Task 4

**Files likely touched:** `app/graph/streaming_graph.py`, `app/core/observability/graph_trace.py`,
`tests/graph/test_retrieval_cache_graph.py`

**Estimated scope:** M

## Checkpoint: L2

- [x] Test graph xanh
- [ ] Chạy tay với `RETRIEVAL_CACHE_REDIS_URL=redis://localhost:6380/3`: 3 lượt cùng câu độc lập → log `miss_first`, `miss_second`, `hit`

## Task 6: Hook invalidation

**Description:** `embed_chunks` sau `_drop_stale_points`: `purge_document` + `bump_generation`
(public hoặc department). `document_index_service.update_index_metadata` và `delete_index`:
`purge_document`. `documents.delete_indexed_chunk`: `purge_document`. Tất cả fail-open.

**Acceptance criteria:**

- [x] Ingest tài liệu public → `gen:public` tăng, `gen:dept:*` không đổi; tài liệu CNTT → `gen:dept:CNTT` tăng.
- [x] Sửa quyền / xoá tài liệu / xoá chunk → entry chứa tài liệu đó bị xoá, entry khác còn.
- [x] Cache tắt → các hook là no-op, test ingestion/document_index cũ xanh.

**Verification:**

- [x] `uv run pytest tests/test_retrieval_cache_invalidation.py tests/test_ingestion_embedding.py tests/test_document_index.py -q`

**Dependencies:** Task 2

**Files likely touched:** `app/worker/tasks/ingestion.py`, `app/services/document_index_service.py`,
`app/api/v1/documents.py`, `tests/test_retrieval_cache_invalidation.py`

**Estimated scope:** M

## Checkpoint: Invalidation

- [x] Test xanh
- [ ] Chạy tay: hit → "Chunk & embed lại" → lượt sau miss

## Task 7: Nối L1 vào streaming_graph (node 03)

**Description:** Trước `classify_intent`: nếu cache bật, `history` rỗng, `resume` None →
`lookup_l1` (key parts: q, role, profile, scope, coll, emb, gen, cfg, date). Hit → dựng
`IntentClassification` + `sub_queries` + `questions` + `TurnRerankResult` từ bundle, nhảy
vào `_classified_turn` với cờ bỏ qua transform/retrieval. Miss → sau LLM rerank thành công,
nếu mọi task `academic_advisory` và `metadata_updates` rỗng → `store_l1` nền.

**Acceptance criteria:**

- [x] Lần 3 cùng câu gốc, không lịch sử: không gọi classifier model, không gọi retrieval; web search và generation vẫn chạy với `tasks` từ bundle.
- [x] Có history / resume / `metadata_updates` / intent khác → không đọc hoặc không ghi đúng như spec.
- [x] `ClassifiedTask` dump giữ `hyde_text`, `sub_queries`, `web`, `sub_query_web`.

**Verification:**

- [x] `uv run pytest tests/graph -q`

**Dependencies:** Task 5, Task 6

**Files likely touched:** `app/graph/streaming_graph.py`, `app/graph/nodes/retrieval_cache.py`,
`app/schemas/retrieval_cache.py`, `tests/graph/test_retrieval_cache_graph.py`

**Estimated scope:** M

## Checkpoint: L1

- [x] Test xanh
- [ ] Chạy tay: 3 hội thoại mới cùng câu → lần 3 trace không có classifier

## Task 8: CLI và tài liệu

**Description:** `app/tools/retrieval_cache.py` với `list` (khoá + mô tả + TTL còn lại),
`show --key` (bundle), `stats [--days N]` (hit/miss_first/miss_second/stale/bypass/error theo
ngày và tầng, số entry, bộ nhớ ước tính), `clear [--document ID] [--disable|--enable]`.
`task db:retrieval-cache` trong `taskfiles/db.yml` theo mẫu `web-cache`. Cập nhật
`docs/guide/cau-hinh.md`, `.env.example`, `.env.eval.example` (sửa luôn ghi chú DB số cũ).

**Acceptance criteria:**

- [x] `task db:retrieval-cache env=eval -- stats` chạy được, in bảng theo ngày.
- [x] `clear --disable` đặt kill switch, graph bypass; `--enable` gỡ.

**Verification:**

- [x] Chạy tay 4 lệnh; `uv run ruff check app`

**Dependencies:** Task 2

**Files likely touched:** `app/tools/retrieval_cache.py`, `taskfiles/db.yml`,
`docs/guide/cau-hinh.md`, `.env.example`, `.env.eval.example`

**Estimated scope:** M

## Task 9: Kill switch trong graph và lock stampede

**Description:** Graph đọc `rc:v1:disabled` một lần mỗi lượt (cùng `MGET` với generation);
bật → `bypass`. Trước khi tính miss: `SET rc:v1:lock:{digest} NX PX 10000`; thua lock → vẫn
tính, không ghi.

**Acceptance criteria:**

- [x] Kill switch bật → không get/set bundle, `bypass` tăng.
- [x] Hai lượt miss đồng thời cùng khoá → chỉ một ghi.

**Verification:**

- [x] `uv run pytest tests/test_retrieval_cache.py tests/graph/test_retrieval_cache_graph.py -q`

**Dependencies:** Task 5

**Files likely touched:** `app/integrations/retrieval_cache.py`, `app/graph/nodes/retrieval_cache.py`,
`tests/test_retrieval_cache.py`

**Estimated scope:** S

## Checkpoint: Hoàn tất

- [ ] `uv run pytest -q`, `uv run ruff check app tests`, `uv run ruff format --check app tests`, `uv run mypy app tests`
- [ ] Spec, plan, cau-hinh.md khớp code
- [ ] Commit theo từng task, PR lên `main`
