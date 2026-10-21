# Spec: Retrieval Cache hai tầng, exact-match (unisage-agent)

> Ticket UNISAGE-104, nhánh `feature/huydh-unisage-104-retrieved-chunks-cache`.
> Spec này **chỉ thêm**, không đổi hành vi của retrieval, rerank hay generation khi
> cache miss hoặc cache tắt. Tắt cache (URL rỗng hoặc kill switch) thì pipeline chạy
> y như hôm nay.
>
> Nền tảng: báo cáo nghiên cứu `reports/Cache retrieval trong RAG production.md`
> (20/10/2026) và buổi phỏng vấn chốt ý định ngày 20/10/2026.

## Ý định đã chốt (từ buổi phỏng vấn)

- **Outcome:** giảm chi phí và độ trễ cho các câu hỏi học vụ lặp lại giữa nhiều sinh
  viên, mà không làm sai hay cũ câu trả lời.
- **Người hưởng lợi:** sinh viên hỏi FAQ ("học phí kỳ này bao nhiêu", "điều kiện tốt
  nghiệp") ở nhiều phiên khác nhau; đội vận hành trả ít tiền LLM hơn.
- **Thành công:** sau 1–2 tuần log thật, hit rate tầng 1 trên các lượt đầu hội thoại
  thuộc intent học vụ ≥ 20 %, và stale-serve = 0 (không lần nào hit mà fetch lại
  thiếu chunk hay bị filter quyền loại). Độ trễ và chi phí được log nhưng không đặt
  mục tiêu cứng. Không cần A/B test có nhóm đối chứng.
- **Quyết định thêm (20/10):** chỉ ghi cache khi một khoá đã miss lần thứ hai trong
  cửa sổ TTL, để câu hỏi một lần không chiếm chỗ.
- **Ràng buộc cứng:** chỉ khớp chính xác (exact-match). **Không** semantic match,
  **không** cache câu trả lời cuối, **không** cache follow-up theo câu gõ thô.
- **Ngoài phạm vi:** semantic cache, answer cache, khoá theo slot canonical của
  classifier (cân nhắc sau khi có số đo, chạy shadow trước), prompt caching phía nhà
  cung cấp, re-warm cache sau cutover, dùng chung entry giữa scope khi scope không có
  tài liệu riêng (tối ưu sau), thay đổi ở backend-java hay web.

## Assumptions (sửa ngay nếu sai, không thì tôi làm theo)

1. Lưu lượng chủ yếu là lượt đầu hội thoại; follow-up ít hơn và được phục vụ bởi
   tầng 2 khi câu độc lập trùng.
2. `RetrievalService.fetch(chunk_ids, security)` (đã có, dùng cho clarification
   resume) là đường fetch-by-id chuẩn: có áp `build_access_filter`, chunk mất hoặc
   không còn thấy được thì vắng mặt trong kết quả.
3. `chunk_id` trong payload Qdrant có dạng `{document_id}:{chunk_index}`, nên từ
   `chunk_id` suy ra được `document_id` mà không cần lưu riêng.
4. Redis instance `redis-cache` (cổng 6380, 256 MB, `volatile-lru`, không persist)
   dùng chung với web search cache là đủ cho v1; retrieval cache dùng logical DB
   khác (`/2` dev, `/3` eval). Eviction là theo instance, chấp nhận ở quy mô hiện tại
   (ước tính < 30 MB ở 50.000 lượt/ngày, TTL 6 h).
5. Bộ eval (`app/eval`, `.env.eval`) chạy với cache **tắt** mặc định để đo pipeline
   thật; bật bằng tay khi muốn đo riêng cache.
6. Thời gian hiện tại (`append_current_time`) ảnh hưởng tới output của classifier
   ("kỳ này" → học kỳ cụ thể), nên tầng 1 phải có ngày (ICT) trong khoá.

## Objective

### Vấn đề

Mỗi lượt chat học vụ chạy 3 lượt LLM (classifier, LLM rerank, generation), 1 lượt
embedding và 1 lượt Qdrant. Với FAQ lặp, bốn bước đầu tính lại cùng một kết quả cho
cùng một câu hỏi. Web search đã có cache; retrieval thì chưa.

### Giải pháp: hai tầng cache, đều exact-match, đều chỉ lưu ID chunk

| Tầng | Khoá theo | Áp dụng cho | Hit thì bỏ qua | Còn phải gọi LLM |
|---|---|---|---|---|
| **L1 "turn bundle"** | câu hỏi gốc của sinh viên (chuẩn hoá) + ngữ cảnh người hỏi | lượt **không có lịch sử** (`history` rỗng, không `resume`) | classifier, HyDE, embedding, Qdrant, score rerank, LLM rerank | 1/3 (generation) |
| **L2 "retrieval bundle"** | danh sách câu độc lập do classifier/HyDE viết ra | **mọi** lượt học vụ, kể cả follow-up | embedding, Qdrant, score rerank, LLM rerank | 2/3 |

Lượt đầu hội thoại miss L1 thì vẫn ghi L2, nên L2 được nuôi bởi toàn bộ lưu lượng.

### Vì sao an toàn

- Cache lưu **ID chunk và điểm**, không lưu nội dung. Khi hit, nội dung được fetch lại
  từ Qdrant bằng `RetrievalService.fetch` **với filter quyền của người đang hỏi**.
  Tài liệu sửa → nội dung mới tự xuất hiện. Tài liệu xoá, re-chunk, đổi quyền, hoặc
  người hỏi không có quyền → thiếu chunk → coi là **miss**, xoá entry, chạy pipeline
  đầy đủ.
- Generation vẫn chạy với câu hỏi thật và chunk thật. Không có câu trả lời nào được
  trả lại nguyên văn từ cache.
- Khoá có scope phân quyền, tên collection đang dùng, identity model embedding, bộ
  đếm generation theo phòng ban và hash cấu hình. Một trong các thứ đó đổi là khoá
  đổi, không có chuyện hit vào kết quả của cấu hình hay dữ liệu cũ.
- Khớp chính xác: hai câu khác nhau một chữ là hai khoá. Rủi ro "gần nghĩa mà khác
  ý" của semantic cache (2,5–5 % hit sai ở ngưỡng 0,97–0,99 theo vCache) không tồn
  tại ở đây.

### User stories / acceptance

- Sinh viên A (khách) hỏi "Điều kiện tốt nghiệp là gì?" lúc 9:00; sinh viên B (khách)
  hỏi đúng câu đó lúc 9:30 → lượt của B hit L1: không gọi classifier, embedding,
  Qdrant, LLM rerank; câu trả lời có cùng tập chunk; log ghi `layer=l1 hit`.
- Sinh viên C thuộc phòng CNTT hỏi cùng câu → **không** dùng chung entry với A và B
  (scope khác), tự tạo entry riêng.
- Admin sửa tài liệu "Quy chế tốt nghiệp" (đổi nội dung hoặc đổi quyền) → trong vòng
  một thao tác ghi, mọi entry đang tham chiếu tài liệu đó bị xoá; lượt hỏi tiếp theo
  là miss và retrieval chạy lại.
- Admin ingest tài liệu mới thuộc phòng CNTT → entry của scope có CNTT miss; entry
  của khách (chỉ public) vẫn hit.
- Re-index sang collection mới và cutover → mọi entry cũ ngừng khớp mà không cần
  xoá tay.
- Redis sập → mọi lookup/ghi đều thành no-op có log cảnh báo, lượt chat vẫn trả lời
  bình thường, độ trễ thêm tối đa bằng socket timeout (≤ 50 ms).
- Follow-up "còn khóa 2025 thì sao?" → L1 bỏ qua (có lịch sử); classifier chạy, viết
  ra câu độc lập; nếu câu độc lập đó đã có trong L2 (cùng scope) → hit L2.

## Tech Stack

Python 3.12, FastAPI, Pydantic v2, `redis[asyncio]` (đã có, dùng cho web search cache),
Qdrant client, pytest + pytest-asyncio, ruff, mypy. Không thêm dependency mới.

## Commands

```
Test (toàn bộ):   task test            # = uv run pytest
Test (một file):  task test:file -- tests/test_retrieval_cache.py
Lint:             task code:lint       # = uv run ruff check app tests
Format:           task code:format     # = uv run ruff format app tests
Type check:       uv run mypy app tests
Dev server:       task be:dev env=eval
Worker ingest:    task be:worker:ingest env=eval -- --pool=solo -n ingest1@%h
Cache CLI (mới):  task db:retrieval-cache -- list|show|stats|clear
```

Nếu venv thiếu dev deps: `uv run --with pytest --with pytest-asyncio --with aiosqlite --with docker pytest ...`.

## Project Structure

```
app/integrations/retrieval_cache.py      → RetrievalCache protocol + RedisRetrievalCache
                                            (get/set/purge_document/bump_generation,
                                            normalize, cache_key, describe_key) - soi gương
                                            app/integrations/web_search_cache.py
app/schemas/retrieval_cache.py           → CachedTurnBundle (L1), CachedRetrievalBundle (L2),
                                            CachedChunkRef(chunk_id, score), KeyParts
app/graph/nodes/retrieval_cache.py       → hai hàm nhỏ cho graph: lookup_l2 / store_l2,
                                            lookup_l1 / store_l1; rebuild TurnRerankResult
                                            từ bundle + chunks đã fetch
app/graph/streaming_graph.py             → gọi các hàm trên tại 03 (L1) và 08 (L2)
app/rag/retrieval/service.py             → không đổi API; dùng fetch() có sẵn
app/services/document_index_service.py   → gọi purge_document sau update_index_metadata
                                            và delete_index
app/api/v1/documents.py                  → gọi purge_document sau delete_indexed_chunk
app/worker/tasks/ingestion.py            → sau _drop_stale_points: purge_document +
                                            bump_generation(scope của tài liệu)
app/core/config.py                       → RETRIEVAL_CACHE_REDIS_URL, _TTL_SECONDS, _TIMEOUT_MS,
                                            _KEY_VERSION, _WRITE_ON_SECOND_SIGHTING
app/tools/retrieval_cache.py             → CLI list/show/stats/clear (như web_search_cache.py)
taskfiles/db.yml                         → task db:retrieval-cache
.env.example / .env.eval.example         → biến mới + ghi chú DB số
docs/specs/SPEC-retrieval-cache.md       → spec này
docs/guide/cau-hinh.md                   → bảng biến môi trường
tests/test_retrieval_cache.py            → unit: normalize, key, serialize, get/set/purge/bump (fakeredis hoặc fake client)
tests/graph/test_retrieval_cache_node.py → graph: hit/miss/bypass/stale cho L1 và L2
tests/test_retrieval_cache_invalidation.py → hook: ingest/edit/delete/chunk-delete
```

## Thiết kế chi tiết

### Chuẩn hoá câu hỏi (`normalize_question`)

Thứ tự bắt buộc: Unicode **NFC** → `casefold()` → gộp mọi khoảng trắng thành một dấu
cách → `strip()` → bỏ dấu câu ở đầu/cuối (`?!.,;:…"'`). **Không** bỏ số, mã môn,
học kỳ, năm. Không bỏ dấu tiếng Việt. Áp dụng cho câu hỏi gốc (L1) và từng câu độc
lập (L2).

### Khoá

Khoá Redis = `rc:v1:{layer}:` + SHA-256 hex của JSON canonical (`sort_keys=True`,
`separators=(",", ":")`, `ensure_ascii=False`) của các trường sau. Thiếu bất kỳ
trường nào là lỗi thiết kế, không phải tối ưu.

| Trường | L1 | L2 | Giá trị | Lý do |
|---|---|---|---|---|
| `v` | ✓ | ✓ | `settings.RETRIEVAL_CACHE_KEY_VERSION` (str, mặc định `"1"`) | bump tay khi đổi cách dựng khoá/giá trị |
| `q` | ✓ | | câu hỏi gốc đã chuẩn hoá | đầu vào tất định duy nhất |
| `qs` | | ✓ | danh sách câu độc lập đã chuẩn hoá, **giữ thứ tự** | thứ tự sub-query ảnh hưởng merge |
| `role` | ✓ | | `security.role` | classifier đổi cách hỏi/ghi hồ sơ theo vai |
| `profile` | ✓ | | `canonical_profile(confirmed_metadata)` sau khi sort | `<declared_attributes>` đổi output classifier |
| `scope` | ✓ | ✓ | `sorted((d.department_id, d.access_level))` + `role == "KHACH"` | hit chia sẻ trong cùng tập quyền, không xuyên quyền |
| `coll` | ✓ | ✓ | `active_collection()` | cutover blue-green đổi khoá nguyên tử |
| `emb` | ✓ | ✓ | `embedder.identity_key` | vector khác model không so được |
| `gen` | ✓ | ✓ | `{ "public": <gen:public>, "<D>": <gen:dept:D> ... }` cho mọi D trong scope | ingest mới chỉ làm miss scope thấy nó |
| `cfg` | ✓ | ✓ | hash của: `CHAT_RETRIEVAL_MAX_CHUNKS`, `CHAT_CONTEXT_MAX_CHUNKS`, `CHAT_RERANK_SCORE_THRESHOLD`, `CHAT_LLM_RERANK_ENABLED`, `CHAT_LLM_RERANK_RESCUE_*`, SHA-256 của template `reranker_compressor`; L1 thêm template `message_classification`, `message_classification_retrieval`, `hyde_generator`, `multi_query_decomposer`, và `CHAT_CLASSIFY_WITH_RETRIEVAL` | đổi cấu hình/prompt không được hit kết quả cũ |
| `date` | ✓ | | ngày ICT `YYYY-MM-DD` | `<current_time>` đổi cách classifier hiểu "kỳ này" |

Mỗi entry có một khoá đồng hành `rc:v1:q:{digest}` lưu `describe_key(...)` dạng văn
bản (câu hỏi, scope, coll, gen...) cùng TTL, để CLI đọc lại được và để debug miss
("khoá khác ở phần nào"). Giống `query_key` của web search cache.

### Giá trị

Pydantic model, serialize JSON, có `schema_version: 1`.

**L2 `CachedRetrievalBundle`:**

```python
class CachedChunkRef(BaseModel):
    chunk_id: str          # "{document_id}:{chunk_index}" - payload chunk_id
    score: float

class CachedRetrievalBundle(BaseModel):
    schema_version: Literal[1] = 1
    created_at: datetime
    questions: list[str]                     # câu độc lập, thô (để debug)
    per_query: list[list[CachedChunkRef]]    # sau LLM rerank, theo thứ tự câu hỏi
    best_scores: list[float]                 # TurnRerankResult.best_scores
    merged: list[str]                        # chunk_id theo thứ tự TurnRerankResult.chunks
```

Rebuild: fetch mọi `chunk_id` duy nhất bằng `retrieval_service.fetch(ids, security=...)`;
nếu **thiếu bất kỳ id nào** → stale → `DEL` entry, tăng `stats:stale`, trả miss. Đủ thì
gán lại `score` từ bundle và dựng `TurnRerankResult` bằng `turn_result(per_query_kept,
best_scores=...)` rồi sắp `chunks` theo `merged`.

**L1 `CachedTurnBundle`:** mở rộng L2 với:

```python
    tasks: list[dict]        # ClassifiedTask.model_dump(exclude=False) từng task
    confidence: float | None
    sub_queries: list[tuple[str, str, int]]   # (question, retrieval_text, task_index)
```

Điều kiện **ghi** L1 (tất cả phải đúng): `history` rỗng, `resume is None`, mọi task
là `academic_advisory`, `classification.metadata_updates == {}`, LLM rerank chạy
thành công (không có `LLM_RERANK_FAILED`/`LLM_RERANK_UNAVAILABLE` warning),
`rerank_result.has_valid_context` đúng. Điều kiện **đọc** L1: `history` rỗng và
`resume is None`. Hit L1 → bỏ qua `classify_intent`, `transform_tasks`,
`retrieve_chunks`, `rerank_chunks`, `llm_rerank`; `trace.node("03_MessageClassificationNode",
cached=True)` và `trace.node("08_RetrievalFilteringNode", cached=True)`; web search và
generation chạy như thường với `tasks`/`sub_queries` từ bundle.

Điều kiện **ghi** L2: có ít nhất một chunk sau rerank, LLM rerank không lỗi. Không
cache kết quả rỗng, lỗi, hay lượt bị kill switch.

### Chính sách ghi: chỉ ghi khi thấy lần thứ hai

Để cache không đầy các câu chỉ hỏi một lần (nhất là câu độc lập của follow-up),
cả L1 và L2 chỉ ghi bundle khi **cùng một khoá** đã miss trước đó trong cửa sổ TTL:

1. Miss lần đầu: `SET rc:v1:seen:{layer}:{digest} 1 EX <TTL> NX`. Không ghi bundle.
   Log `outcome=miss seen=first`.
2. Miss lần sau mà marker còn: ghi bundle (và khoá đồng hành, reverse index), `DEL`
   marker. Log `outcome=miss seen=second`.
3. Lần sau nữa: hit.

Marker ~100 byte, hết hạn cùng TTL nên không tích luỹ. Hệ quả chấp nhận được: một câu
phải xuất hiện **3 lần** trong cửa sổ TTL mới có lần hit đầu tiên, thay vì 2; đổi lại
chỉ câu thực sự lặp mới chiếm chỗ. Bật/tắt bằng `RETRIEVAL_CACHE_WRITE_ON_SECOND_SIGHTING`
(mặc định `true`); tắt thì ghi ngay ở miss đầu. Bộ đếm `stats` tách `miss_first` và
`miss_second` để thấy tỷ lệ câu dùng một lần.

### TTL

`RETRIEVAL_CACHE_TTL_SECONDS` mặc định **21600** (6 giờ, chốt 20/10), jitter ±10 % khi
ghi, **tuyệt đối** (không gia hạn khi hit). Dấu `seen` cùng TTL; khoá đồng hành và
reverse index dùng TTL + 300 s; bộ đếm `stats` 30 ngày; bộ đếm generation không hết
hạn. TTL chỉ là lưới an toàn cuối: chống cũ chính là reverse index + generation +
fetch-by-id. Nới quá 6 giờ cần hỏi trước.

### Invalidation

| Sự kiện | Nơi móc | Hành động |
|---|---|---|
| Embed xong (ingest mới hoặc "Chunk & embed lại") | `embed_chunks` sau `_drop_stale_points` | `purge_document(document_id)` rồi `bump_generation("public" if is_public else department_id)` |
| Sửa quyền/phòng ban tài liệu | `document_index_service.update_index_metadata` | `purge_document` |
| Xoá tài liệu / đổi file | `document_index_service.delete_index` | `purge_document` |
| Xoá một chunk khỏi chỉ mục | `documents.delete_indexed_chunk` | `purge_document` |
| Đổi quyền của user | không cần | `scope` trong khoá đổi theo JWT |
| Cutover re-index | không cần | `coll` trong khoá đổi |
| Đổi model embedding | không cần | `emb` đổi (và `coll` đổi) |
| Đổi prompt/cấu hình | không cần | `cfg` đổi theo hash template/settings |
| Sự cố | kill switch `rc:v1:disabled` = "1" (CLI `clear --disable` / `--enable`), hoặc `clear` toàn bộ |

Reverse index: khi ghi entry, với mỗi `document_id` xuất hiện trong bundle:
`SADD rc:v1:doc:{document_id} {entry_key}` + `EXPIRE` = TTL + 300. `purge_document`
= `SMEMBERS` → `DEL` các entry + khoá đồng hành → `DEL` set. O(số entry chứa tài liệu).

Generation: `INCR rc:v1:gen:public` / `rc:v1:gen:dept:{D}`, không TTL. Khi đọc để
dựng khoá: `MGET` một lượt; khoá vắng = 0. **Không** bump generation cho sửa/xoá (đã có
reverse index + fetch-by-id), chỉ cho embed xong.

### Fail-open, stampede, bypass

- Mọi thao tác Redis bọc `try/except Exception` → log warning → coi như miss / no-op.
  `socket_timeout` và `socket_connect_timeout` = `RETRIEVAL_CACHE_TIMEOUT_MS` (mặc định
  50 ms). Một kết nối ngắn mỗi lần gọi như web search cache (lý do: event loop khác
  nhau giữa request và background).
- Ghi cache **không chặn** generation: `asyncio.create_task` + `_finish_in_background`
  như các job web search.
- Stampede (v1, tối thiểu): `SET rc:v1:lock:{digest} NX PX 10000` trước khi tính; không
  lấy được lock thì **vẫn tính** (không chờ) nhưng không ghi. Chờ có giới hạn là việc
  sau nếu đo thấy cần.
- Bypass: cache tắt (URL rỗng), kill switch bật, `history` không rỗng (chỉ L1),
  `resume` có (L1 và L2), task không phải `academic_advisory`, eval mode mặc định.

### Quan sát

- Log mỗi lookup: `retrieval_cache layer=l1|l2 outcome=hit|miss_first|miss_second|stale|bypass|error
  key=<digest> ms=<n>`; miss kèm `describe_key` ở mức DEBUG.
- Bộ đếm trong Redis: `rc:v1:stats:{YYYY-MM-DD}:{layer}:{outcome}` (`INCR`, TTL 30 ngày)
  để `task db:retrieval-cache -- stats` in hit rate theo ngày và theo tầng. Đây là nguồn
  số liệu cho tiêu chí thành công.
- `trace.node(..., cached=True)` để `rag-response-trace` thấy lượt nào dùng cache.
- CLI `stats` in thêm số entry và dung lượng ước tính theo prefix (`SCAN` + `MEMORY USAGE`
  mẫu).

## Code Style

Soi gương `app/integrations/web_search_cache.py`: Protocol + một lớp Redis, hàm thuần
cho key/normalize để test không cần Redis, docstring nói *vì sao*, log ở mức warning
khi fail-open. Ví dụ mẫu:

```python
class RetrievalCache(Protocol):
    async def get_l2(self, parts: L2KeyParts) -> CachedRetrievalBundle | None: ...
    async def set_l2(self, parts: L2KeyParts, bundle: CachedRetrievalBundle) -> None: ...
    async def purge_document(self, document_id: str) -> int: ...
    async def bump_generation(self, scope: str) -> None: ...


def cache_key(layer: Literal["l1", "l2"], parts: Mapping[str, object]) -> str:
    """`rc:v1:{layer}:{sha256}` of the canonical JSON of `parts` - every field that
    can change the result is in `parts`; see SPEC-retrieval-cache.md "Khoá"."""

    digest = hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return f"{KEY_PREFIX}{layer}:{digest}"
```

Quy ước: type hint đầy đủ, `ruff` sạch, `mypy` sạch, tên biến tiếng Anh, docstring và
comment tiếng Anh như phần còn lại của `app/` (tài liệu `docs/`, `changes/` tiếng Việt).

## Testing Strategy

- **Unit (`tests/test_retrieval_cache.py`)**: chính sách lần-thấy-thứ-hai (miss đầu
  chỉ đặt marker, miss thứ hai ghi bundle và xoá marker, marker hết hạn thì đếm lại
  từ đầu, tắt cờ thì ghi ngay); `normalize_question` (NFC vs NFD, casefold,
  khoảng trắng, dấu câu, giữ số); `cache_key` ổn định và nhạy với từng trường (đổi
  scope/coll/gen/cfg/date → khoá khác); serialize/deserialize bundle; `get/set/purge/bump`
  với một fake client in-memory tự viết (repo không có `fakeredis`; cùng tinh thần
  với `tests/core/test_registry_subscriber.py`); fail-open khi client ném lỗi.
- **Graph (`tests/graph/test_retrieval_cache_node.py`)**: dùng pattern mock LLM của
  `tests/llm_mocks.py`. Ca: L2 hit bỏ qua embed/search/LLM rerank; L2 stale (fetch
  thiếu id) → miss + DEL; L1 hit bỏ qua classifier; L1 bypass khi có history, khi
  có `metadata_updates`, khi intent khác; không ghi khi rerank rỗng hoặc LLM rerank
  lỗi; kill switch.
- **Invalidation (`tests/test_retrieval_cache_invalidation.py`)**: embed xong →
  purge + bump đúng scope; `update_index_metadata`/`delete_index`/`delete_indexed_chunk`
  → purge; entry của scope khác không bị đụng.
- **Hiện có**: toàn bộ `tests/graph`, `tests/test_ingestion_embedding.py`,
  `tests/test_document_index.py` phải xanh với cache tắt (mặc định trong test).
- Coverage: giữ `task test:cov:check` ≥ 80 %.

## Boundaries

- **Always:** fetch lại chunk qua `RetrievalService.fetch` có `build_access_filter`;
  chỉ lưu `chunk_id` + `score`; mọi thao tác Redis fail-open; chạy `task test`,
  `task code:lint`, `task code:format`, `uv run mypy app tests` trước khi commit;
  cache tắt trong test và eval mặc định.
- **Ask first:** thêm dependency (kể cả `fakeredis`); đổi TTL mặc định > 6 giờ; đổi
  Redis instance/DB số; thêm chờ-lock cho stampede; thêm re-warm sau cutover; đụng vào
  backend-java hay web.
- **Never:** semantic/approximate match; cache câu trả lời cuối; cache lượt
  `resume`/clarification, `academic_calculation` hay intent cá nhân; lưu nội dung
  chunk hay payload; bỏ qua filter quyền khi hit; bump generation toàn cục cho mỗi
  sửa/xoá; cache kết quả rỗng/lỗi; commit `.env`.

## Success Criteria

1. Với cache tắt: mọi test hiện có xanh, không đổi hành vi, không đổi response.
2. L2 hit: không có lời gọi embedding, Qdrant search, LLM rerank trong lượt đó; bộ chunk
   trả cho generation **bằng** bộ chunk của lượt gốc (so bằng chunk_id và thứ tự).
3. L1 hit: thêm không có lời gọi classifier/HyDE; `tasks`/`sub_queries` giống lượt gốc.
4. Stale: xoá một chunk của tài liệu đã cache → lượt kế tiếp là miss và không trả chunk
   đã xoá; sửa quyền tài liệu → entry bị purge trong cùng request sửa.
5. Scope: hai user khác `department_access` không bao giờ dùng chung entry (test với
   canary chunk chỉ một phòng thấy).
6. Fail-open: Redis không kết nối được → lượt chat vẫn thành công, log một warning,
   thời gian thêm ≤ 2 × `RETRIEVAL_CACHE_TIMEOUT_MS`.
7. Vận hành (sau deploy, đo bằng `stats`): hit rate L1 trên lượt đầu học vụ ≥ 20 %
   trong 2 tuần; `stale` = 0; p95 lookup < 50 ms. Hit rate tính trên mọi lookup, kể cả
   hai lần miss đầu của chính sách lần-thấy-thứ-hai.

## Quyết định đã chốt (20/10/2026, duyệt spec)

1. TTL khởi điểm 6 giờ.
2. CLI `task db:retrieval-cache` (`list|show|stats|clear`) nằm trong v1 vì `stats` là
   cách đo tiêu chí thành công.
3. Giữ `profile` trong khoá L1: an toàn trước, đo sau.
4. `scope` luôn nằm trong khoá: khách và sinh viên có phòng ban không dùng chung entry,
   vì hai filter cho hai tập chunk khác nhau (chất lượng, không chỉ bảo mật).
5. L2 phụ thuộc cách viết của classifier nên hit rate không hứa trước; plan có task đo
   độ ổn định câu độc lập (chạy bộ eval 3 lần) trước khi code L2, dưới 50 % thì cắt L2
   khỏi v1.

## Open Questions

Không còn. Câu hỏi mới phát sinh trong lúc làm ghi vào `changes/20-10-2026-RetrievalCache/plan.md`.

## Known gaps (ghi 21/10/2026, sẽ làm lại sau)

### Semantic cache cho L1

**Khoảng trống:** L1 chỉ hit khi câu hỏi gốc trùng *nguyên văn* sau chuẩn hoá. Hai sinh viên hỏi
"học phí ngành CNTT bao nhiêu" và "ngành cntt học phí nhiêu v" là hai khoá khác nhau. Đo 20/10
trên câu kiểu sinh viên gõ (không dấu, viết tắt): 0/49 câu cho cùng câu độc lập cả 3 lần, nên cả
L1 lẫn L2 đều khó hit với đầu vào kiểu này. Hit rate thật của L1 chưa đo được vì chưa có traffic.

**Chưa làm vì:** (1) chưa có số đo hit rate exact-match để biết có cần không; (2) rủi ro trả sai
cho câu "gần giống nhưng khác ý" (2,5–5 % hit sai ở ngưỡng 0,97–0,99 theo vCache) trong khi một
câu trả lời sai về học vụ đắt hơn nhiều 5 giây tiết kiệm; (3) cần thêm một lượt embedding
(~400 ms, có phí) cho mỗi miss để tìm láng giềng, và RAM Redis (6 KB/vector) trên instance 256 MB
dùng chung; (4) Redis hiện tại không có vector search, phải thêm Redis Stack / RediSearch hoặc
dùng chính Qdrant làm nơi lưu vector câu hỏi.

**Cổng mở lại:** sau 1–2 tuần log thật, `task db:retrieval-cache -- stats` cho hit rate L1 trên lượt
đầu hội thoại **< 20 %** (tiêu chí 1 của spec không đạt) và `stale = 0`.

**Phác thảo khi làm:**
- Vẫn giữ exact-match làm lớp đầu; semantic chỉ chạy khi L1 miss.
- Vector của câu hỏi gốc sau `normalize_question` (không phải HyDE, không phải câu độc lập) để
  khoá độc lập với LLM; lưu vào một collection Qdrant riêng `question_cache` với payload = các
  thành phần khoá hiện có (scope, coll, emb, gen, cfg, date) làm filter cứng, chỉ so cosine trong
  cùng filter.
- Ngưỡng cao và bảo thủ (bắt đầu 0,97), chỉ chấp nhận khi top-1 vượt ngưỡng và top-2 cách
  top-1 một khoảng rõ; đo tỷ lệ "hit sai" bằng cách so câu trả lời của hit với câu trả lời
  chạy thật trên bộ eval trước khi bật.
- Ghi vẫn theo chính sách thấy-lần-thứ-hai; TTL, invalidation, reverse index, kill switch
  dùng lại nguyên của L1 (entry semantic trỏ vào cùng bundle chunk_id + điểm).
- Loại trừ ngay từ đầu: câu có số/mã (mã học phần, số quyết định, năm, K18), vì khác một chữ số
  là khác câu; câu có intent tính toán; câu có từ phủ định khác nhau ("không", "chưa").
- Đo trước/sau bằng `evals.run` với bộ `question_user` (câu không dấu) để thấy hit tăng mà
  recall/leak không đổi.
