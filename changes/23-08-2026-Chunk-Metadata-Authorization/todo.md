# Task List: Bỏ hardcode department/access_level trong chunk metadata (unisage-agent)

Xem bối cảnh và quyết định kiến trúc đầy đủ tại `plan.md`.

---

## Task 0: Reset Qdrant collection

**Description:** Xoá collection Qdrant hiện tại (`settings.QDRANT_COLLECTION`) bằng
`client.delete_collection(...)` (script/lệnh chạy tay 1 lần, không phải code production) — dữ liệu hiện
có toàn là dữ liệu test, người dùng đã xác nhận không cần giữ. Sau khi xoá, lần chạy tiếp theo
`ensure_collection` sẽ tự tạo lại collection sạch, chưa có point nào — tránh lẫn payload `access_level`
dạng chuỗi cũ với dạng số mới trong cùng collection.

**Trạng thái:** Kiểm tra qua `client.collection_exists(settings.QDRANT_COLLECTION)` — trả `False`, tức
collection chưa từng được tạo trên instance Qdrant hiện kết nối được. Không có dữ liệu cũ (dạng chuỗi)
để xoá — task này coi như no-op, không cần chạy `delete_collection`.

**Acceptance criteria:**
- [x] Collection `settings.QDRANT_COLLECTION` không còn tồn tại (hoặc đã rỗng hoàn toàn) sau khi chạy
- [ ] Lần gọi `/ingestion/embedding` kế tiếp tự tạo lại collection qua `ensure_collection` như bình
      thường, không lỗi — chưa test thủ công qua request thật

**Verification:**
- [x] Manual check: `client.collection_exists(settings.QDRANT_COLLECTION)` trả `False`
- [ ] Gọi thử `/ingestion/embedding` sau đó — collection được tạo lại, point mới có `access_level` dạng
      số — chưa làm (cần OpenAI key thật/mock thủ công ngoài test suite)

**Dependencies:** None — nên làm trước Task 1 để tránh phải chạy lại 2 lần

**Files likely touched:**
- Không sửa file source nào — chạy script/lệnh 1 lần (ví dụ 1 dòng Python tương tác, hoặc thêm 1 script
  nhỏ tạm thời trong `scripts/` nếu project đã có thư mục đó, xoá sau khi dùng xong)

**Estimated scope:** XS (thao tác vận hành, không phải thay đổi code lâu dài)

---

## Task 1: EmbeddingRequest nhận department_id/access_level thật, đổi access_level sang số

**Description:** Thêm `department_id: str`, `access_level: int` (bắt buộc, có validate `ge=1`/`le=5` hoặc
tương đương nếu range đã biết) vào `EmbeddingRequest` trong `app/schemas/ingestion.py`. Đổi
`ChunkPoint.access_level` (`app/rag/vectorstore/qdrant_store.py`) từ `str` sang `int`; đổi payload Qdrant
trong `upsert_chunk` theo. Đổi tham số `department: str, access_level: str` của task Celery `embed_chunks`
(`app/worker/celery_app.py`) thành `department_id: str, access_level: int`. Sửa handler `embed_document`
(`app/api/v1/ingestion.py`) truyền `request.department_id`/`request.access_level` xuống
`embed_chunks.delay(...)` thay vì `context.department`/`context.access_level`.

**Acceptance criteria:**
- [x] `EmbeddingRequest` có 2 field mới, bắt buộc (thiếu → 422 validation error tự động của FastAPI)
- [x] `ChunkPoint.access_level` kiểu `int`, payload Qdrant `"access_level"` là số nguyên
- [x] `embed_chunks` task nhận `department_id`/`access_level` (int) làm tham số, không còn đọc từ context
      người gọi
- [x] Gọi `/ingestion/embedding` với `department_id`/`access_level` trong body → chunk trong Qdrant có
      đúng 2 giá trị đó (không phải giá trị của người gọi)

**Verification:**
- [x] `ruff check app tests`
- [x] `mypy app tests`
- [x] Test tự động: `pytest tests/test_ingestion_schemas.py tests/test_embed_chunks_task.py
      tests/test_qdrant_store.py` (sau khi cập nhật ở Task 5)

**Dependencies:** None

**Files likely touched:**
- `app/schemas/ingestion.py`
- `app/rag/vectorstore/qdrant_store.py`
- `app/worker/celery_app.py`
- `app/api/v1/ingestion.py`

**Estimated scope:** M (4 file)

---

## Task 1b: PreviewRequest/ChunkingRequest thêm department_id

**Description:** Thêm `department_id: str` (bắt buộc) vào `PreviewRequest` và `ChunkingRequest` trong
`app/schemas/ingestion.py`. Chỉ thêm field ở task này — chưa validate gì (không đọc `TrustedContext`,
không so sánh với `department_access`); validate thật diễn ra ở Task 4c, sau khi có đủ helper
(`require_department_membership`, Task 4a). Tách riêng khỏi Task 4c để mỗi task chỉ đổi 1 việc: đây là
thay đổi schema/contract thuần, Task 4c là thay đổi hành vi endpoint.

**Acceptance criteria:**
- [x] `PreviewRequest.department_id: str`, bắt buộc (thiếu → 422)
- [x] `ChunkingRequest.department_id: str`, bắt buộc (thiếu → 422) — cạnh `document_id` đã có sẵn từ
      Decision 7 trước đó, không thay thế
- [x] Chưa có logic validate nào dùng field này ở task này (đúng phạm vi, để Task 4c làm)

**Verification:**
- [x] `ruff check app tests`, `mypy app tests`
- [x] Test tự động: `pytest tests/test_ingestion_schemas.py` (sau khi cập nhật ở Task 5b) — case thiếu
      `department_id` ở cả 2 request → 422

**Dependencies:** None (độc lập, có thể làm song song Task 1)

**Files likely touched:**
- `app/schemas/ingestion.py`

**Estimated scope:** XS (1 file)

---

## Task 2: TrustedContext đọc claim department_access dạng mảng

**Description:** Đổi `TrustedContext` trong `app/api/deps.py`: bỏ field `department: str`/
`access_level: str`, thay bằng `department_access: list[DepartmentAccessEntry]` (dataclass/model mới,
mỗi entry `{department_id: str, access_level: int}`). `get_trusted_context` đọc header
`X-User-Department-Access` (JSON-encoded array) thay vì 2 header `X-User-Department`/
`X-User-Access-Level`; parse JSON lỗi hoặc thiếu field → raise lỗi rõ ràng (400), không phải 500 hay
`MissingTrustedContextException` generic như cũ (cân nhắc thêm exception riêng
`InvalidTrustedContextException` nếu JSON hỏng, khác với "thiếu hẳn header").

**Acceptance criteria:**
- [x] `TrustedContext.department_access: list[DepartmentAccessEntry]`, không còn field `department`/
      `access_level` đơn
- [x] Thiếu header `X-User-Department-Access` → lỗi rõ ràng (giữ hành vi cũ: fail loudly)
- [x] Header có JSON hỏng hoặc entry thiếu field → lỗi rõ ràng, không phải 500 không rõ nguyên nhân
- [x] Header hợp lệ (mảng 1 hoặc nhiều entry) → `TrustedContext.department_access` parse đúng

**Verification:**
- [x] `ruff check app tests`
- [x] `mypy app tests`
- [x] Test tự động: `pytest tests/test_trusted_context.py` (sau khi cập nhật ở Task 5) — case: thiếu
      header, JSON hỏng, entry thiếu field, hợp lệ 1 entry, hợp lệ nhiều entry

**Dependencies:** None (độc lập với Task 1, có thể làm song song)

**Files likely touched:**
- `app/api/deps.py`
- `app/core/exceptions.py` (nếu thêm exception mới cho JSON hỏng)

**Estimated scope:** S (1-2 file)

---

## Task 3: Validate request theo department_access trước khi embed

**Description:** Trong `embed_document` (`app/api/v1/ingestion.py`), trước khi gọi
`embed_chunks.delay(...)`, kiểm tra `request.department_id` có nằm trong
`context.department_access` (theo `department_id`) không, và nếu có, `request.access_level` có
`<=` `access_level` được cấp cho đúng department đó không. Không thoả điều kiện nào → raise lỗi 403 rõ
ràng (không phải 500), theo convention lỗi hiện có của `unisage-agent` (kiểm tra
`app/core/exceptions.py` để dùng đúng pattern sẵn có thay vì tự chế mới nếu đã có exception tương tự).

**Acceptance criteria:**
- [x] `department_id` không nằm trong `department_access` của người gọi → 403, không dispatch Celery task
- [x] `department_id` có trong `department_access` nhưng `access_level` yêu cầu vượt mức được cấp → 403,
      không dispatch Celery task
- [x] `department_id`/`access_level` hợp lệ (khớp và trong trần) → 202 như hành vi cũ, task được dispatch

**Verification:**
- [x] `ruff check app tests`
- [x] `mypy app tests`
- [x] Test tự động: `pytest tests/test_ingestion_embedding.py` (sau khi cập nhật ở Task 5) — 3 case ở
      Acceptance criteria

**Dependencies:** Task 1, Task 2 (cần cả field request mới lẫn shape context mới)

**Files likely touched:**
- `app/api/v1/ingestion.py`

**Estimated scope:** S (1 file, nhưng logic so sánh cần test kỹ 3 nhánh)

---

## Task 4a: TrustedContext thêm permissions + 2 dependency helper

**Description:** Thêm `permissions: list[str]` vào `TrustedContext` (`app/api/deps.py`), đọc từ header
mới `X-User-Permissions` (JSON array tên permission, ví dụ `["DOCUMENT_ALL", "DOCUMENT_CREATE"]` — khớp
đúng shape Java `AuthServiceImpl.buildSession` vừa build). Thêm 2 dependency helper cạnh
`get_trusted_context`: `require_document_permission` (raise 403 nếu `context.permissions` không chứa
`"DOCUMENT_ALL"` hoặc `"DOCUMENT_CREATE"`) và `require_department_membership(department_id, context)`
(raise 403 nếu `department_id` không nằm trong danh sách department của `context.department_access` —
chỉ check có mặt, không so `access_level`, khác Task 3's check đầy đủ hơn cho embed).

**Acceptance criteria:**
- [x] `TrustedContext.permissions: list[str]`, parse từ header `X-User-Permissions`
- [x] Thiếu header → lỗi rõ ràng (400, giống cách `department_access` đang fail loudly)
- [x] `require_document_permission` raise 403 khi thiếu cả 2 permission tên trên, không raise khi có ít
      nhất 1 trong 2
- [x] `require_department_membership` raise 403 khi `department_id` không có trong `department_access`
      của người gọi, không raise khi có (bất kể `access_level` bao nhiêu)

**Verification:**
- [x] `ruff check app tests`, `mypy app tests`
- [x] Test tự động: `pytest tests/test_trusted_context.py` (sau khi cập nhật ở Task 5b) — case thiếu
      header, có `DOCUMENT_ALL`, có `DOCUMENT_CREATE`, có permission khác không liên quan (vẫn 403);
      `department_id` có/không có trong `department_access`

**Dependencies:** Task 2 (cùng sửa `app/api/deps.py`, làm nối tiếp cho gọn thay vì conflict)

**Files likely touched:**
- `app/api/deps.py`

**Estimated scope:** S (1 file, 2 helper nhỏ)

---

## Task 4b: Áp permission gate cho preview/chunking/embedding

**Description:** Thêm dependency `require_document_permission` vào cả 3 endpoint:
`preview_document`, `chunk_document` (hiện **chưa có bất kỳ trusted-context dependency nào** — đây là
hành vi mới, vá đúng lỗ hổng "ai gọi qua Gateway cũng preview/chunk được, không kiểm tra quyền
DOCUMENT_*"), và `embed_document` (đã có `get_trusted_context` cho department/access_level — thêm
gate quyền bên cạnh, không thay thế).

**Acceptance criteria:**
- [x] Gọi `/ingestion/preview`, `/ingestion/chunking`, `/ingestion/embedding` thiếu
      `DOCUMENT_ALL`/`DOCUMENT_CREATE` trong `permissions` → 403 cho cả 3
- [x] Gọi đủ quyền → hành vi như cũ (200/202), không đổi response shape

**Verification:**
- [x] `ruff check app tests`, `mypy app tests`
- [x] Test tự động: `pytest tests/test_ingestion_preview.py tests/test_ingestion_chunking.py
      tests/test_ingestion_embedding.py` (sau khi cập nhật ở Task 5b) — case 403 mới cho cả 3 endpoint

**Dependencies:** Task 4a

**Files likely touched:**
- `app/api/v1/ingestion.py`

**Estimated scope:** S (1 file, nhưng đụng 3 endpoint — review kỹ vì đổi hành vi truy cập thật)

---

## Task 4c: Áp department membership check cho preview/chunking

**Description:** Thêm `require_department_membership(request.department_id, context)` vào
`preview_document` và `chunk_document` — validate `department_id` client gửi (Task 1b) nằm trong
`context.department_access` (Task 4a) trước khi cho phép preview/chunk. **Không** áp cho
`embed_document` — endpoint đó đã có check đầy đủ hơn ở Task 3 (gồm cả `access_level` ceiling), thêm lần
nữa ở đây là dư thừa, không sai nhưng không cần thiết. Vá đúng lỗ hổng người dùng phát hiện: trước Task
này, user có quyền `DOCUMENT_*` nhưng chỉ được cấp `department_access` cho 1 phòng ban vẫn preview/chunk
được tài liệu của phòng ban khác — giờ bị chặn ngay từ bước preview.

**Acceptance criteria:**
- [x] Gọi `/ingestion/preview`, `/ingestion/chunking` với `department_id` không thuộc
      `context.department_access` của người gọi → 403, không đọc/không chunk gì cả
- [x] Gọi với `department_id` hợp lệ → hành vi như cũ (200), không đổi response shape
- [x] `/ingestion/embedding` không bị áp thêm check này (giữ nguyên chỉ dùng Task 3)

**Verification:**
- [x] `ruff check app tests`, `mypy app tests`
- [x] Test tự động: `pytest tests/test_ingestion_preview.py tests/test_ingestion_chunking.py` (sau khi
      cập nhật ở Task 5b) — case `department_id` không thuộc quyền → 403; case hợp lệ → 200 như cũ

**Dependencies:** Task 1b, Task 4a

**Files likely touched:**
- `app/api/v1/ingestion.py`

**Estimated scope:** S (1 file, 2 endpoint)

---

## Task 5a: Cập nhật SPEC-ingestion.md

**Description:** Thêm Decision mới vào `docs/specs/SPEC-ingestion.md`, theo đúng pattern đã dùng cho
Decision 7 (`ChunkingRequest.document_id`) — ghi rõ: (a) `EmbeddingRequest` gains `department_id: str`,
`access_level: int` (breaking change, liệt kê các file cần cập nhật giống Decision 7 đã làm); (b)
`TrustedContext` đổi shape từ 2 field string đơn sang `department_access: list[...]`, đọc từ header mới
`X-User-Department-Access` thay vì `X-User-Department`/`X-User-Access-Level`; (c) endpoint
`/ingestion/embedding` giờ validate `department_id`/`access_level` trong request theo
`context.department_access` trước khi dispatch — 403 nếu không hợp lệ; (d) cả 3 endpoint ingest
(`preview`/`chunking`/`embedding`) giờ gate theo permission `DOCUMENT_ALL`/`DOCUMENT_CREATE` qua header
`X-User-Permissions`; (e) `PreviewRequest`/`ChunkingRequest` gains `department_id` bắt buộc, 2 endpoint
này giờ validate department membership (không check `access_level`) trước khi cho phép preview/chunk.

**Acceptance criteria:**
- [x] `SPEC-ingestion.md` có Decision mới mô tả đầy đủ 5 thay đổi trên, đúng format các Decision trước đó
- [x] Phần mô tả `TrustedContext`/header cũ trong spec (nếu có, ví dụ Decision 3) được cập nhật/tham chiếu
      chéo tới Decision mới, không để 2 mô tả mâu thuẫn nhau trong cùng 1 file
- [x] Ghi rõ trạng thái sequencing hiện tại: Gateway (`gateway-claim-forwarding`) đã forward 2 header cần
      thiết (xác nhận qua đọc code 2026-08-23), nên permission gate chạy được thật qua Gateway — chỉ còn
      thiếu test tự động phía Gateway (không chặn runtime), xem Risks trong `plan.md`

**Verification:**
- [x] Manual check: đọc lại toàn bộ `SPEC-ingestion.md` sau khi sửa, xác nhận không còn đoạn nào mô tả
      hành vi cũ (2 header đơn, `EmbeddingRequest` không có `department_id`, preview/chunking không gate
      quyền) như thể vẫn đang là hiện tại

**Dependencies:** Task 1, Task 1b, Task 2, Task 3, Task 4a, Task 4b, Task 4c (spec ghi lại quyết định đã
implement, không viết trước)

**Files likely touched:**
- `docs/specs/SPEC-ingestion.md`

**Estimated scope:** S (1 file, nội dung nhiều hơn bản trước do gộp thêm phần permission)

---

## Task 5b: Cập nhật test bị ảnh hưởng

**Description:** Rà và cập nhật 7 file test đã xác nhận bị ảnh hưởng, theo contract mới của Task
1-4c: `test_trusted_context.py` (shape `TrustedContext` mới — cả `department_access` lẫn `permissions`,
header mới, 2 helper mới), `test_embed_chunks_task.py` (tham số `embed_chunks` mới, `access_level` là
số), `test_qdrant_store.py` (`ChunkPoint.access_level` kiểu `int`), `test_ingestion_schemas.py`
(`EmbeddingRequest`/`PreviewRequest`/`ChunkingRequest` có field mới, validate bắt buộc),
`test_ingestion_embedding.py` (case 403 mới từ Task 3 và Task 4b, field mới trong request body test),
`test_ingestion_preview.py`, `test_ingestion_chunking.py` (case 403 mới từ Task 4b lẫn Task 4c — 2 file
test này trước giờ không cần set header gì, giờ phải set `X-User-Permissions` +
`X-User-Department-Access` và gửi `department_id` trong body mới gọi được).

**Acceptance criteria:**
- [x] Cả 7 file test cập nhật xong, không còn assert theo contract cũ (string access_level, 2 header đơn,
      `EmbeddingRequest`/`PreviewRequest`/`ChunkingRequest` thiếu field mới, preview/chunking không cần
      permission/department header)
- [x] `pytest` toàn bộ suite (không chỉ 7 file trên) xanh — bắt các chỗ phụ thuộc gián tiếp có thể đã bỏ
      sót

**Verification:**
- [x] `pytest` — toàn bộ suite
- [x] `ruff check app tests`, `ruff format --check app tests`, `mypy app tests`

**Dependencies:** Task 1, Task 1b, Task 2, Task 3, Task 4a, Task 4b, Task 4c

**Files likely touched:**
- `tests/test_trusted_context.py`
- `tests/test_embed_chunks_task.py`
- `tests/test_qdrant_store.py`
- `tests/test_ingestion_schemas.py`
- `tests/test_ingestion_embedding.py`
- `tests/test_ingestion_preview.py`
- `tests/test_ingestion_chunking.py`

**Estimated scope:** M (7 file, chủ yếu cập nhật fixture/assert — vượt nhẹ khuyến nghị ~5 file/task vì
đều là thay đổi cơ học cùng loại, không tách thêm sẽ hợp lý hơn là tách vụn)

---

## Checkpoint: Complete

- [x] `pytest` toàn bộ suite xanh
- [x] `ruff check app tests`, `ruff format --check app tests`, `mypy app tests` sạch
- [x] Test thủ công: gọi `/ingestion/embedding` với header `X-User-Department-Access` +
      `X-User-Permissions` tự set (giả lập Gateway) — case hợp lệ ra 202 và payload Qdrant có
      `department`/`access_level` đúng số; case department/level không được cấp ra 403; case thiếu
      permission ra 403 — covered qua `pytest tests/test_ingestion_embedding.py` (TestClient tự set
      header, không phải gọi tay qua HTTP client thật, nhưng cùng kịch bản)
- [x] Test thủ công: gọi `/ingestion/preview`, `/ingestion/chunking` thiếu `X-User-Permissions` → 403
      (hành vi mới) — covered qua `pytest tests/test_ingestion_preview.py`/`test_ingestion_chunking.py`
- [x] Test thủ công: gọi `/ingestion/preview`, `/ingestion/chunking` với `department_id` không thuộc
      `department_access` của người gọi → 403 (hành vi mới, Task 4c) — covered qua cùng 2 file test trên
- [x] `SPEC-ingestion.md` phản ánh đúng thực tế sau khi implement
- [ ] Test thủ công qua Gateway thật ít nhất 1 lần (không chỉ giả lập header) trước khi coi Phase 4 là
      xong hẳn — Gateway đã forward đúng header (đọc code xác nhận) nhưng chưa có lần chạy end-to-end
      thật nào được xác minh
- [ ] Review với người dùng trước khi merge
