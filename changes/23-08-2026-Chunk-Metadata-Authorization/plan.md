# Implementation Plan: Trusted Context + Permission Gate cho Ingestion (unisage-agent)

Phần việc thuộc module `python-trusted-context` trong capability map "Ingestion Authorization
(JWT-embedded trusted context)" đã duyệt. Phạm vi ban đầu chỉ gồm `department`/`access_level`; đã mở
rộng thêm permission gate (Phase 4 dưới đây) sau khi xác nhận **module `java-jwt-claims` đã implement
xong claim `permissions`** (`AuthServiceImpl.buildSession` giờ build `permissionNames` từ
`mapToPermissionInfo(user)`, truyền vào `jwtUtil.generateAccessToken(user, departmentAccessClaim,
permissionNames)`) — Gateway module (`gateway-claim-forwarding`) là bên forward claim này thành header.
**Cập nhật 2026-08-23**: đọc code thật xác nhận `api-gateway`'s `AuthenticationFilter`/`JwtValidator` đã
implement xong phần forward header (`X-User-Department-Access`, `X-User-Permissions`) — xem
`api-gateway/changes/23-08-2026-JwtClaimForwarding/plan.md` (Task 1-2 done, Task 3-4 — test — vẫn chưa
làm). Code Python ở đây không cần đợi Gateway xong mới viết được (test tự set header giả lập), nhưng giờ
thực tế cả 2 lớp trên đã có dữ liệu thật để nối end-to-end nếu muốn test thủ công qua Gateway thật.

## Overview

Đọc code thật (`app/api/deps.py`, `app/api/v1/ingestion.py`, `app/schemas/ingestion.py`,
`app/worker/celery_app.py`, `app/rag/vectorstore/qdrant_store.py`) xác nhận bug cụ thể, không phải suy
đoán:

- `EmbeddingRequest` **không có field `department_id`/`access_level`** — endpoint `POST
  /ingestion/embedding` lấy 2 giá trị này thẳng từ `TrustedContext` (tức context của **người gọi API**,
  đọc từ header `X-User-Department`/`X-User-Access-Level`), rồi stamp y nguyên lên **mọi chunk của bất kỳ
  document nào** người đó embed. Đây là lỗi khái niệm: "phòng ban/level của người thao tác" bị nhầm với
  "phòng ban/level tài liệu này nên được gắn" — hai thứ không nhất thiết giống nhau, và một khi
  `department_access` trở thành mảng nhiều phòng ban (theo thiết kế JWT đã chốt ở module
  `java-jwt-claims`), `TrustedContext.department: str` đơn lẻ hiện tại không còn đủ biểu diễn — không
  còn biết nên chọn phòng ban nào trong nhiều phòng ban người dùng có quyền.
- `access_level` hiện là **`str`** xuyên suốt (`TrustedContext.access_level`, `ChunkPoint.access_level`,
  payload Qdrant) — ví dụ giá trị thật đang thấy là `"STUDENT"` (chuỗi tên level), không phải số. Không
  so sánh `<=` được với `access_levels.level` (Integer) bên Java — cần cho việc lọc vector DB theo quyền
  ở pipeline retrieval sau này.
- `TrustedContext`/`get_trusted_context` đọc 2 header đơn `X-User-Department`/`X-User-Access-Level` —
  không khớp với thiết kế claim JWT mới (`department_access: [{department_id, access_level}]`, dạng mảng)
  đã chốt ở module `java-jwt-claims`. Gateway module (`gateway-claim-forwarding`, **đã implement phần
  forward header** — xem cập nhật đầu file) forward claim này thành 1 header JSON
  (`X-User-Department-Access`), không phải 2 header string đơn như code Python hiện tại đang đọc.

## Architecture Decisions

- **Client gửi `department_id` + `access_level` (số) trực tiếp trong `EmbeddingRequest`** — không suy ra
  từ context người gọi, không gọi ngược sang Java lấy `documents.min_access_level_id` (giữ đúng boundary
  "Python không gọi Java" đã chốt ở `SPEC-ingestion-resume.md`). Giống hệt cách Java tự làm với
  `documents.min_access_level_id` — client chọn, server validate theo quyền người gọi.
- **`access_level` đổi từ `str` sang `int` toàn hệ thống Python** (request, `TrustedContext`,
  `ChunkPoint`, payload Qdrant, tham số `embed_chunks`) — khớp kiểu `access_levels.level` bên Java. Python
  không có bảng catalog `access_levels` riêng (đúng nguyên tắc DB tách biệt của
  `SPEC-ingestion-resume.md`) nên không tự validate số này hợp lệ hay không — chỉ tin theo những gì JWT
  claim (đã ký bởi Java) xác nhận người gọi thực sự có.
- **`TrustedContext` đổi từ 2 field đơn sang 1 field mảng** `department_access:
  list[DepartmentAccessEntry]` (mỗi entry `{department_id: str, access_level: int}`), đọc từ 1 header
  mới `X-User-Department-Access` (JSON-encoded array) thay vì 2 header string cũ. Parse lỗi (JSON hỏng,
  thiếu field) → 400 rõ ràng, không phải 500.
- **Validate ở tầng handler, trước khi dispatch Celery**: `embed_document` phải kiểm tra
  `(request.department_id, request.access_level)` nằm trong `context.department_access` của người gọi
  (department khớp, và access_level yêu cầu ≤ access_level được cấp cho đúng department đó) — 403 nếu
  không, **trước khi** gọi `embed_chunks.delay(...)`. Không kiểm tra sau khi task đã chạy nền (tránh vừa
  tốn compute vừa phát hiện muộn).
- **Module `gateway-claim-forwarding` đã implement phần forward header (Task 1-2), test riêng (Task 3-4)
  của module đó vẫn chưa làm** — code Python ở đây vẫn viết/test độc lập được (tự set header trong test
  client), nhưng giờ có thể test thủ công end-to-end qua Gateway thật nếu muốn (không bắt buộc phải chờ
  Task 3-4 của Gateway xong, vì đó là test tự động, không phải hành vi runtime).
- **Permission gate (mở rộng — Java đã sẵn sàng)**: `TrustedContext` thêm field `permissions:
  list[str]`, đọc từ header mới `X-User-Permissions` (JSON array tên permission, ví dụ
  `["DOCUMENT_ALL", "DOCUMENT_CREATE", ...]` — khớp đúng shape Java vừa build). Theo quyết định đã chốt
  trước đó (ingest và CRUD tài liệu dùng chung 1 cụm quyền, không tách theo hành động): **cả 3 endpoint**
  `/ingestion/preview`, `/ingestion/chunking`, `/ingestion/embedding` đều gate như nhau — có
  `"DOCUMENT_ALL"` hoặc `"DOCUMENT_CREATE"` trong `context.permissions` mới được gọi, 403 nếu không.
  **Lưu ý quan trọng phát hiện khi đọc lại code**: hiện tại `preview_document`/`chunk_document` **không
  có dependency `get_trusted_context` nào cả** — chỉ `embed_document` mới có (dùng cho
  department/access_level, không phải để gate quyền). Nghĩa là thêm permission gate cho 2 endpoint này là
  hành vi mới hoàn toàn (trước giờ ai gọi được `verify_internal_secret` — tức đi qua đúng Gateway — là
  preview/chunk được, không kiểm tra quyền DOCUMENT_* gì cả). Đây đúng là lỗ hổng đã nêu ở đầu phiên làm
  việc ("hiện tại là ai cũng xem được tiến trình embed hết hã") — Phase 4 vá đúng lỗ hổng đó.
  Dùng chung 1 dependency helper (`require_document_permission`, đặt cạnh `get_trusted_context` trong
  `app/api/deps.py`) áp cho cả 3 endpoint để tránh lặp logic 3 lần.
- **Preview và chunking cũng nhận `department_id` (bắt buộc), validate theo membership — không cần
  access_level ceiling** (khác với embed, xem Task 3): người dùng phát hiện lỗ hổng — hiện tại chỉ embed
  mới kiểm tra department, nên user có `DOCUMENT_CREATE` nhưng chỉ được cấp `department_access` cho 1
  phòng ban vẫn preview/chunk được tài liệu của phòng ban khác, chỉ bị chặn ở bước cuối cùng (embed).
  Thêm `department_id: str` (bắt buộc) vào `PreviewRequest`/`ChunkingRequest`, validate
  `request.department_id` nằm trong danh sách department của `context.department_access` (chỉ kiểm tra
  **có mặt**, không so `access_level` — vì access_level là mức được *gán* cho nội dung, khái niệm đó chỉ
  có ý nghĩa ở bước embed, chưa tồn tại ở bước preview/chunk thuần đọc dữ liệu). Dùng chung 1 helper mới
  `require_department_membership` (khác `require_document_permission` — 1 cái check quyền, 1 cái check
  phạm vi department), đặt cùng `app/api/deps.py`.
- **Reset collection Qdrant thay vì để dữ liệu cũ tồn tại song song** — người dùng xác nhận: dữ liệu
  Qdrant hiện tại toàn là dữ liệu test, không cần giữ. Xoá hẳn collection (`client.delete_collection`)
  trước khi đổi kiểu `access_level` sang `int`, để `ensure_collection` tạo lại sạch — tránh payload cũ
  (`access_level` dạng chuỗi) tồn tại lẫn với payload mới (dạng số) trong cùng collection (Qdrant payload
  không ép kiểu ở tầng collection, dữ liệu cũ/mới lẫn lộn sẽ âm thầm gây lỗi so sánh `<=` phía retrieval
  sau này thay vì báo lỗi ngay).

## Task List

### Phase 1: Reset Qdrant collection + EmbeddingRequest nhận department_id/access_level thật
- [ ] Task 0: Xoá collection Qdrant hiện tại (`QDRANT_COLLECTION`, dữ liệu test — đã xác nhận không cần
      giữ) để tránh lẫn payload `access_level` dạng chuỗi cũ với dạng số mới
- [ ] Task 1: Thêm `department_id: str`, `access_level: int` (bắt buộc) vào `EmbeddingRequest`; đổi
      `access_level` từ `str` sang `int` trong `ChunkPoint`, payload Qdrant, tham số `embed_chunks`;
      `embed_document` truyền `request.department_id`/`request.access_level` xuống Celery thay vì
      `context.department`/`context.access_level`
- [ ] Task 1b: Thêm `department_id: str` (bắt buộc) vào `PreviewRequest`, `ChunkingRequest` — chỉ thêm
      field vào schema ở task này, chưa validate (validate ở Task 4c)

### Checkpoint: Phase 1
- [ ] Collection Qdrant rỗng, không còn point nào có `access_level` dạng chuỗi
- [ ] `ruff check app tests`, `mypy app tests` sạch
- [ ] `pytest tests/test_ingestion_schemas.py tests/test_embed_chunks_task.py tests/test_qdrant_store.py`
      xanh (sau khi cập nhật ở Task 5)

### Phase 2: TrustedContext đọc claim department_access dạng mảng
- [ ] Task 2: Đổi `TrustedContext`/`get_trusted_context` sang đọc header `X-User-Department-Access`
      (JSON array), field `department_access: list[DepartmentAccessEntry]`; bỏ 2 header cũ
      `X-User-Department`/`X-User-Access-Level`

### Checkpoint: Phase 2
- [ ] `ruff check`, `mypy` sạch
- [ ] `pytest tests/test_trusted_context.py` xanh (sau khi cập nhật ở Task 5)

### Phase 3: Validate request theo department_access trước khi embed
- [ ] Task 3: `embed_document` validate `(request.department_id, request.access_level)` khớp
      `context.department_access` — 403 (`ErrorCode`/exception phù hợp, theo convention lỗi hiện có của
      `unisage-agent`) nếu không thuộc phòng ban được cấp hoặc vượt access_level được cấp cho phòng ban
      đó

### Checkpoint: Phase 3
- [ ] `pytest tests/test_ingestion_embedding.py` xanh (thêm case: department không được cấp → 403; access
      level vượt trần → 403; hợp lệ → 202 như cũ)

### Phase 4: Permission gate + department membership cho preview/chunking/embedding
- [ ] Task 4a: Thêm `permissions: list[str]` vào `TrustedContext`, đọc header `X-User-Permissions` (JSON
      array); thêm 2 dependency helper trong `app/api/deps.py`: `require_document_permission` (check
      `DOCUMENT_ALL`/`DOCUMENT_CREATE`) và `require_department_membership` (check `department_id` nằm
      trong `context.department_access`, không so `access_level`)
- [ ] Task 4b: Áp `require_document_permission` cho cả 3 endpoint `/ingestion/preview`,
      `/ingestion/chunking`, `/ingestion/embedding` (2 endpoint đầu hiện chưa có `get_trusted_context`
      nào — đây là hành vi mới)
- [ ] Task 4c: Áp `require_department_membership(request.department_id, context)` cho
      `/ingestion/preview`, `/ingestion/chunking` (không áp cho `/ingestion/embedding` — đã có check đầy
      đủ hơn ở Task 3, gồm cả `access_level` ceiling, làm lại ở đây là dư thừa)

### Checkpoint: Phase 4
- [ ] `pytest tests/test_ingestion_preview.py tests/test_ingestion_chunking.py
      tests/test_ingestion_embedding.py` xanh (sau khi cập nhật ở Task 5) — thêm case thiếu quyền
      `DOCUMENT_*` → 403 cho cả 3 endpoint; case `department_id` không thuộc `department_access` → 403
      cho preview/chunking
- [ ] `ruff check`, `mypy` sạch

### Phase 5: Spec + test tổng hợp
- [ ] Task 5a: Cập nhật `SPEC-ingestion.md` — thêm Decision mới (theo đúng pattern Decision 7 cho
      `ChunkingRequest.document_id` trước đây) ghi lại: `EmbeddingRequest` gains `department_id`/
      `access_level`; `PreviewRequest`/`ChunkingRequest` gains `department_id`; `TrustedContext` đổi
      shape sang `department_access` mảng + thêm `permissions`; cả 3 endpoint ingest giờ đều gate theo
      `DOCUMENT_ALL`/`DOCUMENT_CREATE`, preview/chunking thêm gate theo department membership, embedding
      gate theo department + access_level ceiling
- [ ] Task 5b: Rà và cập nhật toàn bộ test bị ảnh hưởng
      (`test_trusted_context.py`, `test_embed_chunks_task.py`, `test_qdrant_store.py`,
      `test_ingestion_schemas.py`, `test_ingestion_embedding.py`, `test_ingestion_preview.py`,
      `test_ingestion_chunking.py`) theo contract mới

### Checkpoint: Complete
- [ ] `pytest` toàn bộ suite xanh
- [ ] `ruff check app tests`, `ruff format --check app tests`, `mypy app tests` sạch
- [ ] Test thủ công: gọi `/ingestion/embedding` với header `X-User-Department-Access` tự set (giả lập
      Gateway) — case hợp lệ ra 202 và payload Qdrant có `department`/`access_level` đúng số; case
      department/level không được cấp ra 403
- [ ] Test thủ công: gọi `/ingestion/preview`, `/ingestion/chunking` với `department_id` không thuộc
      `department_access` → 403
- [ ] Review với người dùng trước khi merge

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Test hiện có hardcode `access_level` dạng chuỗi (`"STUDENT"` v.v.) rải rác nhiều file | Medium | Task 5 quét đủ 5 file test đã xác nhận qua `grep`; chạy `pytest` toàn suite ở Checkpoint Complete để bắt sót |
| ~~`gateway-claim-forwarding` chưa xong nên header thật sự chưa ai gửi ngoài đời~~ — **đã giải quyết**: Gateway đã implement forward `X-User-Department-Access`/`X-User-Permissions` (đọc code xác nhận 2026-08-23) | — | Không còn rủi ro này; Gateway vẫn thiếu test tự động riêng (Task 3-4 của module đó) nhưng không chặn hành vi runtime |
| Xoá collection Qdrant (Task 0) xoá luôn dữ liệu embed thật nếu đã lỡ có (không chỉ dữ liệu test) | Low | Xác nhận với người dùng trước khi chạy Task 0 rằng collection hiện tại đúng là chỉ có dữ liệu test — không hoàn tác được sau khi xoá |
| Merge Phase 4 (permission gate) — preview/chunking đổi từ "không cần header" sang "bắt buộc `X-User-Permissions`, `X-User-Department-Access`" | Medium (giảm từ High) | Gateway đã forward đủ 2 header (đọc code xác nhận) nên không còn chặn cứng như trước; vẫn **chưa có test end-to-end thật chạy qua** (chỉ đọc code, chưa build/chạy được `api-gateway` qua device bridge — xem `api-gateway/changes/23-08-2026-JwtClaimForwarding/plan.md`) — nên test thủ công 1 lần qua Gateway thật (Checkpoint Complete của cả 2 plan) trước khi coi là xong hẳn |
| `PreviewRequest`/`ChunkingRequest` thêm `department_id` bắt buộc — client hiện tại (frontend) chưa chắc đã gửi field này ở 2 bước đó (trước giờ chỉ `EmbeddingRequest` cần) | Medium | Breaking change có chủ đích, giống hệt tiền lệ `ChunkingRequest.document_id` (Decision 7) — cần báo phía frontend cập nhật trước khi merge, ghi rõ trong Task 5a (SPEC) |
| `require_department_membership` chỉ check "có mặt trong `department_access`", không check `access_level` — nếu sau này có yêu cầu phân quyền preview/chunk theo level cụ thể (không chỉ theo department) thì thiết kế này không đủ | Low | Đã cân nhắc có chủ đích (xem Architecture Decisions — access_level là khái niệm gắn với nội dung được *embed*, chưa có ý nghĩa ở bước đọc/chunk thuần) — nếu nhu cầu thay đổi, mở lại thiết kế này chứ không âm thầm mở rộng |

## Open Questions

Không còn open question chặn tiến độ — `permissions` xác nhận chưa implement (để lại làm task riêng sau,
không mở rộng scope lần này); Qdrant collection xác nhận reset (Task 0).
