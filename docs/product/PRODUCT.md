# UniSage Agent — luật của sản phẩm

**Service này làm gì:** nhận câu hỏi học vụ của một người dùng (sinh viên, giảng viên, cán bộ, hoặc
khách vãng lai), quyết luồng xử lý phù hợp (chào hỏi/xã giao/từ chối ngoài phạm vi/tra cứu quy chế),
truy xuất đúng văn bản người đó được phép đọc từ kho tri thức đã ingest, và trả lời có trích dẫn
nguồn — hoặc từ chối trả lời khi không tìm được văn bản đủ tin cậy. Đồng thời, service vận hành toàn
bộ đường ingest tài liệu (nộp file → xem trước → chia đoạn → nạp vào Qdrant).

**File này là gì:** những điều luôn đúng về sản phẩm — ai dùng nó, dữ liệu gồm những gì, cái gì quyết
cái gì, và cái gì cố ý không làm. Cùng với `DECISIONS.md`, đây là toàn bộ tài liệu sản phẩm.

**File này không phủ hết repo.** README.md, CONTEXT.md ở gốc repo đang mô tả một kiến trúc cũ
(pgvector, "provider-free fallback") không còn đúng — coi hai file đó là lịch sử, không phải nguồn sự
thật. Chỗ nào không ghi ở đây thì **code là nguồn sự thật**, cho tới khi có người ghi luật vào đây.

**Nguồn nghiệp vụ:** thiết kế graph gốc ở `D:\KLTN\RAG_Graph\KLTN` (ảnh flow_design, `nodes/`,
`prompt_template/`) — chuẩn cho cấu trúc luồng và cách lắp prompt, không phải chuẩn cho câu chữ prompt
đã có sẵn trong code. `AGENTS.md` — quy ước kiến trúc và coding. `docs/specs/` — spec ingest chi
tiết và các khoảng trống đã biết.

**Lý do đằng sau từng luật** nằm ở `DECISIONS.md`, kèm những hướng đã cân rồi loại.

## Actors

- **Người dùng cuối** — sinh viên, giảng viên, cán bộ phòng ban đã đăng nhập, hoặc **khách vãng lai**
  (`role = "KHACH"`, chưa đăng nhập). Không ai gọi thẳng vào `unisage-agent`: mọi request đi qua API
  Gateway, Gateway giải mã JWT rồi bơm 5 header đã xác thực (`X-User-Id`, `X-User-Role`,
  `X-User-Code`, `X-User-Department-Access`, `X-User-Permissions`) — service tin các header này, không
  tự giải mã JWT.
- **`unisage-backend` (Java)** — chủ sở hữu conversation/message, xác thực người dùng, phân quyền
  department/access_level. `unisage-agent` gọi ngược sang Java (`BackendJavaClient`) để tạo/cập nhật
  message, không bao giờ tự lưu nội dung hội thoại. Từ Model Registry (2026-09), Java cũng là chủ sở
  hữu duy nhất của cấu hình model LLM (`ChatModel`) — `unisage-agent` không có UI/API nào để tự đăng
  ký hay sửa model, chỉ đọc.
- **Super Admin (SA)** — đăng ký/sửa/xoá `ChatModel` (provider, key, base URL, vai trò CHAT/EMBEDDING/
  EXTRACTION) qua `unisage-web`, gọi `unisage-backend`. Không gọi thẳng `unisage-agent`.
- **Người vận hành/triển khai** — chỉnh biến môi trường, chạy migration, tạo lại Qdrant collection khi
  hình dạng payload đổi; từ Model Registry còn phải chạy lệnh bootstrap danh tính embedding
  (`register_embedding_index_identity`) trước khi bật registry cho collection đã có sẵn vector.
- **Người viết prompt** — chỉnh nội dung `prompt_templates/`, `known_metadata_fields.json`; đổi hành
  vi của một node LLM mà không đụng logic Python.
- **System (graph orchestrator)** — chạy graph 11 node theo đúng flow_design, tự quyết dừng ở đâu
  (fast path, off-topic, fallback, hay sinh câu trả lời); không ai can thiệp giữa chừng một lượt chat.
- **Model registry verifier (Celery Beat + worker, `unisage-agent`)** — định kỳ claim job verify
  credential mới do SA nhập, gọi thử provider, báo kết quả về Java. Không tự quyết trạng thái
  credential — chỉ Java chuyển trạng thái (xem `ChatModel` trong Objects).

## Objects

| Object | Owned by | States |
|---|---|---|
| Conversation / Message | `unisage-backend` | `unisage-agent` chỉ tạo/patch qua API, không giữ bản chính. Trạng thái một message: `PENDING` → `STREAMING` → `COMPLETED` \| `ERROR` |
| `AcademicSecurityContext` | Gateway (từ JWT) | `user_id`, `role`, `department_access: [{department_id, access_level}]`, `permissions` — dựng lại mỗi request từ header, không cache |
| `PendingClarification` | `unisage-agent` (Postgres, 1 dòng/conversation) | một vòng "cần hỏi thêm" đang mở: `missing_fields`, `options`, `retry_count`, `origin_node`, `original_query`. `None` = không có gì đang treo |
| `confirmed_metadata` | `unisage-agent` (Postgres, cùng bảng) | thuộc tính sinh viên **tự khai**, tích luỹ qua hội thoại (VD hệ đào tạo, khoá). Không bao giờ dùng để mở rộng quyền đọc tài liệu — chỉ để chọn đúng nhánh quy định khi trả lời |
| Document / Chunk (ingest) | `unisage-agent` (Postgres draft) + Qdrant (đã embed) | `DocumentProcessLog.current_step`: `PENDING` → `CHUNKED` → `EMBEDDING` → (kết thúc, dòng bị xoá); `DocumentChunk` là bản nháp trước khi client duyệt và gọi embed |
| Chunk point (Qdrant) | `unisage-agent` | payload gồm `department`, `access_level`, `is_public`, `embedding_identity_key`, 3 vector (`content`/`summary`/`questions`) + metadata cấu trúc (trang, heading, bảng) |
| `RetrievedChunk` | ranh giới retrieval → generation | điểm số đã chuẩn hoá `[0,1]`, nguồn, trang, `source_locator` |
| Citation | sinh ra ở `GenerationSynthesisNode` | chỉ dựng từ `RetrievedChunk` thật sự được LLM trích `[n]`, không bao giờ do LLM tự bịa nguồn |
| `ChatModel` (model registry) | `unisage-backend` | credential + cấu hình 1 model LLM, gắn 1 trong 3 `modelPurpose` (`CHAT`/`EMBEDDING`/`EXTRACTION`). Vòng đời: `PENDING → ACTIVE/INACTIVE/DISABLED`, độc lập với `isActive` (soft-delete). Chỉ row `is_active = true AND status = 'ACTIVE'` được dùng để gọi provider |
| `chat_model_verifications` (job verify) | `unisage-backend` | 1 job/lần SA tạo hoặc sửa credential ứng viên; `unisage-agent` claim (pull), verify, báo kết quả. 7 trạng thái, xem Glossary |
| `embedding_index_identity` | `unisage-backend` | danh tính (provider/model/dimension/fingerprint) của vector đang nằm trong 1 collection Qdrant; bất biến sau khi tạo (`INSERT ... ON CONFLICT DO NOTHING`, trigger chặn UPDATE/DELETE) |
| Model registry snapshot | dựng ở `unisage-backend`, cache ở `unisage-agent` (RAM mỗi worker) | toàn bộ `ChatModel` ACTIVE theo purpose + `version`; `unisage-agent` không tự lưu bản lâu dài, poll/reload qua Redis pub/sub |

## Source of truth

- **Nội dung hội thoại: bản sống lâu chỉ có một, ở `unisage-backend`.** `unisage-agent` không lưu
  message; nó gọi `POST/PATCH /messages` để Java giữ bản chính, và đọc lại lịch sử qua
  `GET /conversations/{id}/messages` khi cần dựng `<history_message>`.
- **Quyền đọc tài liệu: chỉ đến từ `AcademicSecurityContext.department_access`, không bao giờ từ
  `confirmed_metadata`.** Một chunk hiện ra khi `is_public == True`, hoặc khi `department` của chunk
  nằm trong `department_access` của người hỏi ở `access_level` đủ thấp (xem `build_access_filter`,
  `app/rag/vectorstore/qdrant_store.py`). Lời tự khai trong hội thoại ("em là học viên cao học") không
  bao giờ mở thêm một tài liệu nào — nó chỉ giúp LLM chọn đúng nhánh quy định để diễn giải.
- **`PendingClarification`/`confirmed_metadata`: `unisage-agent` giữ, `unisage-backend` không biết.**
  Một bảng riêng (`conversation_clarification_states`), khoá theo `conversation_id`. Đây là trạng
  thái đặc thù của luồng hỏi-lại-thuộc-tính, không phải một phần nội dung hội thoại Java cần biết.
- **Chunking draft: Postgres (`unisage-agent`) là nguồn cho đến khi client gọi embed.** Sau khi embed,
  Qdrant là nguồn cho nội dung đã index; Postgres draft không bị xoá ngay (dùng để đối chiếu
  `chunking_version` khi phát hiện chunk cũ theo sơ đồ lỗi thời).
- **File gốc: MinIO, dùng chung với `backend-java`.** `unisage-agent` không sở hữu bucket riêng.
- **Cấu hình model LLM (provider/key/base URL): chỉ `unisage-backend`, bảng `chat_models`.**
  `unisage-agent` không còn đọc `.env` (`OPENAI_API_KEY`/`OPENAI_MODEL`/...) làm nguồn credential khi
  registry bật — nó đọc snapshot từ Java qua API nội bộ `/api/v1/internal/model-registry/**`, cache
  trong RAM mỗi worker process, và tự đảo bản cache khi Java báo `version` mới qua Redis pub/sub. Java
  không bao giờ gọi provider LLM thay Python.
- **Danh tính của vector embedding đang nằm trong Qdrant: bảng `embedding_index_identity` ở
  `unisage-backend`, không phải cấu hình credential đang ACTIVE.** Vẫn đúng kể cả khi không có
  credential EMBEDDING nào ACTIVE. Mọi đường có thể đổi embedding model đang chạy (rotate, activate,
  swap) phải so khớp danh tính này trước khi áp dụng — lệch thì chặn, không bao giờ âm thầm đổi vector
  space.

## Business rules

- **Chào hỏi lượt đầu không tốn token LLM.** Regex bắt câu chào thuần tuý ở lượt đầu tiên
  (`conversation_history` rỗng), trả template tĩnh ngay. Câu chào từ lượt 2 trở đi đi qua phân loại
  intent bình thường (`social_chat`) để không bỏ sót câu hỏi thật đi kèm ("Chào bot, cho em hỏi...").
- **Một tin nhắn được phân vào đúng một trong các nhánh: xã giao, ngoài phạm vi (gồm cả kiến thức phổ
  thông), tư vấn học vụ (đơn hoặc đa truy vấn), tính toán, chào hỏi.** `academic_comparison` và
  `general_knowledge` không còn là nhãn riêng — so sánh 2+ thực thể đi vào tư vấn học vụ với
  `routing_mode = MULTI`; kiến thức phổ thông gộp vào "ngoài phạm vi".
- **Không tìm thấy văn bản đủ tin cậy thì từ chối trả lời, không đoán.** Ngưỡng điểm sau rerank quyết
  định — dưới ngưỡng thì đi thẳng vào TicketFallback, không bao giờ để LLM tự suy luận quy chế.
- **Quyền hạn không bao giờ nâng lên qua hội thoại.** Một người tự nhận vai trò cao hơn, hay một câu
  hỏi giả định ("nếu mình là giảng viên thì sao") đều bị từ chối, không đổi phạm vi tài liệu.
- **Không tra cứu hay suy đoán dữ liệu cá nhân của khách vãng lai** — hệ thống không có hồ sơ của họ.
  Không trả dữ liệu cá nhân (điểm, GPA, MSSV) của người khác, kể cả khi người hỏi tự nhận là giảng
  viên hay phụ huynh.
- **Không lộ cơ chế nội bộ trong câu trả lời** — không nhắc `rerank_score`, `access_level`,
  `department_access`, tên node/tool, thẻ XML nội bộ, hay việc một văn bản đã bị lọc vì phân quyền
  (số lượng bị lọc cũng là rò rỉ thông tin).
- **Mỗi khẳng định quy chế phải trích dẫn nguồn `[n]`, dựng từ chunk thật sự được truy xuất** — không
  bao giờ do LLM tự bịa tên văn bản.
- **Hỏi lại tối đa `CHAT_CLARIFICATION_MAX_RETRY` lần cho cùng một field** trước khi buộc trả lời theo
  hướng liệt kê mọi phương án thay vì hỏi tiếp vô hạn.
- **Chunk chưa gắn `department`/`access_level`/`is_public` (dữ liệu trước migration) bị từ chối theo
  mặc định** — không khớp bất kỳ nhánh nào trong bộ lọc phân quyền.
- **Mỗi chunk chỉ gán đúng một cách chunking (chiến lược do client chọn lúc preview/chunking)** —
  không có phương án dự phòng tự động thử chiến lược khác khi một chiến lược thất bại.
- **Draft chunk theo `chunking_version` cũ hơn hiện tại bị chặn embed**, phải chunk lại trước — tránh
  trộn hai hình dạng payload khác nhau trong cùng collection.
- **`unisage-agent` không tự xác thực JWT** — tin hoàn toàn 5 header Gateway đã bơm sẵn. Một request
  không mang đúng `X-Internal-Secret` bị từ chối ở tầng router, trước khi chạm route handler nào.
- **Mọi lỗi trả về mang một `code` ổn định**, mirror `ErrorCode.java` bên `unisage-backend` — client có
  thể dùng chung một bảng tra mã lỗi cho cả hai backend.
- **Embedding model không bao giờ tự động failover hay tự động đổi.** Failover tự động chỉ áp cho CHAT
  và EXTRACTION. Đổi embedding model (kể cả cùng số chiều) đưa vector vào không gian ngữ nghĩa khác —
  một lỗi âm thầm không có cách nào tự phục hồi. Tại một thời điểm chỉ đúng 1 credential EMBEDDING
  ACTIVE (ép bằng unique index DB); lỗi thì dừng ingest job và báo Super Admin, không tự chuyển sang
  model khác.
- **Endpoint `/api/v1/internal/model-registry/**` của Java không bao giờ trả secret ra ngoài đường
  nội bộ.** Xác thực bằng `X-Internal-Secret` (không qua JWT/RBAC); Gateway chặn mọi request từ ngoài
  tới path này ở tầng filter sớm nhất, trước cả xử lý JWT. Mọi response của namespace này mang
  `Cache-Control: no-store`.
- **`unisage-agent` không tự expose endpoint "nội bộ" nào cho registry.** Gateway gắn
  `X-Internal-Secret` cho mọi request `/api/v1/ai/**` đi qua nó, nên một endpoint "nội bộ" ở Python
  thực chất ai qua Gateway cũng gọi được — không phải hàng rào thật. Verify credential luôn đi theo
  chiều Python gọi Java (pull), không có chiều ngược lại.
- **Danh sách `llmProvider` được phép tạo (`openai`, `google`, cộng
  `SELF_HOSTED` cho server tương thích OpenAI) chỉ gồm provider đã chứng minh nhận được HTTP
  client đã pin SSRF của Python — không phải mọi provider mà `pydantic-ai` hỗ trợ.** `anthropic`
  chưa vào danh sách vì SDK của nó chỉ nhận `httpx2.AsyncClient`, khác hẳn client `httpx` đang
  dùng để pin; `xai` không có client HTTP nào để pin (SDK dùng gRPC); `deepseek` cố định sẵn base
  URL nên không khớp cách factory hiện tại truyền `base_url` theo credential. Java và Python giữ
  đúng cùng danh sách này — SA không thể tạo được một credential mà Python chắc chắn không dùng
  được. `groq` và `mistral` đã bị gỡ khỏi danh sách (quyết định sản phẩm, không phải lý do
  SSRF); model groq/mistral tạo từ trước bị chuyển sang INACTIVE và không được định tuyến nữa.

## Glossary

- **node** — một bước trong graph (VD `GreetingDetectionNode`, `QueryTransformationNode`). Không phải
  lớp `pydantic_graph.BaseNode`: graph hiện là một hàm async duy nhất (`run_graph`) rẽ nhánh bằng
  `if`/routing map thuần Python, đặt tên node chỉ để log trace và đối chiếu với thiết kế gốc.
- **HyDE (Hypothetical Document Embeddings)** — LLM sinh một đoạn văn bản giả định trả lời câu hỏi
  bằng văn phong hành chính, rồi dùng chính đoạn đó để tìm kiếm ngữ nghĩa — không tìm bằng câu hỏi thô.
- **`confirmed_metadata`** — thuộc tính sinh viên **tự khai** (hệ đào tạo, khoá...), khác hẳn
  `AcademicSecurityContext` (xác thực từ JWT). Chỉ dùng để chọn nhánh quy định, không bao giờ dùng
  làm điều kiện lọc tài liệu.
- **`pending_clarification`** — một vòng hỏi-lại đang mở khi LLM cần thêm thuộc tính mới trả lời tiếp
  được. Có hai nguồn: Type A (CalculationNode tự biết thiếu gì trước khi gọi generation) và Type B
  (GenerationSynthesisNode tự phát hiện lúc đọc văn bản, ghi vào output dưới dạng khối
  ```json ask_user_form```).
- **rerank** — bước chấm lại điểm liên quan giữa câu hỏi và từng chunk sau khi retrieval trả về. Hiện
  chỉ lọc ngưỡng trên điểm cosine có sẵn, chưa có cross-encoder thật (xem `known-gaps.md`).
- **`is_public`** — cờ trên một chunk (kế thừa từ `Document.isPublic` bên `unisage-backend`): `true`
  thì ai cũng đọc được, không phân biệt phòng ban hay `access_level`.
- **wildcard department (`*`)** — một entry `department_access` với `department_id = "*"` cấp
  `access_level` của nó cho **mọi** phòng ban, không chỉ một.
- **model registry snapshot** — toàn bộ `ChatModel` đang `ACTIVE` (theo `CHAT`/`EMBEDDING`/
  `EXTRACTION`), kèm key plaintext và `version`, do Java tổng hợp qua
  `GET /internal/model-registry/snapshot`. `unisage-agent` cache trong RAM, không ghi xuống đĩa/DB.
- **verification job** — 1 lần Python thử gọi provider bằng credential ứng viên SA vừa nhập, để Java
  quyết có promote (áp dụng) hay không. 7 trạng thái: `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`,
  `SUPERSEDED` (bị thay bởi thay đổi mới hơn trước khi verify xong), `CANCELLED`,
  `REINDEX_REQUIRED` (chỉ EMBEDDING — ứng viên dùng được nhưng đổi danh tính vector, cần re-index
  trước khi áp dụng).
- **staged credential rotation** — sửa credential của một `ChatModel` đang ACTIVE không ghi thẳng vào
  row; giá trị mới nằm ở verification job dưới dạng ứng viên, row cũ (và Python) vẫn dùng credential
  cũ cho tới khi ứng viên verify xong và được promote — tránh downtime khi SA đổi key.
- **embedding identity guard** — cơ chế chặn mọi đường có thể đổi embedding model đang chạy (rotate,
  activate, swap) nếu ứng viên không khớp danh tính (`embedding_index_identity`) của vector đang nằm
  trong Qdrant; khớp thì cho qua, lệch thì job kết thúc ở `REINDEX_REQUIRED`.

## Human decisions

- Ngưỡng `CHAT_RERANK_SCORE_THRESHOLD` — chưa có số đo chính thức, hiện để tạm ở `.env` máy dev
  (`0.4`, khác mặc định code `0.70`) — người vận hành/nghiệp vụ quyết định sau khi có bộ câu hỏi đo.
- Có làm hybrid search (BM25) và cross-encoder rerank thật hay không — đã cân nhắc và hoãn, xem
  `docs/specs/known-gaps.md`; mở lại khi có số đo cho thấy dense-only không đủ.
- Cách phân biệt "công khai" cho một tài liệu (`is_public`, không phải `access_level = 0`) — do người
  dùng chốt sau khi review thiết kế ban đầu, khớp với `Document.isPublic` đã có sẵn bên
  `unisage-backend`.
- Wildcard `department_id = "*"` có áp dụng cho pre-filter chat hay không — chốt: có, cùng nghĩa với
  `TrustedContext.granted_access_level` bên ingestion, để một token không mang hai nghĩa khác nhau ở
  hai endpoint.

## The bet

Với câu hỏi học vụ tiếng Việt, **HyDE + dense retrieval trên 3 biểu diễn (nội dung/tóm tắt/câu hỏi mẫu)
đủ chính xác mà không cần BM25 hay cross-encoder** — nếu sai (câu hỏi có mã văn bản/số hiệu bị trượt
nhiều, hoặc chunk đúng lấy được nhưng xếp sai thứ hạng), phải bổ sung hybrid search và rerank thật,
đổi lại chi phí vận hành (re-index, thêm hạ tầng rerank) tăng đáng kể.

## Not this product

- Không tự xác thực người dùng — luôn tin header Gateway đã xác thực sẵn.
- Không lưu nội dung hội thoại — đó là việc của `unisage-backend`.
- Không tự nâng quyền qua bất kỳ hình thức tự khai nào trong hội thoại.
- Không tính toán học vụ thật (GPA, học phí) — `CalculationNode` hiện chỉ là placeholder, trả thông
  báo "đang phát triển", chưa gọi Calculator Tool hay lấy dữ liệu điểm sinh viên.
- Không có kho lưu trữ hội thoại lâu dài riêng của mình.
- Không tạo phòng ban/quyền hạn — đọc từ `department_access` do Gateway bơm, không tự định nghĩa.
- Không tự đăng ký hay sửa cấu hình model LLM — đó là việc của Super Admin qua `unisage-backend`.
- Không tự động đổi embedding model dưới bất kỳ hình thức nào — luôn cần re-index có chủ đích, ngoài
  scope của registry.

## Open

- `OPEN — chủ sản phẩm`: ngưỡng rerank đúng nên là bao nhiêu, sau khi có bộ câu hỏi đo thật? — hiện
  chưa có default: chờ số đo.
- `OPEN — nghiệp vụ`: `CalculationNode` lấy dữ liệu điểm/tín chỉ/học phí từ đâu — gọi ngược
  `unisage-backend`, hay chỉ tính từ số liệu người dùng tự nhập trong hội thoại? — default hiện tại:
  placeholder, chưa quyết.
