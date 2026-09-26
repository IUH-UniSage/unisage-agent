# Vì sao sản phẩm lại thế này

Luật của sản phẩm nằm ở `PRODUCT.md`. File này trả lời câu **"vì sao lại làm thế"** cho từng luật,
kèm những hướng đã cân rồi loại. Mục đích: người vào sau không mở lại một cuộc bàn đã xong, và **mở
lại được** khi lý do cũ hết đúng — mỗi mục đều nói rõ nó dựa trên điều gì.

Hai hệ được nhắc suốt file này: **service** là repo hiện tại (`unisage-agent`), **`unisage-backend`**
là hệ Java giữ người dùng, quyền hạn, và toàn bộ nội dung hội thoại.

## Ranh giới với `unisage-backend`

### Vì sao `unisage-agent` không tự lưu nội dung hội thoại?

`unisage-backend` đã có bảng `conversations`/`messages` cùng cơ chế sở hữu (ai được đọc conversation
nào). Nhân đôi bảng đó ở Python nghĩa là hai nơi cùng giữ một sự thật, và phải đồng bộ mỗi lần đổi.
`BackendJavaClient` chỉ forward `Authorization` gốc của người gọi sang Java để Java tự xác thực lại
và tự chấm quyền sở hữu — Python không cấp thêm bằng chứng nào cho việc đó ngoài
`X-Internal-Secret` (xác nhận chính `unisage-agent` là bên gọi, không phải xác nhận người dùng cuối).

Điều này kéo theo: guest (`KHACH`) tạo conversation qua một `X-Guest-Session-Token` riêng, vì không có
JWT nào để forward — Java cần một cách khác để chấm quyền sở hữu cho phiên khách.

Luật nằm ở: `PRODUCT.md` › Source of truth, Actors.

### Vì sao `pending_clarification`/`confirmed_metadata` có bảng riêng ở Postgres của `unisage-agent`, không gửi sang Java?

Đây là trạng thái xử lý nội bộ của luồng hỏi-lại-thuộc-tính, không phải nội dung hội thoại — Java
không cần biết một field còn đang "chờ người dùng trả lời" để làm bất cứ việc gì của nó.
`ConversationClarificationState` khoá theo `conversation_id` **không có FK** vào bảng
`conversations` của Java (hai schema tách biệt, không FK xuyên service), cùng mẫu với
`DocumentProcessLog` (theo dõi tiến độ ingest) — Python giữ trạng thái quy trình của chính nó trong
schema của chính nó.

Đã cân và loại: gửi `pending_clarification` như một field mở rộng của message bên Java (đẩy logic xử
lý sang bên không sở hữu nó, và mỗi lần đổi hình dạng field phải sửa cả hai repo).

Luật nằm ở: `PRODUCT.md` › Objects, Source of truth.

### Vì sao `unisage-agent` tin thẳng 5 header do Gateway bơm, không tự giải mã JWT?

Giải mã và xác thực JWT là việc Gateway đã làm, dùng chung một khoá bí mật mà `unisage-agent` không
cần biết tới. Tự giải mã lại ở đây là nhân đôi logic xác thực ở một nơi không sở hữu nó, và mọi lần
đổi thuật toán/khoá JWT phải sửa thêm một service nữa.

Đổi lại, `unisage-agent` chỉ còn một hàng rào duy nhất: `X-Internal-Secret` phải khớp
`APP_INTERNAL_SECRET_KEY`. Hàng rào này xác nhận **request tới từ Gateway**, không xác nhận **người dùng
là ai** — hai việc tách biệt có chủ đích, vì service không có cách nào tự kiểm tra vế thứ hai mà
không giải mã lại JWT.

Đã cân và loại: tự giải mã JWT ở cả hai service (hai nơi cùng phải đồng bộ khoá và thuật toán, sai một
bên là lỗ hổng).

Luật nằm ở: `PRODUCT.md` › Actors, Business rules.

## Phân quyền tài liệu

### Vì sao bộ lọc phân quyền chỉ dựng từ `AcademicSecurityContext`, tuyệt đối không đọc `confirmed_metadata`?

`confirmed_metadata` là thứ người dùng **tự khai** trong hội thoại, chưa qua xác minh — một sinh viên
gõ "em là học viên cao học" thì giá trị đó đi thẳng vào `confirmed_metadata`. Nếu giá trị này lọt vào
điều kiện lọc Qdrant, gõ đúng một câu là mở được tài liệu ngoài quyền thật — biến kênh hỏi-lại-metadata
(vốn chỉ để chọn đúng nhánh quy định khi trả lời) thành một lỗ hổng leo thang đặc quyền.

`build_access_filter(security: AcademicSecurityContext)` vì vậy chỉ nhận đúng một tham số, không có
cách nào truyền `confirmed_metadata` vào kể cả vô tình — ranh giới được khoá bằng chữ ký hàm, không
chỉ bằng quy ước. Có test riêng dựng `confirmed_metadata` chứa giá trị leo thang, khẳng định filter
sinh ra không đổi.

Luật nằm ở: `PRODUCT.md` › Source of truth, Business rules.

### Vì sao "công khai" là cờ `is_public` riêng, không phải `access_level == 0`?

Thiết kế ban đầu dùng `access_level == 0` làm điều kiện công khai. Sai vì khách vãng lai (`KHACH`)
không có `access_level` nào để so sánh — họ chỉ có `department_access = []`, không có một con số
"cấp 0" nào gắn với danh tính của họ. "Công khai" phải là một thuộc tính của **tài liệu**, độc lập
hoàn toàn với `department`/`access_level`, không phải một mức trong cùng thang đo với chúng.

Đổi sang field `is_public: bool` riêng, mirror đúng `@Column(name = "is_public") private Boolean
isPublic` đã có sẵn trên entity `Document` bên `unisage-backend` — hai hệ dùng chung một khái niệm,
không bịa ra một khái niệm mới cho Python.

Cái giá: field này chưa từng tồn tại ở đâu trong `unisage-agent`, nên phải kéo xuyên suốt cả pipeline
ingest chứ không chỉ sửa ở retrieval — không làm vậy thì không có cách nào đánh dấu một chunk là công
khai, và mọi câu hỏi của khách vãng lai sẽ luôn rơi vào TicketFallback:
`EmbeddingRequest.is_public` (mặc định `False`) → tham số Celery task `embed_chunks` →
`ChunkPoint.is_public` → payload Qdrant → điều kiện `should` đầu tiên trong `build_access_filter`.

Đã cân và loại: giữ `access_level == 0` và quy ước "khách luôn có access_level ngầm định = 0" (phải
bịa một giá trị không tồn tại trong dữ liệu thật, và không khớp với cách `unisage-backend` đã mô hình
hoá tài liệu công khai).

Luật nằm ở: `PRODUCT.md` › Objects, Source of truth, Glossary.

### Vì sao mỗi entry `department_access` được ràng buộc `department` và `access_level` thành một cặp riêng, không tách phẳng thành hai điều kiện `should` độc lập?

Nếu tách phẳng (một `should` cho tập `department` hợp lệ, một `should` khác cho `access_level ≤ max`
trong mọi entry), người có `[{KHOA_CNTT, 2}, {PHONG_DAOTAO, 1}]` sẽ vô tình đọc được chunk
`PHONG_DAOTAO` ở `access_level = 2` — mức đó chỉ được cấp cho KHOA_CNTT, không phải PHONG_DAOTAO.
Gói `department` và `access_level` của cùng một entry vào một khối `must` riêng giữ đúng: mỗi phòng
ban chỉ dùng mức được cấp cho chính nó, mức của phòng ban khác không "tràn" sang.

Luật nằm ở: `PRODUCT.md` › Business rules.

### Vì sao wildcard `department_id = "*"` được áp dụng cho retrieval chat, giống hệt cách ingestion đang hiểu nó?

`TrustedContext.granted_access_level` (phía ingestion, `app/api/deps.py`) đã coi một entry `*` là
được cấp `access_level` của nó trên **mọi** phòng ban. Nếu `build_access_filter` (phía chat) hiểu
khác đi — ví dụ coi `*` chỉ là một chuỗi department bình thường — thì cùng một token JWT mang hai
nghĩa khác nhau tuỳ endpoint nào đọc nó, một nguồn nhầm lẫn âm thầm và khó phát hiện qua test tách
biệt từng phía.

Luật nằm ở: `PRODUCT.md` › Business rules, Glossary.

## Phân loại ý định và luồng graph

### Vì sao `general_knowledge` và `academic_comparison` không còn là nhãn intent riêng?

Theo cập nhật flow_design 2026-08 của thiết kế gốc: `general_knowledge` (kiến thức phổ thông, VD
"Python là gì?") không cần tra quy chế nhà trường nên xử lý giống hệt một câu ngoài phạm vi — gộp vào
`off_topic`. `academic_comparison` (so sánh 2+ thực thể) đã luôn route tới cùng một đích
(`QueryTransformationNode`) như `academic_advisory`, chỉ khác ở việc có phân rã thành nhiều sub-query
hay không — sự khác biệt đó được `routing_mode = MULTI` diễn đạt đủ, không cần một nhãn intent riêng.

Kéo theo: `DirectLLMNode` (nhánh trả lời trực tiếp không qua RAG cho `general_knowledge`) bị xoá hoàn
toàn — không còn intent nào route tới đó.

Đã cân và loại: giữ `general_knowledge` như một nhãn riêng nhưng route chung đích với `off_topic`
(thêm một nhánh trong routing map mà không đổi hành vi, chỉ làm bảng routing dài hơn không cần
thiết).

Luật nằm ở: `PRODUCT.md` › Business rules, Glossary (node).

### Vì sao không làm hybrid search (BM25), RRF, và cross-encoder rerank?

Thiết kế gốc mô tả node truy xuất = pre-filter + dense + BM25 + RRF, và node rerank = cross-encoder
`bge-reranker-base` + ngưỡng 0.70. Code hiện chỉ làm pre-filter + dense search trên 3 vector đặt tên
(nội dung/tóm tắt/câu hỏi mẫu, giữ điểm cao nhất mỗi chunk) + lọc ngưỡng trên điểm cosine sẵn có. Đây
là quyết định có chủ đích:

- **BM25:** lợi ích chính là khớp chính xác mã/số hiệu văn bản (`QĐ-45/2023`, mã môn học), trong khi
  câu hỏi sinh viên chủ yếu hỏi theo nghĩa. Chi phí chạy gần như bằng 0, nhưng cần thêm sparse vector
  vào collection Qdrant — tức sửa ingestion và **re-index toàn bộ tài liệu đã có**. Chưa có số liệu
  cho thấy dense search đang trượt loại câu hỏi này.
- **RRF:** điểm tính theo thứ hạng (`1/(60 + rank)`, cỡ 0.01–0.03), không cùng thang với ngưỡng
  `CHAT_RERANK_SCORE_THRESHOLD` (một ngưỡng cosine). Bật RRF khi chưa có cross-encoder chấm lại điểm sẽ
  khiến mọi chunk đều dưới ngưỡng — mọi câu hỏi rơi vào TicketFallback. RRF chỉ có nghĩa khi đi cùng
  một bước rerank thật.
- **Cross-encoder:** phần tốn nhất — phí API mỗi câu hỏi, hoặc RAM/CPU/GPU để tự host. Lợi ích bị
  giảm bớt vì vector `questions` (câu hỏi mẫu sinh sẵn cho mỗi chunk) đã khớp khá sát câu hỏi thật
  của sinh viên, và mỗi lượt chỉ lấy tối đa `RETRIEVAL_MAX_CHUNKS` (8) chunk để xếp lại.

Rủi ro còn lại: `CHAT_RERANK_SCORE_THRESHOLD` đang áp lên điểm cosine của `text-embedding-3-small`, trong
khi thiết kế gốc đặt ngưỡng này cho điểm cross-encoder — hai thang điểm không tương đương. `.env` máy
dev hiện để `0.4` (thấp hơn mặc định code `0.70`), cho thấy đã có quan sát thực tế là `0.70` quá cao
cho cosine, nhưng chưa có số đo chính thức để chốt lại.

Đã cân và loại: bật RRF trước khi có rerank (mọi câu hỏi rơi fallback, xem trên) · tự host
`bge-reranker-base` nguyên bản (yếu với tiếng Việt; nếu làm nên dùng `bge-reranker-v2-m3` hoặc một
API rerank đa ngôn ngữ).

Khi nào làm lại: bộ câu hỏi đánh giá cho thấy trượt nhiều câu hỏi có mã/số hiệu → thêm BM25; chunk
đúng lấy được nhưng xếp sai thứ hạng, hoặc ngưỡng cosine không tách được đúng/sai → thêm rerank thật
(và khi đó mới bật RRF).

Luật nằm ở: `PRODUCT.md` › Business rules, The bet, Human decisions. Chi tiết kỹ thuật:
`docs/specs/known-gaps.md`.

## Chunking và ingest

### Vì sao các field cấu trúc mới của `ChunkPoint` (`source_type`, `heading_path`, `chunking_version`...) đều có giá trị mặc định thay vì bắt buộc?

Một chunk được upsert từ trước khi các field này tồn tại (phase cũ) sẽ thiếu hoàn toàn các khoá này
trong payload gốc. Nếu các field bắt buộc, mọi thao tác đọc lại chunk cũ sẽ phải tự bịa giá trị hoặc
raise lỗi. Mặc định `None`/`"legacy"`/`[]`/`False` để một điểm thiếu các khoá này vẫn parse được
thành `RetrievedChunk` hợp lệ — chỉ đơn giản là không có thông tin trang/heading/bảng cho chunk đó,
không phải một lỗi.

`is_public` (thêm sau, xem mục trên) theo đúng mẫu này: mặc định `False`, không bắt buộc phải khai
báo ở mọi lời gọi `EmbeddingRequest`/`ChunkPoint` cũ.

Luật nằm ở: `PRODUCT.md` › Objects.

### Vì sao draft chunk có `chunking_version` khác `settings.CHUNKING_VERSION` bị chặn embed, phải chunk lại?

`CHUNKING_VERSION` đại diện cho hình dạng chunk (cách chia đoạn, cách gắn metadata cấu trúc) tại một
thời điểm. Nếu cho phép embed một draft cũ hình dạng lẫn với chunk mới trong cùng collection, không
có cách nào phân biệt được payload nào đủ tin cậy cho field nào khi đọc lại — mỗi lần đổi
`CHUNKING_VERSION` là một lần đổi hợp đồng payload, không thể âm thầm trộn hai hợp đồng.

Luật nằm ở: `PRODUCT.md` › Business rules.

## Dynamic Model Registry

Chi tiết kỹ thuật đầy đủ (contract, state machine, security flow) nằm ở
`backend-java/changes/23-09-2026-Dynamic-Model-Registry-Runtime-Failover/plan.md`. Mục này chỉ ghi
"vì sao", cho người không đọc hết plan 1000+ dòng.

### Vì sao Embedding không auto-failover, trong khi Chat/Extraction có?

Failover tự động nghĩa là: credential A lỗi → tự chuyển sang credential B còn hoạt động, không cần
người can thiệp. Với Chat/Extraction, B trả lời khác A một chút không sao — cả hai đều sinh text. Với
Embedding, B tạo vector trong một không gian ngữ nghĩa khác A (kể cả khi cùng số chiều: hai model
384-dim từ hai nhà cung cấp khác nhau không so sánh cosine được với nhau một cách có nghĩa). Nếu để
failover tự động, hệ thống sẽ âm thầm bắt đầu ghi vector "sai không gian" vào cùng collection với
vector cũ, và retrieval degrade dần mà không có lỗi nào báo — phát hiện được chỉ khi người dùng phàn
nàn câu trả lời sai, lúc đó dữ liệu đã lẫn.

Vì vậy embedding lỗi thì dừng hẳn (ingest job fail, báo SA), không tự chọn credential khác. Ràng buộc
này ép bằng unique partial index DB (`chat_models WHERE model_purpose = 'EMBEDDING' AND status =
'ACTIVE'`), không chỉ bằng code — tại một thời điểm chỉ có đúng 1 credential EMBEDDING ACTIVE.

Luật nằm ở: `PRODUCT.md` › Business rules.

### Vì sao có "embedding identity guard" (bảng `embedding_index_identity`) thay vì chỉ tin credential đang ACTIVE?

Ban đầu thiết kế cho rotate/activate embedding chạy giống hệt Chat: verify ứng viên xong thì promote.
Review chỉ ra lỗ hổng: verify OK chỉ chứng minh ứng viên **gọi được**, không chứng minh ứng viên tạo
ra vector **cùng không gian** với vector đã có trong Qdrant. SA đổi `EMBEDDING` từ
`text-embedding-3-small` sang `text-embedding-3-large` (cùng verify OK, khác hẳn không gian vector) sẽ
promote êm ru rồi âm thầm phá retrieval — đúng lỗi mà quyết định "không auto-failover" ở trên cố tránh,
chỉ là qua một đường khác (SA chủ động sửa, không phải hệ thống tự chọn).

Giải pháp: tách "credential nào đang chạy" (rotation, có thể đổi) khỏi "vector trong Qdrant thuộc về
danh tính nào" (`embedding_index_identity`, gần như bất biến — đo bằng fingerprint 3 câu probe cố
định). Mọi đường có thể đổi embedding đang chạy (rotate, SA activate, swap, bật lại sau DISABLED) phải
so khớp hai thứ này; lệch thì chặn (`REINDEX_REQUIRED`), không có đường "UI cảnh báo rồi cho qua".
Bảng này khoá theo `collection_name` (không phải một singleton toàn cục) và không có method
update/delete nào ở tầng repository, cộng thêm trigger DB chặn UPDATE/DELETE — ba lớp, vì đây là loại
lỗi (âm thầm, phát hiện muộn) mà một lớp code review không đủ để bắt hết theo thời gian khi có người
mới sửa code sau này.

Đã cân và loại: chỉ cảnh báo UI rồi để SA tự quyết định có đổi hay không (SA không phải lúc nào cũng
biết đổi provider/model có nghĩa là "không gian vector khác", nhất là khi verify vẫn báo OK).

Luật nằm ở: `PRODUCT.md` › Source of truth, Objects, Glossary (embedding identity guard).

### Vì sao danh sách `llmProvider` không gồm mọi provider mà `pydantic-ai` hỗ trợ?

SSRF guard là điều kiện bắt buộc trước khi gọi bất kỳ URL nào lấy từ registry — không có ngoại lệ
"tạm chấp nhận rồi hardening sau". Guard đó hoạt động bằng cách pin socket layer của một
`httpx.AsyncClient`/`httpx.Client` cụ thể, nên một provider chỉ được thêm vào danh sách khi SDK của
nó thực sự nhận client đó qua tham số khởi tạo.

Từng kiểm và loại: `anthropic` — SDK hiện tại chỉ nhận `httpx2.AsyncClient` (một package HTTP client
khác hẳn `httpx`), nên không có cách gắn transport đã pin vào nó; `xai` — SDK của xAI dùng gRPC,
không tồn tại một HTTP client nào để pin; `deepseek` — SDK chấp nhận `httpx.AsyncClient` bình
thường, nhưng constructor cố định sẵn base URL, không nhận `base_url` theo từng credential như
cách factory hiện tại dựng client. Google (Gemini), Groq, Mistral đều đã thử thật và nhận đúng
client đã pin, nên được thêm vào cùng OpenAI.

Luật nằm ở: `PRODUCT.md` › Business rules.

### Vì sao verify credential đi theo chiều Python gọi Java (pull), không phải Java gọi Python (push)?

Thiết kế ban đầu định để Java, sau khi SA nhập credential mới, gọi thẳng sang Python để nhờ thử
provider, rồi Python gọi lại Java báo kết quả — một vòng Java→Python→Java. Vấn đề: không có cách nào
để Java biết chắc Python đã nhận việc (Python có thể đang restart, network đứt giữa chừng), nên phải tự
bịa thêm một lớp retry/timeout riêng cho chính cuộc gọi đó, tách biệt với vòng đời job.

Đảo thành pull: Java chỉ tạo job ở trạng thái `QUEUED`, Python (Celery Beat) tự định kỳ tới lấy job
(`claim`, có lease + fencing token) và tự báo kết quả về. Java không cần biết Python có đang sống hay
không — job cứ nằm đó tới khi có người tới lấy hoặc lease hết hạn thì job khác lấy lại. Toàn bộ vấn đề
"Java gọi Python nhưng Python không phản hồi" biến mất, đổi lại độ trễ tối đa là 1 chu kỳ Beat (15s)
thay vì tức thời — chấp nhận được vì verify không nằm trên đường phục vụ người dùng cuối.

Kéo theo: `unisage-agent` không expose endpoint "nội bộ" mới nào cho registry — Gateway đã gắn
`X-Internal-Secret` cho mọi request `/api/v1/ai/**` nó forward, nên một endpoint "nội bộ" ở Python thực
chất ai qua Gateway cũng gọi được, không phải hàng rào thật. Giữ nguyên tắc "mọi luồng Java↔Python của
registry đều là Python gọi Java" loại bỏ nhu cầu đó hoàn toàn.

Đã cân và loại: Java gọi Python trực tiếp kèm retry riêng (thêm một cơ chế đáng tin cậy phải tự xây,
trong khi pull-based đạt cùng mục đích bằng đúng cơ chế job-queue đã quen thuộc) · webhook Python báo
Java khi xong việc (vẫn cần Java→Python để giao việc trước, không giải quyết được vấn đề gốc).

Luật nằm ở: `PRODUCT.md` › Business rules, Actors (model registry verifier).

### Vì sao sửa credential của một `ChatModel` đang ACTIVE không ghi thẳng vào row (staged rotation)?

Ghi thẳng nghĩa là: SA bấm lưu key mới → row đổi ngay → nếu key mới sai (gõ nhầm, key đã bị thu hồi),
mọi request Chat/Extraction tiếp theo lỗi ngay lập tức, cho tới khi SA phát hiện và sửa lại — một cửa
sổ downtime hoàn toàn có thể tránh được bằng cách verify trước.

Giải pháp: giá trị mới trở thành "ứng viên" trong một verification job, row (và snapshot Python đang
dùng) giữ nguyên giá trị cũ cho tới khi ứng viên verify xong. Verify OK mới promote (ghi đè bằng
compare-and-set, so cả `revision` lẫn `candidate_generation` để chặn race khi SA sửa liên tiếp nhiều
lần trước khi lần verify trước kịp xong — job dở bị đánh `SUPERSEDED`, không bao giờ promote nhầm giá
trị cũ hơn). Verify FAIL thì báo lỗi ngay cho SA, row không đổi, hệ thống tiếp tục chạy bằng key cũ.

Đã cân và loại: verify đồng bộ ngay trong request lưu credential của SA (chặn UI vài giây chờ gọi
provider thật, và không giải quyết được trường hợp Python đang không sống để verify ngay lúc đó).

Luật nằm ở: `PRODUCT.md` › Glossary (staged credential rotation).

## Cấu hình

### Vì sao mỗi biến môi trường mang một tiền tố theo nhóm (`APP_`, `DB_`, `CHAT_`, `INGEST_`...) thay vì một `env_prefix` riêng cho từng class `BaseSettings`?

Trước 2026-09, toàn bộ ~26 biến nằm trần trong một class `Settings` duy nhất, không tiền tố — dễ đọc
khi danh sách còn ngắn, nhưng không còn phân biệt được biến nào ảnh hưởng một lượt chat (đọc lại mỗi
request) với biến nào chỉ ảnh hưởng lúc ingest (đọc một lần khi chunk/embed) chỉ bằng cách nhìn tên.
Ví dụ thực tế của sự mập mờ đó: `CHUNKING_VERSION` và `RERANK_SCORE_THRESHOLD` nằm cạnh nhau trong
`config.py` dù một cái là chuyện ingest, một cái là chuyện chat — không có gì trong tên nói lên sự
khác biệt đó.

Không tách theo `env_prefix` riêng từng class `BaseSettings` như một số service khác trong hệ
UniSage (`APP_`, `OCR_`... — mỗi class là một tiến trình/mối quan tâm lớn) vì `unisage-agent` chỉ có
một tiến trình FastAPI + một Celery worker **dùng chung đúng một cấu hình** — tách class không tạo ra
ranh giới triển khai thật nào, chỉ tạo thêm một lớp gián tiếp. Thay vào đó, tiền tố gắn trực tiếp vào
tên field trong cùng một `Settings`, đặt theo hai trục:

- **Gọi một hệ ngoài cụ thể** → tiền tố theo tên hệ đó (`OPENAI_`, `MINIO_`, `QDRANT_`) — dùng được ở
  cả ingest lẫn chat (`OPENAI_*`) thì giữ tiền tố theo hệ ngoài, không ép về `CHAT_`/`INGEST_`.
- **Chỉ tinh chỉnh hành vi của chính graph/pipeline, không gọi hệ ngoài nào** → `INGEST_` (chỉ đọc lúc
  ingest tài liệu) hoặc `CHAT_` (chỉ đọc trong một lượt chat). `APP_` cho phần còn lại: tên service,
  môi trường, debug logging, khoá bí mật cấp tiến trình.

Cái giá: đổi tên ~13 biến đang có nghĩa là một lần rà toàn repo (mọi `settings.X`, mọi
`monkeypatch.setattr(settings, "X", ...)`, `.env.example`, Taskfile, `.devcontainer/docker-compose.yml`,
`migrations/env.py`) — đã làm trong đợt 2026-09. **`.env` thật trên máy dev không nằm trong git nên
không tự đổi theo được** — người triển khai phải tự sửa lại theo `.env.example` mới, nếu không
`Settings` sẽ âm thầm rơi về giá trị mặc định trong code (nguy hiểm nhất với `DB_URL`: kết nối sai
database mà không có lỗi nào báo ở bước khởi động).

Đã cân và loại: `env_prefix` riêng theo class (thêm một lớp gián tiếp không cần thiết cho một tiến
trình duy nhất) · giữ tên trần và chỉ ghi chú nhóm trong comment (comment không việc gì bắt được lệch
tên field mới thêm sau này quên đặt tiền tố — tiền tố trong chính tên field mới là ràng buộc mà
review/`grep` bắt được ngay).

Luật nằm ở: `docs/guide/cau-hinh.md` › Biến môi trường.

### Vì sao `APP_HOST`/`APP_PORT` nằm trong `.env` nhưng không phải field của `Settings`?

Hai biến này chỉ có ý nghĩa với tiến trình **khởi chạy** server (uvicorn/gunicorn bind address), không
phải với logic ứng dụng — không node nào, không service nào bên trong `app/` cần biết mình đang nghe ở
địa chỉ nào. Taskfile đọc thẳng `.env` qua `dotenv: ['.env']` rồi truyền vào `--host`/`--port`, tách
biệt khỏi `Settings` (Pydantic) mà code Python đọc.

Đã cân và loại: thêm `APP_HOST`/`APP_PORT` vào `Settings` dù không ai đọc (giữ lại "làm tài liệu" — đúng thứ
`docs/guide/cau-hinh.md` cấm).

Luật nằm ở: `docs/guide/cau-hinh.md` › Biến môi trường.

### Vì sao chưa có hàng rào chặn khởi động khi `APP_ENV=production` mà cấu hình còn thiếu an toàn (VD `APP_INTERNAL_SECRET_KEY` vẫn để mặc định)?

Đây là khoảng trống thật, chưa phải một quyết định có chủ đích — `Settings` hiện không đọc `APP_ENV`
để gate bất cứ điều gì. Ghi nhận ở đây để không ai tưởng nhầm im lặng này là "đã cân nhắc rồi bỏ qua".

Luật nằm ở: `docs/guide/cau-hinh.md` › Biến môi trường; `docs/specs/known-gaps.md`.
