# Hướng dẫn cấu hình

Áp dụng cho mọi cấu hình của `unisage-agent`. Đọc trước khi thêm một biến môi trường, một file
prompt, hay một trường trong `known_metadata_fields.json`.

## Hai loại cấu hình thật, một loại chưa có

Mẫu chuẩn cho service trong hệ UniSage có ba loại (bí mật/theo môi trường → biến môi trường; ngưỡng
vận hành sửa tay → YAML trong `configs/`; dữ liệu máy sinh → JSON có số phiên bản cạnh code nạp nó).
`unisage-agent` hôm nay chỉ có **hai loại đầu tiên**, và loại hai không nằm trong một thư mục
`configs/` riêng như mẫu — nó là chính nội dung prompt LLM. Đừng suy diễn có một `configs/` đang ẩn
đâu đó; không có.

| Loại | Ví dụ | Giữ ở đâu | Định dạng | Ai sửa | Đổi có cần build lại image? |
|---|---|---|---|---|---|
| Bí mật và thứ đổi theo môi trường | `OPENAI_API_KEY`, `DB_URL`, `CHAT_RERANK_SCORE_THRESHOLD` | biến môi trường, `.env` khi chạy máy dev | `KEY=value`, đọc qua `Settings` (Pydantic) | người triển khai | Không |
| Nội dung prompt LLM, cùng vòng đời với code | system prompt từng node, quy tắc bảo mật, từ vựng field | `app/rag/prompting/prompt_templates/`, `known_metadata_fields.json` | YAML (`template`/`content`), JSON | người viết/tinh chỉnh prompt | Có — nạp lúc import, không đọc lại khi chạy |
| Dữ liệu máy sinh có số phiên bản | *(chưa có trong repo này)* | — | — | — | — |

Không trộn hai loại đang có. Một khoá API không bao giờ nằm trong YAML prompt; một câu quy tắc
bảo mật không bao giờ nằm trong `.env`.

## Biến môi trường

### Tiền tố — quy tắc

Toàn bộ biến nằm trong một class `Settings(BaseSettings)` duy nhất ở `app/core/config.py`, nhưng
**mỗi tên đều mang một tiền tố nói rõ nó thuộc nhóm nào**, theo đúng thứ tự khai báo trong
`config.py`:

| Tiền tố | Ý nghĩa | Đọc ở đâu |
|---|---|---|
| `APP_` | Cấp tiến trình: tên service, môi trường, debug logging, địa chỉ bind, khoá bí mật dùng chung với Gateway | Mọi nơi trong `app/` cần biết mình là ai/đang chạy ở đâu |
| `DB_` | Database Postgres của riêng `unisage-agent` | `app/database/session.py`, `migrations/env.py` |
| `OPENAI_` | Provider LLM/embedding — dùng chung cho cả ingest lẫn chat | Cả `app/rag/` lẫn `app/graph/` |
| `MINIO_` | Kho object cho file gốc đã ingest | `app/rag/ingestion/minio_client.py` |
| `QDRANT_` | Vector store | `app/rag/vectorstore/` |
| `TAVILY_` | Web search API cho WebSearchNode | `app/integrations/tavily_client.py` |
| `REDIS_URL`, `BACKEND_JAVA_BASE_URL` | Hệ ngoài chỉ có đúng một biến — tự tên đã đủ rõ, không cần gói thành nhóm | Celery/broker; `BackendJavaClient` |
| `INGEST_` | Chỉ đọc lúc **ingest tài liệu** (chunking, enrichment) — không bao giờ đọc trong một lượt chat | `app/rag/chunking/`, `app/rag/enrichment/` |
| `CHAT_` | Chỉ đọc trong một **lượt chat** (các node của graph) | `app/graph/`, `app/api/v1/chat.py` |

Một biến dùng được ở **cả hai** nhánh ingest và chat (ví dụ provider LLM) thì **không** ép vào
`INGEST_`/`CHAT_` — giữ tiền tố theo tên hệ ngoài nó gọi tới (`OPENAI_`). Chỉ dùng `INGEST_`/`CHAT_`
cho tham số **tinh chỉnh hành vi của chính graph/pipeline**, không phải tên một hệ ngoài.

Khi thêm biến mới, tự hỏi theo đúng thứ tự: nó gọi một hệ ngoài cụ thể? → tiền tố theo tên hệ đó. Nó
chỉ có ý nghĩa lúc ingest? → `INGEST_`. Chỉ có ý nghĩa trong một lượt chat? → `CHAT_`. Còn lại (tên
service, môi trường, bind address, khoá bí mật cấp tiến trình) → `APP_`.

### Quy tắc chung

- Tên biến viết `UPPER_SNAKE_CASE`, luôn có đúng một trong các tiền tố ở trên. `model_config` đặt
  `extra="ignore"`: biến lạ trong `.env` không làm service từ chối khởi động, khác mẫu (mẫu cho
  service kia chặn khởi động khi CORS mở hoặc khoá dùng chung rỗng ở `production`) —
  `unisage-agent` **chưa có** hàng rào tương tự cho `APP_ENV=production`; đây là khoảng trống thật,
  không phải lựa chọn có chủ đích, xem `docs/product/DECISIONS.md`.
- Không phải mọi biến trong `.env`/`.env.example` đều được `Settings` đọc. `APP_HOST` và `APP_PORT`
  có mặt ở đó nhưng **không** là field của `Settings` — chúng được Taskfile đọc trực tiếp qua
  `dotenv: ['.env']` (`taskfiles/backend.yml`) để truyền vào `uvicorn --host --port`/`gunicorn -b`.
  Đây là quy ước duy nhất trong repo: một biến trong `.env` có thể thuộc về **tiến trình khởi chạy**
  (Taskfile) thay vì về **Settings** của ứng dụng, dù vẫn mang tiền tố `APP_` như mọi biến cấp tiến
  trình khác. Khi thêm một biến mới, tự hỏi nó thuộc bên nào trước khi đặt tên và viết comment.
- Mỗi biến có comment ngay phía trên (trong cả `config.py` lẫn `.env.example`) giải thích *dùng để
  làm gì*, và với biến dễ hiểu nhầm thì giải thích cả *vì sao giá trị này*. Biến không ai đọc thì xoá
  khỏi cả hai file — không giữ lại làm tài liệu.
- `.env.example` là hợp đồng: mọi biến `Settings` đọc thật đều có mặt ở đó với giá trị mẫu chạy được
  ngay cho máy dev (trừ khoá bí mật thật như `OPENAI_API_KEY`, để placeholder). Thêm một field vào
  `Settings` mà quên thêm vào `.env.example` là để lại một khoảng trống hợp đồng.
- `tests/test_config.py` là nơi kiểm tra "giá trị mặc định còn hợp lý" và "override qua env hoạt
  động" — không phải nơi khớp `.env.example` với `Settings` (repo chưa có test đó, xem known-gaps).

> **Đã đổi tên 2026-09:** trước đây các biến không gọi thẳng một hệ ngoài đều để trần, không tiền
> tố (`DEBUG`, `INTERNAL_SECRET_KEY`, `DATABASE_URL`, `CLARIFICATION_MAX_RETRY`,
> `RETRIEVAL_MAX_CHUNKS`, `RERANK_SCORE_THRESHOLD`, `HISTORY_MESSAGE_LIMIT`, `ALLOW_REPAIR_JSON`,
> `SEMANTIC_MAX_TOKEN_FACTOR`, `TABLE_CHUNK_MAX_TOKENS`, `CHUNKING_VERSION`,
> `MULTI_REP_LLM_MODEL`, `MULTI_REP_QUESTION_COUNT`, `HOST`, `PORT`). Đã đổi hết sang bảng tiền tố ở
> trên (`APP_DEBUG`, `APP_INTERNAL_SECRET_KEY`, `DB_URL`, `CHAT_CLARIFICATION_MAX_RETRY`,
> `CHAT_RETRIEVAL_MAX_CHUNKS`, `CHAT_RERANK_SCORE_THRESHOLD`, `CHAT_HISTORY_MESSAGE_LIMIT`,
> `CHAT_ALLOW_REPAIR_JSON`, `INGEST_SEMANTIC_MAX_TOKEN_FACTOR`, `INGEST_TABLE_CHUNK_MAX_TOKENS`,
> `INGEST_CHUNKING_VERSION`, `INGEST_MULTI_REP_LLM_MODEL`, `INGEST_MULTI_REP_QUESTION_COUNT`,
> `APP_HOST`, `APP_PORT`). **File `.env` thật trên máy dev (không nằm trong git) vẫn giữ tên cũ —
> phải tự sửa lại theo `.env.example` mới, nếu không `Settings` sẽ âm thầm rơi về giá trị mặc định
> trong code thay vì đọc được giá trị đã chỉnh.** Rủi ro rõ nhất là `DB_URL`: nếu Postgres máy dev
> không khớp giá trị mặc định trong `config.py`, service sẽ kết nối sai database mà không báo lỗi
> nào ở bước khởi động.

### Danh mục biến hiện có (`app/core/config.py`)

**`APP_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `APP_NAME` | `UniSage AI Agent Service` | Tên hiển thị, log khởi động |
| `APP_ENV` | `development` | Chưa gate hành vi nào theo giá trị này (xem khoảng trống ở trên) |
| `APP_DEBUG` | `True` | Chỉ bật log chi tiết của **chính service này** (`app/core/graph_trace.py` dump `academic_metadata`/`prepared_context`/HyDE output) — cố tình **không** nâng root log level, để log DEBUG của httpx/OpenAI SDK/SQLAlchemy không nhấn chìm dòng log node đang chạy (xem `app/main.py`) |
| `APP_INTERNAL_SECRET_KEY` | `unisage-internal-secret-key-2026` | Phải khớp `X-Internal-Secret` mà API Gateway gắn vào mọi request chuyển tới; `app/core/security.py::verify_internal_secret` chặn mọi request không mang đúng giá trị này ở tầng router — service chỉ vào được qua Gateway, không được gọi thẳng |
| `APP_HOST`/`APP_PORT` | `0.0.0.0`/`8402` (`.env.example`); `127.0.0.1`/`8402` (mặc định Taskfile) | **Không** là field `Settings` — Taskfile đọc thẳng qua `dotenv`, truyền vào `uvicorn`/`gunicorn` lúc khởi chạy |

**`DB_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `DB_URL` | trỏ `unisage_agent_db` trên cổng `5433` | Cùng server Postgres vật lý với `backend-java` nhưng **khác database** (`unisage_agent_db`, không phải `assistant_DB` của Java) — không schema chung, không FK xuyên service (xem `docs/specs/SPEC-ingestion-resume.md`) |

**`OPENAI_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `OPENAI_API_KEY` | rỗng | Bắt buộc phải điền để chạy bất kỳ node LLM nào |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model cho 4 node LLM của graph (classification, HyDE, generation — xem `GraphModels`) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model embed cho cả ingest lẫn truy vấn — đổi model này **phải** re-index toàn bộ collection Qdrant vì chiều vector đổi |

**`MINIO_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `MINIO_ENDPOINT`/`MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY`/`MINIO_BUCKET`/`MINIO_SECURE` | instance MinIO dùng chung với `backend-java` | Kho object cho file gốc đã ingest |

**`QDRANT_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `QDRANT_HOST`/`QDRANT_PORT`/`QDRANT_COLLECTION` | `localhost:6333`, `unisage_chunks` | Vector DB duy nhất — không còn pgvector dù README/CONTEXT.md cũ còn nhắc (xem known-gaps) |

**`TAVILY_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `TAVILY_API_KEY` | rỗng | Khoá Tavily cho WebSearchNode. Rỗng thì bỏ qua web search (log warning), luồng đi thẳng tới TicketFallbackNode |
| `TAVILY_BASE_URL` | `https://api.tavily.com` | Host cố định do người triển khai đặt — không qua SSRF guard, cùng mức tin cậy với `SLACK_APIKEY_ALERT_WEBHOOK_URL` |
| `TAVILY_INCLUDE_DOMAINS` | `iuh.edu.vn` | Danh sách domain phân tách bằng dấu phẩy, gửi làm `include_domains` — chỉ tìm trên trang chính thức của trường (và subdomain), không bao giờ diễn đàn hay trường khác |
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` tốn 1 credit/lần tìm, `advanced` 2 credit nhưng snippet dài và sát hơn |
| `TAVILY_TIMEOUT_SECONDS` | `15` | Hạn chót cho **cả** lần gọi (không phải từng giai đoạn kết nối/đọc như timeout của httpx). Tavily thường trả lời trong ~3 giây nhưng có lúc vọt quá 10 giây; quá hạn thì coi như không có kết quả web, lượt chat vẫn đi tiếp tới TicketFallbackNode |

**Hệ ngoài chỉ một biến**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Broker + result backend của Celery (tác vụ embed nền), và kênh publish sự kiện tiến độ ingest |
| `BACKEND_JAVA_BASE_URL` | `http://localhost:8401/api/v1` | `BackendJavaClient` gọi thẳng `backend-java`, **bỏ qua** API Gateway — luôn kèm cả `Authorization` gốc (forward nguyên văn) lẫn `X-Internal-Secret` để Java xác thực đúng service gọi |

**`INGEST_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `INGEST_MULTI_REP_LLM_MODEL` | `gpt-4o-mini` | Model sinh `summary`/`questions` lúc ingest (`MultiRepresentationEnricher`) — tách riêng khỏi `OPENAI_MODEL` vì đây là việc ngoài luồng, có thể chạy model rẻ hơn hoặc khác hẳn mà không ảnh hưởng chất lượng trả lời |
| `INGEST_MULTI_REP_QUESTION_COUNT` | `3` | Số câu hỏi mẫu sinh sẵn cho mỗi chunk, dùng làm `questions_vector` khi retrieval fan-out 3 vector |
| `INGEST_SEMANTIC_MAX_TOKEN_FACTOR` | `1.5` | Chiến lược chunking `semantic`: trần token mềm = `target_tokens * hệ số này`, trước khi buộc cắt |
| `INGEST_TABLE_CHUNK_MAX_TOKENS` | `800` | Trần token mặc định cho một chunk bảng, ghi đè được qua `params.table_max_tokens` của từng request chunking |
| `INGEST_CHUNKING_VERSION` | `2026-09-structural-v2` | Dán vào mọi chunk mới ingest; `ChunkPoint.chunking_version` mặc định `"legacy"` cho chunk cũ hơn phase này. `app/api/v1/ingestion.py` so sánh giá trị này với `chunk.chunking_version` của draft để phát hiện draft chunk theo sơ đồ cũ trước khi cho embed — đổi giá trị này là đổi **cả một lứa** chunk cũ thành "cần chunk lại", nên chỉ đổi khi thật sự đổi hình dạng chunk, không đổi tuỳ hứng |

**`CHAT_`**

| Biến | Mặc định | Dùng ở đâu / vì sao |
|---|---|---|
| `CHAT_CLARIFICATION_MAX_RETRY` | `2` | Clarification Guard (SecurityContextExtractionNode): số lần hỏi lại tối đa cho một field trước khi buộc trả lời an toàn theo hướng "so sánh phương án" thay vì hỏi tiếp mãi |
| `CHAT_RETRIEVAL_MAX_CHUNKS` | `8` | Số chunk tối đa trả về sau RetrievalFilteringNode, trước khi qua ngưỡng rerank |
| `CHAT_RERANK_SCORE_THRESHOLD` | `0.70` | Ngưỡng lọc ở PostRetrievalRerankNode. **Đang áp lên điểm cosine của `text-embedding-3-small`**, không phải điểm cross-encoder như thiết kế gốc (chưa có cross-encoder) — xem rủi ro ở `docs/specs/known-gaps.md` |
| `CHAT_MAX_SUB_QUERIES` | `3` | Số câu hỏi con tối đa khi decomposer tách một câu so sánh (task `MULTI`); tối thiểu 2. Mỗi câu hỏi con tốn thêm một lần embedding + tìm Qdrant, và chia nhỏ quota chunk của RetrievalFilteringNode |
| `CHAT_WEB_SEARCH_ENABLED` | `False` | Bật WebSearchNode: tìm web (Tavily) cho mỗi câu hỏi con mà rerank không còn chunk nào, trước khi rơi xuống TicketFallbackNode |
| `CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB` | `5` | `max_results` gửi Tavily cho mỗi câu hỏi con trượt. Tavily tính credit theo lượt tìm, không theo số kết quả, nên lấy nhiều ứng viên không tốn thêm — trang đúng thường không nằm ở top 2 với câu hỏi tiếng Việt |
| `CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN` | `2` | Tổng số trang web tối đa đưa vào `<websearch>` một lượt — chia round-robin: mỗi câu hỏi con trượt được trang tốt nhất trước, phần còn lại theo điểm. Lấy `PER_SUB` ứng viên rồi chỉ giữ ngần này trang điểm cao nhất |
| `CHAT_WEB_SEARCH_MAX_QUERIES` | `2` | Số lần tìm Tavily tối đa một lượt. Tin nhắn nhiều task có thể có nhiều câu hỏi con trượt, nhưng chỉ `PER_TURN` trang vào prompt nên tìm thêm chỉ tốn credit. Ưu tiên câu hỏi con trượt nặng nhất (chunk tốt nhất có điểm thấp nhất) |
| `CHAT_WEB_SEARCH_MIN_SCORE` | `0.5` | Bỏ kết quả có điểm liên quan của Tavily dưới ngưỡng này |
| `CHAT_WEB_SEARCH_RESULT_MAX_CHARS` | `1500` | Cắt nội dung mỗi kết quả — cùng `PER_TURN` giới hạn phần web trong system prompt ở khoảng 3000 ký tự, bất kể câu hỏi dài hay tách thành bao nhiêu câu hỏi con |
| `CHAT_ALLOW_REPAIR_JSON` | `True` | Bật/tắt lệnh gọi LLM sửa lỗi lần 2 khi câu trả lời quên khối `ask_user_form` bắt buộc (`generation_synthesis.py::_repair_missing_ask_form`). Heuristic phát hiện có lỗ hổng biết trước (câu mời đặt điều kiện ở cuối câu, kiểu "..., nếu bạn cần...", không bị nhận diện là câu không ràng buộc) khiến lần gọi sửa đôi khi bịa ra một form không ai hỏi. Tắt thì bỏ hẳn lần gọi sửa: một form thật sự bị quên sẽ không được vá, nhưng không bao giờ bịa form giả |

### Ví dụ thêm một biến mới

```python
# app/core/config.py — biến chỉ có ý nghĩa lúc ingest → tiền tố INGEST_
# Trần số chunk mỗi request chunking preview trả về — vì sao 200: tránh
# response phình to khi tài liệu vài nghìn trang, xem docs/product/DECISIONS.md.
INGEST_CHUNKING_PREVIEW_MAX_CHUNKS: int = 200
```

```env
# .env.example — comment giống hệt bên config.py, có giá trị mẫu
INGEST_CHUNKING_PREVIEW_MAX_CHUNKS=200
```

## Nội dung prompt LLM (YAML/JSON, cùng vòng đời với code)

- Nằm trong `app/rag/prompting/prompt_templates/`, chia ba thư mục:
  - `agents/` — system prompt của các LLM nhỏ một việc (phân loại intent, sinh HyDE...). Nạp
    **nguyên văn**, dùng thẳng làm `system_prompt` của `Agent`, **không** qua `.format()` — nhiều file
    chứa ví dụ JSON có dấu `{}`, `.format()` sẽ ném `KeyError` ngay ví dụ đầu tiên.
  - `main/` — khung system prompt chính cho từng nhánh của graph (advisory, multi-intent, ticket
    fallback...), có placeholder `{tên}` để `.format()` lồng các khối `common/` cùng dữ liệu động vào.
  - `common/` — khối dùng chung nhiều khung (`header`, `academic_metadata`, `task_1`, `task_2`,
    `security_access_control`...). Một số khối là chuỗi tĩnh dùng thẳng, một số cần `.format()` thêm
    một lớp bên trong `builder.py` trước khi lồng vào khung chính (xem docstring `builder.py`).
- Mỗi file YAML có `description` (một dòng, nói *file này để làm gì*) và trường nội dung tên
  `template` hoặc `content` — `loader.py::_load_yaml_template` chấp nhận cả hai tên, ưu tiên
  `template`. Không dùng tên khác.
- `loader.get_templates()` nạp và cache toàn bộ trong bộ nhớ tiến trình **một lần**, lúc gọi đầu
  tiên — sửa file YAML rồi **phải khởi động lại tiến trình** (hoặc gọi
  `reset_templates_cache()` — chỉ dùng trong test) mới thấy hiệu lực, không có hot-reload.
- `known_metadata_fields.json` (cạnh `loader.py`) là từ vựng khái niệm → tên field chuẩn, render vào
  `ask_user_form_guide.yaml` qua `builder.build_known_metadata_fields_section()`. **Không phải
  whitelist đóng** — LLM vẫn được tự đặt tên field mới cho khái niệm chưa liệt kê; file chỉ ép dùng
  lại đúng tên khi khái niệm trùng một dòng đã có, để tránh một khái niệm bị phân mảnh thành hai tên
  field khác nhau qua các lượt hỏi (bug đã quan sát được: `nganh_hoc` và `nganh_dao_tao` cùng nghĩa
  "ngành" nhưng khác tên, làm giá trị cũ bị bỏ quên — xem comment `_comment` đầu file). Khác dữ liệu
  máy sinh thật sự: file này **không có** `version` hay lệnh sinh lại — sửa tay trực tiếp, review qua
  PR như mọi thay đổi code khác.
- Vì đọc một lần lúc import và không phân biệt "đã build" hay chưa, đổi nội dung một file trong
  `prompt_templates/` (hay `known_metadata_fields.json`) luôn cần build lại image / khởi động lại
  tiến trình để có hiệu lực — coi như một phần của code, không phải cấu hình vận hành đổi được lúc
  service đang chạy.

### Thêm một file prompt mới

1. Thêm file YAML vào đúng thư mục (`agents/`, `main/`, hay `common/`).
2. Thêm field tương ứng vào `PromptTemplates` (`app/rag/prompting/schema.py`) — đặt tên
   `agent_<tên_file>` cho `agents/`, tên trần cho `main/`/`common/`, theo đúng quy ước đang có.
3. Thêm dòng nạp trong `loader._load_all_templates()`.
4. Nếu file cần dữ liệu động, viết hàm `build_*` trong `builder.py` rồi export qua
   `app/rag/prompting/__init__.py`.
5. Thêm test khẳng định file nạp được và giữ đúng placeholder (`tests/rag/test_prompt_loader.py`).

## Quy ước chung

- Một nguồn cho mỗi con số. `CHAT_RETRIEVAL_MAX_CHUNKS`, `CHAT_RERANK_SCORE_THRESHOLD`... mỗi cái
  chỉ khai một chỗ trong `Settings`, không lặp lại hằng số ở nơi khác.
- `tests/test_config.py` kiểm giá trị mặc định "còn hợp lý" và việc override qua biến môi trường hoạt
  động — không kiểm `.env.example` khớp `Settings` (chưa có, xem `docs/specs/known-gaps.md`).
  `tests/rag/test_prompt_loader.py` kiểm mọi file trong `prompt_templates/` nạp được, giữ đúng
  placeholder, và (với `agents/`) không có `.format()` nào âm thầm nuốt mất một khối JSON mẫu.
- Đổi giá trị một biến môi trường không cần build lại image (chỉ cần khởi động lại tiến trình đọc
  `.env` mới). Đổi nội dung một file `prompt_templates/`, thêm một field `Settings` mới, hay đổi hình
  dạng `known_metadata_fields.json` đều cần build lại — vì cả ba đều là code, đi qua review và test
  như mọi thay đổi khác, không phải cấu hình vận hành người dùng cuối chỉnh được lúc chạy.
