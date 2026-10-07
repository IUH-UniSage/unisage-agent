# Pipeline chat RAG của unisage-agent — tra cứu khi trace

Mọi đường dẫn tính từ `/home/huy/Main/unisage-agent`. Nếu số dòng lệch, grep theo tên hàm hoặc tên node.

## Mục lục
1. Chuỗi node và ý nghĩa
2. Retrieval: những gì quyết định chunk nào được lấy ra
3. Ingest: chunk được tạo ra như thế nào
4. Log: cái gì có, cái gì KHÔNG có
5. Prompt sinh câu trả lời
6. Cấu hình ảnh hưởng kết quả

## 1. Chuỗi node (`app/graph/streaming_graph.py`, hàm `run_graph`)

| Node trong log | Việc làm | Dấu hiệu lỗi |
|---|---|---|
| `01_GreetingDetectionNode` | Chỉ ở lượt đầu, regex chào hỏi → template tĩnh | Câu hỏi thật bị coi là chào hỏi |
| `02_SecurityContextExtractionNode_ClarificationGuard` | Đang có form hỏi lại thì nhảy thẳng sang advisory | Hội thoại cũ còn form treo làm lệch luồng |
| `03_MessageClassificationNode` | LLM phân loại: `greeting`, `social_chat`, `academic_advisory`, `academic_calculation`, `off_topic`; tối đa 3 task. JSON lỗi hoặc intent lạ → advisory SINGLE | Phân loại sai. Kết quả phân loại (kèm `hyde_text` / `sub_queries`) chỉ được log khi `APP_DEBUG=true` |
| `04_IntentRouting_SocialChat` | Template tĩnh `SOCIAL_CHAT_TEMPLATE` (`app/graph/nodes/intent_routing.py`) | Câu học vụ bị trả lời xã giao |
| `05_OffTopicRejectNode` | Template tĩnh (`app/graph/nodes/off_topic.py`) | Câu học vụ bị từ chối "ngoài phạm vi" |
| `07_CalculationNode` | Lượt chỉ có tính toán | |
| `06_QueryTransformationNode` | SINGLE → HyDE (`agents/hyde_generator.yaml`): câu hỏi viết lại + một câu "tài liệu giả định". MULTI → tách sub-query (`agents/multi_query_decomposer.yaml`, tối đa `CHAT_MAX_SUB_QUERIES`). Khi `CHAT_CLASSIFY_WITH_RETRIEVAL=true` và node 03 đã viết sẵn HyDE / sub-query thì node này không gọi LLM | **Toàn bộ text HyDE được embed**, kể cả câu giả định. Câu giả định bịa số liệu, tên phòng ban hoặc đối tượng sai thì kéo về chunk sai |
| `08_RetrievalFilteringNode` | Dense search Qdrant + filter quyền | Không log gì |
| `09_PostRetrievalRerankNode` | Chỉ cắt theo ngưỡng cosine `CHAT_RERANK_SCORE_THRESHOLD` (`app/rag/reranking/cross_encoder.py`, tên gây hiểu nhầm, không có cross-encoder) | Không log gì |
| `09a_LLMRerankNode` | Model EXTRACTION đọc 800 ký tự đầu mỗi chunk, giữ chunk nó nêu được lý do (`agents/reranker_compressor.yaml`). Lỗi thì giữ nguyên kết quả theo score | Loại nhầm chunk đúng vì đoạn đáp án nằm sau ký tự thứ 800 |
| `09b_WebSearchNode` | Tavily, chỉ cho sub-query không còn chunk nào | Web lấp chỗ trống bằng nguồn ngoài |
| `11_TicketFallbackNode` | Không còn chunk và không có web → mẫu "chưa có thông tin, tạo ticket" | Từ chối dù tài liệu có đáp án |
| `10_GenerationSynthesisNode` | Sinh câu trả lời từ `<academic_context>` | Có chunk đúng nhưng đọc sai, tổng hợp sai hoặc từ chối |

Không có intent "unanswerable". Câu không có đáp án phải bị chặn ở ngưỡng hoặc LLM rerank (dẫn tới TicketFallback), hoặc bởi luật từ chối trong prompt sinh.

## 2. Retrieval (`app/rag/vectorstore/qdrant_store.py`, `app/rag/retrieval/service.py`)

- Collection `unisage_chunks`. Mỗi point có 3 vector tên `content_vector`, `summary_vector`, `questions_vector` (cosine, 1536 chiều).
- `search_chunks_batch` gửi mọi sub-query × 3 vector trong một `query_batch_points`, lấy điểm cao nhất của point qua 3 vector, rồi cắt top-k theo từng sub-query.
- Không có BM25 hay hybrid. **Câu hỏi chứa mã, số hiệu hoặc từ khoá hiếm dễ trượt** vì chỉ có dense search.
- top-k = `CHAT_RETRIEVAL_MAX_CHUNKS` (16). Có n sub-query thì mỗi sub-query lấy `ceil(16/n)`. Số chunk vào prompt bị chặn riêng bởi `CHAT_CONTEXT_MAX_CHUNKS` (8).
- Filter quyền (`build_access_filter`) là OR của:
  - `is_public`
  - từng mục (department, `access_level <= x`)
  - mục wildcard `department_id="*"`
- **SUPER_ADMIN** có claim `{"department_id":"*","access_level":100}`, nên thấy mọi chunk có `access_level <= 100`. Chunk thiếu `department`, `access_level` hoặc `is_public` trong payload thì **bị từ chối kể cả với SA**.
- Guest (`user_id=guest` trong log) chỉ thấy chunk public.

## 3. Ingest (`app/worker/tasks/ingestion.py`, `app/rag/chunking/strategy.py`, `app/rag/ingestion/parser.py`)

- Ingest thủ công qua wizard web. MinIO object key có dạng `<uuid>_<tên file gốc>` và chính là `source` / `object_key` của chunk. Khi đối chiếu với manifest, so theo **tên file** (`file_name`) hoặc `file_id` (khi file được upload dưới tên `<file_id>.pdf`).
- `chunk_id = "<document_id>:<chunk_index>"`.
- Chiến lược mặc định là `markdown_aware`: chia đệ quy 800 ký tự, chồng lấn 120. Bảng được chia theo dòng (`TableRowChunker`, tối đa 800 token). `heading_path` được ghép vào đầu content.
- Lúc embed, mỗi chunk có thêm `summary` và 3 câu hỏi mẫu (model EXTRACTION), làm `summary_vector` và `questions_vector`.
- **Không có OCR.** Parser dùng `pymupdf` / `pymupdf4llm`. PDF trong manifest có `quality` là `scanned`, `garbled_ocr`, `broken_encoding` hoặc `no_diacritics` sẽ ra chunk rỗng, rác hoặc mất dấu.
- Xem chunk thật trên server: `GET /api/v1/ai/documents/{document_id}/chunks/indexed` (cần quyền DOCUMENT_ALL hoặc DOCUMENT_CREATE), hoặc trang quản lý tài liệu trên web.

## 4. Log (stderr của uvicorn, format `%(asctime)s %(levelname)s %(name)s: %(message)s`)

Có:
- `unisage.graph`: `node=<tên> model=<model> conversation_id=… message_id=<id tin ASSISTANT> user_id=<id|guest> ip=…`, mỗi node một dòng.
- `unisage.graph`: `node_done=<tên> elapsed_ms=…` khi node kế tiếp bắt đầu; `first_token … ttft_ms=… total_ms=…`; `graph_done total_ms=…`.
- `unisage.graph`: `prompt node=03_MessageClassificationNode …:\n<JSON tasks>`. Chỉ khi `APP_DEBUG=true`.
- `unisage.graph`: `prompt node=06_QueryTransformationNode …:\n<text embed>`. **Chỉ khi `APP_DEBUG=true`.**
- `unisage.graph`: `prompt node=10_GenerationSynthesisNode …:\n<academic_metadata>\n<prepared_context>`. Chỉ khi `APP_DEBUG=true`. Mỗi chunk có dạng `  [n] (<object_key>, tr. X-Y) <content>`. Không có chunk thì là `(không có tài liệu liên quan)`.
- `app.graph.nodes.llm_rerank`: `LLM rerank SQ<n> '<sub-query>': kept [C<i> <tên file> > <heading> (<score>): <lý do>; …]; dropped [C<i> … (<score>); …]`, sau đó là `LLM rerank kept X of Y chunk(s); sub-queries without any: …`. Đây là **nơi duy nhất thấy được ứng viên kèm score**, nhưng chỉ là các ứng viên đã qua ngưỡng.
- `app.graph.nodes.web_search`: query, số trang và điểm.
- `app.graph.streaming_session`: `graph execution failed … ref=<8 hex>`. `ref` cũng hiện trong thông báo lỗi người dùng thấy.

KHÔNG có:
- Danh sách chunk của bước 08 và điểm trước ngưỡng 09. Chunk bị ngưỡng cắt thì không để lại dấu vết.
- `chunk_id`.
- Prompt đầy đủ và câu trả lời.
- Token. Token nằm ở bảng `request_usage_logs` / `request_usage_lines` bên Java, xem qua `GET /usage-logs`.

Hệ quả: tài liệu kỳ vọng không xuất hiện trong dòng rerank thì log **không phân biệt được** ba khả năng:
1. tài liệu chưa được ingest hoặc ingest ra chunk rỗng;
2. tài liệu có nhưng nằm ngoài top-k;
3. tài liệu nằm trong top-k nhưng dưới ngưỡng.

Phải kiểm tra thêm ở server (mục "Khi log không đủ" trong SKILL.md).

## 5. Prompt sinh câu trả lời (`app/rag/prompting/prompt_templates/`)

- `main/chat_academic_advisory.yaml` dùng cho 1 câu hỏi; `main/chat_multi_intent_synthesis.yaml` dùng khi có nhiều sub-query; `main/chat_ticket_fallback.yaml` dùng cho fallback.
- `common/task_1.yaml`: không dùng kiến thức ngoài; ý nào thiếu thì nói "hiện chưa có thông tin cho ý đó"; thiếu bằng chứng thì nói "hiện chưa xác nhận được quy định cụ thể" và gợi ý tạo ticket.
- `common/security_access_control.yaml`: dùng câu "Hiện chưa có quy định cụ thể về…" thay vì nhắc tới "ngữ cảnh".
- `common/citation_rules.yaml`: số liệu và điều kiện nào cũng phải có `[n]`; không được bịa chỉ số.
- `common/prepared_context.yaml`: khung `<academic_context>` và `<current_date>`. Model dùng ngày hiện tại để chọn năm học, nên tài liệu cũ hoặc mới có thể bị coi là hết hoặc chưa tới hiệu lực.
- `[n]` trong câu trả lời ứng với `[n]` trong context, đánh số theo thứ tự sau rerank. Web được đánh số tiếp sau chunk.

## 6. Cấu hình (`app/core/config.py`, `.env`)

| Biến | Mặc định | Ghi chú |
|---|---|---|
| `CHAT_RETRIEVAL_MAX_CHUNKS` | 16 | top-k cả lượt |
| `CHAT_CONTEXT_MAX_CHUNKS` | 8 | số chunk tối đa vào prompt |
| `CHAT_CLASSIFY_WITH_RETRIEVAL` | true | node 03 viết sẵn HyDE / sub-query cho node 06 |
| `CHAT_RERANK_SCORE_THRESHOLD` | 0.70 trong code | `.env` local đang để 0.3. Server có thể khác, cần hỏi |
| `CHAT_LLM_RERANK_ENABLED` | true | |
| `CHAT_LLM_RERANK_SNIPPET_CHARS` | 800 | rerank chỉ đọc đoạn đầu của chunk |
| `CHAT_WEB_SEARCH_ENABLED` | false | `.env` local đang để true |
| `APP_DEBUG` | true | tắt thì mất log phân loại, HyDE và context |

Tên model (CHAT / EMBEDDING / EXTRACTION) nằm trong bảng `chat_models` bên Java, trang admin "Cấu hình AI". Model hiện trong dòng `node=… model=…`.
