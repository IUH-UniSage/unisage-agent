# Plan: Giảm latency node retrieve (08_RetrievalFilteringNode)

## Vấn đề

Log 14-10-2026 09:12 (câu "học phí ngành cnnt", máy Windows): `08_RetrievalFilteringNode elapsed_ms=6323`.
Node này chỉ embed câu hỏi rồi gửi một request `query_batch_points` sang Qdrant, bình thường
khoảng 0.5-1.5s. Đây là node chậm nhất của lượt (classify 1.6s, rerank 2.7s, web 3.1s).

Đã loại trừ: `localhost` trên Windows chậm do thử IPv6 trước. Đo bằng `curl.exe` tới Qdrant:
`localhost` 40ms, `127.0.0.1` 27ms.

Chưa biết thời gian nằm ở đâu: không có log thời gian chi tiết trong node. Mỗi lượt hiện làm:

1. `deps.py` tạo `RetrievalService()` mới mỗi request, kéo theo embedder mới, SDK client mới,
   kết nối TCP + TLS mới tới API embedding.
2. `qdrant_store.get_client()` tạo `QdrantClient` mới mỗi lượt: một request kiểm tra phiên bản
   (vì vậy cảnh báo "Qdrant client 1.19.1 is incompatible" hiện ở mọi lượt), một request
   `collection_exists`, rồi mới tới request tìm kiếm.
3. `embed()` là hàm đồng bộ gọi từ bên trong event loop: `to_thread` → `ThreadPoolExecutor` →
   `asyncio.run` (event loop mới) → Redis client mới → reserve budget, acquire, gọi provider,
   release, settle, LPUSH usage (khoảng 5-6 lượt Redis tuần tự).

## Nguyên tắc

- Đo trước khi tối ưu sâu: T1 thêm log thời gian để lần chạy tiếp theo trên Windows chỉ ra chỗ
  chậm. T2-T4 là các sửa an toàn, đáng làm dù nguyên nhân chính là gì.
- Không đổi cách ghi usage và budget: embedding vẫn là một request `purpose=EMBEDDING` riêng như
  hiện nay.
- Không đổi kết quả retrieval (cùng vector, cùng filter quyền, cùng thứ tự).

## Task

### T1 - Log thời gian chi tiết của retrieval

- `RetrievalService`: log một dòng mỗi lượt `Retrieval timing: queries=N embed_ms=… qdrant_ms=…`.
- `OpenAIEmbedder` / `GoogleEmbedder` `_embed_recorded` (đường chat, không phải ingestion): log
  `budget_reserve_ms`, `provider_ms` (gồm acquire/release budget), `recorder_close_ms`.
- **AC:** một lượt chat in đủ hai dòng; `embed_ms - provider_ms` cho biết phần overhead
  (thread, event loop, Redis).
- **Files:** `app/rag/retrieval/service.py`, `app/rag/embeddings/openai_embedder.py`,
  `app/rag/embeddings/google_embedder.py`.
- **Size:** S.

### T2 - Một Qdrant client cho mỗi process, bỏ `collection_exists` trên đường chat

- `get_client()` trả về một client dùng chung cho mỗi process (theo `os.getpid()`, an toàn với
  Celery prefork). Kiểm tra phiên bản chỉ còn chạy một lần khi tạo client.
- `search_chunks_batch` và `fetch_chunks_by_ids` bỏ lời gọi `collection_exists`; collection chưa
  có (Qdrant trả 404) thì vẫn trả rỗng như cũ.
- **AC:** hai lần `get_client()` trong cùng process trả về cùng object; search khi collection
  chưa tồn tại trả `[[]…]`; lỗi Qdrant khác 404 vẫn được raise.
- **Files:** `app/rag/vectorstore/qdrant_store.py`, `tests/test_retrieval.py`,
  `tests/test_qdrant_store.py`.
- **Size:** S.

### T3 - Embed trên event loop của server (`embed_async`)

- Thêm `embed_async()` vào `EmbeddingProvider` và hai embedder: cùng phần ghi usage và budget như
  `embed()` nhưng await trên loop hiện tại, không dùng thread + `asyncio.run`. Lần resolve
  credential đầu tiên (có thể probe provider và Qdrant, đồng bộ) chạy trong `to_thread`.
- `RetrievalService.retrieve_many_async()`: `await embed_async`, Qdrant search trong `to_thread`.
  Node `retrieve_chunks` gọi hàm này. `retrieve_many()` đồng bộ giữ nguyên cho caller khác.
- Điều kiện để làm T4: client async của Gemini (`httpx.AsyncClient`) gắn với event loop đã mở
  kết nối, nên chỉ dùng chung được khi mọi lần embed chạy trên cùng một loop.
- **AC:** test graph và retrieval hiện có vẫn pass; `embed_async` ghi đúng một usage line như
  `embed()`.
- **Files:** `app/rag/embeddings/provider.py`, `app/rag/embeddings/openai_embedder.py`,
  `app/rag/embeddings/google_embedder.py`, `app/rag/retrieval/service.py`,
  `app/graph/nodes/retrieval_filtering.py`, `tests/llm_mocks.py`, `tests/test_retrieval.py`.
- **Size:** M.

### T4 - Dùng lại `RetrievalService` giữa các request

- `deps.py`: một `RetrievalService` dùng chung, tạo lại khi `snapshot.version` đổi (đổi
  credential embedding thì embedder mới). Giữ kết nối keep-alive tới API embedding, bỏ TCP + TLS
  handshake ở mỗi lượt.
- **AC:** hai request cùng snapshot nhận cùng instance; snapshot đổi thì nhận instance mới.
- **Files:** `app/api/deps.py`, `tests/api/test_deps.py`.
- **Size:** S.
- **Phụ thuộc:** T3.

### Checkpoint - đo trên Windows

- Chạy lại câu "học phí ngành cnnt", gửi các dòng `Retrieval timing` và `Embedding timing`.
- Quyết định T5 theo số liệu:
  - `budget_reserve_ms` / `recorder_close_ms` lớn → Redis là nguyên nhân (Redis có đang chạy và
    truy cập được từ Windows không?).
  - `provider_ms` lớn → do provider embedding (vị trí server, model); cân nhắc đổi model/region.
  - `qdrant_ms` lớn → kiểm tra Qdrant (index, payload, phiên bản server 1.15 vs client 1.19).

## Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Client Qdrant dùng chung bị mang qua fork (Celery prefork) | Trung bình | Cache theo pid, process con tự tạo client mới |
| Client async Gemini dùng chung giữa nhiều event loop | Cao | T3 đưa embed của chat về loop server trước khi làm T4; `retrieve_many()` đồng bộ không dùng instance dùng chung |
| Bỏ `collection_exists` làm lộ lỗi khi collection chưa tạo | Thấp | Bắt 404 trả rỗng, có test |

## Câu hỏi mở

- Redis trên máy Windows có chạy không (`Test-NetConnection 127.0.0.1 -Port 6379`)? Nếu Redis
  không chạy, mỗi lượt budget sẽ thất bại rồi bỏ qua (fail open), có thể mất vài giây.
- Nâng Qdrant server lên 1.18+ hay pin client về 1.15.x cho khớp phiên bản?
