# Todo: Giảm latency node retrieve

Chi tiết (AC, files, size) xem `plan.md`.

- [x] T1 Log thời gian: `Retrieval timing` (embed_ms / qdrant_ms) + `Embedding timing` (reserve / provider / close)
- [x] T2 Một Qdrant client mỗi process, bỏ `collection_exists` trên đường chat (404 → rỗng), test
- [x] T3 `embed_async` cho hai embedder + `retrieve_many_async`, node retrieve await trên loop server, test (gồm `evals/run/recording.py`)
- [x] T4 `RetrievalService` dùng chung theo `snapshot.version` trong `deps.py`, test
- [x] Checkpoint: pytest (bỏ `tests/e2e`, đã lỗi import từ trước) 2104 passed; ruff/format sạch trên file đã sửa; mypy không thêm lỗi mới (437 lỗi có sẵn)
- [ ] Checkpoint: đo lại trên Windows, gửi log timing → quyết định T5

Đã loại trừ (14-10-2026): `localhost` chậm do IPv6 (curl 40ms vs 27ms); Redis không chạy (`Test-NetConnection 127.0.0.1 -Port 6379` thành công).
