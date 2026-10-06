# Implementation Plan: Guardrails và dọn nợ sau đánh giá unisage-agent

## Overview

Sau khi rà unisage-agent (06-10-2026), còn 4 nhóm việc chưa có ai làm:

1. Prompt chưa coi văn bản quy chế là dữ liệu, và chưa có dấu hiệu nào ghi lại khi người dùng thử prompt injection.
2. Test fail vì phụ thuộc `.env` và dependency thật của máy dev.
3. Tài liệu kiến trúc và quyết định đã cũ, có chỗ mâu thuẫn với code.
4. Ngưỡng rerank chưa có số đo; việc này phải chờ bộ eval.

Plan này **không** làm lại bộ eval. Bộ eval đã có spec và todo riêng ở UNISAGE-95, nhánh
`feature/huydh-unisage-95-rag-evaluation` (`unisage-backend/changes/30-09-2026-RAG-Evaluation/`).
Phase 3 dùng `report.md` của UNISAGE-95 làm đầu vào.

## Architecture Decisions

- **Injection chỉ ghi log và đếm, không chặn.** Regex với tiếng Việt dễ bắt nhầm, nên phải đo tỉ lệ
  dương tính giả trước rồi mới quyết có chặn hay không. Lớp phòng thủ chính là prompt: mọi khối
  context được đánh dấu là dữ liệu, không phải chỉ dẫn.
- **Không kiểm tra grounding lúc runtime.** Câu trả lời đi qua SSE nên người dùng đã thấy hết trước
  khi kiểm tra xong. Faithfulness được đo offline bằng chỉ số #2 của UNISAGE-95.
- **Ngưỡng rerank chỉ đổi dựa trên `report.md`.** Trước khi có số đo, chỉ sửa tài liệu cho khớp
  với code (mặc định `0.70`); `.env` của máy dev không phải nguồn sự thật.
- **Không thêm giới hạn tần suất theo người dùng.** Usage limit hiện có đã lo phần này.
- **Một contract độ dài câu hỏi:** `ChatStreamRequest.message max_length=2000` → `400`/`4009`. Bỏ việc
  `sanitize_input_text` cắt âm thầm ở 1000 ký tự; không thêm biến cấu hình thứ hai cho cùng con số.
- **Test concurrency chạy trên SQLite file-backed** chỉ chứng minh upsert không ném lỗi; không thay thế kiểm
  chứng trên PostgreSQL production.
- **Viết lại docs cũ, không xoá.** README giữ quick start; CONTEXT.md và rag-pipeline.md mô tả
  đúng graph 11 node + Qdrant hiện tại và trỏ sang `docs/product/PRODUCT.md`.

## Task List

Chi tiết từng task ở `todo.md`.

### Phase 1: Guardrail và test (làm ngay, độc lập với nhau)
- [x] Task 1: Test không phụ thuộc môi trường máy
- [x] Task 2: Đánh dấu context là dữ liệu trong prompt
- [x] Task 3: Ghi log khi nghi prompt injection (log-only)
- [ ] Task 4: Một contract độ dài duy nhất cho câu hỏi (2000 ký tự, `4009`)

### Checkpoint 1
- [ ] `pytest` (bỏ e2e) xanh trên máy có `.env` dev, ruff/mypy không vượt baseline

### Phase 2: Docs
- [ ] Task 5: Sửa các tài liệu sản phẩm đang mâu thuẫn với code
- [ ] Task 6: Viết lại README.md, CONTEXT.md, rag-pipeline.md

### Checkpoint 2
- [ ] Không còn tài liệu nào nói pgvector, "provider-free fallback" hay "chưa có startup guard"

### Phase 3: Theo số đo (chặn bởi UNISAGE-95 Checkpoint 4)
- [ ] Task 7: Chốt `CHAT_RERANK_SCORE_THRESHOLD` từ kết quả eval
- [ ] Task 8: Ghi quyết định BM25 / cross-encoder theo số đo

### Checkpoint 3
- [ ] `PRODUCT.md` › Open không còn câu hỏi về ngưỡng; known-gaps cập nhật theo số đo

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mẫu injection tiếng Việt bắt nhầm câu hỏi hợp lệ ("bỏ qua môn này có sao không") | Low (chỉ log) | Mẫu phải là cụm nhắm vào hệ thống/chỉ dẫn, có test âm tính; xem log 1–2 tuần trước khi bàn chuyện chặn |
| Log chứa nguyên văn câu hỏi người dùng | Med | Chỉ log tên mẫu khớp, role, conversation id, độ dài; không log nội dung |
| Văn bản quy chế chứa chuỗi `</academic_context>` làm vỡ khung XML | Low | Task 2 escape thẻ đóng trong nội dung chunk và web |
| UNISAGE-95 chậm làm Phase 3 treo | Med | Phase 1–2 không phụ thuộc; Phase 3 tách hẳn |
| unisage-web không giới hạn ô nhập, người dùng gõ quá 2000 ký tự chỉ thấy lỗi chung `4009` | Low | Ngoài scope repo này; ghi lại để web thêm `maxLength=2000` |

## Open Questions

- ~~Giới hạn độ dài câu hỏi~~ → **Đã chốt:** 2000 ký tự theo schema, lỗi `4009` (review 06-10-2026).
- Có thêm nhóm câu hỏi `injection` vào bộ câu hỏi của UNISAGE-95 để đo Task 2 không? Nếu có, đó là
  việc của UNISAGE-95, không phải plan này.
