# Capability Map: UNISAGE-99 Calculation flow + Clarification panel

Nhánh: `feature/huydh-unisage-99-calculation-flow` (agent, backend, web dùng cùng tên nhánh).

## Ý định đã chốt (interview 09-10-2026)

- Sinh viên hỏi tính toán học vụ → bot **hiện công thức + thế số từng bước**; mọi phép tính do
  Python làm, LLM không tự tính.
- 3 công thức cài sẵn (GPA thang 4, ĐTKHP tích hợp LT/TH, quy đổi 10 → chữ → 4) nằm chung 1 file,
  không cần trích nguồn. Công thức khác lấy từ Qdrant; không tìm thấy → nói không tìm thấy, không bịa.
- Bỏ form chip cũ. Thay bằng **panel nhiều tab phía trên ô chat**, gộp câu hỏi của cả calculation
  lẫn advisory trong cùng lượt, bắt buộc trả lời hết mới gửi được, có nút Huỷ, reload vẫn còn panel.
- Sau khi gửi, lượt của user hiện thành **card có border** liệt kê câu hỏi → câu trả lời.
- Ngoài phạm vi: lấy điểm thật từ backend, trích nguồn cho 3 công thức cài sẵn.

## Modules

| Module id | Repo | Trách nhiệm | Phụ thuộc |
|---|---|---|---|
| `calc-engine` | agent | `app/calculation/formulas.py` (3 công thức + bảng quy đổi + làm tròn half-up), `expression.py` (evaluator AST an toàn cho công thức Qdrant), cả hai trả `CalculationResult` gồm từng bước đã thế số. Thuần Python, không LLM, không I/O. | — |
| `message-metadata` | backend | `StartTurnRequest.metadata` (chỉ key `clarification_answers`, dùng cho card) và endpoint nội bộ `PATCH /internal/messages/{id}/clarification` (chỉ dùng cho huỷ: `open → cancelled`). Cột jsonb đã có, không cần migration. Thêm bảng `calculation_traces` (chỉ staff, nhận qua `/internal/calculation-traces`), `POST /messages/{id}/calculation-feedback` (Đúng/Sai) và ticket `AI_CALCULATION_WRONG` riêng cho từng item (partial unique index). | — |
| `clarification-panel` | agent | Contract chặt (`ClarificationPanel`, `PendingRound` v2, `Answer`), state machine `OPEN/PROCESSING` có `claim_token` + lease trong `conversation_clarification_states`, SSE `clarification`/`clarification_closed`, `FenceRedactor` stateful, bản chiếu vào metadata. Xoá Guard so khớp text. | `message-metadata` |
| `calculation-node` | agent | Extractor (viết lại `calculation_extractor.yaml`), nhánh Qdrant (retrieve → LLM chép công thức → `expression.py`), prompt `main/chat_calculation.yaml`, nối vào graph, gộp kết quả với advisory trong node 10. | `calc-engine`, `clarification-panel` |
| `clarification-panel-ui` | web | Panel tab (choice có mô tả/Đề xuất/Khác, number, text, bảng môn), khoá composer, Huỷ/Esc, thu gọn; card có border cho câu trả lời; dựng lại panel từ metadata khi reload. Xoá `AskUserFormCard` + parse fence. | `clarification-panel` (contract) |

## Build order

```
calc-engine ─┐
             ├──► calculation-node
message-metadata ──► clarification-panel ─┤
                                          └──► clarification-panel-ui
```

1. `calc-engine` ∥ `message-metadata` (độc lập, làm song song)
2. `clarification-panel` (chốt contract SSE/request trước để web làm song song)
3. `calculation-node` ∥ `clarification-panel-ui`

## Specs

| Module id | Spec |
|---|---|
| `calc-engine` | `docs/specs/SPEC-calc-engine.md` |
| `clarification-panel` + `message-metadata` | `docs/specs/SPEC-clarification-panel.md` |
| `calculation-node` | `docs/specs/SPEC-calculation-node.md` |
| `clarification-panel-ui` | `unisage-web/docs/specs/SPEC-clarification-panel-ui.md` |

## Quyết định kiến trúc

- **Source of truth của panel là bảng `conversation_clarification_states` của agent.**
  `messages.metadata.clarification` bên Java chỉ là bản chiếu để web hiển thị và dựng lại sau reload.
  Agent không đọc bản chiếu để ra quyết định.
- **Chống submit cũ hoặc trùng** bằng state machine `OPEN → PROCESSING(claim_token, lease) → complete/restore`; không bao giờ dùng `NULL` cho "đang xử lý". Mọi lỗi 4xx của panel trả về
  trước `start_turn`, nên không tạo message và không trừ quota.
- **Bản chiếu không best-effort:** dữ liệu card đi cùng `start_turn` (cùng transaction); panel mới và lệnh huỷ đều phải PATCH thành công trước khi gửi event, lỗi thì thu hồi hoặc khôi phục state.
- **Huỷ không tạo lượt chat:** claim → PATCH bản chiếu qua `/internal/**`. Không quota, không LLM.
  Cờ bỏ qua quota trên `/messages/turn` bị loại, vì client có thể tự gọi endpoint đó qua route master.
- **Advisory vẫn báo thiếu thông tin bằng fence trong output của LLM**, nhưng `FenceRedactor` lọc fence
  ngay trong stream, nên fence không bao giờ tới client hay nằm trong `content` được lưu.
- **Con số chỉ do Python sinh ra, và Python render luôn khối tính.** LLM chỉ viết nhận xét.
- **Thứ tự triển khai:** backend → web (đọc được cả form legacy lẫn panel) → agent.
- **Công thức Qdrant không có eval gate:** công tắc `CHAT_CALC_RETRIEVED_FORMULA_ENABLED`, trace `metadata.calculation`, phản hồi Đúng/Sai tạo ticket `AI_CALCULATION_WRONG` (dùng lại màn support-tickets).
- **Module mới `app/calculation/`** (ngang hàng `app/rag/`), giữ đúng chiều `Graph -> services`.
