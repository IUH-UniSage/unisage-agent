# Implementation Plan: UNISAGE-99 Calculation flow + Clarification panel

## Overview

Thay placeholder `CalculationNode` bằng luồng tính toán thật, trong đó Python làm mọi phép tính. Có 3
công thức cài sẵn, còn lại lấy công thức từ Qdrant và phải qua kiểm tra provenance. Song song với đó,
làm lại toàn bộ cơ chế hỏi lại người dùng: bỏ form JSON trong text và Clarification Guard so khớp text,
thay bằng panel có cấu trúc gồm nhiều tab, gộp câu hỏi của cả calculation lẫn advisory, bắt buộc trả lời
hết hoặc huỷ.

Phạm vi trải trên 3 repo, cùng tên nhánh `feature/huydh-unisage-99-calculation-flow`:

- **agent:** gần như toàn bộ công việc.
- **backend:** 1 endpoint nội bộ.
- **web:** panel, card có border, đường đọc legacy.

Nguồn yêu cầu: [capability-map.md](capability-map.md) và 4 spec:
- `docs/specs/SPEC-calc-engine.md`
- `docs/specs/SPEC-clarification-panel.md`
- `docs/specs/SPEC-calculation-node.md`
- `unisage-web/docs/specs/SPEC-clarification-panel-ui.md`

Chi tiết từng task ở [todo.md](todo.md).

## Architecture Decisions

Bản tóm tắt; lý do đầy đủ nằm trong capability map và các spec.

- **State của panel nằm ở `conversation_clarification_states` (agent).** State machine
  `OPEN → PROCESSING(claim_token, lease) → complete/restore`; không dùng `NULL` cho "đang xử lý".
  Metadata bên Java chỉ là bản chiếu, nhưng **không best-effort**: dữ liệu card đi cùng `start_turn`,
  còn panel mới và lệnh huỷ phải PATCH thành công thì mới gửi event.
- **Mọi lỗi 4xx của panel trả về trước `start_turn`.** Huỷ không tạo lượt chat (không quota, không LLM)
  mà đi qua endpoint `/internal/**` mới bên Java.
- **`FenceRedactor` có trạng thái**, lọc `ask_user_form`/`confirmed_metadata` ngay trong stream.
- **Con số chỉ do `app/calculation/` sinh ra, và khối tính do Python render.** Node 10 không nhận số.
  Nhận xét không stream, có số ngoài whitelist thì thay bằng câu cố định. Barrier trước node 10 giữ đúng
  thứ tự khối tính → advisory.
- **Fail closed:** extractor lỗi thì không rơi sang Qdrant; công thức cài sẵn được định tuyến bằng luật
  trước LLM.
- **Mỗi câu hỏi là một tab, tối đa 12 tab mỗi panel, tối đa 3 panel liên tiếp** cho cùng một câu hỏi
  gốc.
- **Thứ tự triển khai: backend → web → agent.** Web đọc được cả form legacy lẫn panel mới.

## Thứ tự và song song hoá

```
Phase 1  T1 ─ T2 ─ T3 (calc-engine)        T4 (backend)          ← song song hoàn toàn
Phase 2  T5 (contract) ─► T6 ─► T7 ─► T8 ─► T9 ─► T10            ← agent, tuần tự (chung state/graph)
Phase 3  T11 ─► T12 ─► T13 ─► T14 (web)                          ← bắt đầu ngay sau T5 (contract chốt)
Phase 4  T15 ─► T16 ─► T17 ─► T18 ─► T19 (calculation-node)      ← cần T1–T3 và T10
Phase 5  T21 (backend feedback) ∥ T22 (web feedback) ─► T20 (docs + rollout)
```

Task có rủi ro cao được đưa lên sớm:
- **T3** (evaluator an toàn) nằm ở Phase 1.
- **T8** (`FenceRedactor`) là task đầu tiên chạm graph.
- **T6** (state machine claim/lease) làm trước mọi luồng cần đến nó.

## Task List

### Phase 1: Nền tảng (song song)
- [x] T1: calc-engine: rounding, bảng quy đổi, `grade_conversion`, `course_score`
- [x] T2: calc-engine: `gpa`, `ParamSpec`/`missing_params`, `render_markdown`
- [x] T3: calc-engine: evaluator allowlist cho công thức Qdrant
- [x] T4: backend: `StartTurnRequest.metadata` + `PATCH /internal/messages/{id}/clarification` (huỷ)

### Checkpoint 1
- [ ] `tests/calculation` xanh, coverage `app/calculation` ≥ 95%; test backend xanh
- [ ] Review với người dùng: các bước hiển thị của 3 công thức (snapshot render)

### Phase 2: Contract panel (agent)
- [x] T5: Schema v2 + `validate_answers` + `contracts/chat-sse.md`
- [x] T6: Migration + state machine `OPEN/PROCESSING` (claim_token, lease)
- [x] T7: Định tuyến đầu lượt (bảng 2.2), mã lỗi mới, luồng huỷ
- [x] T8: `FenceRedactor` + nối vào generation
- [x] T9: Advisory sinh panel: captured asks → `PendingRound`, thứ tự upsert → event → PATCH
- [x] T10: Luồng submit + resume advisory; gỡ Guard/retry/carry-forward

### Checkpoint 2
- [ ] Luồng advisory hỏi lại chạy end-to-end bằng contract mới (e2e với fake LLM)
- [ ] Không test nào thấy `ask_user_form` trong token hay `content`; toàn bộ test agent xanh

### Phase 3: Web
- [x] T11: Schema zod, SSE events, request variants, `deriveOpenPanel`, legacy read path
- [x] T12: Khung panel + câu hỏi choice/text + khoá composer + huỷ + ADR 0003
- [x] T13: Câu hỏi number/number_list/course_table + validate client + nháp `sessionStorage`
- [x] T14: Card có border + xử lý 4010/4091/4092 + e2e + screenshot

### Checkpoint 3
- [ ] Web chạy với **agent cũ** không lỗi (legacy) và với agent nhánh này (panel)
- [ ] `lint`, `format:check`, `typecheck`, `build`, `test`, `test:e2e` xanh; có screenshot 375/1280

### Phase 4: Calculation node (agent)
- [x] T15: Prompts: viết lại extractor, `calculation_formula`, `chat_calculation`, sửa task_2/guide
- [x] T16: `run_calculation_task`, nhánh công thức cài sẵn
- [x] T17: Nhánh Qdrant + kiểm tra provenance
- [x] T18: Graph wiring: thứ tự stream, gộp panel, nhận xét, `calculation_results` cho node 10
- [x] T19: Resume calculation + chain depth + e2e luồng tính

### Checkpoint 4
- [ ] Đủ 4 tiêu chí "Success Criteria" của SPEC-calculation-node
- [ ] Chạy tay trên máy dev (đủ 3 repo): ĐTKHP thiếu điểm TH, GPA với bảng môn, lượt lẫn advisory + calculation, Huỷ, reload

### Phase 5: Hoàn tất
- [x] T21: backend: `POST /messages/{id}/calculation-feedback` + ticket type `AI_CALCULATION_WRONG`
- [x] T22: web: nút Đúng/Sai + popover lý do
- [x] T20: Docs (known-gaps, PRODUCT, DECISIONS, rag-pipeline) + ghi chú rollout

### Checkpoint cuối
- [ ] Mọi acceptance criteria đã tick; review với người dùng trước khi mở PR (3 PR, ghi rõ thứ tự merge)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LLM chép sai **ý nghĩa** công thức Qdrant mà câu trích vẫn đúng | High | 7 kiểm tra fail-closed; nhãn "Kết quả tham khảo theo quy chế" kèm câu trích và nguồn; **không có eval gate** (đã chốt). Thay bằng công tắc tắt khẩn cấp, trace đầy đủ và phản hồi Đúng/Sai tạo ticket để admin sửa |
| Sinh viên không nhận ra kết quả sai nên không bấm Sai | Med | Đã chấp nhận; trace vẫn đủ để admin chủ động rà soát các ticket hoặc metadata sau này |
| Fence lọt ra stream khi bị chia chunk lạ | High | Test cắt fence ở mọi vị trí ký tự (T8) |
| Triển khai agent trước web khiến sinh viên kẹt ở `409` | High | Thứ tự backend → web → agent ghi trong PR và `DECISIONS.md`; Checkpoint 3 kiểm tra web với agent cũ |
| State machine claim chỉ được test trên SQLite | Med | Câu SQL giống hệt trên Postgres; chạy tay một lần trên Postgres dev ở Checkpoint 4 |
| Panel 12 tab quá dài trên mobile | Med | Tab list cuộn ngang bên trong, `max-h-[60vh]`; screenshot 375px ở T14 |
| Prompt snapshot và test cũ của Guard bị vỡ hàng loạt | Low | T10 xoá test của code bị gỡ cùng commit với code; cập nhật snapshot có chủ đích |
| LLM nhận xét bịa số | Med | Không stream; số ngoài whitelist thì thay bằng câu cố định (T18); node 10 không nhận số |
| Router luật bắt nhầm hoặc bỏ sót | Med | Bộ ≥ 10 câu khớp và ≥ 10 câu không khớp mỗi công thức (T16); log `router_disagreement` để tinh chỉnh |

## Open Questions

Không còn câu hỏi mở. Các quyết định ngày 09-10-2026 đã được ghi vào mục "Decisions" của từng spec.
