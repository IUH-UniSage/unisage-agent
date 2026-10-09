# Todo: UNISAGE-99 Calculation flow + Clarification panel

Nhánh: `feature/huydh-unisage-99-calculation-flow` (agent, backend, web). Commit dùng scope kèm mã
ticket, ví dụ `feat(calculation): [UNISAGE-99] add grade conversion`. Không trộn docs, migration và
code ứng dụng trong một commit (`AGENTS.md`).

Lệnh chung:
- **Agent:** `.venv/bin/pytest -q <path>` (toàn bộ: `.venv/bin/pytest -q --ignore=tests/e2e`) ·
  `.venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests && .venv/bin/mypy app`
- **Backend:** `./mvnw test -Dtest=<Class>` (toàn bộ: `./mvnw test`)
- **Web:** `pnpm test -- <path>` · `pnpm lint && pnpm format:check && pnpm typecheck && pnpm build` · `pnpm test:e2e`

---

## Phase 1: Nền tảng

## T1: calc-engine: rounding, bảng quy đổi, `grade_conversion`, `course_score`

**Description:** Tạo `app/calculation/` với `result.py` (`Step`, `CalculationResult`, `FieldError`,
`CalculationInputError`) và `formulas.py`. File `formulas.py` chứa mọi hằng số nghiệp vụ
(`GRADE_SCALE`, trọng số 20/30/50, số chữ số làm tròn, giới hạn input), cùng các hàm `round_half_up`,
`fmt` (tối đa 2 chữ số, kèm `≈`), `grade_conversion` và `course_score`. Xem SPEC-calc-engine
§Business contract.

**Acceptance criteria:**
- [x] Ví dụ (TBtx 8, GK 7, CK 6.5, TH [9, 8], 2 TC LT + 1 TC TH) cho ra `7.5 / B / 3.0`, có đủ các bước ĐLT, ĐTH, ĐTKHP và quy đổi
- [x] Mọi biên của bảng quy đổi và các ca half-up trong spec đều pass; không có `float` nào trong module
- [x] Các trường hợp chỉ có LT, chỉ có TH, tổng tín chỉ bằng 0, điểm ngoài khoảng đều raise `CalculationInputError` đúng field

**Verification:** `.venv/bin/pytest -q tests/calculation/test_formulas.py` · ruff/mypy `app/calculation`

**Dependencies:** Không · **Files:** `app/calculation/{__init__,result,formulas}.py`, `tests/calculation/test_formulas.py` · **Scope:** M

## T2: calc-engine: `gpa`, `ParamSpec`/`missing_params`, `render_markdown`

**Description:** Thêm `gpa` (nhập điểm số hoặc điểm chữ, môn F vẫn tính, 1–30 môn, làm tròn 2 chữ số),
`ParamSpec` cho 3 công thức, `missing_params` (có `required` có điều kiện, ví dụ `th` chỉ bắt buộc khi
`tcth > 0`), dispatcher `calculate`, và `render.py` với `render_markdown`.

**Acceptance criteria:**
- [x] GPA đúng với tổ hợp có môn F và có điểm chữ; 31 môn bị từ chối
- [x] `missing_params` trả về đúng thứ tự và đúng điều kiện
- [x] Có snapshot markdown cho 3 công thức; cùng input luôn ra cùng output

**Verification:** `.venv/bin/pytest -q tests/calculation` · coverage `app/calculation` ≥ 95%

**Dependencies:** T1 · **Files:** `app/calculation/{formulas,render}.py`, `tests/calculation/{test_formulas,test_render}.py`, `tests/calculation/snapshots/*` · **Scope:** M

## T3: calc-engine: evaluator allowlist cho công thức Qdrant

**Description:** Viết `app/calculation/expression.py` gồm `FormulaVariable`, `RetrievedFormula`,
`validate` và `evaluate`, theo SPEC-calc-engine §expression.py: grammar allowlist, giới hạn
độ dài/node/độ sâu/literal, `Decimal` localcontext có trap, kiểm tra biến khớp hai chiều, `round` dùng
half-up.

**Acceptance criteria:**
- [x] Toàn bộ danh sách chuỗi tấn công trong spec bị `validate` từ chối (không chuỗi nào được `eval`/`compile`)
- [x] Chia cho 0 raise lỗi với field rõ ràng; kết quả vượt `1e12` bị từ chối
- [x] Công thức hợp lệ (`so_tc * don_gia + phi`, có `round`/`min`/`max`) tính đúng và có các bước

**Verification:** `.venv/bin/pytest -q tests/calculation/test_expression.py`

**Dependencies:** T1 (dùng `result.py`, `round_half_up`) · **Files:** `app/calculation/expression.py`, `tests/calculation/test_expression.py` · **Scope:** S

## T4: backend: `StartTurnRequest.metadata` + `PATCH /internal/messages/{id}/clarification`

**Description:** Theo SPEC-clarification-panel §7. Repo `unisage-backend`, tạo nhánh cùng tên.
- §7.1: `POST /messages/turn` nhận `metadata` (chỉ key `clarification_answers`, tối đa 32 KB) và lưu
  vào message USER trong cùng transaction.
- §7.2: endpoint nội bộ chỉ dùng cho huỷ (`open → cancelled`).

**Acceptance criteria:**
- [x] `start_turn` có `clarification_answers` thì message USER mang đúng metadata đó; key lạ hoặc > 32 KB trả `400`; không có metadata thì hành vi y như cũ
- [x] Endpoint huỷ: thiếu secret bị chặn; sai conversation, message USER hoặc không có panel đều trả `404`; `cancelled` hai lần là idempotent; chuyển trạng thái khác trả `409`; các key khác trong metadata giữ nguyên
- [x] Không đụng `content`/`status`/usage của message ASSISTANT

**Verification:** `./mvnw test -Dtest='*Clarification*,MessageControllerTest,MessageServiceImplTest'`

**Dependencies:** Không · **Files:** `dto/request/StartTurnRequest.java`, `controller/internal/InternalMessageController.java` (hoặc controller internal sẵn có), `dto/request/internal/CancelClarificationRequest.java`, `service/conversation/MessageService{,Impl}.java`, tests · **Scope:** M

### Checkpoint 1
- [x] `tests/calculation` xanh, coverage ≥ 95%; `./mvnw test` xanh (trừ `ModelPricingServiceImplTest.getHistory_filtersAndPagesNewestFirst`, fail sẵn trên `main`)
- [ ] Người dùng duyệt snapshot render của 3 công thức

---

## Phase 2: Contract panel (agent)

## T5: Schema v2 + `validate_answers` + `contracts/chat-sse.md`

**Description:** Viết lại `app/schemas/clarification.py` theo SPEC-clarification-panel §1:
- `ChoiceOption`, `NumberConstraint`, `Question` (validator theo kind), `ClarificationPanel`,
  `PublicClarificationPanel`
- `PendingAdvisoryTask`, `PendingCalculationTask` (với `CalculationPlan` tạm là model tối thiểu),
  `PendingRound`
- `Answer`, `ClarificationSubmit`/`Cancel`, và `ChatStreamRequest` mới ở `app/schemas/chat.py`

Thêm `validate_answers`. Schema phải khớp `contracts/chat-sse.md` (contract canonical, **đã viết**); test
dùng chính các ví dụ JSON trong contract làm fixture. PendingClarification
v1 **chưa** xoá ở task này (để T10 xoá); đặt tên mới song song để code cũ vẫn chạy.

**Acceptance criteria:**
- [x] Mọi model đều `extra="forbid"`; mỗi kind có ca hợp lệ và không hợp lệ; panel có 12 câu hợp lệ, 13 câu bị từ chối
- [x] `validate_answers` bắt được: thiếu, trùng, id lạ, sai kind, ngoài khoảng, sai step, "Khác" khi `allow_other=False`, bảng 31 dòng
- [x] `ChatStreamRequest` bắt buộc có đúng một trong `message` hoặc `clarification`

**Verification:** `.venv/bin/pytest -q tests/schemas/test_clarification_schema.py tests/api/test_chat_stream_endpoint.py`

**Dependencies:** Không (có thể làm song song Phase 1) · **Files:** `app/schemas/{clarification,chat}.py`, `app/graph/clarification_answers.py`, tests (fixture từ `contracts/chat-sse.md`) · **Scope:** M

## T6: Migration + state machine `OPEN/PROCESSING` trong repository

**Description:** Alembic migration thêm 4 cột `pending_status`, `pending_panel_id`, `claim_token`,
`claim_expires_at` (SPEC §2.1). Thêm setting `CHAT_CLAIMED_TURN_DEADLINE_SECONDS = 150` và
`CHAT_CLARIFICATION_LEASE_SECONDS = 210`, kèm validator từ chối khởi động khi `lease < deadline + 60`.
`ClarificationStateRepository` có thêm:
- `get_round`: row legacy → None, kèm log; lease hết hạn thì dọn row
- `upsert_open(round)`: chỉ ghi khi không có round
- `claim(cid, panel_id, token) -> PendingRound | None`
- `restore(token)` và `complete(token, new_round | None)`: đều có điều kiện `claim_token`
- `revoke_open(panel_id)`: dùng khi PATCH finalize lỗi

**Acceptance criteria:**
- [x] Claim `OPEN → PROCESSING`; claim sai `panel_id` hoặc đang `PROCESSING` thì trả `None`; hai claim đồng thời chỉ một thắng
- [x] `restore`/`complete` với token sai không ghi được gì; lease hết hạn bị dọn; `upsert_open` không ghi đè round đang có
- [x] `alembic upgrade head` rồi `downgrade -1` chạy sạch

**Verification:** `.venv/bin/pytest -q tests/database/test_clarification_state_repository.py` · `.venv/bin/alembic upgrade head`

**Dependencies:** T5 · **Files:** `migrations/versions/2026_10_09_*_add_pending_panel_id.py`, `app/database/models.py`, `app/database/repositories/clarification_state.py`, test · **Scope:** M

## T7: Định tuyến đầu lượt, mã lỗi mới, luồng huỷ

**Description:** Trong `chat.py`, luôn đọc state **trước** `start_turn` và áp dụng đủ 9 dòng của bảng
SPEC §2.3. Thêm 4 mã lỗi (`4010`, `4091`, `4092`, `4093`) và giới hạn body 16 KB (`413`). Thêm
`ClarificationClosedItem` và serializer cho `clarification_closed`. Thêm
`BackendJavaClient.cancel_clarification(...)` (thử lại 3 lần, backoff 200/400/800 ms). Luồng huỷ theo
§2.5: claim → PATCH (**bắt buộc thành công**) → `complete` → `clarification_closed` → `done`; PATCH lỗi
thì `restore` rồi trả `503`. Ở task này, submit hợp lệ tạm trả `501`; T10 sẽ thay.

**Acceptance criteria:**
- [x] Đủ các dòng của bảng 2.3 (trừ submit hợp lệ) có test; mọi lỗi 4xx không gọi `start_turn`
- [x] Huỷ không gọi `start_turn`, không gọi LLM, không ghi usage; PATCH lỗi 3 lần thì trả `503` và state về `OPEN`
- [x] Body 17 KB nhận `413`

**Verification:** `.venv/bin/pytest -q tests/api/test_chat_stream_clarification.py tests/api/`

**Dependencies:** T4 (contract), T6 · **Files:** `app/api/v1/chat.py`, `app/core/errors/error_codes.py`, `app/graph/queue_items.py`, `app/integrations/backend_java_client.py`, `tests/api/test_chat_stream_clarification.py` · **Scope:** M

## T8: `FenceRedactor` + nối vào generation

**Description:** Viết `app/graph/fence_redactor.py` theo SPEC §4. Bọc quanh `token_sink` của
`run_generation_synthesis` và luồng multi-intent; `response_text` lấy từ phần đã redact. Khối do
`_repair_missing_ask_form` sinh ra đi thẳng vào phần đã bắt. Sửa `build_missing_metadata_block` để gửi
label thật cho LLM.

**Acceptance criteria:**
- [x] Cắt fence ở mọi vị trí ký tự (và cả từng ký tự một chunk): output hiện ra luôn giống nhau và không chứa `ask_user_form`
- [x] Code block thật và JSON khác được giữ nguyên; fence chưa đóng được xử lý đúng
- [x] `GraphOutput.response_text` và token gửi ra đều không có fence; `confirmed_metadata` vẫn được cập nhật như trước

**Verification:** `.venv/bin/pytest -q tests/graph/test_fence_redactor.py tests/graph/test_generation_synthesis_node.py tests/rag`

**Dependencies:** Không (có thể làm song song T5–T7) · **Files:** `app/graph/fence_redactor.py`, `app/graph/nodes/generation_synthesis.py`, `app/rag/prompting/builder.py`, tests · **Scope:** M

## T9: Advisory sinh panel

**Description:** Chuyển các khối `ask_user_form` đã bắt thành `Question(origin="advisory", kind="choice", allow_other=True)`,
rồi dựng `PendingRound` (hàm `build_round` dùng chung với T18). `GraphOutput` có thêm `pending_round`.
Trong `streaming_session.run_and_persist`, thứ tự theo SPEC §2.6: ghi state `OPEN` → PATCH finalize
kèm `metadata.clarification` (`open`, thử lại 3 lần) → `ClarificationItem` (`event: clarification`) →
`done`. Ghi state lỗi thì PATCH không kèm panel; PATCH lỗi thì `revoke_open` và không gửi event.

**Acceptance criteria:**
- [x] Lượt advisory thiếu thông tin chỉ gửi một event `clarification`, và event đó đi sau cả ghi state lẫn PATCH thành công
- [x] PATCH finalize mang `content` đã redact và metadata `open` với đúng `panel_id` đang lưu trong state
- [x] PATCH lỗi thì round bị thu hồi và không có event `clarification`; ghi state lỗi thì không có metadata panel

**Verification:** `.venv/bin/pytest -q tests/graph/test_streaming_session.py tests/graph/test_graph_wiring.py`

**Dependencies:** T5, T6, T8 · **Files:** `app/graph/clarification_round.py`, `app/graph/streaming_state.py`, `app/graph/streaming_session.py`, `app/graph/streaming_graph.py`, tests · **Scope:** M

## T10: Luồng submit + resume advisory; gỡ Guard

**Description:** Submit hợp lệ theo SPEC §2.4: claim (`PROCESSING`) → `start_turn` với bản tóm tắt
và `metadata.clarification_answers` (lỗi thì `restore(token)` rồi trả lỗi) → resume các
`PendingAdvisoryTask` (merge `confirmed_metadata`, chạy lại `origin_task`) → `complete(token, new_round | None)`. Có `chain_depth`: đến 3
thì không hỏi thêm. Xoá `resolve_clarification_guard`, `_match_reply`, `retry_count`,
`CHAT_CLARIFICATION_MAX_RETRY`, `_carry_forward_unanswered`, `_resume_retrieval_query`,
PendingClarification v1, cùng các test của chúng.

**Acceptance criteria:**
- [x] Submit hai lần: lần đầu chạy, lần sau `409`; chỉ có một message USER, và nó mang `clarification_answers`
- [x] `message` gửi trong lúc đang xử lý nhận `4093`; `start_turn` lỗi `429` thì state về `OPEN` (đúng token)
- [x] Phần sau claim nằm trong `asyncio.timeout(deadline)`: quá hạn thì cancel, không có PATCH và không ghi state; mất quyền sở hữu (`still_owner`) trước finalize thì bỏ PATCH
- [x] Không còn tham chiếu nào tới Guard/retry trong `app/`; toàn bộ test agent xanh

**Verification:** `.venv/bin/pytest -q --ignore=tests/e2e` · `.venv/bin/pytest -q tests/e2e/test_advisory_flow_e2e.py` · ruff/mypy

**Dependencies:** T7, T9 · **Files:** `app/api/v1/chat.py`, `app/graph/streaming_graph.py`, `app/graph/nodes/security_context.py`, `app/graph/nodes/generation_synthesis.py`, `app/core/config.py` (+ xoá test tương ứng) · **Scope:** M (chủ yếu là xoá)

### Checkpoint 2
- [ ] E2E advisory: hỏi → panel → submit → trả lời; reload metadata đúng
- [x] Không có `ask_user_form` trong token hay `content`; toàn bộ test agent xanh; ruff/mypy không vượt baseline
- [ ] Review với người dùng trước khi sang Phase 4

---

## Phase 3: Web (repo `unisage-web`, bắt đầu được ngay sau T5)

## T11: Schema zod, SSE events, request variants, `deriveOpenPanel`, legacy

**Description:** `schemas/clarification-schemas.ts` (`.strict()`); thêm `onClarification` và
`onClarificationClosed` cho `use-chat-stream.ts`; 3 dạng body của `/chat/stream`; `use-chat-workspace`
ghi `metadata.clarification` vào message đang stream khi nhận event; `utils/clarification-state.ts` (`deriveOpenPanel`, `readAnsweredCard`); đổi tên `ask-user-form.ts` thành `legacy-ask-user-form.ts` (chỉ giữ strip và parse read-only).

**Acceptance criteria:**
- [x] `deriveOpenPanel`/`readAnsweredCard` đúng cho các trạng thái open/cancelled, message cuối là USER, message ERROR, metadata hỏng
- [x] Stream có event `clarification` thì message trong cache có metadata `open`; event lạ bị bỏ qua
- [x] Message cũ có fence: text không lộ JSON

**Verification:** `pnpm test -- src/features/chat` · `pnpm typecheck`

**Dependencies:** T5 (contract) · **Files:** `schemas/clarification-schemas.ts`, `hooks/use-chat-stream.ts`, `hooks/use-chat-workspace.ts`, `utils/clarification-state.ts`, `utils/legacy-ask-user-form.ts` (+ tests) · **Scope:** M

## T12: Khung panel + choice/text + khoá composer + huỷ + ADR

**Description:** `clarification-panel.tsx` (Tabs variant "line", ⌄ thu gọn, ✕/Esc huỷ có xác nhận khi
đã có nháp, nút gửi kèm số câu còn thiếu), `question-choice.tsx` (thêm shadcn `radio-group`; mỗi option
có label, mô tả, "(Đề xuất)", "Khác" kèm input), `question-text.tsx`. Đặt panel trong
`ActiveConversation`, và disable `ChatComposer` khi panel đang mở. Viết `docs/adr/0003-clarification-panel.md`.
Xoá `AskUserFormCard`.

**Acceptance criteria:**
- [x] Nút gửi chỉ bật khi đủ câu trả lời; "Khác" bắt buộc có text; dùng được hoàn toàn bằng bàn phím
- [x] Huỷ gửi `{action: "cancel"}`; composer mở lại ngay sau `clarification_closed`
- [x] Không dùng màu hex tuỳ ý; product tour anchor còn nguyên

**Verification:** `pnpm test -- src/features/chat/components/clarification` · `pnpm lint && pnpm typecheck`

**Dependencies:** T11 · **Files:** `components/clarification/{clarification-panel,question-choice,question-text}.tsx`, `components/chat-conversation-content.tsx`, `components/ui/radio-group.tsx`, `docs/adr/0003-clarification-panel.md` · **Scope:** M

## T13: number/number_list/course_table + validate client + nháp

**Description:** `question-number.tsx`, `question-number-list.tsx`, `question-course-table.tsx` (dạng
thẻ trên mobile); `utils/clarification-answers.ts` (cùng quy tắc với server, dựng `Answer[]`); nháp
lưu `sessionStorage` theo `panel_id`, có try/catch.

**Acceptance criteria:**
- [x] Ngoài khoảng hoặc sai step thì báo lỗi ngay trong tab; bảng môn thêm/xoá được dòng, tối đa 30, nhận điểm chữ
- [x] Reload thì nháp được khôi phục; `sessionStorage` lỗi thì panel vẫn chạy
- [x] `Answer[]` dựng ra khớp contract (test so với fixture lấy từ `contracts/chat-sse.md`)

**Verification:** `pnpm test -- src/features/chat`

**Dependencies:** T12 · **Files:** `components/clarification/{question-number,question-number-list,question-course-table}.tsx`, `utils/clarification-answers.ts` (+ tests) · **Scope:** M

## T14: Card có border + xử lý lỗi + e2e + screenshot

**Description:** `answered-clarification-card.tsx` thay bong bóng USER khi `readAnsweredCard` có dữ
liệu (message USER lạc quan cũng mang sẵn `clarification_answers`). Xử lý lỗi theo contract §4: `4010`,
`4091`, `4092`, `4093`, và `503` khi huỷ. E2E
`e2e/chat-clarification.spec.ts` (desktop + mobile), chụp screenshot ở 375 và 1280.

**Acceptance criteria:**
- [x] Sau khi gửi, card có border hiện đúng các cặp câu hỏi → câu trả lời (bảng môn dùng `ui/table`); reload vẫn y hệt
- [x] Reload khi panel đang mở: panel hiện lại → Huỷ → composer mở lại
- [x] `pnpm lint && pnpm format:check && pnpm typecheck && pnpm build && pnpm test && pnpm test:e2e` xanh

**Verification:** như trên, kèm screenshot đính vào PR

**Dependencies:** T13 · **Files:** `components/clarification/answered-clarification-card.tsx`, `components/chat-conversation-content.tsx`, `hooks/use-chat-workspace.ts`, `e2e/chat-clarification.spec.ts`, `e2e/support/chat.ts` · **Scope:** M

### Checkpoint 3
- [ ] Web chạy với agent `main` (legacy) không lỗi, và chạy với agent nhánh này (panel)
- [ ] Người dùng xem screenshot và duyệt UI

---

## Phase 4: Calculation node (agent)

## T15: Prompts

**Description:** Viết lại `agents/calculation_extractor.yaml` (danh sách công thức và param render từ
`ParamSpec`). Thêm `agents/calculation_formula.yaml` (status found/ambiguous/not_found, provenance),
`agents/calculation_formula_verifier.yaml` và `main/chat_calculation.yaml`. Thêm `{calculation_results}` vào `chat_academic_advisory.yaml` và
`chat_multi_intent_synthesis.yaml`. Sửa `common/task_2.yaml` và `common/ask_user_form_guide.yaml` (bỏ
Type A cũ và tham chiếu tới file không tồn tại). Cập nhật `schema.py`, `loader.py` và snapshot.

**Acceptance criteria:**
- [x] Mọi template load được; placeholder trong template khớp với tham số `format`
- [x] Không còn nhắc `tuition_calculation`/`credit_check`/`chat_calculation_result.yaml`
- [x] Snapshot advisory được cập nhật có chủ đích (chỉ thêm khối `calculation_results`)

**Verification:** `.venv/bin/pytest -q tests/rag`

**Dependencies:** T2 · **Files:** 5–6 file YAML, `app/rag/prompting/{schema,loader,__init__}.py`, `tests/rag/*` · **Scope:** M

## T16: `run_calculation_task`, nhánh công thức cài sẵn

**Description:** Viết lại `app/graph/nodes/calculation.py` gồm `CalculationPlan`, `TaskOutcome`
(`Computed`/`NeedsInput`/`Unresolved`), `build_calculation_extractor_agent` và `run_calculation_task`
cho 3 công thức cài sẵn. Câu hỏi được dựng từ `missing_params`; param bị Python từ chối thì câu hỏi kèm
lý do.

**Acceptance criteria:**
- [x] Đủ tham số thì ra `Computed` với kết quả đúng; thiếu `th` khi `tcth > 0` thì ra `NeedsInput` với đúng câu hỏi `number_list`
- [x] Điểm 11 thì câu hỏi có lý do; JSON hỏng, `formula_id` lạ hoặc LLM lỗi thì ra `Unresolved("extraction_failed")` (fail closed, không retrieve)
- [x] Router luật `BUILTIN_TRIGGERS` khoá `formula_id`: câu GPA mà LLM trả `retrieved` vẫn chạy `gpa`; có `test_builtin_triggers.py` với ≥ 10 câu khớp và ≥ 10 câu không khớp cho mỗi công thức
- [x] Gọi extractor qua `run_agent_text_with_failover` và `usage_recorder.bind("CalculationNode")`

**Verification:** `.venv/bin/pytest -q tests/graph/test_calculation_node.py`

**Dependencies:** T2, T5, T15 · **Files:** `app/graph/nodes/calculation.py`, `app/calculation/formulas.py` (`BUILTIN_TRIGGERS`), `tests/graph/test_calculation_node.py`, `tests/calculation/test_builtin_triggers.py`, `tests/llm_mocks.py` · **Scope:** M

## T17: Nhánh Qdrant + provenance

**Description:** Với `formula_id == "retrieved"`: chạy `retrieve_chunks` (có access filter, limit 5,
không HyDE), gọi LLM `calculation_formula`, rồi chạy 7 kiểm tra của SPEC-calculation-node §2 theo thứ
tự: chunk id, câu trích là chuỗi con, `expression.validate`, values, neo hằng số, neo biến, verifier LLM
độc lập. Kết quả là `Computed` (khối có nhãn "Kết quả tham khảo theo quy chế", câu trích và nguồn) hoặc
`NeedsInput` hoặc `Unresolved`. Thêm công tắc `CHAT_CALC_RETRIEVED_FORMULA_ENABLED` (SPEC §7.1): khi tắt,
chỉ chạy kiểm tra 1–2 rồi hiện câu trích (`status = "quote_only"`).

**Acceptance criteria:**
- [x] Mỗi kiểm tra trong 7 kiểm tra khi trượt đều ra `Unresolved("formula_invalid")` và không gọi `evaluate`; verifier lỗi hoặc timeout cũng bị coi là trượt
- [x] `ambiguous` trả về candidates kèm nguồn; `not_found` hoặc không có chunk thì không tính
- [x] Công thức hợp lệ thiếu biến thì `NeedsInput` với câu hỏi `number` theo min/max của biến
- [x] Công tắc tắt: chỉ hiện câu trích, không gọi `evaluate`, không có panel

**Verification:** `.venv/bin/pytest -q tests/graph/test_calculation_retrieved.py`

**Dependencies:** T3, T16 · **Files:** `app/graph/nodes/calculation.py` (hoặc `calculation_retrieved.py`), `tests/graph/test_calculation_retrieved.py` · **Scope:** M

## T18: Graph wiring

**Description:** Theo SPEC-calculation-node §3:
- Tách `_run_advisory_flow` thành `prepare_advisory` và `generate_advisory`.
- Các task calculation chạy song song với `prepare_advisory`. **Barrier** `await calc_future` đặt trước
  node 10.
- Thứ tự stream: khối tính → `Unresolved` → node 10 (`{calculation_results}` chỉ có tiêu đề, không có
  số) → nhận xét (chỉ khi lượt không có advisory; không stream; số ngoài whitelist thì thay bằng câu cố
  định) → panel qua `build_round`.
- Trace theo SPEC-calculation-node §7.2: push trace đầy đủ sang `POST /internal/calculation-traces` trước finalize (thử lại 3 lần, lỗi thì chỉ log); `metadata.calculation` trong PATCH finalize chỉ có phần công khai. `prompt_versions` được tính lúc load template; `models` lấy credential sau failover.
- Xoá `CALCULATION_PLACEHOLDER_TEMPLATE`.

**Acceptance criteria:**
- [x] Lượt chỉ calculation đủ tham số: có khối tính và nhận xét, không có panel; thiếu tham số: câu dẫn và panel, không có lời gọi generation
- [x] Lượt lẫn mà cả hai đều thiếu: **một** panel có câu hỏi của cả hai origin; tối đa 12 câu, phần dư được log
- [x] `metadata.calculation` chỉ chứa các field công khai (test assert **không có** `question_raw`, `inputs`, `expression`, `source_quote`, `models`, `prompt_versions`); trace đầy đủ được push đúng 1 lần mỗi lượt, push lỗi thì lượt vẫn chạy
- [x] Advisory chuẩn bị xong trước vẫn không gửi token nào trước khối tính; nhận xét có số ngoài whitelist (hoặc LLM timeout) được thay bằng câu cố định, client không bao giờ thấy số lạ

**Verification:** `.venv/bin/pytest -q tests/graph`

**Dependencies:** T9, T16, T17 · **Files:** `app/graph/streaming_graph.py`, `app/graph/clarification_round.py`, `app/rag/prompting/__init__.py`, `tests/graph/test_graph_wiring.py`, `tests/graph/test_calculation_node.py` · **Scope:** M

## T19: Resume calculation + chain + e2e

**Description:** Khi resume một `PendingCalculationTask`: `known_params ∪ answers`, rồi gọi
`calculate`/`evaluate`, không gọi lại extractor, không retrieve lại. Resume chạy song song với advisory.
Còn thiếu thì tạo panel mới với `chain_depth + 1`, tối đa 3. E2E `tests/e2e/test_calculation_flow_e2e.py`.

**Acceptance criteria:**
- [x] Resume không gọi LLM extractor/formula (kiểm tra bằng mock: 0 lời gọi)
- [x] Ở `chain_depth = 3` không hỏi thêm, câu trả lời nêu rõ còn thiếu gì
- [x] E2E: ĐTKHP thiếu TH → `clarification` → submit → `7.5 / B / 3.0`

**Verification:** `.venv/bin/pytest -q tests/graph tests/e2e/test_calculation_flow_e2e.py` · toàn bộ `--ignore=tests/e2e` xanh

**Dependencies:** T10, T18 · **Files:** `app/graph/streaming_graph.py`, `app/graph/nodes/calculation.py`, `tests/graph/test_graph_wiring.py`, `tests/e2e/test_calculation_flow_e2e.py`, `tests/e2e/fake_llm_provider/app.py` · **Scope:** M

### Checkpoint 4
- [ ] Đủ Success Criteria của SPEC-calculation-node
- [ ] Chạy tay trên dev với cả 3 repo (Postgres thật): ĐTKHP thiếu TH, GPA bảng môn, học phí (Qdrant), lượt lẫn, Huỷ, reload, submit hai lần

---

## Phase 5: Hoàn tất

## T21: backend: bảng trace, phản hồi Đúng/Sai, ticket theo từng item

**Description:** Theo SPEC-calculation-node §7.2(b), §7.3 và contract §5b. Flyway migration mới gồm:
- bảng `calculation_traces` (FK `message_id` `ON DELETE CASCADE`, `UNIQUE (message_id, item_id)`);
- `tickets.calculation_item_id`;
- bỏ `UNIQUE (message_id)`, thay bằng 2 partial unique index;
- type `AI_CALCULATION_WRONG` cùng CHECK ràng buộc type ↔ item.

Endpoint:
- `POST /internal/calculation-traces` (upsert);
- `POST /messages/{messageId}/calculation-feedback`: ghi `metadata.calculation_feedback` không kèm note.
  Khi `WRONG` và người gọi là user đăng nhập thì tạo hoặc cập nhật ticket **của item đó**, description
  dựng từ trace. Đổi sang `CORRECT` thì ticket `OPEN` chuyển sang `CLOSED`.

**Acceptance criteria:**
- [x] T1 và T2 cùng sai thì có 2 ticket; Report thường vẫn tạo được trên cùng message; 2 Report thường trên một message vẫn bị chặn
- [x] Ticket chứa lý do, ghi chú và trace đọc từ `calculation_traces`; không có trace thì ghi rõ; `GET /messages/...` không bao giờ trả trace hay `note`
- [x] Validation: `reason` bắt buộc khi `WRONG` và phải `null` khi `CORRECT`; `OTHER` bắt buộc có note; item không phải retrieved đã tính thì `404`; ticket đã xử lý xong thì `409`; body > 2 KB thì `400`; `/internal/calculation-traces` thiếu secret bị chặn

**Verification:** `./mvnw test -Dtest='*CalculationFeedback*,TicketServiceImplTest'`

**Dependencies:** T4 (cùng nhánh backend), contract §5b · **Files:** `db/migration/V34__calculation_traces_and_item_tickets.sql`, `entity/{CalculationTrace,Ticket}.java` + `enums/TicketType.java`, `controller/internal/InternalCalculationTraceController.java`, `controller/MessageController.java`, `service/conversation/CalculationFeedbackService{,Impl}.java` + tests · **Scope:** L, nên tách khi làm thành T21a (migration + trace ingest) và T21b (feedback + ticket)

## T22: web: nút Đúng/Sai + popover lý do

**Description:** Theo SPEC-clarification-panel-ui §Phản hồi: `components/calculation-feedback.tsx`,
mutation qua `httpClient`, cập nhật cache của messages, trạng thái đọc từ `metadata.calculation_feedback`.

**Acceptance criteria:**
- [x] Nút chỉ hiện cho phần tử `retrieved` đã tính; Sai bắt buộc chọn 1 trong 5 lý do; "Khác" bắt buộc ghi chú; có dòng thông báo dữ liệu sẽ được gửi
- [x] Reload vẫn giữ lựa chọn; `409` thì khoá nút
- [x] `pnpm test`, `lint`, `typecheck`, `build` xanh; e2e mock endpoint feedback

**Verification:** `pnpm test -- src/features/chat` · `pnpm test:e2e -- e2e/chat-clarification.spec.ts`

**Dependencies:** T14, contract §5b · **Files:** `components/calculation-feedback.tsx`, `components/chat-conversation-content.tsx`, `queries/use-mutations.ts`, `api/chat-api.ts`, `schemas/chat-schemas.ts` (+ tests) · **Scope:** M

## T20: Docs + ghi chú rollout

**Description:**
- `known-gaps.md`: bỏ hoặc viết lại 4 mục đang gắn "Đang xử lý ở UNISAGE-99". Thêm các mục:
  - "LLM chép sai ý nghĩa công thức Qdrant: không có eval gate, dựa vào 7 kiểm tra, công tắc và phản hồi
    Đúng/Sai"
  - "Guest bấm Sai không tạo được ticket"
  - "Graph lỗi sau khi claim thì không khôi phục round"
  - "Process bị đóng băng lâu hơn lease"
- `PRODUCT.md`: cập nhật Not this product, Open và Glossary (`pending_clarification` → `PendingRound`
  và panel).
- `DECISIONS.md`: ghi các quyết định của UNISAGE-99, kèm thứ tự triển khai.
- `docs/architecture/rag-pipeline.md`: cập nhật sơ đồ node 07 và luồng panel.

**Acceptance criteria:**
- [x] Không còn tài liệu nào nói CalculationNode là placeholder hoặc nói tới Clarification Guard so khớp text
- [x] `DECISIONS.md` ghi rõ thứ tự backend → web → agent

**Verification:** `grep -rn "placeholder\|Clarification Guard" docs/` chỉ còn những chỗ có chủ đích

**Dependencies:** T19, T21, T22 · **Files:** `docs/specs/known-gaps.md`, `docs/product/{PRODUCT,DECISIONS}.md`, `docs/architecture/rag-pipeline.md` · **Scope:** S

### Checkpoint cuối
- [ ] Mọi acceptance criteria đã tick; 3 PR (backend, web, agent) ghi rõ thứ tự merge và triển khai
- [ ] Người dùng review trước khi mở PR
