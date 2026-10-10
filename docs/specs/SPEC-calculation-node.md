# Spec: calculation-node (unisage-agent)

> **Status: Implemented** (thiết kế chốt lại 09-10-2026: Python tính xuôi 3 công thức, LLM tính phần còn lại) (UNISAGE-99, nhánh `feature/huydh-unisage-99-calculation-flow`). Rủi ro còn
> lại nằm ở `unisage-agent/docs/specs/known-gaps.md`.

Module id `calculation-node` trong [capability map](../../changes/09-10-2026-calculation-flow/capability-map.md).
Phụ thuộc: [SPEC-calc-engine](SPEC-calc-engine.md) và [SPEC-clarification-panel](SPEC-clarification-panel.md).

## Objective

Thay placeholder `CalculationNode` bằng luồng thật cho các task `academic_calculation`. Chốt lại
(09-10-2026, sau khi thử bộ giải ngược và 7 kiểm tra công thức Qdrant):

1. **Python chỉ tính XUÔI 3 công thức cài sẵn** (GPA, ĐTKHP, quy đổi điểm): LLM chỉ chọn công thức và
   chép số; thiếu/sai số thì hỏi trên panel; Python render công thức và từng bước.
2. **Mọi phép tính khác do LLM tự tính** (`main/chat_calculation_llm.yaml`): câu hỏi ngược ("cuối kỳ cần
   bao nhiêu để được A+"), công thức trong tài liệu Qdrant (quy chế, học phí, xét tuyển, bài học...), câu
   nối tiếp cần lịch sử chat ("quay lại môn A"). Kết quả luôn có nhãn **"Kết quả do AI tự tính, có thể
   sai"** và nút Đúng/Sai (mục 7) - chủ sản phẩm chấp nhận LLM có thể tính sai, bắt lỗi bằng phản hồi.
3. Câu hỏi còn thiếu (của cả hai đường) gộp với câu hỏi advisory thành một panel.

## Luồng của một task calculation

```
 task.query ─► calculation_extractor (LLM): {formula_id: gpa|course_score|grade_conversion|previous|llm,
                                             params, retrieval_query}
                 │ cài sẵn / previous (tính xuôi)            │ llm
                 ▼                                           ▼
   formulas.missing_params / calculate            retrieve_chunks(retrieval_query) nếu có (access filter)
                 │                                           ▼
     NeedsInput(ParamSpec) | Computed            chat_calculation_llm (không stream): builtin rules +
                                                 tài liệu [C1..] + known_values + lịch sử chat
                                                             │
                                          ask_user_form → NeedsInput (panel) | text → LlmAnswered
```

### Trạng thái của task (`app/graph/nodes/calculation.py`)

```python
class CalculationPlan:            # field `plan` của PendingCalculationTask
    formula_id: Literal["gpa", "course_score", "grade_conversion", "llm"]
    retrieval_query: str | None   # chỉ cho "llm"

TaskOutcome = (
    Computed(task_id, result: CalculationResult, plan, params)            # Python, cài sẵn
  | LlmAnswered(task_id, query, text, plan, known_params, sources)        # LLM tự tính
  | NeedsInput(task_id, plan, known_params, questions, lead)
  | Unresolved(task_id, reason: Literal["llm_disabled", "llm_failed"])
)
```

`run_calculation_task` (lượt thường) và `resume_calculation_task` (lượt submit) là hai hàm graph gọi.

## 1. Extractor: `agents/calculation_extractor.yaml`

```json
{"formula_id": "course_score", "params": {"tbtx": 8, "gk": 7, "ck": 6.5, "th": [9, 8], "tclt": 2, "tcth": 1}, "retrieval_query": null}
```

- Công thức cài sẵn **chỉ khi tính xuôi** đúng công thức đó (kể cả chưa đủ số). `previous`: tính xuôi lại
  phép tính trong `<previous_calculation>` với số mới. `llm`: mọi trường hợp khác, kèm `retrieval_query`
  (câu tìm tài liệu chứa công thức) hoặc `null` khi chỉ cần công thức cài sẵn (VD hỏi ngược điểm cuối kỳ).
- Chỉ trích những số **người dùng đã nói**. Prompt liệt kê 3 công thức cài sẵn từ `ParamSpec`.
- `parse_extraction`: `llm` luôn thắng; còn lại một công thức cài sẵn duy nhất do router luật khớp sẽ
  thắng (kể cả khi output hỏng); output hỏng/không rõ mà router không khớp → `llm` (LLM vẫn hỏi lại được).
  `previous` chỉ nhận khi có `last_calculation` cài sẵn.

### 1.1 Định tuyến công thức cài sẵn bằng luật (trước LLM)

`BUILTIN_TRIGGERS` trong `formulas.py` (NFC, chữ thường). Khớp đúng một → khoá công thức cài sẵn đó
(trừ khi LLM chọn `llm`); khớp nhiều → đưa danh sách vào `<allowed_formula_ids>` (luôn kèm `previous`
khi có). Test: `tests/calculation/test_builtin_triggers.py`.

Câu hỏi cho công thức cài sẵn được dựng từ `ParamSpec` (`number`, `number_list`, `number_or_list`,
`course_table`); số Python từ chối được hỏi lại kèm lý do, ví dụ "Điểm cuối kỳ (bạn nhập 11, ...)".

## 2. LLM tự tính: `main/chat_calculation_llm.yaml`

- Input: `<builtin_rules>` (`describe_builtin_rules()`: trọng số, mọi bước làm tròn, bảng quy đổi - lấy từ
  hằng số trong `formulas.py`), `<academic_context>` (≤ 5 chunk đã qua `build_access_filter`, đánh số `[1]..`,
  trích dẫn theo `common/citation_rules.yaml`),
  `<known_values>` (số đã biết + câu trả lời panel), lịch sử chat gần đây, câu hỏi.
- Gọi **không stream** (`CHAT_CALC_LLM_TIMEOUT_SECONDS`, mặc định 60 s) để tách khối `ask_user_form` bằng
  `FenceRedactor` trước khi hiện. Trình bày: công thức (kèm `[n]` nếu từ tài liệu), thay số, kết quả;
  câu hỏi ngược phải kiểm tra bằng giá trị tìm được và giá trị liền kề.
- Không có công thức trong tài liệu → LLM nói rõ, không tự nghĩ công thức.
- Thiếu số / nhiều trường hợp → khối `ask_user_form` (`number`, `number_list`, `choice`, `text`) →
  `llm_questions` dựng câu hỏi panel (field bỏ dấu bằng `slug`; field đã có trong known hoặc trùng thì bỏ
  - chống hỏi vòng); câu dẫn của LLM hiện phía trên panel. Số liệu phụ thuộc một lựa chọn (phương thức,
  trường hợp) thì lượt đầu chỉ hỏi `choice`, sau đó mới hỏi số của nhánh đã chọn; mỗi field đúng một số.
  Lúc resume, câu trả lời `choice` vào `known_values` bằng **nhãn** lựa chọn (LLM không biết id `o1..`).
- Hiển thị: `**Kết quả do AI tự tính, có thể sai - bạn kiểm tra lại giúp mình nhé**` rồi text của LLM.
  Nguồn là các chỉ số `[n]` trong text, client biến thành link qua `citations` của message
  (`number_citations`: đánh số lại cho cả message, nối tiếp sau nguồn của câu trả lời advisory trong lượt
  hỗn hợp; chỉ số tài liệu LLM không thấy bị bỏ).
- Công tắc `CHAT_CALC_LLM_ENABLED=false` → câu cố định "Hiện mình chỉ tự tính được GPA, điểm tổng kết
  học phần và quy đổi điểm..."; lỗi/timeout → câu "Mình chưa tính được câu này lúc này...".

## 3. Graph wiring (`streaming_graph.py`)

### Lượt thường: barrier trước node 10

```
plan_route
  ├─ calc_future = gather(run_calculation_task(t) for t in calculation_tasks)         (bắt đầu ngay)
  └─ advisory: query transformation → retrieval → rerank → web search                 (song song với calc)
                                    │
                         ══ BARRIER: await calc_future ══
                                    │
     1. stream khối tính cho mỗi Computed (Python render) / LlmAnswered (nhãn AI), theo thứ tự T1..T3
     2. stream câu cố định cho mỗi Unresolved
     3. node 10 generation (qua FenceRedactor) → token ra client
     4. lượt chỉ có calculation: nhận xét đã kiểm tra số (mục 3.2)
     5. build_round → panel (theo thứ tự SPEC-clarification-panel §2.6)
```

- Node 10 là bước **duy nhất** sinh token của luồng advisory, và nó chỉ bắt đầu sau barrier. Vì vậy thứ
  tự "khối tính trước, advisory sau" luôn đúng mà không cần buffer token của advisory. Phần tốn thời
  gian nhất (HyDE, retrieval, rerank, web search) vẫn chạy song song với calculation.
- `_run_advisory_flow` được tách thành `prepare_advisory(...)` (đến hết rerank/web search, không gửi
  token nào) và `generate_advisory(prepared, calculation_titles, sink)`.
- **Node 10 không nhận con số nào từ phép tính.** Prompt chỉ nhận `{calculation_results}` là danh sách
  tiêu đề các phép tính đã hiện ở trên (ví dụ "Điểm tổng kết học phần (đã tính ở trên)"), kèm yêu cầu
  không nhắc lại hay tính lại. Không có số thì LLM không thể chép sai số.
- Nếu **mọi** task calculation đều `NeedsInput` và không có advisory, lượt chỉ gửi một câu dẫn cố định
  ("Mình cần thêm vài thông tin để tính giúp bạn:") rồi đến panel. Không gọi LLM.

### 3.2 Nhận xét (`main/chat_calculation.yaml`): không bao giờ hiện số chưa kiểm tra

Chỉ dùng cho lượt **chỉ có** calculation và có ít nhất một `Computed` (công thức cài sẵn). Kết quả LLM tự tính đã có lời giải thích của chính nó nên không có nhận xét riêng.

1. Gọi LLM **không stream** (`run_agent_text_with_failover`). Nhận xét ngắn (tối đa 2 câu, `max_tokens` thấp),
   nên chờ trọn vẹn rồi mới gửi không ảnh hưởng trải nghiệm.
2. Trích mọi số trong text bằng regex `\d+(?:[.,]\d+)?`, chuẩn hoá `,` thành `.`, bỏ số 0 thừa.
3. Whitelist gồm: các số sinh viên đã nhập (`inputs`) cùng các hằng `{0, 4, 10}` (tên thang điểm).
   **Số của kết quả (`outputs`) bị loại khỏi whitelist** - kết quả đã nằm ở dòng "Kết quả" ngay phía
   trên, nhận xét chỉ nói ý nghĩa của nó (đã chốt 09-10-2026, sau phản hồi về thông tin bị lặp).
4. Có số **ngoài whitelist** (số bịa hoặc nhắc lại kết quả) → **bỏ hẳn nhận xét**, không có câu thay thế
   (câu thay thế cũng chỉ lặp lại kết quả), ghi log `calculation.commentary_rejected`. LLM lỗi hoặc
   timeout cũng bỏ nhận xét.
5. Text gửi ra luôn là text đã qua bước 4. **Không có số nào chưa qua kiểm tra tới được client.**

## 4. Gộp câu hỏi thành một panel

```python
def build_round(outcomes, advisory_asks, *, original_query, chain_depth, assistant_message_id) -> PendingRound | None
```

- Thứ tự câu hỏi: các task theo thứ tự `T1..T3`; trong mỗi task, giữ thứ tự của `missing_params` hoặc
  của `fields`.
- Gán `id` từ `q1` trở đi, mỗi câu hỏi là một tab. **Mọi câu hỏi đều được hỏi** trong cùng panel; chỉ
  khi vượt 50 câu (payload bất thường) mới cắt và log `clarification.questions_truncated`.
- Mỗi task có câu hỏi trở thành một `PendingAdvisoryTask` hoặc `PendingCalculationTask`.
  `known_params` giữ lại những gì extractor đã lấy được, để lúc resume **không gọi lại extractor**.

## 5. Resume (lượt submit)

```
answers → nhóm theo question.task_id
  PendingCalculationTask: values = known_params ∪ {question.field: answer}
      cài sẵn → formulas.calculate (không gọi LLM)
      llm     → chat_calculation_llm lại với known_values mới (không gọi extractor, không qua classifier)
  PendingAdvisoryTask: confirmed_metadata ∪ {field: option_id | other_text} → _run_advisory_flow
```

Còn thiếu thì tạo panel mới với `chain_depth + 1` (không giới hạn; field đã trả lời không bao giờ bị hỏi lại).

## 6. Prompts

| File | Nội dung |
|---|---|
| `agents/calculation_extractor.yaml` | mục 1: cài sẵn (xuôi) / previous / llm |
| `main/chat_calculation_llm.yaml` | mục 2: LLM tự tính, `ask_user_form` khi thiếu số |
| `main/chat_calculation.yaml` | nhận xét sau khối tính cài sẵn (mục 3.2) |
| `agents/message_classification.yaml` | quy tắc 1c nối tiếp phép tính, `<previous_calculation_turn>` (mục 9) |
| `common/response_style.yaml` | phép nhân viết `×` |
| `main/chat_academic_advisory.yaml`, `main/chat_multi_intent_synthesis.yaml` | `{calculation_results}`: chỉ tiêu đề phép tính đã hiện |

## 7. Công tắc, trace và phản hồi cho phép tính do LLM

Không có eval trước khi phát hành (đã chốt 09-10-2026): công tắc tắt khẩn cấp, trace đầy đủ, và phản hồi
Đúng/Sai tạo ticket để admin điều tra.

### 7.1 Công tắc `CHAT_CALC_LLM_ENABLED` (mặc định `True`)

`False`: chỉ 3 công thức cài sẵn được tính xuôi; mọi phép tính khác trả câu cố định (mục 2). Đổi qua
biến môi trường rồi restart agent.

### 7.2 Trace: tách phần công khai và phần chỉ staff đọc được

Nguyên tắc: **message mà client nhận về chỉ chứa những gì cần để hiển thị.** Câu hỏi gốc, điểm số, tín
chỉ, biểu thức, câu trích, model và prompt version nằm trong một store **chỉ staff đọc được**, không bao
giờ đi qua `GET /messages/...` hay SSE.

**(a) Phần công khai: `metadata.calculation` trên message ASSISTANT** (agent ghi trong PATCH finalize)

```json
{"calculation": {"schema_version": 1, "items": [
  {"item_id": "T1", "run_id": "<budget.request_id>", "mode": "llm", "status": "computed",
   "result_summary": null,
   "source_summary": {"title": "QĐ-123.pdf", "heading": "Chương II › Điều 8"}}
]}}
```

`result_summary` và `source_summary` vốn đã nằm trong `content` mà người dùng thấy, nên không lộ thêm gì.
`mode`: `builtin` (Python) hoặc `llm`. Không có `question_raw`, `inputs`, `answer`, `models` hay
`prompt_versions`.

**(b) Phần chỉ staff: bảng `calculation_traces` (backend)**

- Agent gọi `POST /internal/calculation-traces` (dưới `/internal/**`, có `X-Internal-Secret`, đi theo tiền
  lệ `InternalUsageLogController`) **trước** PATCH finalize, mỗi lượt một lần, chứa mọi item của lượt đó.
  Thử lại 3 lần; vẫn lỗi thì log `calculation.trace_push_failed` và lượt chat vẫn tiếp tục. Trace là dữ
  liệu chẩn đoán, không nằm trên đường chính.
- Bảng có các cột: `id`, `message_id` (FK → `messages`, `ON DELETE CASCADE`), `item_id`, `run_id`,
  `trace jsonb`, `created_at`. Có `UNIQUE (message_id, item_id)`. Gửi lại cùng một cặp này thì upsert,
  nên idempotent khi thử lại.
- Nội dung `trace` (tối đa 16 KB mỗi item), với `mode = "llm"`:

  ```json
  {"question_raw": "…", "status": "computed", "mode": "llm", "retrieval_query": "công thức học phí",
   "known_params": {"so_tin_chi": 20, "don_gia": "420000"}, "answer": "<toàn bộ lời giải của LLM>",
   "sources": [{"ref": "C1", "chunk_id": "c_123", "document_id": "d_9", "source": "QĐ-123.pdf",
                "heading_path": ["Chương II", "Điều 8"], "chunk_hash": "sha256:<12>"}],
   "models": {"llm": "<provider>/<model>"},
   "prompt_versions": {"agent_calculation_extractor": "sha256:<12>", "chat_calculation_llm": "sha256:<12>"}}
  ```

  Với `mode = "builtin"`: `formula_id`, `formula_hash`, `inputs`, `outputs`.

- Không có API public nào đọc bảng này. Staff xem trace **thông qua ticket** (7.3), không đọc thẳng
  bảng. Dữ liệu tự xoá theo message nhờ cascade; chính sách lưu trữ riêng (nếu cần) để ticket sau.
- `chunk_hash` là hash của `content` chunk tại thời điểm tính, dùng thay số phiên bản tài liệu (payload
  Qdrant chưa có). `prompt_versions` là hash nội dung file YAML, tính một lần lúc load template. `models`
  lấy từ credential thực sự được dùng, sau failover.
- Builtin cũng có trace, để tiện debug, dù không có nút phản hồi.

### 7.3 Phản hồi Đúng/Sai

Áp dụng cho **mỗi item có `mode = "llm"` và `status = "computed"`**. Công thức cài sẵn (Python) không có
nút này.

- Khối kết quả có nhãn đầu là **"Kết quả do AI tự tính, có thể sai..."**, cuối khối có hai nút **Đúng**
  và **Sai**.
- Bấm **Sai** thì bắt buộc chọn **một** lý do: `WRONG_FORMULA` (Sai công thức), `WRONG_RESULT` (Sai kết
  quả), `WRONG_SOURCE` (Sai nguồn/quy chế), `MISSING_INFO` (Thiếu thông tin), `OTHER` (Khác). Có thêm ô
  ghi chú không bắt buộc, tối đa 500 ký tự; chọn `OTHER` thì ô này thành bắt buộc. Form có dòng "Câu hỏi
  và các số bạn đã nhập sẽ được gửi cho bộ phận hỗ trợ để kiểm tra."
- Mỗi item có một phản hồi. Đổi được Đúng ↔ Sai cho đến khi ticket của item đó được xử lý.

**Backend: `POST /messages/{messageId}/calculation-feedback`** (route master, kiểm tra quyền sở hữu
user hoặc guest giống `startTurn`)

```json
{"itemId": "T1", "verdict": "CORRECT" | "WRONG", "reason": "WRONG_FORMULA" | … | null, "note": "…" | null}
```

1. Message phải là ASSISTANT, thuộc người gọi, và có `metadata.calculation.items[itemId]` với
   `mode = "llm"` và `status = "computed"`. Không thoả thì trả `404`.
2. Ghi `metadata.calculation_feedback[itemId] = {verdict, reason, at}`. Không lưu `note` ở đây; `note` chỉ
   nằm trong ticket. Áp dụng cho cả user lẫn guest.
3. `verdict = WRONG` **và người gọi là user đã đăng nhập** → tạo hoặc cập nhật **một ticket riêng cho item
   đó**:
   - `type = AI_CALCULATION_WRONG`, `calculation_item_id = itemId`, `title = "Tính sai (T1): <lý do>"`.
   - `description` do backend dựng từ lý do, ghi chú, `run_id` và trace đọc từ
     `calculation_traces (message_id, item_id)`. Không có trace (lần push bị lỗi) thì ghi "Không có trace
     cho run_id …".
   - Ticket của item đó đã có và còn `OPEN` thì cập nhật lý do và ghi chú (người dùng đổi lý do). Ticket
     đã `RESOLVED`/`CLOSED` thì trả `409`.
4. Đổi từ `WRONG` sang `CORRECT`: ticket còn `OPEN` thì chuyển sang `CLOSED` với
   `resolution = "Người dùng đổi đánh giá thành Đúng"`.
5. Guest không tạo được ticket (`tickets.user_id NOT NULL`). Phản hồi của guest chỉ nằm trong
   `metadata.calculation_feedback`.
6. Request tối đa 2 KB; `itemId` khớp `^T[1-3]$`.

**Ràng buộc unique của ticket** (Flyway migration mới):

- Thêm cột `calculation_item_id VARCHAR(4) NULL`.
- Bỏ `tickets_message_id_key UNIQUE (message_id)` và thay bằng hai partial unique index:
  - `UNIQUE (message_id) WHERE calculation_item_id IS NULL`: vẫn giữ "một Report cho mỗi message" như cũ;
  - `UNIQUE (message_id, calculation_item_id) WHERE calculation_item_id IS NOT NULL`: mỗi item sai có ticket
    riêng, trạng thái riêng.
- Thêm `AI_CALCULATION_WRONG` vào `tickets_type_check`, cùng CHECK: `calculation_item_id IS NOT NULL`
  **khi và chỉ khi** `type = 'AI_CALCULATION_WRONG'`.
- Report thường và ticket tính sai cùng tồn tại được trên một message. T1 và T2 cùng sai thì có hai
  ticket.

Admin điều tra trong màn `support-tickets` hiện có, lọc theo type `AI_CALCULATION_WRONG`. Trace đầy đủ nằm
trong `description`, nên không cần màn hình mới, và staff chỉ thấy trace qua quyền xem ticket sẵn có.

## 8. Phép tính trước (`last_calculation`)

- Cột JSON trên `conversation_clarification_states` (migration `a4b5c6d7e8f9`), ghi cùng `confirmed_metadata`:
  phép tính cuối của lượt. Cài sẵn: kèm `params` (câu nối tiếp tính xuôi `formula_id="previous"` gộp số cũ
  với số mới). LLM: chỉ `title` (câu hỏi) - số của nó nằm trong lịch sử chat, LLM đọc lại khi cần.
- `title` đi vào classifier (`<previous_calculation_turn>`, mục 9). Lượt không có phép tính giữ nguyên.

## 9. Classifier qua nhiều lượt

Nguyên tắc: **câu trả lời cho câu hỏi của bot không bao giờ được phân loại lại.**

1. Mọi câu hỏi lại (thiếu số, nhiều công thức, thuộc tính advisory) đi qua panel; lượt submit vào
   thẳng `_run_resume`, không gọi classifier. Panel đang mở thì tin nhắn thường bị chặn (`409`).
2. Câu gõ tự do sau một phép tính: classifier nhận `<previous_calculation_turn>` (tên phép tính trong
   `last_calculation`) - tín hiệu có cấu trúc thay vì đoán từ lịch sử chat.
3. Quy tắc 1c trong `message_classification.yaml`: câu bổ sung dữ kiện/đổi số/hỏi ngược về phép tính
   trước giữ `academic_calculation`, `query` viết lại thành câu đầy đủ (bước tính chỉ đọc `query`);
   câu hỏi quy định mới vẫn là advisory. Test giữ quy tắc và ví dụ trong prompt.

## Commands

```
.venv/bin/pytest -q tests/graph tests/rag tests/calculation
.venv/bin/pytest -q tests/e2e/test_calculation_flow_e2e.py      # dùng fake_llm_provider
.venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests && .venv/bin/mypy app
```

## Testing Strategy

- **Extractor** (`tests/graph/test_calculation_task.py`): `llm` không bị router ghi đè; router khoá công
  thức cài sẵn kể cả khi output hỏng; output hỏng không có router → `llm`; `previous` cần
  `last_calculation`; `llm_questions` bỏ field đã biết/trùng/sai kiểu.
- **Công thức cài sẵn:** đủ số → `Computed`; thiếu → `NeedsInput` từ `ParamSpec`; điểm 11 → hỏi lại kèm lý do.
- **Graph** (`tests/graph/test_calculation_node.py`): lượt chỉ có tính toán (khối tính + nhận xét đã
  kiểm tra số); thiếu số → panel; lượt hỗn hợp → một panel, khối tính trước advisory; câu hỏi ngược sau
  một phép tính → LLM nhận builtin rules + lịch sử, có nhãn AI, `mode = "llm"`; LLM hỏi số bằng
  `ask_user_form` → panel → resume không gọi classifier/extractor, có `Nguồn:` cho `[C1]`.
- **Classifier:** `<previous_calculation_turn>` được gửi; quy tắc 1c và ví dụ còn trong prompt.

## Boundaries

- **Always:** Python tính xuôi 3 công thức cài sẵn; mọi kết quả LLM tự tính có nhãn AI + nút Đúng/Sai;
  chunk qua access filter trước khi tới LLM; mọi câu hỏi lại đi qua panel.
- **Ask first:** thêm công thức cài sẵn thứ 4; bỏ nhãn AI; stream câu trả lời LLM tự tính.
- **Never:** để LLM nghĩ câu hỏi cho công thức cài sẵn; gọi lại extractor khi resume; đưa con số phép
  tính vào prompt node 10.

## Success Criteria

- [x] Câu ĐTKHP đủ số → khối tính `7.5 / B / 3.0`, không có panel.
- [x] GPA không kèm điểm → panel `course_table`; submit → GPA 2 chữ số thập phân.
- [x] "Cuối kỳ cần bao nhiêu để được A+" → LLM tự tính, có nhãn AI và nút Đúng/Sai.
- [x] Học phí/xét tuyển: LLM tính theo tài liệu tìm được (kèm nguồn) hoặc nói không tìm thấy công thức;
      thiếu số hoặc nhiều trường hợp → panel.

## Decisions (09-10-2026)

- Giữ bước nhận xét LLM (`chat_calculation.yaml`) cho lượt chỉ có calculation.
- (Review lần 2) Barrier trước node 10 thay cho buffer token; nhận xét không stream, không nhắc lại kết quả
  và bị bỏ hẳn nếu có số ngoài whitelist; node 10 chỉ nhận tiêu đề phép tính; extractor fail closed; công
  thức cài sẵn được định tuyến bằng luật trước LLM.
- (Đã thay bằng LLM tự tính, xem dòng cuối) Công thức Qdrant vẫn được tính, nhưng phải qua 7 kiểm tra (neo hằng số, neo biến, verifier LLM độc lập),
  luôn gắn nhãn "Kết quả tham khảo theo quy chế", không hỏi sinh viên xác nhận.
- Không làm eval trước khi phát hành. Thay bằng công tắc `CHAT_CALC_RETRIEVED_FORMULA_ENABLED`, trace và
  nút Đúng/Sai với 5 lý do.
- (Review lần 4) Trace tách đôi: message chỉ chứa `run_id`, `item_id`, `status`, `result_summary` và
  `source_summary`. Trace đầy đủ (câu hỏi gốc, điểm số, biểu thức, câu trích, model, prompt version) nằm ở
  bảng `calculation_traces` bên backend, và staff chỉ xem qua ticket. Mỗi item sai có một ticket riêng nhờ
  partial unique index `(message_id, calculation_item_id)`.
- (09-10-2026, sau khi thử) **Bỏ bộ giải ngược (`solver.py`) và 7 kiểm tra công thức Qdrant**
  (`expression.py`, `provenance.py`, formula agent, verifier). Python chỉ tính xuôi 3 công thức cài sẵn;
  mọi phép tính khác do LLM tự tính với nhãn "AI tự tính, có thể sai" và nút Đúng/Sai. Công tắc đổi
  tên thành `CHAT_CALC_LLM_ENABLED`; `mode` của item đổi từ `retrieved` thành `llm`.
