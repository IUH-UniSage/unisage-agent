# Spec: calculation-node (unisage-agent)

> **Status: Implemented** (UNISAGE-99; chốt lại 26-10-2026 trong UNISAGE-106: **Python tính mọi phép
> tính, xuôi và ngược, theo registry công thức có version; LLM chỉ định tuyến; bỏ nhánh LLM tự tính**).
> Rủi ro còn lại nằm ở `unisage-agent/docs/specs/known-gaps.md`.

Module id `calculation-node` trong [capability map](../../changes/09-10-2026-calculation-flow/capability-map.md).
Phụ thuộc: [SPEC-calc-engine](SPEC-calc-engine.md) và [SPEC-clarification-panel](SPEC-clarification-panel.md).

## Objective

Luồng thật cho các task `academic_calculation`. Chốt lại 26-10-2026 (sau 4 lần đo 38 câu calculation:
LLM tính nhẩm sai 4-5 câu mỗi lần, nhiễu ±3-4 câu giữa hai lần chạy cùng prompt - xem
`changes/26-10-2026-FormulaRegistry/plan.md`):

1. **Python tính mọi phép tính**, xuôi (`formulas.calculate`) và ngược (`solver.solved_result`: "cuối kỳ
   cần bao nhiêu để được A+", "ĐHB tối thiểu để XT1 đạt 24", "tối đa mấy tín để học phí ≤ 8 triệu"),
   theo **registry công thức có version và nguồn** (SPEC-calc-engine). LLM chỉ chọn công thức, chép số
   người dùng đã nói, và nêu ẩn số + mục tiêu khi hỏi ngược; thiếu/sai số thì hỏi trên panel; Python
   render công thức, từng bước và dòng kiểm tra.
2. **Số học thuần** (chính sách 3 mức, chốt 26-10-2026 tối): số người dùng tự nêu ("18 tín kỳ này, 15
   tín kỳ sau, tổng bao nhiêu") → extractor trả `arithmetic` + biểu thức → code kiểm tra mọi số hạng có
   trong câu hỏi rồi tính (`app/calculation/arithmetic.py`), không qua LLM. Số liệu phải lấy từ văn bản
   → câu đi nhánh advisory, node 10 **không tự tính** mà viết khối `calc_request` (biểu thức + nhãn +
   đơn vị + [n]) để `app/graph/calc_requests.py` kiểm tra căn cứ (mọi số hạng có trong câu hỏi hoặc chunk
   được dẫn) và nối khối Python dưới câu trả lời; từ chối → dòng cố định + cảnh báo admin
   `CALC_REQUEST_REJECTED`. Kết luận nghiệp vụ (phải đóng bao nhiêu, đủ điều kiện không) và công thức có
   hệ số/điều kiện riêng → chỉ trích dẫn, giải thích biến. Grounding là một phần của kiểm tra căn cứ,
   không phải quyền tính.
3. **Câu hỏi không có công thức trong registry** → extractor trả `none` → task **đi nhánh advisory**
   ngay đầu lượt (mục 2 ở trên áp dụng). Không còn nhánh "AI tự tính".
4. Câu hỏi còn thiếu gộp với câu hỏi advisory thành một panel.

## Luồng của một task calculation

```
 task.query ─► calculation_extractor (LLM): {formula_id: <registry id>|previous|arithmetic|none,
                                             params, solve_for, target, arithmetic, retrieval_query}
                 │ registry / previous        │ arithmetic              │ none → Deferred
                 ▼                            ▼                         ▼
   fill_from_documents (lookup)       số hạng ⊆ số trong câu hỏi?   task nhập vào advisory_tasks
   missing_params (trừ ẩn số)         có → evaluate → Computed      (node 10 trích dẫn; số học thuần
   → NeedsInput; đủ số →              không → Deferred               trên số đã dẫn → calc_request →
   calculate | solver → Computed                                      calc_requests.py tính, nối khối)
```

### Trạng thái của task (`app/graph/nodes/calculation.py`)

```python
class CalculationPlan:            # field `plan` của PendingCalculationTask
    formula_id: Literal[<9 id trong registry>, "llm"]   # "llm": round cũ (trước 26-10) còn đọc được
    solve_for: str | None         # tính ngược: tham số cần tìm ("ck", "diem[4]")
    target: CalculationTarget | None   # {output, op (>=, <=, ==), value (số hoặc điểm chữ)}

TaskOutcome = (
    Computed(task_id, result: CalculationResult, plan, params, notes, sources)   # Python
  | NeedsInput(task_id, plan, known_params, questions, lead)
  | Unresolved(task_id, reason: Literal["legacy_llm", "bad_target", "not_resumable"])
)
Deferred(task_id, query, retrieval_query)   # không có công thức: graph chuyển sang advisory
ArithmeticTask(task_id, query, request, evaluation)   # số học thuần trên số người dùng nêu
```

Graph gọi `plan_calculation` (extractor, một lần cho mọi task, trước retrieval advisory) rồi
`run_planned_task` (lượt thường) và `resume_calculation_task` (lượt submit); `run_calculation_task` gộp
hai bước đầu cho test.

## 1. Extractor: `agents/calculation_extractor.yaml`

```json
{"formula_id": "course_score", "params": {"tbtx": 8, "gk": 7, "tclt": 2, "tcth": 1, "th": [9, 8]},
 "solve_for": "ck", "target": {"output": "dtkhp", "op": ">=", "value": "A+"}, "retrieval_query": null}
```

- Một `formula_id` của registry khi câu hỏi là phép tính của đúng công thức đó, xuôi (`solve_for: null`)
  hay ngược (`solve_for` = tham số cần tìm, `diem[4]` cho một vị trí trong danh sách; `target.output` =
  key đầu ra, `op`, `value` số hoặc điểm chữ). Danh sách công thức + tham số + đầu ra render từ registry
  (`describe_builtin_formulas()`, kèm version). `previous`: phép tính trong `<previous_calculation>` với
  số mới (có thể kèm `solve_for`). `none`: không công thức nào khớp, kèm `retrieval_query` (câu tìm tài
  liệu chứa công thức/số liệu) - task đi nhánh advisory.
- Tin nhắn gửi extractor: `query` (bản classifier đã chuẩn hoá), `<original_message>` (nguyên văn người
  dùng gõ ở lượt này, chỉ khi khác `query`; lượt submit panel không có) và `<current_time>`. Hai bản khác
  nghĩa thì tin nguyên văn; số và chữ phủ định lấy theo nguyên văn (13-10-2026: classifier từng viết
  "không biết tổng" thành "biết tổng").
- Chỉ trích những số **người dùng đã nói**. Prompt liệt kê các công thức cài sẵn từ `ParamSpec`. Câu học phí (kỳ / năm / cả khoá) đi vào `tuition_*`, không còn qua `llm`; câu hỏi ngược về học phí vẫn là `llm`.
- Học phí: `pham_vi` = cụm phạm vi nguyên văn trong câu ("2 năm đầu", "học kỳ 1 năm 2"); bỏ khi người
  dùng không nói hoặc chỉ nói đúng phạm vi mặc định. Nhiều năm nhưng chưa phải cả khoá ("2 năm đầu") →
  `tuition_year` kèm `pham_vi`; form hỏi "Số tín chỉ lý thuyết của 2 năm đầu" và kết quả có tiêu đề
  "Học phí 2 năm đầu" (SPEC-calc-engine, mục học phí). Lý do (13-10-2026): trước đó câu "2 năm đầu" ra
  form và kết quả "một năm học", lệch với câu hỏi.
- `parse_extraction` (26-10-2026): lựa chọn công thức registry của model được giữ, kể cả khi router
  luật khớp công thức khác (router không phân biệt được "cuối kỳ cần bao nhiêu để được B+" với quy đổi
  điểm; chỉ log `router_disagreement`); `none` của model được giữ; output hỏng/id lạ → công thức duy nhất
  router khớp, không có hoặc nhiều → `none`. `solve_for`/`target` hỏng → log `solve_dropped`, tính xuôi.
  `previous` chỉ nhận khi phép tính gần nhất (`recent_calculations[-1]`) là công thức registry.

### 1.1 Định tuyến công thức cài sẵn bằng luật (trước LLM)

`BUILTIN_TRIGGERS` trong `formulas.py` (NFC, chữ thường). Khớp đúng một → dự phòng khi output
extractor hỏng; khớp nhiều → đưa danh sách vào `<allowed_formula_ids>` (luôn kèm `previous` khi có).
Test: `tests/calculation/test_builtin_triggers.py`.

Câu hỏi cho công thức cài sẵn được dựng từ `ParamSpec` (`number`, `number_list`, `number_or_list`,
`course_table`, `text`) bằng `question_for(spec, known)`, nhãn lấy từ `label_for` theo các param đã biết;
số Python từ chối được hỏi lại kèm lý do, ví dụ "Điểm cuối kỳ (bạn nhập 11, ...)".

### 1.2 Tra tài liệu trước khi hỏi (`builtin_task`, thêm 13-10-2026)

Mọi công thức cài sẵn, ở lượt hỏi lẫn lượt submit, đi qua `builtin_task`:

1. `fill_from_documents`: các tham số còn thiếu có `ParamSpec.lookup` (VD đơn giá học phí) được tìm qua
   `_search` (Qdrant + ngưỡng điểm + LLM rerank), rồi `agents/calculation_value_extractor.yaml` chép từng
   con số kèm số đoạn `[n]`. **Python chỉ nhận số khi đoạn được dẫn thật sự chứa con số đó** (`_amounts`,
   đọc dấu chấm/phẩy là phân cách hàng nghìn); sai thì bỏ, để panel hỏi.
2. Biểu phí có nhiều dòng theo khoá tuyển sinh / khối ngành mà chưa biết sinh viên thuộc dòng nào → model
   trả `ask`; hỏi tiêu chí đó trước (choice từ các giá trị tài liệu ghi, `allow_other`), **cùng lượt** với
   các số của riêng sinh viên còn thiếu. Lượt submit tra lại với tiêu chí vừa trả lời.
3. Còn thiếu → panel hỏi đúng các `ParamSpec` đó. Đủ → Python tính; `Computed.notes` liệt kê từng số lấy
   từ tài liệu kèm `[n]` (đánh số chung với các nguồn khác của tin nhắn trong `number_citations`), thêm ghi
   chú đơn giá áp dụng cho đại học chính quy tại trụ sở chính. Trace lưu các chunk nguồn.

Không có tham số `lookup` nào còn thiếu thì không tìm Qdrant, không gọi model. Ẩn số của phép ngược
không bao giờ được tra hay hỏi (`skip`).

### 1.3 Tính ngược (`solver.solved_result`, 26-10-2026)

`builtin_outcome(..., unknown, target)`: thiếu tham số (trừ ẩn số) → panel như tính xuôi; đủ → solver
tìm giá trị nhỏ nhất/lớn nhất của ẩn số trên lưới của nó (bước của tham số hoặc bước làm tròn công thức
áp dụng - điểm CK tìm theo 0,5) để đầu ra đạt mục tiêu, trả `CalculationResult` gồm dòng "Giá trị cần
tìm", các bước tính xuôi tại nghiệm và dòng "Kiểm tra" với giá trị liền kề; không có nghiệm → nói rõ
kèm giá trị biên (vẫn là `Computed`). `plan.solve_for`/`plan.target` được lưu để lượt submit tính nốt.
Lỗi không gắn được với tham số nào (mục tiêu hỏng) → `Unresolved("bad_target")`.

## 2. Không có công thức: `Deferred` → nhánh advisory (thay "LLM tự tính", 26-10-2026)

`_classified_turn` gọi extractor cho mọi task calculation **trước** khi retrieval advisory bắt đầu
(`_plan_calculations`, ~1 lần gọi phụ trợ). Task `Deferred` thành một `ClassifiedTask(academic_advisory,
SINGLE)` nối vào `advisory_tasks` của lượt; `l1_pending`/cache coi lượt như advisory khi không còn task
tính được. Node 10 trả lời từ tài liệu với luật (cuối `chat_academic_advisory.yaml`): câu cần tính mà
hệ thống không có công thức → trích số liệu/công thức kèm `[n]`, nêu điều kiện và cách thay số, KHÔNG tự
ra kết quả. Prompt phân loại cũng liệt kê registry (`{calculation_formulas}`) để các phép cộng/trừ trên số
liệu văn bản được gán thẳng `academic_advisory`.

Phần dưới đây mô tả nhánh đã bỏ, giữ để đọc round cũ:

### 2.1 (Đã bỏ) LLM tự tính: `main/chat_calculation_llm.yaml`

- Input: `<builtin_rules>` (`describe_builtin_rules()`: trọng số, mọi bước làm tròn, bảng quy đổi - lấy từ
  hằng số trong `formulas.py`), `<academic_context>` (≤ 5 chunk đã qua `build_access_filter`, đánh số `[1]..`,
  trích dẫn theo `common/citation_rules.yaml`), `<current_time>` (ngày + học kỳ hiện tại theo lịch dự kiến,
  `common/current_time.yaml`), và dưới câu hỏi một dòng tin nhắn gốc của sinh viên khi khác `query`,
  `<known_values>` (số đã biết + câu trả lời panel), lịch sử chat gần đây, câu hỏi.
- Gọi **không stream** (`CHAT_CALC_LLM_TIMEOUT_SECONDS`, mặc định 60 s) để tách khối `ask_user_form` bằng
  `FenceRedactor` trước khi hiện. Trình bày: công thức (kèm `[n]` nếu từ tài liệu), thay số, kết quả;
  câu hỏi ngược phải kiểm tra bằng giá trị tìm được và giá trị liền kề.
- Không có công thức trong tài liệu → LLM nói rõ, không tự nghĩ công thức.
- Sinh viên nói **không biết** một giá trị → không hỏi lại giá trị đó; LLM tính nó từ số liệu khác (VD tổng
  học phí kỳ = tín chỉ × đơn giá theo tài liệu) và chỉ hỏi các số liệu cấu thành còn thiếu.
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
  ├─ planned = gather(plan_calculation(t) for t in calculation_tasks)   (extractor, ~1 s; Deferred → advisory_tasks)
  ├─ calc_future = gather(run_planned_task(t) for t in planned if PlannedTask)        (bắt đầu ngay)
  └─ advisory: query transformation → retrieval → rerank → web search                 (song song với calc)
                                    │
                         ══ BARRIER: await calc_future ══
                                    │
     1. stream khối tính cho mỗi Computed (Python render), theo thứ tự T1..T3
     2. stream câu cố định cho mỗi Unresolved
     3. node 10 generation (qua FenceRedactor) → token ra client
     4. build_round → panel (theo thứ tự SPEC-clarification-panel §2.6)

  Lượt chỉ có calculation: không có node 10; bước kiểm tra và trả lời (mục 3.2) chạy trước, phần AI
  viết đứng đầu tin nhắn, rồi mới đến khối tính (1, 2).
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

### 3.2 Kiểm tra và trả lời (`main/chat_calculation_review.yaml`, thay nhận xét 13-10-2026)

Chỉ dùng cho lượt **chỉ có** calculation và có ít nhất một `Computed` (công thức cài sẵn). Kết quả LLM
tự tính đã có lời giải thích của chính nó; lượt có cả câu hỏi tra cứu thì node 10 viết phần chữ, không
gọi bước này. **Số do Python tính, AI chỉ kiểm tra và viết lời**: AI không tính lại, không đổi số.

1. `review(computed, question, deps)` gọi LLM **không stream** (`run_agent_text_with_failover`,
   `CHAT_AUX_CALL_TIMEOUT_SECONDS`) với câu hỏi gốc (lượt submit: `original_query` của round), lịch sử gần
   đây và mỗi kết quả Python (tiêu đề, đầu vào, các bước đã thế số, kết quả). Một lần gọi cho mọi kết quả
   của lượt; thay đúng chỗ lần gọi nhận xét cũ, không thêm lần gọi LLM.
2. Đầu ra JSON `{"fits": bool, "issue": str | null, "answer": str}` (`Review`):
   - `fits`: phép tính có trả lời đúng điều được hỏi (đại lượng, phạm vi, đối tượng). Prompt chỉ cho báo
     `false` khi khác hẳn câu hỏi, để tránh cảnh báo thừa.
   - `issue`: một câu nói chỗ lệch khi `fits = false` (thiếu thì coi như output hỏng).
   - `answer`: 1-3 câu trả lời thẳng câu hỏi bằng con số Python đã tính; khi lệch thì nói rõ kết quả là
     cho cái gì và gợi ý hỏi lại.
3. Kiểm tra số: trích số trong `issue` và `answer` (`_NUMBER`: số có dấu chấm phân cách hàng nghìn đọc
   là một số, "165.200.000" → 165200000; còn lại `\d+(?:[.,]\d+)?`, `,` → `.`). Tập được phép gồm **mọi
   số Python đã hiển thị** cho các kết quả đó (tiêu đề, công thức, các bước, dòng kết quả, ghi chú số liệu
   từ tài liệu), **số trong câu hỏi** (để nói được "bạn hỏi 2 năm đầu") và `{0, 4, 10}`. Có số ngoài tập
   → `rejected`, bỏ toàn bộ phần AI viết, log `calculation.review_rejected`.
4. JSON hỏng, lỗi gọi model, timeout → bỏ phần AI viết (log `calculation.review_failed` /
   `calculation.review_unreadable`). Không bao giờ chặn câu trả lời: khối Python vẫn hiện.
5. `fits = false` → **giữ khối Python và hiện "Lưu ý"** (26-10-2026; trước đó LLM tính lại
   `redo_with_llm` - đã bỏ cùng nhánh LLM tự tính).
6. Thứ tự hiển thị (`Review.lead`):
   - `fits = false` → `**Lưu ý:** {issue}`, rồi `answer`, rồi khối Python.
   - `fits = true` → `answer`, rồi khối Python.
   - Không có phần AI viết → chỉ khối Python.
7. Vì phần AI viết đứng trước, khối Python chỉ được stream sau khi bước kiểm tra xong: token đầu tiên đến
   muộn hơn khoảng một lần gọi LLM. Phần review hiện ra luôn qua kiểm tra số (bước 3).
8. Trace staff (7.2) của mỗi item: `review: {fits, issue, rejected}` và
   `prompt_versions.chat_calculation_review`.

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
  mọi câu trả lời: field → canonical_field (tên chuẩn trong known_metadata_fields.json)
      field scope profile (hoặc field advisory lạ) → hồ sơ, giá trị chuẩn hoá (khoa_nhap_hoc = năm)
  PendingCalculationTask: values = hồ sơ (tham số scope profile) ∪ known_params ∪ {field: answer}
      registry → builtin_task: tra tài liệu lại nếu cần, calculate hoặc solver theo plan.solve_for/target
      llm (round lưu trước 26-10-2026) → Unresolved("legacy_llm"): câu cố định, mời hỏi lại
  PendingAdvisoryTask: hồ sơ đã cập nhật → _run_advisory_flow
```

Còn thiếu thì tạo panel mới với `chain_depth + 1` (không giới hạn; field đã trả lời không bao giờ bị hỏi lại).

## 6. Prompts

| File | Nội dung |
|---|---|
| `agents/calculation_extractor.yaml` | mục 1: registry (xuôi / ngược: `solve_for`, `target`) / previous / none |
| `main/chat_academic_advisory.yaml`, `main/chat_multi_intent_synthesis.yaml` (luật cuối) | mục 2: số học thuần trên số đã dẫn → khối `calc_request` (code tính, nối khối dưới câu trả lời); kết luận nghiệp vụ/công thức có điều kiện → chỉ trích dẫn |
| `main/chat_calculation_review.yaml` | kiểm tra kết quả cài sẵn có khớp câu hỏi và viết câu trả lời mở đầu (mục 3.2) |
| `agents/message_classification.yaml` | quy tắc 1c nối tiếp phép tính, `<previous_calculation_turn>` (mục 9); `{calculation_formulas}` = registry, quy tắc 5: chỉ công thức registry mới là `academic_calculation` |
| `common/response_style.yaml` | phép nhân viết `×` |
| `main/chat_academic_advisory.yaml`, `main/chat_multi_intent_synthesis.yaml` | `{calculation_results}`: chỉ tiêu đề phép tính đã hiện |

## 7. Trace và phản hồi

### 7.1 (Đã bỏ 26-10-2026) Công tắc `CHAT_CALC_LLM_ENABLED`

Không còn nhánh LLM tự tính nên không còn công tắc. Mọi item có `mode = "builtin"`; `mode = "llm"` chỉ
còn trong message cũ (client vẫn nhận enum cả hai).

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

  Với `mode = "builtin"` (mọi item từ 26-10-2026): `formula_id`, `formula_version`, `formula_hash`
  (hash của `id@version`), `solve_for`, `target`, `inputs`, `outputs`, `sources` (chunk của số liệu tra
  từ tài liệu), và khi lượt chỉ có calculation đã qua bước kiểm tra (mục 3.2): `review: {"fits",
  "issue", "rejected"}`.

- Không có API public nào đọc bảng này. Staff xem trace **thông qua ticket** (7.3), không đọc thẳng
  bảng. Dữ liệu tự xoá theo message nhờ cascade; chính sách lưu trữ riêng (nếu cần) để ticket sau.
- `chunk_hash` là hash của `content` chunk tại thời điểm tính, dùng thay số phiên bản tài liệu (payload
  Qdrant chưa có). `prompt_versions` là hash nội dung file YAML, tính một lần lúc load template. `models`
  lấy từ credential thực sự được dùng, sau failover.
- Builtin cũng có trace, để tiện debug, dù không có nút phản hồi.

### 7.3 Phản hồi Đúng/Sai

Áp dụng cho **mỗi item có `mode = "llm"` và `status = "computed"`** - từ 26-10-2026 không còn item nào
như vậy được tạo (mọi phép tính do Python); backend và web giữ nguyên để message cũ vẫn hiển thị.

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

## 8. Phép tính gần đây (`recent_calculations`, 13-10-2026)

- Cột JSON trên `conversation_clarification_states` (migration `b5c6d7e8f9a0`, thay `last_calculation`):
  tối đa 5 phép tính gần nhất của hội thoại (cũ trước), **mọi** phép tính của lượt được nối vào. Mỗi phép
  tính giữ số của riêng nó: cài sẵn kèm `params`; LLM kèm các giá trị đã biết (tên chuẩn nếu có trong sổ,
  nhãn câu hỏi nếu không; bỏ lý do review). Lượt không có phép tính giữ nguyên danh sách.
- Phần tử cuối là `GraphInput.last_calculation`: câu nối tiếp tính xuôi `formula_id="previous"` gộp số cũ
  với số mới, và `title` đi vào classifier (`<previous_calculation_turn>`, mục 9).
- Thứ tự ưu tiên tham số của công thức cài sẵn: số gõ ở lượt này (extractor) > hồ sơ sinh viên (tham số
  scope `profile`: `khoa_nhap_hoc`, `khoi_nganh`) > số của phép tính trước. Câu giả định "khoá 2021 thì
  sao" không bị hồ sơ ghi đè; khoá đã sửa trong hồ sơ thay khoá cũ của phép tính trước.
- Node 10 không nhận số: `{calculation_results}` chỉ có tên phép tính kèm `[đã có: <tên field>]`, và câu
  hỏi form của node 10 có field nằm trong danh sách đó bị bỏ.

## 9. Classifier qua nhiều lượt

Nguyên tắc: **câu trả lời cho câu hỏi của bot không bao giờ được phân loại lại.**

1. Mọi câu hỏi lại (thiếu số, nhiều công thức, thuộc tính advisory) đi qua panel; lượt submit vào
   thẳng `_run_resume`, không gọi classifier. Panel đang mở thì tin nhắn thường bị chặn (`409`).
2. Câu gõ tự do sau một phép tính: classifier nhận `<previous_calculation_turn>` (tên phép tính trong
   phép tính gần nhất của `recent_calculations`) - tín hiệu có cấu trúc thay vì đoán từ lịch sử chat.
3. Quy tắc 1c trong `message_classification.yaml`: câu bổ sung dữ kiện/đổi số/hỏi ngược về phép tính
   trước giữ `academic_calculation`, `query` gộp yêu cầu tính ban đầu với dữ kiện mới; câu hỏi quy định
   mới vẫn là advisory. Test giữ quy tắc và ví dụ trong prompt.
4. Mục "Cách viết `query`" (13-10-2026) áp dụng cho mọi task: giữ nguyên từ ngữ, con số và mọi chữ phủ
   định/điều kiện; sửa lỗi chính tả, viết tắt khi chắc nghĩa; chỉ thêm ngữ cảnh lịch sử khi tin nhắn
   không đứng riêng được; chuyển ý định thì không kéo ngữ cảnh cũ; "kỳ này" giữ nguyên, không thay bằng
   học kỳ tự suy.
5. Cập nhật hồ sơ (13-10-2026): classifier nhận `<student_profile>` và bảng field `profile`, trả thêm
   `metadata_updates` khi sinh viên tự khai/sửa thông tin của mình (SPEC-clarification-panel §2.8).

## Thời gian hiện tại (`<current_time>`, 13-10-2026)

Mọi luồng chat (classifier, HyDE, decomposer, node 10 hai khung, ticket fallback, extractor, value
extractor, LLM tự tính, review) nhận cùng khối `common/current_time.yaml`: ngày hôm nay (ICT) và học kỳ
hiện tại từ `app/core/academic_calendar.py` (lịch dự kiến chung 2026-2027: HK I 07/09 - nửa đầu 01,
HK II nửa cuối 02 - cuối 05, HK III đầu 06 - giữa 07; ngày rơi vào kỳ nghỉ ghi "đang nghỉ, học kỳ tới").
Lịch khác theo khoa nên câu trả lời dựa vào học kỳ phải kèm một câu lưu ý để sinh viên tự xác nhận;
"kỳ thứ N" của khoá học không được quy đổi sang học kỳ năm học khi chưa biết khoá.

Học kỳ của riêng sinh viên (dòng thứ ba của khối): mỗi năm học 2 học kỳ chính đánh số liên tục từ năm
nhập học, học kỳ hè không đánh số (4 năm = 8, 4,5 năm = 9): HK I → `2 × (năm đầu của năm học − năm nhập
học) + 1`, HK II → `+ 2`; kỳ hè ghi "học kỳ hè sau học kỳ N", kỳ nghỉ ghi học kỳ tới. Khoá IUH: khóa N =
nhập học năm 2004+N. Nguồn là hồ sơ sinh viên `khoa_nhap_hoc` (năm 4 chữ số). Chưa có thì khối kèm
bảng tra năm nhập học → học kỳ và dặn hỏi MỘT câu `choice` field `khoa_nhap_hoc` (`id` = năm, `label`
"Nhập học 2022 (đang học kỳ 9)", cho tự gõ). Câu trả lời (form advisory hay calculation) được lưu vào
hồ sơ sinh viên dưới dạng năm (`parse_intake_year`: năm, "K20", "kỳ 9" quy đổi tại thời điểm trả lời)
và lượt resume dùng ngay. Khối thay cho
`<current_date>` cũ của node 10. Review không được viết số của khối này (kiểm tra số mục 3.2 vẫn chặt).

## Commands

```
.venv/bin/pytest -q tests/graph tests/rag tests/calculation
.venv/bin/pytest -q tests/e2e/test_calculation_flow_e2e.py      # dùng fake_llm_provider
.venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests && .venv/bin/mypy app
```

## Testing Strategy

- **Extractor** (`tests/graph/test_calculation_task.py`): lựa chọn registry của model được giữ; router
  chỉ điền khi output hỏng; `none` giữ `retrieval_query`; `solve_for`/`target` đọc được và hỏng; `previous`
  cần phép tính gần nhất registry; task `none` → `Deferred`; phép ngược → `Computed` với `plan.solve_for`;
  ẩn số không bị hỏi; resume plan ngược; resume plan `llm` cũ → `legacy_llm`.
- **Công thức registry:** đủ số → `Computed`; thiếu → `NeedsInput` từ `ParamSpec`; điểm 11 → hỏi lại kèm lý do.
- **Solver** (`tests/calculation/test_solver.py`): các câu ngược của tập eval (d0308, d0309, d0311, d0313),
  điểm chữ làm mục tiêu, tín chỉ tối đa dưới ngân sách, tham số giảm, mục tiêu `==`, không đạt được.
- **Graph** (`tests/graph/test_calculation_node.py`): lượt chỉ có tính toán (khối tính + nhận xét đã
  kiểm tra số); thiếu số → panel; lượt hỗn hợp → một panel, khối tính trước advisory; câu hỏi ngược sau
  một phép tính → Python giải, trace `solve_for`/`target`/`formula_version`; câu không có công thức → đi
  advisory, không có item calculation; lookup tài liệu được lọc như advisory.
- **Classifier:** `<previous_calculation_turn>` được gửi; quy tắc 1c và ví dụ còn trong prompt.

## Boundaries

- **Always:** mọi con số kết quả do Python tính từ công thức registry (xuôi hoặc solver); chunk qua access
  filter trước khi tới LLM; mọi câu hỏi lại đi qua panel; mỗi công thức có `version` và `source`.
- **Ask first:** thêm/đổi công thức registry (nghiệp vụ); đổi cách hiển thị "không có công thức".
- **Never:** để LLM tính nhẩm hay nghĩ câu hỏi cho công thức registry; gọi lại extractor khi resume; đưa
  con số phép tính vào prompt node 10.

## Success Criteria

- [x] Câu ĐTKHP đủ số → khối tính `7.5 / B / 3.0`, không có panel.
- [x] GPA không kèm điểm → panel `course_table`; submit → GPA 2 chữ số thập phân.
- [x] "Cuối kỳ cần bao nhiêu để được A+" → Python giải trên lưới 0,5, có dòng kiểm tra giá trị liền kề.
- [x] Điểm xét tuyển 2026 (xuôi: 25,4; ngược ĐHB: 25,5), học bạ 3 học kỳ (HK1 Lý: 5,5), trọng số đề
      cương (6,8; ngược 7,1 / 9,5) do Python tính; thiếu số → panel.
- [x] Câu không có công thức ("tổng số đơn phúc khảo") → trả lời advisory từ tài liệu, không có item
      calculation, không có con số do model tính.

## Decisions (09-10-2026)

- Giữ bước nhận xét LLM (`chat_calculation.yaml`) cho lượt chỉ có calculation. (13-10-2026: thay bằng
  bước kiểm tra và trả lời `chat_calculation_review.yaml`, mục 3.2.)
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
- (26-10-2026, UNISAGE-106) **Đảo lại: bỏ nhánh LLM tự tính, quay về Python cho cả tính ngược** - nhưng
  với bộ giải mới đơn giản (tìm nghiệm trên lưới của tham số bằng chính công thức xuôi, không có
  evaluator biểu thức) và registry công thức có version thay vì chép công thức Qdrant. Lý do: 4 lần đo 38
  câu (19 → 26/38) cho thấy LLM tính nhẩm sai 4-5 câu mỗi lần và nhiễu ±3-4 câu giữa hai lần chạy cùng
  prompt; một câu "đậu" chỉ nhờ cách chấm toàn văn. Câu không có công thức registry không được tính hộ
  mà hiển thị công thức/số liệu từ tài liệu. Thêm `admission_score_2026`, `transcript_score_3hk`,
  `weighted_score`.
