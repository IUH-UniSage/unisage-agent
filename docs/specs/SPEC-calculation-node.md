# Spec: calculation-node (unisage-agent)

> **Status: Implemented** (UNISAGE-99, nhánh `feature/huydh-unisage-99-calculation-flow`). Rủi ro còn
> lại nằm ở `unisage-agent/docs/specs/known-gaps.md`.

Module id `calculation-node` trong [capability map](../../changes/09-10-2026-calculation-flow/capability-map.md).
Phụ thuộc: [SPEC-calc-engine](SPEC-calc-engine.md) và [SPEC-clarification-panel](SPEC-clarification-panel.md).

## Objective

Thay placeholder `CalculationNode` bằng luồng thật cho các task `academic_calculation`:

1. Nhận diện cần tính gì và trích xuất tham số từ câu hỏi (LLM, chỉ đọc hiểu).
2. Chọn công thức: một trong 3 công thức cài sẵn, hoặc tìm trong quy chế (Qdrant).
3. Tham số nào còn thiếu hoặc không hợp lệ thì thành câu hỏi trên panel, gộp chung với câu hỏi của
   advisory trong cùng lượt.
4. Python tính (`calc-engine`), rồi **Python render** công thức và từng bước. LLM chỉ viết phần nhận
   xét ngắn, không viết lại con số nào.

## Luồng của một task calculation

```
              ┌──────────────── calculation_extractor (LLM, 1 lần/task) ───────────────┐
 task.query ─►│ {formula_id: gpa|course_score|grade_conversion|retrieved, params, retrieval_query} │
              └─────────────────────────────────────────────────────────────────────────┘
                 │ formula cài sẵn                         │ retrieved
                 │                                         ▼
                 │                    retrieve_chunks(retrieval_query) + access filter
                 │                                         ▼
                 │                    calculation_formula (LLM): status found|ambiguous|not_found
                 │                                         ▼
                 │                    kiểm tra provenance + expression.validate()
                 │                         │ found & hợp lệ          │ ambiguous / not_found / sai
                 ▼                         ▼                         ▼
        formulas.missing_params / validate tham số              TaskOutcome.unresolved (không tính)
                 │ thiếu/sai                    │ đủ
                 ▼                              ▼
     TaskOutcome.needs_input(questions)    TaskOutcome.computed(CalculationResult)
```

### Trạng thái của task (`app/graph/nodes/calculation.py`)

```python
@dataclass(frozen=True)
class CalculationPlan:            # cũng là field `plan` trong PendingCalculationTask
    formula_id: Literal["gpa", "course_score", "grade_conversion", "retrieved"]
    retrieved: RetrievedFormulaWithSource | None   # chỉ khi formula_id == "retrieved"

TaskOutcome = (
    Computed(task_id, result: CalculationResult, source: Citation | None)
  | NeedsInput(task_id, plan: CalculationPlan, known_params: dict, questions: list[Question])
  | Unresolved(task_id, reason: Literal["extraction_failed", "formula_not_found", "formula_ambiguous", "formula_invalid"],
               candidates: list[FormulaCandidate])
)
```

`run_calculation_task(task, ...) -> TaskOutcome` là hàm duy nhất graph gọi cho mỗi task.

## 1. Extractor: viết lại `agents/calculation_extractor.yaml`

Output contract mới. LLM **không** tự nghĩ câu hỏi; danh sách câu hỏi do `formulas.missing_params`
dựng ra.

```json
{
  "formula_id": "course_score",
  "params": {"tbtx": 8, "gk": 7, "ck": 6.5, "th": [9, 8], "tclt": 2, "tcth": 1},
  "retrieval_query": null
}
```

- Prompt liệt kê đúng 3 công thức cài sẵn kèm tên param, lấy từ `ParamSpec` (render vào template lúc
  load, nên không phải khai báo hai nơi). Không còn `tuition_calculation`, `credit_check`.
- Không khớp công thức cài sẵn nào thì trả `formula_id: "retrieved"` và `retrieval_query` là câu tìm
  kiếm văn phong quy chế, ví dụ "công thức tính học phí theo số tín chỉ đăng ký".
- Chỉ trích những số **người dùng đã nói**. Không suy đoán, không điền mặc định.
- Parse theo pattern sẵn có (`Agent[None, str]`, `_load_json_object`, failover). **Fail closed:** JSON
  hỏng, `formula_id` lạ hoặc LLM lỗi thì ra `Unresolved("extraction_failed")` với câu cố định "Mình
  chưa hiểu bạn cần tính gì, bạn nói rõ hơn giúp mình (ví dụ: tính GPA, tính điểm tổng kết học phần)
  nhé." **Không** rơi sang nhánh Qdrant.

### 1.1 Định tuyến công thức cài sẵn bằng luật (trước LLM)

`formulas.py` khai báo `BUILTIN_TRIGGERS: dict[formula_id, re.Pattern]`, chạy trên câu hỏi đã chuẩn hoá
(NFC, chữ thường):

| formula_id | Pattern (rút gọn) |
|---|---|
| `gpa` | `\bgpa\b`, `điểm trung bình (tích lũy\|học kỳ\|chung)`, `thang (điểm )?4` kèm nhiều môn |
| `course_score` | `(điểm )?tổng kết (học phần\|môn)`, `lý thuyết.*thực hành`, `\b(tx\|gk\|ck)\b` |
| `grade_conversion` | `quy đổi`, `điểm chữ`, `(được\|là) (điểm )?[abcdf]\+?\b` |

- Khớp **đúng một** pattern → `formula_id` bị khoá theo luật. LLM extractor chỉ còn trích `params`;
  `formula_id` mà LLM trả về bị bỏ qua (khác luật thì ghi log `calculation.router_disagreement`).
- Khớp **nhiều** pattern → đưa danh sách ứng viên vào prompt; LLM chỉ được chọn trong danh sách đó,
  không được chọn `retrieved`.
- Không khớp pattern nào → LLM được chọn tự do giữa 3 công thức cài sẵn và `retrieved`.

Nhờ vậy một câu hỏi GPA không thể bị đưa nhầm sang công thức lấy từ Qdrant. Bộ pattern được test bằng
một danh sách câu hỏi mẫu (`tests/calculation/test_builtin_triggers.py`), mỗi công thức ≥ 10 câu khớp
và ≥ 10 câu không khớp.
- Param mà Python từ chối (sai kiểu, ngoài khoảng) **vẫn được hỏi lại**, kèm lý do trong `prompt` của
  câu hỏi, ví dụ "Điểm cuối kỳ (bạn nhập 11, điểm phải từ 0 đến 10)".

Câu hỏi được dựng từ `ParamSpec`:

| `ParamSpec.kind` | `Question.kind` | Ghi chú |
|---|---|---|
| `number` | `number` | `min`/`max`/`step` lấy từ spec, `tab_label` lấy từ label rút gọn |
| `number_list` | `number_list` | Dùng cho điểm thực hành TH1..THn |
| `course_table` | `course_table` | Dùng cho GPA |

## 2. Nhánh Qdrant: `agents/calculation_formula.yaml` (mới)

Input: các chunk đã qua `build_access_filter` (dùng chung `retrieve_chunks`, `limit = 5`, không HyDE),
kèm câu hỏi gốc và các param mà extractor đã trích được. Output:

```json
{
  "status": "found",
  "formula": {
    "expression": "so_tc * don_gia_tc",
    "variables": [{"name": "so_tc", "label": "Số tín chỉ đăng ký", "unit": "TC", "min": 1, "max": 40},
                  {"name": "don_gia_tc", "label": "Đơn giá một tín chỉ", "unit": "đồng", "min": 0, "max": 10000000}],
    "result_label": "Học phí học kỳ",
    "values": {"so_tc": 20},
    "source_chunk_id": "c_123",
    "source_quote": "Học phí = số tín chỉ đăng ký × đơn giá tín chỉ"
  },
  "candidates": []
}
```

### Provenance và kiểm tra ngữ nghĩa (fail closed)

Công thức chỉ được dùng khi qua **tất cả** các kiểm tra sau, theo đúng thứ tự (rẻ trước, đắt sau). Trượt
một kiểm tra thì `Unresolved("formula_invalid")` và log `calculation.formula_rejected` kèm số thứ tự của
kiểm tra bị trượt.

1. `source_chunk_id` là một chunk trong danh sách vừa retrieve cho chính người dùng này.
2. `source_quote` (chuẩn hoá NFC, gộp khoảng trắng, không phân biệt hoa thường) là **chuỗi con** của
   `content` của chunk đó. Câu trích không có thật thì bị từ chối.
3. `expression.validate(formula)` pass (grammar allowlist, giới hạn, biến khớp, xem SPEC-calc-engine).
4. `values` chỉ chứa những biến đã khai báo, và mỗi giá trị nằm trong `min`/`max` của biến đó.
5. **Hằng số được neo vào câu trích:** mọi hằng số trong biểu thức (trừ `0`, `1`, và đối số làm tròn
   của `round`) phải xuất hiện trong `source_quote`. Phần trăm trong câu trích được quy đổi thành số
   (`20%` ↔ `0.2`), và `,` được coi như `.`. Hệ số mà câu trích không có thì nghĩa là LLM tự thêm vào.
6. **Biến được neo vào câu trích:** mỗi `variable.label` phải có ít nhất một từ nội dung (≥ 3 ký tự,
   không thuộc stopword tiếng Việt như "của", "các", "số"…) xuất hiện trong `source_quote` đã chuẩn
   hoá. Biến không có gì trong câu trích tương ứng thì nghĩa là LLM tự bịa ra biến.
7. **Verifier LLM độc lập** (`agents/calculation_formula_verifier.yaml`): một lần gọi riêng, **không**
   thấy câu hỏi của người dùng hay output của lần chép, chỉ nhận `source_quote`, `expression`,
   `variables` (name + label) và trả `{"equivalent": true|false, "reason": "..."}`. Prompt yêu cầu
   kiểm tra từng toán tử, từng hệ số, và biến nào ở tử hay ở mẫu. `false`, JSON hỏng hoặc LLM lỗi đều
   bị coi là trượt.

### Khi có nhiều công thức hoặc không có công thức nào

- `status = "ambiguous"`: các chunk có **từ 2 công thức trở lên mâu thuẫn nhau** (khác khoá, khác hệ,
  khác năm). Trả về `candidates: [{summary, source_chunk_id}]`, không tính. Câu trả lời liệt kê các
  công thức kèm nguồn và hỏi sinh viên thuộc trường hợp nào (dạng text thường, không phải panel, vì
  không có cách nào chắc chắn để biến đây thành câu hỏi có options).
- `status = "not_found"`, hoặc không có chunk nào: trả lời cố định "Mình chưa tìm thấy công thức này
  trong quy chế hiện có…" và gợi ý liên hệ Phòng Đào tạo. **Không tính, không đoán công thức.**
- Prompt nói rõ: công thức mơ hồ hoặc thiếu hệ số thì trả `not_found` hoặc `ambiguous`, không tự bổ
  sung cho đủ.

### Trình bày kết quả công thức Qdrant

Khác với 3 công thức cài sẵn, kết quả từ Qdrant **luôn** được trình bày như sau (Python render):

- Nhãn đầu khối: **"Kết quả tham khảo theo quy chế"**, cuối khối có nút Đúng/Sai (mục 7.3).
- `source_quote` nguyên văn, kèm tên văn bản (`chunk.source`, `heading_path`) và citation giống nhánh
  advisory.
- Biểu thức đã chép, rồi các bước thế số và kết quả.

Không hỏi sinh viên xác nhận công thức (đã chốt 09-10-2026). Rủi ro còn lại là cả 7 kiểm tra đều sót
một lỗi ngữ nghĩa. Không làm eval trước khi phát hành (đã chốt 09-10-2026). Thay vào đó là công tắc tắt
khẩn cấp, trace đầy đủ, và phản hồi Đúng/Sai tạo ticket cho admin (mục 7). Rủi ro này được ghi vào
`known-gaps.md`.

## 3. Graph wiring (`streaming_graph.py`)

### Lượt thường: barrier trước node 10

```
plan_route
  ├─ calc_future = gather(run_calculation_task(t) for t in calculation_tasks)         (bắt đầu ngay)
  └─ advisory: query transformation → retrieval → rerank → web search                 (song song với calc)
                                    │
                         ══ BARRIER: await calc_future ══
                                    │
     1. stream khối tính (Python render) cho mỗi Computed, theo thứ tự T1..T3
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

Chỉ dùng cho lượt **chỉ có** calculation và có ít nhất một `Computed`.

1. Gọi LLM **không stream** (`run_agent_text_with_failover`). Nhận xét ngắn (1–3 câu, `max_tokens` thấp),
   nên chờ trọn vẹn rồi mới gửi không ảnh hưởng trải nghiệm.
2. Trích mọi số trong text bằng regex `\d+(?:[.,]\d+)?`, chuẩn hoá `,` thành `.`, bỏ số 0 thừa.
3. Whitelist gồm: mọi giá trị trong `inputs`, `outputs`, `step.display` (bỏ dấu `≈`), cùng các hằng
   `{0, 4, 10}` (tên thang điểm).
4. Có số **ngoài whitelist** → bỏ nhận xét của LLM, thay bằng câu cố định do Python dựng theo
   `formula_id` (ví dụ "Học phần đạt điểm chữ B, tương đương 3.0 trên thang 4."), và ghi log
   `calculation.commentary_rejected`. LLM lỗi hoặc timeout cũng dùng câu cố định này.
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

Input: `PendingRound` đã claim + `dict[question_id, NormalizedAnswer]` đã validate.

```
answers → nhóm theo question.task_id
  PendingCalculationTask: params = known_params ∪ {question.field: answer}
      → formulas.calculate / expression.evaluate (không gọi LLM extractor, không retrieve lại)
      → Computed | NeedsInput (vẫn sai, hiếm: chỉ khi ràng buộc chéo như TCLT + TCTH = 0)
  PendingAdvisoryTask: confirmed_metadata ∪ {field: option_id | other_text}
      → _run_advisory_flow(origin_task) như hiện nay
```

- `other_text` của câu hỏi advisory được ghi vào `confirmed_metadata` dưới dạng text tự do. Prompt
  `<student_declared_attributes>` đã coi metadata là thông tin sinh viên tự khai, không phải bộ lọc.
- Các task calculation và advisory chạy song song như một lượt thường. Thứ tự stream giống mục 3.
- Resume mà vẫn còn thiếu thì tạo panel mới với `chain_depth + 1` (không giới hạn, xem SPEC-clarification-panel §2.6).
- Câu hỏi gốc của task được dùng làm câu hỏi (`original_query`), không dùng bản tóm tắt câu trả lời.

## 6. Prompts

| File | Trạng thái | Nội dung |
|---|---|---|
| `agents/calculation_extractor.yaml` | viết lại | mục 1 |
| `agents/calculation_formula.yaml` | mới | mục 2: chép công thức |
| `agents/calculation_formula_verifier.yaml` | mới | mục 2, kiểm tra 7: so biểu thức với câu trích, không thấy câu hỏi |
| `main/chat_calculation.yaml` | mới | nhận xét cho lượt chỉ có calculation (mục 3.2, không stream, kiểm tra số); dùng `{header}`, `{response_style}`, `{calculation_payload}` (outputs + inputs), `{user_query}` |
| `common/task_2.yaml`, `common/ask_user_form_guide.yaml` | sửa | bỏ phần mô tả Type A cũ và `chat_calculation_result.yaml` không tồn tại; nói rõ form luôn bị hệ thống lọc khỏi câu trả lời |
| `main/chat_academic_advisory.yaml`, `main/chat_multi_intent_synthesis.yaml` | sửa | thêm `{calculation_results}`: **chỉ tiêu đề** các phép tính đã hiện, không có số (chuỗi "Không có" khi không có phép tính) |

Mỗi file mới phải thêm field vào `PromptTemplates` (`schema.py`) và một dòng trong `loader.py`. Snapshot
`tests/rag/fixtures/advisory_prompt_snapshot.txt` được cập nhật.

## 7. Công tắc, trace và phản hồi cho công thức Qdrant

Thay cho bộ eval trước khi phát hành (đã chốt 09-10-2026), có ba thứ: một công tắc tắt khẩn cấp, trace
đầy đủ cho mỗi lần tính, và phản hồi Đúng/Sai từ người dùng để admin điều tra.

### 7.1 Công tắc `CHAT_CALC_RETRIEVED_FORMULA_ENABLED` (mặc định `True`)

- `True`: nhánh Qdrant chạy đầy đủ như mục 2 (7 kiểm tra, rồi Python tính).
- `False`: vẫn retrieve và chạy kiểm tra 1–2 (chunk id, câu trích có thật), nhưng **không tính, không
  hỏi tham số**. Chỉ hiện nguyên văn câu trích kèm nguồn, cùng câu cố định "Bạn thế số vào công thức
  trên để tính nhé, hiện mình chưa tự tính loại công thức này." Ba công thức cài sẵn không bị ảnh hưởng.
- Đổi qua biến môi trường hoặc `.env` rồi restart agent. Không cần màn hình admin.

### 7.2 Trace: tách phần công khai và phần chỉ staff đọc được

Nguyên tắc: **message mà client nhận về chỉ chứa những gì cần để hiển thị.** Câu hỏi gốc, điểm số, tín
chỉ, biểu thức, câu trích, model và prompt version nằm trong một store **chỉ staff đọc được**, không bao
giờ đi qua `GET /messages/...` hay SSE.

**(a) Phần công khai: `metadata.calculation` trên message ASSISTANT** (agent ghi trong PATCH finalize)

```json
{"calculation": {"schema_version": 1, "items": [
  {"item_id": "T1", "run_id": "<budget.request_id>", "mode": "retrieved", "status": "computed",
   "result_summary": "Học phí học kỳ: 8.400.000 đồng",
   "source_summary": {"title": "QĐ-123.pdf", "heading": "Chương II › Điều 8"}}
]}}
```

`result_summary` và `source_summary` vốn đã nằm trong `content` mà người dùng thấy, nên không lộ thêm gì.
Không có `question_raw`, `inputs`, `expression`, `source_quote`, `models` hay `prompt_versions`.

**(b) Phần chỉ staff: bảng `calculation_traces` (backend)**

- Agent gọi `POST /internal/calculation-traces` (dưới `/internal/**`, có `X-Internal-Secret`, đi theo tiền
  lệ `InternalUsageLogController`) **trước** PATCH finalize, mỗi lượt một lần, chứa mọi item của lượt đó.
  Thử lại 3 lần; vẫn lỗi thì log `calculation.trace_push_failed` và lượt chat vẫn tiếp tục. Trace là dữ
  liệu chẩn đoán, không nằm trên đường chính.
- Bảng có các cột: `id`, `message_id` (FK → `messages`, `ON DELETE CASCADE`), `item_id`, `run_id`,
  `trace jsonb`, `created_at`. Có `UNIQUE (message_id, item_id)`. Gửi lại cùng một cặp này thì upsert,
  nên idempotent khi thử lại.
- Nội dung `trace` (tối đa 16 KB mỗi item):

  ```json
  {"question_raw": "…", "formula_id": "retrieved", "expression": "so_tc * don_gia_tc",
   "variables": [{"name": "so_tc", "label": "Số tín chỉ đăng ký"}], "formula_hash": "sha256:<12>",
   "inputs": [{"name": "so_tc", "label": "Số tín chỉ đăng ký", "value": "20", "origin": "message"}],
   "outputs": [{"label": "Học phí học kỳ", "value": "8400000"}],
   "source": {"chunk_id": "c_123", "document_id": "d_9", "source": "QĐ-123.pdf",
              "heading_path": ["Chương II", "Điều 8"], "chunk_hash": "sha256:<12>"},
   "source_quote": "…", "checks": [{"n": 1, "passed": true}],
   "models": {"extractor": "…", "formula": "…", "verifier": "…"},
   "prompt_versions": {"calculation_extractor": "sha256:<12>"}}
  ```

- Không có API public nào đọc bảng này. Staff xem trace **thông qua ticket** (7.3), không đọc thẳng
  bảng. Dữ liệu tự xoá theo message nhờ cascade; chính sách lưu trữ riêng (nếu cần) để ticket sau.
- `chunk_hash` là hash của `content` chunk tại thời điểm tính, dùng thay số phiên bản tài liệu (payload
  Qdrant chưa có). `prompt_versions` là hash nội dung file YAML, tính một lần lúc load template. `models`
  lấy từ credential thực sự được dùng, sau failover.
- Builtin cũng có trace, để tiện debug, dù không có nút phản hồi.

### 7.3 Phản hồi Đúng/Sai

Áp dụng cho **mỗi item có `mode = "retrieved"` và `status = "computed"`**. Công thức cài sẵn không có
nút này.

- Khối kết quả có nhãn đầu là **"Kết quả tham khảo theo quy chế"**, cuối khối có hai nút **Đúng** và
  **Sai**.
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
   `mode = "retrieved"` và `status = "computed"`. Không thoả thì trả `404`.
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

## Commands

```
.venv/bin/pytest -q tests/graph tests/rag tests/calculation
.venv/bin/pytest -q tests/e2e/test_calculation_flow_e2e.py      # dùng fake_llm_provider
.venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests && .venv/bin/mypy app
```

## Testing Strategy

- **Unit, `run_calculation_task`** (LLM giả bằng `tests/llm_mocks.py`): công thức cài sẵn đủ tham số
  → `Computed`; thiếu `th` khi `tcth > 0` → `NeedsInput` với đúng câu hỏi; điểm 11 → câu hỏi có lý do;
  extractor trả JSON hỏng, `formula_id` lạ hoặc LLM lỗi → `Unresolved("extraction_failed")`, **không retrieve**
  (mock retrieval có 0 lời gọi).
- **Provenance:** mỗi kiểm tra 1–7 có ít nhất một ca trượt riêng, gồm: chunk id lạ; câu trích bịa;
  biểu thức bị allowlist từ chối; giá trị ngoài khoảng; hệ số `0.4` không có trong câu trích; biến
  "phí bảo hiểm" không có trong câu trích; verifier trả `false`, JSON hỏng hoặc timeout. Thêm các ca
  `ambiguous`, `not_found`, không có chunk nào, và ca hợp lệ: khối có nhãn "Kết quả tham khảo" và câu
  trích.
- **Graph:** chỉ calculation, đủ tham số (khối tính + nhận xét, không có panel); chỉ calculation, thiếu
  tham số (câu dẫn cố định + panel, không gọi LLM generation); có cả hai loại đều thiếu (**một** panel
  có câu hỏi của cả hai origin); có cả hai loại, calculation đủ còn advisory thiếu; resume chỉ chạy các
  task còn pending, không gọi lại extractor; chain 3 panel thì dừng.
- **Kiểm tra số trong nhận xét:** số lạ, số viết kiểu `7,5`, số có `≈`, LLM timeout → đều ra câu cố định; nhận xét hợp lệ được giữ nguyên.
- **Barrier:** advisory chuẩn bị xong trước calculation vẫn không gửi token nào trước khi khối tính được gửi (mô phỏng bằng calc chậm).
- **Fail closed + router luật:** extractor trả JSON hỏng thì ra `extraction_failed`, không retrieve; câu GPA khớp luật mà LLM trả `retrieved` thì vẫn chạy `gpa`.
- **Trace và công tắc:** `metadata.calculation` **không** chứa `question_raw`/`inputs`/`expression`/`source_quote`/`models` (test kiểm tra từng key); trace đầy đủ được push sang `/internal/calculation-traces` trước finalize; push lỗi thì lượt vẫn chạy và có log; `prompt_versions` đổi khi file YAML đổi; `models` phản ánh credential sau failover. Công tắc tắt: chỉ hiện câu trích, không gọi `evaluate`, không có panel.
- **E2E** (`tests/e2e/test_calculation_flow_e2e.py`): hỏi ĐTKHP thiếu điểm TH → event `clarification`
  → submit → khối tính ra `7.5 / B / 3.0`.

## Boundaries

- **Always:** con số chỉ đến từ `calc-engine`; khối tính được render bằng Python; chunk qua access
  filter trước khi tới LLM công thức.
- **Ask first:** thêm công thức cài sẵn thứ 4; bỏ hoặc nới bất kỳ kiểm tra nào trong 7 kiểm tra
  provenance; bỏ nhãn "tham khảo"; cho LLM tự viết phần các bước.
- **Never:** tính khi không tìm thấy hoặc không chắc công thức; để LLM nghĩ câu hỏi cho công thức cài
  sẵn; gọi lại extractor hay retrieve lại khi resume; gửi ra client text LLM có số chưa qua whitelist;
  đưa con số phép tính vào prompt node 10; rơi sang nhánh Qdrant khi extractor lỗi.

## Success Criteria

- [ ] Câu "điểm TX 8, GK 7, CK 6.5, TH 9 và 8, môn 2 TC LT 1 TC TH thì tổng kết bao nhiêu" trả về khối
      tính đúng `7.5 / B / 3.0` mà không có panel.
- [ ] Câu hỏi GPA không kèm điểm → panel có một tab `course_table`; submit → GPA đúng 2 chữ số thập phân.
- [ ] Câu hỏi học phí: công thức có trong Qdrant thì tính và hiện nguồn; không có thì trả câu
      "chưa tìm thấy công thức", và log cho thấy không có lời gọi calc-engine nào.
- [ ] Lượt có cả hai loại câu hỏi đều thiếu thông tin → một panel → submit → một câu trả lời gồm khối
      tính, sau đó là phần advisory.
- [ ] `known-gaps.md` bỏ 3 mục "CalculationNode is a placeholder", "two branches at once" và
      "Calculation results don't reach node 10", thêm mục rủi ro "LLM chép sai ý nghĩa công thức".
      `PRODUCT.md` cập nhật mục Not this product / Open. `DECISIONS.md` ghi các quyết định của
      UNISAGE-99.

## Decisions (09-10-2026)

- Giữ bước nhận xét LLM (`chat_calculation.yaml`) cho lượt chỉ có calculation.
- `ambiguous` trả lời bằng text; đưa vào panel là ticket khác.
- (Review lần 2) Barrier trước node 10 thay cho buffer token; nhận xét không stream và bị thay bằng câu
  cố định nếu có số ngoài whitelist; node 10 chỉ nhận tiêu đề phép tính; extractor fail closed; công
  thức cài sẵn được định tuyến bằng luật trước LLM.
- Công thức Qdrant vẫn được tính, nhưng phải qua 7 kiểm tra (neo hằng số, neo biến, verifier LLM độc lập),
  luôn gắn nhãn "Kết quả tham khảo theo quy chế", không hỏi sinh viên xác nhận.
- Không làm eval trước khi phát hành. Thay bằng công tắc `CHAT_CALC_RETRIEVED_FORMULA_ENABLED`, trace và
  nút Đúng/Sai với 5 lý do.
- (Review lần 4) Trace tách đôi: message chỉ chứa `run_id`, `item_id`, `status`, `result_summary` và
  `source_summary`. Trace đầy đủ (câu hỏi gốc, điểm số, biểu thức, câu trích, model, prompt version) nằm ở
  bảng `calculation_traces` bên backend, và staff chỉ xem qua ticket. Mỗi item sai có một ticket riêng nhờ
  partial unique index `(message_id, calculation_item_id)`.
