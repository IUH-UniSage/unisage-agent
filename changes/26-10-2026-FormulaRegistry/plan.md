# Registry công thức có version, bỏ nhánh LLM tự tính (26/10/2026)

Ticket: UNISAGE-106 (tiếp theo Phase 6). Nhánh `feature/huydh-unisage-106-formula-registry`, xếp chồng
trên `feature/huydh-unisage-106-benchmark-cross-encoder` (PR #28).

## Quyết định (người dùng, 26/10 tối)

Sau 4 lần chạy 38 câu calculation (19 → 26/38) kết luận: LLM tính nhẩm trên số liệu KB vẫn sai 4-5 câu
mỗi lần, nhiễu ±3-4 câu giữa hai lần chạy cùng prompt, và một câu "đậu" chỉ nhờ cách chấm toàn văn
(d0309). Quyết định:

1. **Bỏ hẳn nhánh LLM tự tính** trong node calculation (`chat_calculation_llm.yaml`, `LlmAnswered`,
   `_ask_llm`, review-redo, `CHAT_CALC_LLM_*`). LLM chỉ còn hai việc: **định tuyến** tới đúng công thức
   trong registry và **chép tham số** người dùng đã nói.
2. **Registry công thức Python có version**: mỗi công thức là một hàm có `formula_id`, `version`,
   `source` (văn bản gốc), tham số (`ParamSpec`), đầu ra (`OutputSpec`), và được tính **xuôi** (ra kết
   quả) lẫn **ngược** (cho kết quả mong muốn, tìm một tham số) bằng code.
3. Công thức chỉ có trong KB mà registry chưa cài: **hiển thị công thức/số liệu từ tài liệu và để người
   dùng tự tính** (đi nhánh advisory, không ra con số do LLM tính). Thiếu dữ liệu / nhiều phiên bản quy
   định: hỏi lại, không đoán.
4. Triển khai thêm **công thức điểm xét tuyển 2026** của Trường (nhiều tầng: quy đổi ĐGNL về thang 30,
   ĐK = max, XT1/XT2/XT3, ĐXT = max) - xuôi và ngược.

## Thiết kế

### Registry (`app/calculation/formulas.py`, `app/calculation/solver.py`)

```python
@dataclass(frozen=True)
class OutputSpec:            # một đầu ra có thể làm mục tiêu khi tính ngược
    key: str                 # "dxt", "xt1", "dtkhp", "gpa", "hoc_phi"
    label: str

@dataclass(frozen=True)
class Formula:
    formula_id: FormulaId
    version: str             # "2026.1": năm văn bản áp dụng + lần sửa cài đặt
    source: str              # văn bản gốc, hiện trong kết quả và trace
    title, description, params, outputs, compute
```

`CalculationResult.values: dict[str, Decimal]` - giá trị số của từng đầu ra (solver và trace dùng),
`CalculationResult.version` - hiện trong khối kết quả ("Công thức ... (phiên bản 2026.1, nguồn: ...)").

| formula_id | version | nguồn | tham số | đầu ra |
|---|---|---|---|---|
| `course_score`, `gpa`, `grade_conversion`, `tuition_*` | 1 | quy chế đào tạo / biểu phí (như cũ) | như cũ | thêm `values` |
| `admission_score_2026` (mới) | 2026.1 | "Công thức tính điểm xét tuyển đại học chính quy năm 2026" | `dtn` (tổng 3 môn thi TN, 0-30), `dgnl` (điểm ĐGNL, tuỳ chọn), `dtk` (điểm thủ khoa ĐGNL, cần khi có `dgnl`), `dhb` (tổng điểm học bạ 3 môn, 0-30), `uu_tien` (mặc định 0), `diem_cong` (mặc định 0) | `ddgnl`, `dk`, `xt1`, `xt2`, `xt3`, `dxt` |
| `transcript_score_3hk` (mới) | 2020.1 | "Bản mô tả CTĐT ngành Kiểm toán 2020 §4.1" (xét học bạ lớp 10, 11, HK1 lớp 12) | 3 môn × (TB lớp 10, TB lớp 11, TB HK1 lớp 12), `nguong` (mặc định 19,5) | `mon1..mon3`, `tong` |
| `weighted_score` (mới) | 1 | đề cương học phần (trọng số do người dùng/tài liệu nêu) | `trong_so` (list %), `diem` (list, một ô có thể để trống khi tính ngược), `lam_tron_thanh_phan` (tuỳ chọn 0,5) | `diem_hp` |

Solver (`solve(formula_id, params, unknown, target)`): tham số chưa biết là một `ParamSpec` kiểu số (hoặc
một ô trống trong list), mục tiêu là `(output_key, op ∈ {>=, <=, ==}, value)`; `value` có thể là điểm chữ
(A+, B...) → quy về ngưỡng thang 10 theo `GRADE_SCALE`. Tìm kiếm trên lưới `[min, max]` theo `step`
(nhị phân, các công thức đều đơn điệu theo từng tham số), trả `CalculationResult` gồm phép tính xuôi tại
nghiệm + dòng **kiểm tra** với giá trị liền kề ("CK 7,5 → 7,2 đạt; CK 7,0 → 6,9 chưa đạt"). Không có
nghiệm trong miền → kết quả nói rõ "không đạt được dù tối đa", kèm giá trị tại biên.

### Node calculation (`app/graph/nodes/calculation.py`)

- Extractor trả `{"formula_id": <id> | "previous" | "none", "params", "solve_for": <param> | null,
  "target": {"output", "op", "value"} | null, "retrieval_query"}`. Danh sách công thức + tham số + đầu
  ra render từ registry (một nguồn sự thật).
- `formula_id == "none"` → task chuyển sang **advisory** ngay đầu lượt (trước khi retrieval advisory
  chạy): node 10 trả lời từ tài liệu theo luật mới "nêu công thức, số liệu và điều kiện áp dụng, không
  tự tính ra con số". Grounding vẫn gắn caution nếu LLM vẫn tính.
- `CalculationPlan` thêm `solve_for`, `target` để lượt submit panel tính nốt phép ngược; `"llm"` giữ
  trong Literal chỉ để đọc được round cũ đang chờ (resume → câu cố định).
- Bỏ: `LlmAnswered`, `_ask_llm`, `llm_outcome`, `_builtin_handoff`, `llm_questions`, `redo_with_llm`,
  `known_for_llm`, `REVIEW_ISSUE_KEY`, `CHAT_CALC_LLM_ENABLED/TIMEOUT`, template `chat_calculation_llm`.
  Review lượt chỉ-tính giữ nguyên (LLM viết câu mở đầu, số bị kiểm tra); không khớp → "Lưu ý".
- Trace/public item: `mode` luôn `builtin` (web chỉ nhận `builtin|llm`), `formula_version` thêm vào
  trace staff.

### Prompt

- `agents/calculation_extractor.yaml`: viết lại cho schema mới, ví dụ xuôi/ngược/none.
- `agents/message_classification.yaml`: `academic_calculation` CHỈ khi câu hỏi là tính xuôi/ngược một
  công thức trong registry (danh sách render qua placeholder `{calculation_formulas}`); mọi phép cộng
  trừ trên số liệu văn bản → `academic_advisory`. Bỏ rule 5b vừa thêm.
- `main/chat_academic_advisory.yaml`: luật "câu hỏi cần tính mà hệ thống không có công thức cài sẵn:
  trích công thức/số liệu + điều kiện, hướng dẫn tự tính, KHÔNG tự ra kết quả".

### Eval

38 câu calculation của tập demo: chỉ 8 câu thuộc công thức registry (d0307-d0313 + d0314 làm tròn);
30 câu còn lại là số học trên số liệu văn bản → theo quyết định mới sẽ được trả lời kiểu advisory
(hiện số liệu, không tính). Chạy lại 38 câu một lần để ghi nhận hiện trạng; tiêu chí `pass` của
category calculation cần gắn lại theo công thức (`formula_id` kỳ vọng) ở ticket sau.

## Bổ sung 26/10 tối: năng lực số học thuần (`pure_arithmetic`)

Chính sách 3 mức do người dùng chốt: (A) số học thuần trên số người dùng nêu → code tính, không cần
registry; (B) phép tính phụ thuộc quy định (đơn giá, trọng số, làm tròn) → công thức registry hoặc hỏi
lại; (C) công thức nghiệp vụ chưa cài, nhiều điều kiện, số liệu mâu thuẫn → chỉ trích dẫn. Grounding
không phải quyền tính: hai quyết định độc lập (được phép tính? căn cứ đủ?).

- `app/calculation/arithmetic.py`: parser whitelist (`+ - × ÷`, ngoặc, thập phân), không eval, Decimal,
  giới hạn; kết quả là `CalculationResult` (formula_id `arithmetic`).
- Node: extractor `formula_id: arithmetic` + `arithmetic.expression/label/unit`; code kiểm tra mọi số
  hạng nằm trong câu hỏi (`missing_operands`), không thì Deferred.
- Advisory: khối ```json {"type": "calc_request", ...}``` (FenceRedactor giữ lại) → `calc_requests.py`
  kiểm tra biểu thức + căn cứ (câu hỏi hoặc chunk [n] được dẫn), nối khối Python dưới câu trả lời, log
  biểu thức/số hạng/kết quả/nguồn; từ chối → dòng cố định + `CALC_REQUEST_REJECTED`. Tối đa 3 khối.
- Prompt phân loại/extractor/advisory/multi-intent cập nhật theo 3 mức.

## Ngoài phạm vi

- Điểm ưu tiên khu vực/đối tượng theo công thức giảm dần của Bộ GD&ĐT (từ 22,5 điểm): văn bản của
  Trường chỉ dẫn chiếu quy định Bộ, không có công thức trong KB → nhập mức đã quy đổi, có ghi chú.
- Điểm thủ khoa ĐGNL (ĐTK) từng năm: chưa có trong KB → hỏi người dùng (tham số), có thể thêm lookup
  tài liệu khi Trường công bố.
- Gắn lại nhãn eval calculation theo `formula_id`, đo lại tập đầy đủ.
