# Spec: calc-engine (unisage-agent)

> **Status: Implemented** (UNISAGE-99, nhánh `feature/huydh-unisage-99-calculation-flow`). Rủi ro còn
> lại nằm ở `unisage-agent/docs/specs/known-gaps.md`.

Module id `calc-engine` trong [capability map](../../changes/09-10-2026-calculation-flow/capability-map.md).
Thuần Python: không gọi LLM, không I/O, không phụ thuộc graph. Graph node `calculation-node` gọi xuống.

## Objective

Tính đúng và **giải thích được** các phép tính học vụ. Mỗi kết quả kèm theo công thức và từng bước đã
thế số, để sinh viên tự đối chiếu. LLM không làm bất kỳ phép tính nào; module này là nơi duy nhất
sinh ra con số.

Có hai nguồn công thức:

1. **Cài sẵn** (`formulas.py`): `course_score`, `grade_conversion`, `gpa`. Không trích nguồn, chỉ hiện
   công thức.
2. **Lấy từ quy chế** (`expression.py`): một biểu thức do LLM chép từ chunk Qdrant, đã qua kiểm tra
   provenance ở `calculation-node`. Module này chỉ chịu trách nhiệm parse an toàn và tính.

## Project Structure

```
app/calculation/
├── __init__.py
├── formulas.py     # TOÀN BỘ business rule: bảng quy đổi, làm tròn, 3 công thức, ParamSpec, BUILTIN_TRIGGERS (router luật)
├── expression.py   # parser/evaluator allowlist cho công thức lấy từ Qdrant
├── result.py       # CalculationResult, Step, CalculationInputError (dataclass, frozen)
└── render.py       # CalculationResult -> markdown (deterministic, không LLM)
tests/calculation/
├── test_formulas.py
├── test_expression.py
└── test_render.py
```

Mọi hằng số nghiệp vụ (bảng điểm, trọng số 20/30/50, số chữ số làm tròn, giới hạn input) chỉ được
khai báo trong `formulas.py`. Các file khác import từ đó.

## Business contract: 3 công thức cài sẵn

### Quy tắc chung

- **Số học:** dùng `decimal.Decimal` cho mọi bước tính. Input được chuyển bằng `Decimal(str(value))`
  để tránh sai số float. Không dùng `float` ở bất kỳ bước nào.
- **Làm tròn:** dùng `round_half_up(x, places)` = `x.quantize(Decimal(10) ** -places, ROUND_HALF_UP)`.
  Không dùng `round()` của Python, vì nó làm tròn kiểu banker (`round(7.45, 1) == 7.4`).
- **Làm tròn điểm thành phần (quy chế, đã chốt 09-10-2026):** mỗi điểm sinh viên nhập cho từng cột TX,
  GK, CK, từng cột TH được làm tròn đến 0.5 trước tiên (`round_to_half_point`); dòng "Làm tròn điểm
  thành phần" chỉ hiện khi có điểm bị đổi. TBtx sinh viên nhập sẵn là giá trị trung bình, không làm tròn
  0.5.
- **Thời điểm làm tròn (đã chốt 09-10-2026):** trong `course_score`, **TBtx, ĐLT và ĐTH đều làm tròn
  đến 0.1 trước khi bước sau dùng tới** (`COMPONENT_PLACES`), rồi ĐTKHP làm tròn 0.1 ở cuối. GPA làm
  tròn 2 chữ số ở cuối; tổng điểm chất lượng giữ nguyên. Dòng thay số ghi rõ bước làm tròn, ví dụ
  `ĐLT = … = 6.95 → 7.0`, `ĐTH = 25 / 3 ≈ 8.33 → 8.3` (dài hơn 2 chữ số thì hiện 2 chữ số kèm `≈`;
  khi 2 chữ số làm người đọc hiểu sai kết quả làm tròn thì hiện 4 chữ số).
- **Thang điểm 10:** mọi điểm thành phần nằm trong `[0, 10]`, tối đa 2 chữ số thập phân.
- **Tín chỉ:** số nguyên. TCLT và TCTH nằm trong `[0, 10]`, tín chỉ của một môn khi tính GPA nằm trong
  `[1, 10]`.

### Bảng quy đổi (`GRADE_SCALE`)

Tra trên điểm **đã làm tròn đến 0.1**. Vì đã làm tròn nên các khoảng liền nhau, không có điểm rơi vào
khe hở giữa hai mức (8.95 → 9.0 → A+).

| Điều kiện (điểm đã làm tròn) | Chữ | Thang 4 |
|---|---|---|
| ≥ 9.0 | A+ | 4.0 |
| ≥ 8.5 | A | 3.8 |
| ≥ 8.0 | B+ | 3.5 |
| ≥ 7.0 | B | 3.0 |
| ≥ 6.0 | C+ | 2.5 |
| ≥ 5.5 | C | 2.0 |
| ≥ 5.0 | D+ | 1.5 |
| ≥ 4.0 | D | 1.0 |
| < 4.0 | F | 0.0 |

### `grade_conversion(score10)`

1. `s = round_half_up(score10, 1)`
2. Tra `GRADE_SCALE` → `(letter, gp4)`

### `course_score(tbtx, gk, ck, th, tclt, tcth)`: ĐTKHP học phần tích hợp

```
TXi, GK, CK, THi = round_to_half_point(điểm sinh viên nhập)   (lẻ < 0.25 → .0; 0.25..< 0.75 → .5; ≥ 0.75 → 1)
TBtx  = round_half_up((TX1 + … + TXn) / n, 1)               (từng cột, trọng số như nhau, 1 ≤ n ≤ 20)
      | round_half_up(TBtx sinh viên nhập sẵn, 1)           (sinh viên chọn 1 trong 2 cách nhập)
ĐLT   = round_half_up(0.2 × TBtx + 0.3 × GK + 0.5 × CK, 1)  (chỉ khi TCLT > 0)
ĐTH   = round_half_up((TH1 + … + THn) / n, 1)               (chỉ khi TCTH > 0, 1 ≤ n ≤ 20)
ĐTKHP = round_half_up((ĐLT × TCLT + ĐTH × TCTH) / (TCLT + TCTH), 1)
→ grade_conversion(ĐTKHP)
```

Param `tbtx` là một số (TBtx nhập sẵn) **hoặc** một mảng (các cột TX). Câu hỏi trên panel là loại
`number_or_list` (xem SPEC-clarification-panel §1.1).

Các trường hợp biên:

| Trường hợp | Xử lý |
|---|---|
| `TCLT = 0`, `TCTH > 0` | Học phần chỉ có thực hành: ĐTKHP = round(ĐTH). Không hỏi TBtx/GK/CK |
| `TCTH = 0`, `TCLT > 0` | Học phần chỉ có lý thuyết: ĐTKHP = round(ĐLT). Không hỏi TH |
| `TCLT + TCTH = 0` | `CalculationInputError(field="tclt", reason="Tổng tín chỉ phải lớn hơn 0")` |
| Thiếu một input bắt buộc (theo 2 dòng trên) | Báo thiếu, xem `missing_params` bên dưới |
| Điểm ngoài `[0, 10]`, tín chỉ âm hoặc không nguyên | `CalculationInputError` cho đúng field đó |

### `gpa(courses)`

`courses`: 1 đến 30 môn, mỗi môn `{name, credits, score10 | letter}`. Không bắt buộc có tên môn.

```
mỗi môn i: gp4_i = grade_conversion(score10_i).gp4      (nếu nhập điểm chữ: tra trực tiếp gp4)
           Q_i   = gp4_i × TC_i
GPA = round_half_up(Σ Q_i / Σ TC_i, 2)
```

- Môn F (gp4 = 0) **vẫn được tính** vào mẫu số.
- Không loại trừ môn nào (GDTC, GDQP...): sinh viên tự chọn môn nào đưa vào bảng.
- `Σ TC_i = 0` không xảy ra được, vì mỗi môn có tín chỉ ≥ 1. Danh sách rỗng thì báo thiếu `courses`.
- Điểm chữ được chấp nhận: `A+ A B+ B C+ C D+ D F`, không phân biệt hoa thường.

## Contract chung cho mọi công thức

```python
@dataclass(frozen=True)
class ParamSpec:
    name: str                 # "tbtx", "th", "courses"...
    label: str                # nhãn tiếng Việt, dùng làm câu hỏi trên panel
    kind: Literal["number", "number_list", "course_table"]
    min: Decimal | None
    max: Decimal | None
    step: Decimal | None      # 0.01 cho điểm, 1 cho tín chỉ
    required: Callable[[Mapping[str, object]], bool]  # VD "th" bắt buộc khi tcth > 0

@dataclass(frozen=True)
class Step:
    label: str                # "Điểm lý thuyết"
    symbolic: str             # "ĐLT = 0.2 × TBtx + 0.3 × GK + 0.5 × CK"
    substituted: str          # "ĐLT = 0.2 × 8 + 0.3 × 7 + 0.5 × 6.5 = 1.6 + 2.1 + 3.25"
    value: Decimal
    display: str              # "6.95" hoặc "≈ 7.47"
    note: str | None = None   # "làm tròn đến 0.1"

@dataclass(frozen=True)
class CalculationResult:
    formula_id: str                    # "course_score" | "gpa" | "grade_conversion" | "retrieved"
    title: str
    formula_text: list[str]            # các dòng công thức hiện cho người dùng
    inputs: list[tuple[str, str]]      # (label, giá trị đã nhập) theo thứ tự hiển thị
    steps: list[Step]
    outputs: list[tuple[str, str]]     # ("ĐTKHP", "7.5"), ("Điểm chữ", "B"), ("Thang 4", "3.0")
    warnings: list[str]

class CalculationInputError(Exception):
    errors: list[FieldError]           # FieldError(field, reason), có thể nhiều field một lúc
```

Hai hàm public của `formulas.py`:

- `missing_params(formula_id, params) -> list[ParamSpec]`: danh sách param bắt buộc còn thiếu, theo
  thứ tự hiển thị. `calculation-node` dùng nó để dựng câu hỏi cho panel một cách deterministic, không
  để LLM tự nghĩ ra câu hỏi.
- `calculate(formula_id, params) -> CalculationResult`: raise `CalculationInputError` khi input sai.

`render.py`: `render_markdown(result) -> str`. Output cố định cho cùng một input, có snapshot test.
**Mỗi giá trị chỉ xuất hiện một lần** (đã chốt 09-10-2026 sau phản hồi về thông tin bị lặp): không có
mục "Số liệu" (số đã có ở card câu trả lời và trong các dòng thay số; `inputs` chỉ còn dùng cho trace và
nhận xét), bước làm tròn gộp vào dòng tính (`… ≈ 2.044 → 2.04`), và quy đổi điểm chữ nằm trong **một**
dòng kết quả (`CalculationResult.summary`), ví dụ
`Kết quả: ĐTKHP **7.5** thuộc khoảng [7.0; 8.0) → điểm chữ **B** → thang 4 **3.0**`.

## `expression.py`: evaluator cho công thức lấy từ Qdrant

### Input

```python
@dataclass(frozen=True)
class FormulaVariable:
    name: str            # ^[a-z][a-z0-9_]{0,31}$
    label: str           # ≤ 80 ký tự
    unit: str | None
    min: Decimal | None
    max: Decimal | None

@dataclass(frozen=True)
class RetrievedFormula:
    expression: str      # ≤ 200 ký tự, VD "so_tc * don_gia + phi_khac"
    variables: list[FormulaVariable]   # 1..10
    result_label: str
```

### Grammar allowlist

Parse bằng `ast.parse(expression, mode="eval")`, sau đó duyệt toàn bộ cây. **Gặp bất kỳ node nào
ngoài danh sách dưới đây thì từ chối cả biểu thức.**

| Được phép | Ràng buộc |
|---|---|
| `Expression` | gốc |
| `BinOp` với `Add`, `Sub`, `Mult`, `Div` | không có `Pow`, `Mod`, `FloorDiv`, toán tử bit |
| `UnaryOp` với `USub`, `UAdd` | |
| `Constant` | chỉ `int` hoặc `float`, **không phải `bool`**, `abs ≤ 1e9`, tối đa 6 chữ số thập phân |
| `Name` (ctx `Load`) | phải có trong `variables` |
| `Call` | `func` là `Name` thuộc `{round, min, max}`, không keyword, không starred. `round(x, n)` thì `n` là hằng nguyên trong `0..4`. `min`/`max` có 2..10 đối số |

Bị từ chối ngay cả khi cú pháp hợp lệ: `Attribute`, `Subscript`, `Compare`, `BoolOp`, `IfExp`,
`Lambda`, comprehension, `JoinedStr`, `Starred`, `Pow`, chuỗi.

### Giới hạn chống DoS

- Độ dài biểu thức ≤ 200 ký tự, kiểm tra **trước khi** gọi `ast.parse`.
- Số node ≤ 50, độ sâu ≤ 10.
- Không có `Pow`, nên không thể tạo số khổng lồ bằng lũy thừa.
- Tính trong `decimal.localcontext(prec=28, Emax=24, Emin=-24, traps=[Overflow, InvalidOperation, DivisionByZero])`.
  Nếu kết quả trung gian có `abs > 1e12` thì báo lỗi.
- Chia cho 0: `CalculationInputError(field=<biến ở mẫu số nếu xác định được, không thì "expression">, reason="Mẫu số bằng 0")`.
- `round` trong biểu thức dùng `round_half_up`.

### Kiểm tra tính nhất quán

- Mọi `Name` trong biểu thức phải được khai báo trong `variables`, và mọi biến khai báo phải xuất hiện
  trong biểu thức. Biến khai báo mà không dùng nghĩa là LLM chép sai, nên từ chối.
- Giá trị của từng biến được kiểm tra theo `min`/`max` của chính biến đó.

### Output

`evaluate(formula, values) -> CalculationResult` với `formula_id="retrieved"`. Các bước gồm biểu thức
dạng ký hiệu, biểu thức đã thế số, rồi kết quả. Với phép chia ở gốc cây thì thêm một bước tử số và một
bước mẫu số. `validate(formula)` được tách riêng để `calculation-node` gọi trước khi hỏi người dùng
bất cứ điều gì.

## `solver.py` + `target.py`: câu hỏi ngược ("cần bao nhiêu để ...") {#solver}

- `solve(forward, Domain(min, max, step), Goal(">=" | "<=", value), want="min" | "max")`: thử lần lượt
  mọi giá trị của biến (≤ 2000 điểm), miền lớn hơn (tiền) thì chia đôi giả định đơn điệu (kiểm tra hai
  đầu trước). Trả giá trị tốt nhất, giá trị liền kề **không** đạt (để chứng minh đó là mức tối thiểu),
  hoặc đầu miền tốt nhất khi không đạt được. `forward` trả `None` = giá trị đó không tính được.
- `forward` luôn là công thức xuôi (`compute(params).primary_value`), nên mọi bước làm tròn được giữ.
- Miền: `ParamSpec.solve_step` (GK/CK bước 0.5, `half_point=True`; TBtx bước 0.1; tín chỉ bước 1);
  biến Qdrant: `min/max` đã khai báo (thiếu `max` → 10^9), bước 1 nếu đơn vị là TC/tín/môn, còn lại 0.01.
- Mục tiêu điểm chữ chỉ nhận cho công thức có `target_is_score10` (ĐTKHP): A+ → `ĐTKHP ≥ 9.0`.
- Một ẩn: bước thay số đầy đủ với giá trị tìm được + dòng "Mức liền kề" + một dòng kết quả, VD
  `Cần CK tối thiểu **10.0** (điểm từ 9.75 trở lên được làm tròn thành 10.0) để ĐTKHP ≥ 9.0 (A+).`
  Không đạt được → nêu kết quả tốt nhất; đã đạt với 0 → "bạn đã chắc chắn đạt".
- Hai ẩn: mức khi hai ẩn bằng nhau + bảng đánh đổi ≤ 5 dòng (chỉ trong khoảng còn đạt được), render
  thành bảng markdown (`CalculationResult.table`).
- `CalculationResult.primary_value`: con số chính (ĐTKHP, GPA, kết quả công thức quy chế) ở độ chính
  xác đầy đủ - thứ solver so với mục tiêu.

## Code Style

Theo `AGENTS.md`: Python 3.12, type hint đầy đủ, dataclass frozen, không thêm abstraction (không
registry/plugin). Ba công thức là ba hàm thường, được dispatch bằng một `dict[str, ...]` trong
`formulas.py`.

```python
def course_score(params: CourseScoreParams) -> CalculationResult:
    steps: list[Step] = []
    if params.tclt > 0:
        dlt = THEORY_WEIGHTS.tx * params.tbtx + THEORY_WEIGHTS.gk * params.gk + THEORY_WEIGHTS.ck * params.ck
        steps.append(Step(label="Điểm lý thuyết", symbolic=..., substituted=..., value=dlt, display=fmt(dlt)))
    ...
```

## Testing Strategy

Dùng pytest, đặt trong `tests/calculation/`, không mock gì vì module là code thuần.

- **Bảng quy đổi:** test tham số hoá tại mọi biên, ví dụ `3.99 → 4.0 → D`, `3.94 → 3.9 → F`,
  `8.45 → 8.5 → A`, `8.44 → 8.4 → B+`, `8.95 → 9.0 → A+`, `10 → A+`, `0 → F`.
- **Làm tròn half-up:** `7.45 → 7.5`, `7.449 → 7.4`, `2.345 → 2.35` (GPA), `0.1 + 0.2` không bị lệch
  do float.
- **ĐTKHP:** ví dụ đầy đủ (TBtx 8, GK 7, CK 6.5, TH [9, 8], 2 + 1 TC) ra `7.5 / B / 3.0`; trường hợp
  chỉ lý thuyết, chỉ thực hành, tổng tín chỉ bằng 0, điểm 10.01, tín chỉ 1.5.
- **GPA:** có môn F, nhập điểm chữ lẫn điểm số, 1 môn, 30 môn, 31 môn bị từ chối.
- **`missing_params`:** đúng thứ tự, `th` chỉ bắt buộc khi `tcth > 0`.
- **Evaluator:** các ví dụ tấn công đều bị từ chối, gồm `__import__('os')`, `().__class__`, `a**b`,
  `9**9**9`, `[1][0]`, `a if b else c`, `"x"`, `True + 1`, `1e400`, biểu thức 201 ký tự, lồng
  ngoặc 11 tầng, `min()` với 11 đối số, biến khai báo mà không dùng, biến dùng mà không khai báo.
  Ngoài ra test chia cho 0 và test các công thức hợp lệ cho kết quả đúng.
- **Render:** snapshot markdown của 3 công thức cài sẵn và 1 công thức retrieved.

## Boundaries

- **Always:** mọi hằng số nghiệp vụ nằm trong `formulas.py`; dùng `Decimal` từ đầu đến cuối; mỗi
  business rule ở trên phải có ít nhất một test.
- **Ask first:** đổi bảng quy đổi, đổi trọng số 20/30/50, đổi số chữ số làm tròn (đây là thay đổi
  nghiệp vụ, không phải refactor).
- **Never:** dùng `eval`/`exec`/`compile` lên chuỗi từ LLM; dùng `float` trong phép tính; để LLM làm
  tròn hoặc tính.

## Success Criteria

- [ ] Mọi test trong `tests/calculation/` pass, coverage của `app/calculation/` ≥ 95%.
- [ ] Ví dụ ĐTKHP ở trên ra đúng `7.5 / B / 3.0`, và các bước hiển thị khớp snapshot.
- [ ] Mọi chuỗi tấn công trong danh sách đều bị `validate` từ chối mà không chạy bất kỳ code nào.
- [ ] `ruff`, `mypy` sạch cho `app/calculation/`.

## Decisions (09-10-2026)

- Giá trị trung gian hiển thị tối đa 2 chữ số thập phân kèm `≈`; tính bằng giá trị đầy đủ.
- GPA không tự loại môn nào (GDTC, GDQP...); sinh viên tự chọn môn đưa vào bảng.
