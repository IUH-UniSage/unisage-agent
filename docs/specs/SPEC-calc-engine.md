# Spec: calc-engine (unisage-agent)

> **Status: Implemented** (UNISAGE-99, nhánh `feature/huydh-unisage-99-calculation-flow`). Rủi ro còn
> lại nằm ở `unisage-agent/docs/specs/known-gaps.md`.

Module id `calc-engine` trong [capability map](../../changes/09-10-2026-calculation-flow/capability-map.md).
Thuần Python: không gọi LLM, không I/O, không phụ thuộc graph. Graph node `calculation-node` gọi xuống.

## Objective

Tính đúng và **giải thích được** mọi phép tính của **registry công thức** (`formulas.py`), xuôi và
ngược: `course_score`, `grade_conversion`, `gpa`, học phí `tuition_semester` / `tuition_year` /
`tuition_program` (13-10-2026), và từ 26-10-2026 `admission_score_2026`, `transcript_score_3hk`,
`weighted_score`. Mỗi kết quả kèm công thức, phiên bản + nguồn, và từng bước đã thế số, để sinh viên tự
đối chiếu. Module này không trích nguồn; số liệu học phí lấy từ tài liệu được `calculation-node` trích
dẫn.

Tính ngược (`solver.py`, 26-10-2026): tìm một tham số để một đầu ra đạt mục tiêu bằng cách chạy chính
công thức xuôi trên lưới của tham số đó. Không có LLM nào tính nhẩm nữa (`describe_builtin_rules()` giữ
lại cho tài liệu/prompt khi cần).

## Project Structure

```
app/calculation/
├── __init__.py
├── formulas.py     # TOÀN BỘ business rule: bảng quy đổi, làm tròn, 9 công thức, ParamSpec, OutputSpec,
│                   # Formula (version, source, rounding_of), FORMULAS (registry), BUILTIN_TRIGGERS (router luật)
├── solver.py       # tính ngược: Unknown, Target, solve/solved_result trên lưới của tham số
├── result.py       # CalculationResult (values, version, source), Step, CalculationInputError (dataclass, frozen)
└── render.py       # CalculationResult -> markdown (deterministic, không LLM), dòng phiên bản + nguồn
tests/calculation/
├── test_formulas.py
├── test_solver.py
├── test_builtin_triggers.py
└── test_render.py  (+ snapshots/)
```

Mọi hằng số nghiệp vụ (bảng điểm, trọng số 20/30/50, số chữ số làm tròn, giới hạn input) chỉ được
khai báo trong `formulas.py`. Các file khác import từ đó.

## Registry công thức có version (26-10-2026)

```python
@dataclass(frozen=True)
class OutputSpec:   key: str; label: str; money: bool = False      # đầu ra làm mục tiêu khi tính ngược
@dataclass(frozen=True)
class Formula:
    formula_id: FormulaId
    version: str      # "<năm văn bản>.<lần sửa>" khi gắn với một văn bản ("2026.1"), "1", "2"... cho quy tắc chung
    source: str       # văn bản gốc, hiện dưới công thức và trong trace staff
    title, description, params: tuple[ParamSpec, ...], outputs: tuple[OutputSpec, ...]
    compute: Callable[[Mapping], CalculationResult]
    rounding_of: Callable[[Mapping, str], Decimal | None] | None   # bước làm tròn công thức áp cho một tham số (CK: 0,5)
```

`calculate()` đóng dấu `version`/`source` lên kết quả; `CalculationResult.values` giữ giá trị số của
từng đầu ra (solver và trace dùng). Đổi quy tắc của một công thức = tăng `version` (và ghi vào Decisions);
công thức gắn với văn bản mới (biểu điểm xét tuyển năm khác) = id mới (`admission_score_2027`), không ghi
đè. Danh sách cho prompt render từ registry (`describe_builtin_formulas()`, `describe_registry()`).

| formula_id | version | nguồn | tham số | đầu ra (`values`) |
|---|---|---|---|---|
| `course_score` | 1 | quy chế đào tạo (quy tắc chốt 09/10/2026) | tclt, tcth, tbtx, gk, ck, th | dtkhp, thang4 |
| `gpa` | 1 | quy chế đào tạo | courses | gpa, tong_tin_chi |
| `grade_conversion` | 1 | quy chế đào tạo | score10 | diem10, thang4 |
| `tuition_semester/_year/_program` | 1 | quyết định mức thu học phí | tc_lt, tc_th, don_gia_lt, don_gia_th (+ khoa_nhap_hoc, khoi_nganh, pham_vi) | hoc_phi (money) |
| `admission_score_2026` | 2026.1 | Công thức tính điểm xét tuyển ĐHCQ 2026 | dtn, dhb, dgnl?, dtk (khi có dgnl), uu_tien = 0, diem_cong = 0 | ddgnl, dk, xt1, xt2, xt3, dxt |
| `transcript_score_3hk` | 2020.1 | Bản mô tả CTĐT Kiểm toán 2020 §4.1 | mon1..3 (tên), m{i}_10, m{i}_11, m{i}_12, nguong = 19,5 | mon1..3, tong |
| `weighted_score` | 1 | trọng số đề cương học phần | trong_so (list %), diem (list), lam_tron_thanh_phan? | diem_hp, thang4 |

### `admission_score_2026`

ĐĐGNL = ĐGNL × 30 / ĐTK; ĐK = Max(ĐTN, ĐĐGNL); XT1 = 0,7 × ĐK + 0,3 × ĐHB + Đ(Kv;Đt) + ĐC;
XT2 = ĐTN + Đ(Kv;Đt) + ĐC; XT3 = ĐĐGNL + Đ(Kv;Đt) + ĐC; ĐXT = Max(XT1, XT2, XT3). Văn bản không nêu làm
tròn → giữ giá trị chính xác, hiển thị ≤ 2 chữ số (`fmt`). ĐTN/ĐHB ∈ [0; 30]; ĐGNL ∈ [0; 1200] và ≤ ĐTK;
ĐTK chỉ cần khi có ĐGNL. Ưu tiên/điểm cộng không nêu → 0, có dòng cảnh báo (mức ưu tiên từ 22,5 điểm
giảm dần theo quy định Bộ: người dùng nhập mức đã quy đổi - không có công thức của Bộ trong KB).

### `transcript_score_3hk`

Điểm môn = (TB cả năm lớp 10 + TB cả năm lớp 11 + TB HK1 lớp 12) / 3 (chính xác, không làm tròn);
tổng 3 môn so với ngưỡng (mặc định 19,5 theo CTĐT Kiểm toán 2020); dòng kết quả nói đủ/chưa đủ điều kiện.

### `weighted_score`

Điểm = Σ (trọng số_i / 100 × điểm_i), làm tròn 0,1, rồi điểm chữ/thang 4 theo `GRADE_SCALE`. Trọng số
nhận % hoặc phân số (tổng ≈ 1), tổng phải bằng 100 % (±0,5). `lam_tron_thanh_phan` (VD 0,5) làm tròn
từng điểm thành phần trước khi nhân, như quy chế làm tròn điểm thi; solver tìm điểm thành phần theo bước
này. Đây là công thức "theo số liệu người dùng/đề cương nêu", không phải quy định riêng của Trường.

## Tính ngược (`solver.py`, 26-10-2026)

```python
Unknown(name, index | None)        # "ck", hoặc "diem[4]" = vị trí 4 (1-based) của danh sách
Target(output, op ∈ {">=", "<=", "=="}, value: Decimal)   # điểm chữ → ngưỡng thang 10 (dtkhp/diem10/diem_hp) hoặc thang 4
solve(formula_id, params, unknown, target) -> Solution(value | None, result, output, checked, maximum)
solved_result(...) -> CalculationResult
```

- Lưới: `[ParamSpec.min; max]` theo bước = max(`ParamSpec.step`, `Formula.rounding_of(params, name)`):
  điểm CK/GK/TH và cột TX của `course_score` tìm theo 0,5; `diem` của `weighted_score` theo
  `lam_tron_thanh_phan`. Vùng hợp lệ (công thức từ chối một đầu: 0 tín chỉ, ĐGNL > ĐTK) tìm bằng nhị
  phân; cả hai đầu bị từ chối → lỗi của các tham số khác được ném ra.
- Chiều đo bằng hai đầu lưới (không giả định); mọi công thức registry đơn điệu theo từng tham số số.
  `>=`/`<=` → giá trị nhỏ nhất (điểm cần đạt) hoặc lớn nhất (tín chỉ tối đa) thoả mục tiêu; `==` tìm như
  `>=` rồi kiểm tra bằng.
- Kết quả: tiêu đề "<tham số> tối thiểu/tối đa để <đầu ra> ≥ <mốc>", bước "Giá trị cần tìm", các bước
  xuôi tại nghiệm, bước "Kiểm tra" (nghiệm → đạt; giá trị liền kề → không đạt, hiện 4 chữ số khi 2 chữ số
  trông bằng nhau), `outputs` có thêm nghiệm, `values["nghiem"]`. Không có nghiệm → tiêu đề không có
  "tối thiểu", dòng kết quả nêu giá trị biên, cảnh báo "Mục tiêu không đạt được".

## Business contract: 3 công thức cài sẵn ban đầu

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

### Học phí: `tuition_semester`, `tuition_year`, `tuition_program` (chốt 13-10-2026)

Một hàm `_tuition(scope)` dùng chung, ba phạm vi chỉ khác tiêu đề và nhãn số tín chỉ:

```
Học phí = tc_lt × don_gia_lt + tc_th × don_gia_th
```

- **Đơn giá theo khoá tuyển sinh, giữ nguyên suốt khoá học**; chỉ khoá mới có đơn giá khác. Vì vậy học
  phí cả khoá là tổng tín chỉ của chương trình nhân đơn giá của khoá, không phải ước tính theo năm.
- `tc_lt`, `tc_th`: **hai tham số bắt buộc riêng**, số nguyên `[0, 300]`, không được cùng bằng 0. Không
  bao giờ hỏi gộp thành "tổng tín chỉ".
- `don_gia_lt`, `don_gia_th`: số nguyên đồng `[0, 100.000.000]`. Chuỗi phải viết đủ số (`980000`,
  `980.000`, `980.000 đ`): dấu chấm/phẩy là phân cách hàng nghìn; "1,6 triệu", "980k" bị từ chối và hỏi
  lại thay vì đọc sai (`parse_money`).
- `khoa_tuyen_sinh`, `khoi_nganh`: kiểu `text`, **không tham gia phép tính**, không bao giờ bị
  `missing_params` hỏi. Chỉ dùng để chọn đúng dòng biểu phí khi tra tài liệu.
- `pham_vi` (thêm 13-10-2026): kiểu `text`, không bắt buộc, không `lookup`, không tham gia phép tính,
  không bao giờ bị hỏi. Là cụm phạm vi **nguyên văn** người dùng nói ("2 năm đầu", "học kỳ 1 năm 2"),
  extractor chép khi có. `tuition_period` dùng nó cho tiêu đề kết quả ("Học phí 2 năm đầu") và nhãn
  tín chỉ trên form ("Số tín chỉ lý thuyết của 2 năm đầu"); không có (hoặc rỗng) thì giữ phạm vi mặc
  định "một học kỳ" / "một năm học" / "cả khoá học". Không có công thức riêng cho "N năm": câu "2 năm
  đầu" đi vào `tuition_year` kèm `pham_vi`, và người dùng nhập tín chỉ của cả quãng đó. Bỏ `*`, tối đa
  60 ký tự (tiêu đề hiển thị in đậm).
- Không gồm QPAN (đơn giá riêng); kết quả luôn kèm cảnh báo này. Tiền hiển thị `38.350.000 đồng`
  (`money`).
- `ParamSpec.lookup`: câu tìm Qdrant cho tham số tra được trong tài liệu (đơn giá; tín chỉ của
  `tuition_program`); `None` = số của riêng sinh viên (tín chỉ một học kỳ / một năm).
- Router: câu có "học phí" chỉ được xét với ba công thức học phí (theo từ chỉ phạm vi kỳ / năm / cả khoá,
  không phụ thuộc thứ tự), không bao giờ rơi vào `course_score` dù có "lý thuyết … thực hành".

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
    lookup: str | None        # câu tìm Qdrant trước khi hỏi; None = số của riêng sinh viên
    label_from: Callable[[Mapping[str, object]], str] | None  # nhãn theo param đã biết (pham_vi)

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
    formula_id: str                    # "course_score" | "gpa" | "grade_conversion"
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
- `label_for(spec, known) -> str`: nhãn câu hỏi của một param theo các param đã biết (`label_from`), mặc
  định `label`. `calculation-node` dùng cho câu hỏi panel và dòng "Số liệu lấy từ tài liệu".

`render.py`: `render_markdown(result) -> str`. Output cố định cho cùng một input, có snapshot test.
**Mỗi giá trị chỉ xuất hiện một lần** (đã chốt 09-10-2026 sau phản hồi về thông tin bị lặp): không có
mục "Số liệu" (số đã có ở card câu trả lời và trong các dòng thay số; `inputs` chỉ còn dùng cho trace và
nhận xét), bước làm tròn gộp vào dòng tính (`… ≈ 2.044 → 2.04`), và quy đổi điểm chữ nằm trong **một**
dòng kết quả (`CalculationResult.summary`), ví dụ
`Kết quả: ĐTKHP **7.5** thuộc khoảng [7.0; 8.0) → điểm chữ **B** → thang 4 **3.0**`.

## Code Style

Theo `AGENTS.md`: Python 3.12, type hint đầy đủ, dataclass frozen. Mỗi công thức là một hàm thường;
`FORMULAS: dict[FormulaId, Formula]` trong `formulas.py` là registry duy nhất (không plugin, không
nạp động).

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
- **Render:** snapshot markdown của 7 công thức (`tests/calculation/snapshots/`).
- **Solver** (`test_solver.py`): d0308 (ĐHB ≥ 25,5), d0309 (HK1 Lý = 5,5), d0311 (cuối kỳ 9,5 theo bước
  0,5), d0313 (7,1 vì 6,95 → 7,0), CK = 10 để A+, 8 tín chỉ tối đa dưới 8 triệu, ĐTK tối đa 1080 (tham số
  giảm), `==`, không đạt được, lỗi tham số khác được ném ra, ẩn số phải là số.

## Boundaries

- **Always:** mọi hằng số nghiệp vụ nằm trong `formulas.py`; dùng `Decimal` từ đầu đến cuối; mỗi
  business rule ở trên phải có ít nhất một test.
- **Ask first:** đổi bảng quy đổi, đổi trọng số 20/30/50, đổi số chữ số làm tròn (đây là thay đổi
  nghiệp vụ, không phải refactor).
- **Never:** dùng `eval`/`exec`/`compile` lên chuỗi từ LLM; dùng `float` trong phép tính; để LLM tính
  thay Python bất kỳ phép tính nào (xuôi hay ngược) của công thức registry; sửa quy tắc một công thức mà
  không tăng `version`.

## Success Criteria

- [ ] Mọi test trong `tests/calculation/` pass, coverage của `app/calculation/` ≥ 95%.
- [ ] Ví dụ ĐTKHP ở trên ra đúng `7.5 / B / 3.0`, và các bước hiển thị khớp snapshot.
- [ ] `ruff`, `mypy` sạch cho `app/calculation/`.

## Decisions (09-10-2026)

- Giá trị trung gian hiển thị tối đa 2 chữ số thập phân kèm `≈`; tính bằng giá trị đầy đủ.
- GPA không tự loại môn nào (GDTC, GDQP...); sinh viên tự chọn môn đưa vào bảng.
- Bỏ `expression.py` (evaluator công thức Qdrant) và bộ giải ngược `solver.py`/`target.py`: phần đó giao
  cho LLM tự tính (nhãn "AI tự tính, có thể sai" + nút Đúng/Sai), xem SPEC-calculation-node.
- (26-10-2026) Quay lại Python cho tính ngược, nhưng bằng tìm nghiệm trên lưới với chính công thức xuôi
  (không evaluator biểu thức, không chép công thức Qdrant); registry có version/nguồn; thêm 3 công thức.
