# Implementation Plan: Công thức học phí có sẵn (kỳ / năm / khoá)

## Mục tiêu

Thêm 3 công thức có sẵn do Python tính: **học phí 1 học kỳ**, **học phí 1 năm học**,
**học phí cả khoá**, cùng một công thức:

```
Học phí = TC lý thuyết × đơn giá LT + TC thực hành × đơn giá TH
```

Số liệu lấy theo thứ tự **tài liệu trước, hỏi sau**:

1. Số người dùng đã nói trong câu hỏi (extractor hiện có).
2. Số trong tài liệu Qdrant (đơn giá theo khoá tuyển sinh, tổng tín chỉ của chương
   trình đào tạo): một lần gọi LLM chỉ để **trích giá trị kèm đoạn nguồn**, Python
   kiểm tra con số có nguyên văn trong đoạn đó.
3. Còn thiếu → hỏi người dùng bằng panel hiện có (`ParamSpec` → `question_for`).

Python tính, nên kết quả không còn nhãn "AI tự tính, có thể sai". Mỗi giá trị lấy từ
tài liệu được trích dẫn `[n]`.

## Quy ước nghiệp vụ (đã chốt với người dùng)

- Đơn giá tín chỉ theo **khoá tuyển sinh** của sinh viên và **giữ nguyên suốt thời
  gian học**; chỉ khoá mới mới có đơn giá khác. Vì vậy học phí cả khoá = tổng TC của
  chương trình × đơn giá của khoá đó, không phải ước tính theo năm.
- Công thức chỉ gồm LT và TH (không có mức thu theo năm, không có QPAN - xem câu hỏi mở 1).
- Không có dữ liệu trong tài liệu → hỏi người dùng (fallback có sẵn).

## Thiết kế

### Tham số (chung cho cả 3 công thức)

| Tham số | Nhãn | Nguồn ưu tiên |
|---|---|---|
| `khoa_tuyen_sinh` | Khoá tuyển sinh (VD 2025-2026) | câu hỏi / metadata đã xác nhận → hỏi |
| `tc_lt` | Số tín chỉ lý thuyết (của kỳ / năm / cả chương trình) | kỳ, năm: người dùng; khoá: tài liệu CTĐT |
| `tc_th` | Số tín chỉ thực hành (như trên) | như trên |
| `don_gia_lt` | Đơn giá 1 TC lý thuyết (đồng) | tài liệu biểu phí |
| `don_gia_th` | Đơn giá 1 TC thực hành (đồng) | tài liệu biểu phí |

- `ParamSpec` thêm trường `source: Literal["user", "document"]`. Chỉ tham số
  `document` mới được tìm trong Qdrant; tham số `user` (số TC của kỳ này) đi thẳng ra
  panel - không đoán số của riêng sinh viên từ tài liệu.
- `khoa_tuyen_sinh` không vào phép tính, chỉ để tìm đúng dòng đơn giá. Đơn giá còn
  thiếu mà chưa biết khoá → hỏi khoá trước (một lượt), như quy tắc "hỏi lựa chọn trước,
  số liệu sau" của luồng LLM.

### Luồng

```
extract_request (formula_id = tuition_semester | tuition_year | tuition_program)
  → builtin_outcome: thiếu tham số `document`?
      → fill_from_documents: _search (Qdrant + ngưỡng điểm + LLM rerank)
          → LLM trích {param: {value, chunk}} → Python kiểm tra nguyên văn
      → còn thiếu → NeedsInput (panel)  |  đủ → Computed (+ nguồn)
resume (panel) → ghép câu trả lời → nếu còn thiếu `document` (VD vừa biết khoá)
  → fill_from_documents lần nữa → tính
```

### Chốt chặn chống bịa số

Giá trị LLM trích chỉ được nhận khi con số (chuẩn hoá `980.000` / `980,000` / `980000`)
xuất hiện trong đúng đoạn được dẫn. Sai → bỏ, coi như thiếu → hỏi người dùng.

## Task

### T1 - Công thức và tham số (`app/calculation/formulas.py`)
- `FormulaId` thêm `tuition_semester`, `tuition_year`, `tuition_program`; một hàm
  `tuition()` dùng chung, `Decimal`, bước thay số hiển thị dạng tiền (`38.350.000 đ`).
- `ParamSpec.source`; giới hạn hợp lý (TC 0-200, đơn giá 0-20.000.000, bước 1).
- `BUILTIN_TRIGGERS`: "học phí" + kỳ/học kỳ, năm/năm học, khoá/cả khoá/toàn khoá.
- `describe_builtin_formulas` / `describe_builtin_rules` thêm quy tắc học phí (để luồng
  LLM trả lời câu ngược "tối đa bao nhiêu tín để không quá 8 triệu").
- **AC:** test đơn vị cho tính đúng, thiếu tham số, giá trị sai giới hạn, router.
- **Size:** S-M.

### T2 - Trích giá trị từ tài liệu (`app/graph/nodes/calculation.py` + prompt mới)
- Prompt `agents/calculation_value_extractor.yaml`: nhận câu hỏi, khoá đã biết, danh
  sách tham số `document` còn thiếu, các đoạn đánh số; trả JSON
  `{"don_gia_lt": {"value": 980000, "chunk": 3}, ...}`; không chắc → bỏ trống.
- `fill_from_documents(formula_id, query, known, deps)` → (params mới, nguồn); dùng
  lại `_search`; câu tìm kiếm ghép từ khoá + ngành + loại số liệu cần.
- Kiểm tra nguyên văn như trên.
- **AC:** test: lấy đúng đơn giá từ đoạn bảng; bỏ giá trị không có trong đoạn dẫn;
  không có đoạn nào → không gọi LLM, trả rỗng.
- **Size:** M.

### T3 - Nối vào luồng tính toán và hiển thị nguồn
- `builtin_outcome` (lượt hỏi và lượt gửi panel) gọi `fill_from_documents` trước khi
  quyết định hỏi; `Computed` thêm `sources`; `number_citations` đánh số cả nguồn của
  `Computed`; phần hiển thị ghi `[n]` cạnh giá trị lấy từ tài liệu.
- **AC:** test graph: (a) đủ số từ tài liệu → tính luôn, có citation; (b) thiếu đơn giá,
  chưa biết khoá → panel hỏi khoá; trả lời xong → tìm lại → tính; (c) Qdrant không có
  gì → panel hỏi đơn giá.
- **Size:** M.

### T4 - Định tuyến trong extractor
- `calculation_extractor.yaml`: câu học phí kỳ/năm/khoá → công thức mới (không còn
  `llm`); ví dụ cho từng loại; câu ngược vẫn `llm`.
- **AC:** test `parse_extraction` + router cho các câu mẫu.
- **Size:** S.

### T5 - Tài liệu
- `docs/specs/SPEC-calc-engine.md` / `SPEC-calculation-node.md`: 3 công thức, quy tắc
  đơn giá theo khoá, nguồn `document`. Commit riêng.

### T6 - Kiểm tra thật trên collection eval
- Hỏi 3 câu (kỳ / năm / khoá) với ngành CNTT; ghi lại: số nào lấy từ tài liệu, số nào
  phải hỏi. Collection eval hiện không có biểu phí (Quyết định 1035) → đơn giá dự kiến
  sẽ phải hỏi; nếu muốn thử nhánh "lấy từ tài liệu" cần ingest biểu phí vào eval.

## Quyết định đã chốt (13-10-2026)

1. **QPAN**: không vào công thức; kết quả luôn kèm cảnh báo "Chưa gồm học phần Giáo dục
   quốc phòng - an ninh".
2. **Ngành → khối**: dùng khi tài liệu ghi rõ; không ghi thì hỏi `khoi_nganh`.
3. **Học phí 1 năm**: số TC lý thuyết / thực hành của năm do người dùng nhập.
4. **Hệ / cơ sở**: mặc định đại học chính quy tại trụ sở chính; ghi rõ trong kết quả khi
   đơn giá lấy từ tài liệu.
5. **TC lý thuyết và TC thực hành luôn là hai câu hỏi riêng**, không bao giờ gộp thành
   "tổng tín chỉ" (yêu cầu của người dùng).

Khi làm, `ParamSpec.source` được thay bằng `ParamSpec.lookup` (câu tìm Qdrant; `None` = số
của riêng sinh viên), và router được sửa để câu có "học phí" không rơi vào `course_score`.
