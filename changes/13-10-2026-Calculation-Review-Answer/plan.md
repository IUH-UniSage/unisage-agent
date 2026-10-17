# Plan: AI kiểm tra kết quả tính có khớp câu hỏi

## Vấn đề

Câu "tính giúp tôi học phí ngành công nghệ thông tin 2 năm đầu" (13-10-2026, collection eval):

1. Extractor chọn `tuition_year` vì không có công thức nào cho "N năm". Form hỏi "Số tín chỉ …
   **của một năm học**", người dùng nhập 100 / 42, Python ra "Học phí **một năm học**
   140.400.000 đồng", lệch với câu hỏi.
2. Lần gọi LLM sau khi Python tính (`commentary`, prompt `main/chat_calculation.yaml`) chỉ được
   dặn viết ≤ 2 câu nhận xét và không nói số lạ. Không bước nào đối chiếu phép tính với câu
   hỏi, nên sự lệch đi thẳng tới người dùng.
3. LLM rerank giữ đoạn mà chính lý do của nó ghi là không dùng được ("không có dòng học phí …
   nhưng được giữ", "Khối Kinh tế … không dùng cho SQ1"). Prompt rerank cũng còn nhắc nội
   dung bị cắt ngắn và dòng "Mục có trong đoạn", đã lỗi thời từ khi rerank đọc nguyên đoạn.

## Nguyên tắc

- **Số do Python tính, AI chỉ kiểm tra và viết lời.** AI không tính lại, không đổi số. Câu trả
  lời AI viết chỉ được chứa số có trong kết quả Python; có số lạ thì bỏ phần AI viết, hiện khối
  Python như hiện nay.
- Không thêm lần gọi LLM: bước kiểm tra thay chỗ lần gọi viết nhận xét.
- Không khớp → vẫn hiện kết quả, cảnh báo đặt lên đầu (phương án a); không hỏi lại, không vòng
  lặp.

## Task

### T1 - Phạm vi học phí theo đúng câu hỏi (`pham_vi`)

Sửa gốc lỗi 1: tiêu đề và nhãn form dùng đúng cụm phạm vi người dùng nói.

- `formulas.py`: ba công thức học phí thêm tham số `pham_vi` (`text`, không bắt buộc, không
  `lookup`, không vào phép tính). Nhãn tín chỉ và tiêu đề kết quả dùng `pham_vi` khi có ("Số
  tín chỉ lý thuyết của 2 năm đầu", "Học phí 2 năm đầu"), không có thì giữ mặc định ("một học
  kỳ" / "một năm học" / "cả khoá học").
- Nhãn form phụ thuộc tham số đã biết: thêm `label_for(spec, known)` trong `formulas.py`;
  `question_for` (calculation.py) nhận `known` để dựng nhãn.
- `calculation_extractor.yaml`: `pham_vi` = cụm phạm vi nguyên văn trong câu ("2 năm đầu",
  "học kỳ 1 năm 2"); "N năm" (N ≥ 2) → `tuition_year` với `pham_vi` (tín chỉ của cả N năm do
  người dùng nhập); ví dụ cho câu "2 năm đầu".
- **AC:** `calculate("tuition_year", {..., "pham_vi": "2 năm đầu"})` có tiêu đề "Học phí 2 năm
  đầu"; form hỏi "Số tín chỉ lý thuyết của 2 năm đầu"; test graph cho đúng câu trong log.
- **Files:** `app/calculation/formulas.py`, `app/graph/nodes/calculation.py`,
  `agents/calculation_extractor.yaml`, `tests/calculation/test_formulas.py`,
  `tests/graph/test_calculation_node.py`.
- **Size:** S.

### T2 - Bước "kiểm tra và trả lời" thay cho câu nhận xét

Lưới an toàn cho mọi công thức có sẵn (không riêng học phí).

- Prompt mới `main/chat_calculation_review.yaml`, thay `main/chat_calculation.yaml` (xoá file cũ,
  `build_calculation_commentary_prompt`, mục trong `loader.py` / `schema.py`). Đầu vào: câu hỏi
  gốc, lịch sử gần đây, mỗi kết quả Python (tiêu đề, đầu vào, các bước đã thế số, kết quả). Đầu
  ra JSON:
  ```json
  {"fits": true, "issue": null, "answer": "Học phí 2 năm đầu của bạn là 140.400.000 đồng."}
  ```
  - `fits`: phép tính có trả lời đúng điều được hỏi không (đúng đại lượng, đúng phạm vi, đúng
    đối tượng).
  - `issue`: một câu nói chỗ lệch khi `fits = false`.
  - `answer`: 1-3 câu trả lời thẳng câu hỏi bằng con số Python đã tính; khi `fits = false` thì
    nói rõ kết quả dưới đây là cho cái gì và gợi ý hỏi lại.
- `calculation_turn.py`: `commentary()` → `review()` trả `Review(fits, issue, answer)`.
  - Tập số được phép: số trong đầu vào **và** mọi số Python đã hiển thị (các bước, kết quả).
    `answer` / `issue` có số ngoài tập → bỏ phần AI viết (log `calculation.review_rejected`).
  - JSON hỏng, lỗi gọi model, timeout → bỏ phần AI viết, không chặn câu trả lời.
- `streaming_graph.py` (`_turn_with_calculations`, lượt chỉ có tính toán):
  - Thứ tự hiển thị: `fits = false` → "**Lưu ý:** {issue}" rồi `answer`, rồi khối Python;
    `fits = true` → `answer` rồi khối Python. Không có phần AI viết → chỉ khối Python.
  - Vì phần AI viết đứng trước, khối Python chỉ được stream sau khi bước kiểm tra xong (token
    đầu tiên đến muộn hơn khoảng một lần gọi LLM; tổng thời gian không đổi).
  - Lượt có cả câu hỏi tra cứu: giữ như hiện tại (node 10 viết phần chữ, không gọi review).
- Trace staff (`trace_items`): thêm `review: {fits, issue, rejected}` cho các mục `builtin`.
- **AC (test graph):** (a) khớp → `answer` đứng trước khối Python; (b) câu "2 năm đầu" với kết
  quả "một năm học" → "Lưu ý:" đứng đầu, kết quả vẫn hiện; (c) `answer` có số lạ → chỉ khối
  Python; (d) model lỗi → chỉ khối Python; (e) lượt có cả tra cứu → không gọi review.
- **Files:** `main/chat_calculation_review.yaml` (mới), `main/chat_calculation.yaml` (xoá),
  `app/rag/prompting/{__init__,loader,schema}.py`, `app/graph/calculation_turn.py`,
  `app/graph/streaming_graph.py`, test tương ứng.
- **Size:** M.

### T3 - LLM rerank: chỉ giữ đoạn thật sự trả lời câu hỏi

- `agents/reranker_compressor.yaml`:
  - Bỏ câu "nội dung có thể bị cắt ngắn", dòng "Mục có trong đoạn" và quy tắc 4 (đoạn bị cắt):
    rerank giờ đọc nguyên đoạn.
  - Thêm quy tắc: lý do nào nói đoạn không chứa thông tin cần ("không có dòng học phí", "không
    dùng cho SQk", "chỉ cùng ngành") thì KHÔNG đưa đoạn vào object; chỉ liệt kê đoạn được giữ.
  - Ví dụ phản diện lấy từ log 13-10-2026 (đoạn "Kế hoạch đào tạo CNTT" cho câu học phí).
- Không đổi code `llm_rerank.py`: dò từ phủ định trong lý do dễ sai.
- **AC:** test prompt loader khẳng định quy tắc mới có trong template và chữ "bị cắt ngắn" /
  "Mục có trong đoạn" không còn; kiểm tra thật ở T5.
- **Size:** XS.

### T4 - Spec

- `docs/specs/SPEC-calculation-node.md` §3.2 (nhận xét → kiểm tra và trả lời), §1 (`pham_vi`);
  `SPEC-calc-engine.md` mục học phí. Commit riêng.

### T5 - Kiểm tra thật trên collection eval (người dùng chạy)

- "tính giúp tôi học phí ngành công nghệ thông tin 2 năm đầu" → form hỏi "… của 2 năm đầu",
  kết quả "Học phí 2 năm đầu", câu trả lời mở đầu đúng câu hỏi.
- Một câu cố ý lệch (VD hỏi "cả khoá" nhưng nhập số tín chỉ một kỳ) → có "Lưu ý:".
- Log rerank không còn đoạn được giữ với lý do phủ định.

## Thứ tự và commit

T1 → T2 → T3 → T4, mỗi task một commit (T4 riêng vì là tài liệu). Chạy toàn bộ test + ruff trước
mỗi commit.

## Rủi ro

- Model kiểm tra báo `fits = false` sai (quá khắt khe): kết quả vẫn hiện, chỉ thêm một câu lưu
  ý; prompt yêu cầu chỉ báo lệch khi đại lượng / phạm vi / đối tượng khác hẳn câu hỏi.
- Token đầu tiên ở lượt chỉ có tính toán đến chậm hơn (chờ bước kiểm tra). Nếu cần, sau này có
  thể stream khối Python trước và đặt phần AI viết bên dưới, đổi lại cảnh báo không nằm đầu.
