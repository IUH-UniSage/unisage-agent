# Báo cáo: registry công thức có version, bỏ nhánh LLM tự tính (26/10/2026)

Nhánh `feature/huydh-unisage-106-formula-registry` (xếp chồng trên PR #28). Quyết định và thiết kế:
`plan.md`; việc đã làm: `todo.md`.

## 1. Đã thay đổi gì

| Phần | Trước | Sau |
|---|---|---|
| Ai tính | Python tính xuôi 6 công thức; mọi phép khác (hỏi ngược, công thức trong tài liệu, số học trên số liệu văn bản) do LLM tự tính, nhãn "AI tự tính" | **Python tính mọi con số**, xuôi (`calculate`) và ngược (`solver.solved_result`), theo registry 9 công thức có `version` + `source`; LLM chỉ định tuyến và chép số |
| Câu không có công thức | LLM tính nhẩm từ tài liệu | Extractor trả `none` → task đi nhánh advisory: node 10 trích công thức/số liệu kèm nguồn, người dùng tự tính |
| Công thức | course_score, gpa, grade_conversion, tuition ×3 | + `admission_score_2026` (ĐGNL quy đổi thang 30, ĐK = Max, XT1/XT2/XT3, ĐXT), `transcript_score_3hk` (học bạ lớp 10, 11, HK1 lớp 12), `weighted_score` (trọng số đề cương) |
| Tính ngược | prompt LLM "thử giá trị liền kề" | solver nhị phân trên lưới của tham số (bước làm tròn của công thức: CK theo 0,5), dòng "Kiểm tra" với giá trị liền kề, báo rõ khi không đạt được |
| Prompt | 32 file; `chat_calculation_llm.yaml` 81 dòng | 31 file (−78 dòng): extractor viết lại (schema `solve_for`/`target`/`none`), phân loại liệt kê registry, advisory thêm luật không tự tính |
| Code | | `app/calculation` +1 213 dòng (formulas +778, solver +431 mới); node/graph/schema −301; prompting py −59; tests −120 ròng (15 test nhánh LLM bỏ, 7 test mới + 26 test formulas/solver) |
| Cấu hình | `CHAT_CALC_LLM_ENABLED`, `CHAT_CALC_LLM_TIMEOUT_SECONDS` | bỏ; sổ field thêm 22 tham số |

Kiểm tra: 2 723 test pass; ruff/mypy sạch trên file đã sửa (lỗi mypy/ruff còn lại là lỗi cũ ngoài phạm vi).

## 2. Chạy lại 38 câu calculation (`runs/calc-registry-v1`, chế độ `none`, 26/10 tối)

### 2.1 Tám câu thuộc công thức registry

| Câu | Công thức | Kỳ vọng | Python trả | Đúng |
|---|---|---|---|---|
| d0307 ĐXT xuôi | admission_score_2026 | 25,4 | XT1 25,4; XT2 24,5; ĐXT 25,4 | ✓ |
| d0308 ĐHB tối thiểu để XT1 ≥ 24 | admission (ngược, `dhb`) | 25,5 | 25,5 (kiểm tra 25,49 → 23,997) | ✓ |
| d0309 HK1 lớp 12 môn Lý tối thiểu | transcript (ngược, `m3_12`) | 5,5 | 5,5 | ✓ (trước đây LLM dừng ở 5,00 / kết luận 7,0) |
| d0310 điểm học phần 20/30/50 | weighted_score | 7,9 | 7,9 / B | ✓ |
| d0311 cuối kỳ để A, điểm thi làm tròn 0,5 | weighted (ngược, `diem[3]`) | 9,5 | 9,5 (lưới 0,5) | ✓ |
| d0312 Interpretation 1 | weighted_score | 6,8 | 6,8 / C+ | ✓ |
| d0313 Final Test tối thiểu để ≥ 7,0 | weighted (ngược, `diem[4]`) | 7,2 | **7,1** (6,95 → 7,0 sau làm tròn 0,1) | ✗ theo đáp án; đúng theo quy tắc làm tròn của Trường |
| d0314 làm tròn 6,2 / 7,6 / 8,8 | không có công thức (quy tắc làm tròn) → advisory | 6,0; 7,5; 9,0 | trả lời từ Sổ tay, không tính | ✗ theo thang đo |

7/8 con số do Python tính đúng và được giải thích từng bước; d0313 là khác biệt về giả định làm tròn
(đáp án không làm tròn điểm học phần). Chạy riêng một câu (d0307) hết 4,8 s; trong lượt chạy 38 câu các
câu này ghi 46–70 s vì rate limit credential eval (max_rpm 15, hai lượt song song, chờ "next slot in
38,6 s") - không phải do code.

### 2.2 Ba mươi câu không có công thức (đi nhánh advisory)

- Tất cả đều được trả lời từ tài liệu, không còn item calculation nào (`calculation_items` rỗng),
  không còn nhãn "AI tự tính". Theo thang đo cũ (số kỳ vọng xuất hiện) pass 14/38 (v4: 26/38) - thang đo
  này chấm phép tính mà hệ thống giờ cố ý không làm, cần gắn lại nhãn theo `formula_id` (ticket sau).
- **Điểm cần quyết định tiếp**: luật mới trong prompt advisory ("không tự tính ra con số kết quả") chưa
  đủ mạnh - trong ~20/30 câu model vẫn cộng trừ ra kết quả (d0237 "12 tín chỉ", d0316 "100 đơn", d0330
  "600.000 đồng"...), và grounding chỉ cảnh báo 2 câu vì tổng/hiệu của các số đã hiện được coi là "suy
  ra được" (thiết kế Phase 6). Hai hướng: (a) chấp nhận LLM cộng trừ trong câu trả lời advisory, giữ
  grounding như hiện tại; (b) siết: với lượt có task `Deferred`, grounding bật chế độ chặt (số suy ra
  cũng bị gắn caution + cảnh báo admin) và đưa luật lên đầu prompt. Chưa làm, chờ chốt.

### 2.3 Tổng hợp

| | calc-none-v4 (LLM tự tính) | calc-registry-v1 |
|---|---|---|
| câu qua node calculation | 32 | 7 (đúng 7 câu có công thức) |
| con số do Python tính | 1 | 7 (6 đúng đáp án, 1 khác giả định làm tròn) |
| con số do LLM tính nhẩm | 31 | 0 trong khối tính; vẫn có trong câu trả lời advisory (xem 2.2) |
| nhãn "AI tự tính" | 32 | 0 |
| cảnh báo grounding | 1 | 2 (`ANSWER_UNGROUNDED_NUMBERS`) + 1 `ANSWER_NO_CITATION` |
| p50 câu công thức (chạy riêng) | 5,1 s | 4,8 s |

## 3. Bổ sung tối 26/10: năng lực số học thuần (`pure_arithmetic`) và grounding chặt

Theo chính sách 3 mức người dùng chốt (A số học thuần → code; B phụ thuộc quy định → registry/hỏi lại;
C chưa cài → chỉ trích dẫn; grounding không phải quyền tính). Triển khai: `app/calculation/arithmetic.py`
(parser whitelist, không eval), node nhận `formula_id: arithmetic` khi mọi số hạng nằm trong câu hỏi,
câu trả lời tư vấn gửi khối `calc_request` để `app/graph/calc_requests.py` kiểm tra căn cứ rồi nối khối
Python; lượt có task không công thức hoặc có `calc_request` bật grounding chặt (tổng/hiệu model tự tính
cũng bị gắn caution + `ANSWER_UNGROUNDED_NUMBERS`).

### 3.1 Chạy lại 38 câu (`runs/calc-registry-v2`, trước khi thêm grounding chặt)

| | registry-v1 | registry-v2 (+ số học thuần) |
|---|---|---|
| pass theo thang đo cũ | 14/38 | 20/37 (1 câu lỗi rate limit) |
| qua node calculation | 7 | 17 (7 công thức + 10 số học thuần trên số trong câu hỏi) |
| câu trả lời tư vấn có khối `calc_request` tính được | 0 | 13 |
| `calc_request` bị từ chối (số hạng không có căn cứ / không phải số) | — | 4 (d0329, d0330, d0332, d0336: đếm ngày, giờ:phút - không phải biểu thức số) |
| p50 cả lượt | 12,5 s | 8,4 s |

Số học thuần trên số trong câu hỏi tính đúng cả 10 câu (d0244 100 − 10 = 90; d0322 140 − (20 + 25 + 21
+ 22 + 21) = 31; d0333 0,5 × 20 + 1 × 2 = 12...). Khối `calc_request` tính đúng khi số hạng có căn cứ
(d0321 15 − 6 = 9; d0237 3 + 4 + 2 + 3 = 12).

Vấn đề còn lại: model vẫn **tự nêu kết quả trong câu chữ** ở một số câu dù đã có khối (d0330 "590.000"
sai, khối tính đúng 600.000; d0317 "382" sai; d0332 "33 giờ 20 phút" sai). Chạy grounding chặt offline
trên 20 câu trả lời tư vấn của v2: chế độ thường gắn caution 2 câu, chế độ chặt 8 câu (bắt đủ d0317,
d0330, d0332, d0319) - đã bật cho các lượt này (chưa chạy lại eval).

### 3.2 Giới hạn

- Đếm ngày giữa hai mốc và cộng giờ:phút không phải biểu thức số → `calc_request` từ chối (đúng thiết
  kế); nếu cần, thêm công thức registry "khoảng ngày" sau.
- Model có thể vẫn viết kết quả tự tính trong câu chữ; grounding chặt chỉ gắn caution, không xoá.
- Thang đo eval cũ vẫn chưa gắn nhãn theo `formula_id`/`arithmetic`.

## 4. Ngoài phạm vi / việc tiếp

- Gắn lại nhãn eval calculation theo `formula_id`, tách 30 câu số học văn bản sang category riêng.
- Chốt hướng (a)/(b) ở 2.2.
- Điểm ưu tiên theo công thức giảm dần của Bộ GD&ĐT, điểm thủ khoa ĐGNL từng năm: tham số người dùng
  nhập; cài thêm khi có văn bản trong KB.
- Công thức trong tài liệu hay được hỏi (điểm rèn luyện, quy đổi chứng chỉ, hệ số đề tài) → thêm vào
  registry với version/nguồn khi cần.
