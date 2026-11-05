# Báo cáo đầy đủ: benchmark CE chỉ retrieval trên bộ official (03/11/2026)

Tóm tắt, bảng tổng, hình, khuyến nghị: `report.md`. File này: từng bước, đo phụ, từng câu. Bảng đầy
đủ của từng truy vấn: `report.xlsx` (sheet "5 Lỗi ce40", "6 Cứu-rơi ce40", "Theo câu") và
`results/summary.json` (khoá `ce40_misses`, `changed_ce40`, `all_rows`).

## 1. Bước 1: chuẩn bị bộ câu

- 592 dòng; 454 có `expected_doc_ids` (439 một tài liệu, 15 hai tài liệu), 0 tài liệu thiếu `document_id` trong manifest.
- Persona của 352 câu thấy được: guest 227, same_department_exact_level 76, wildcard_max_level 37, other_department_max_level 12.
- 102 câu không được thấy (kiểm tra rò): same_department_one_level_below 37, other_department_max_level 37, guest 28.

## 2. Bước 2: lấy ứng viên

- 908 truy vấn (704 + 204), embed theo batch 16, `_search` ở độ sâu 64 và 40. Pool luôn đủ 64.
- 668/704 truy vấn có tài liệu đúng trong pool 64 sau sàn; 679/704 trước sàn (sàn lấy mất 11).
- Pool 40 lấy riêng so với 40 chunk đầu của pool 64: trùng tập 98,5 %, trùng hoàn toàn thứ tự 74/704, 8 chunk đầu trùng 599/704, recall@8 cấp tài liệu bằng nhau (0,901).

## 3. Bước 3: chấm CE

- 704 × 3 request, 0 lỗi, khoảng 8 phút. Passage bị cắt ở 512 token: 36–42 %.
- Đo phụ: `ce40` không sàn, 704 request, p50 217 ms, p95 228 ms.

## 4. Đo phụ: bỏ sàn khi bật CE (recall@8)

| Nhóm | ce40 sàn 0,45 | ce40 không sàn | none không sàn | tốt hơn / kém hơn |
|---|---|---|---|---|
| tất cả (704) | 0,917 | **0,933** | 0,901 | 11 / 0 |
| question (352) | 0,973 | 0,973 | 0,977 | 0 / 0 |
| question_user (352) | 0,861 | **0,892** | 0,825 | 11 / 0 |
| normal (414) | 0,934 | 0,941 | 0,912 | 3 / 0 |
| calculation (118) | 0,856 | 0,898 | 0,847 | 5 / 0 |
| access (172) | 0,919 | 0,936 | 0,913 | 3 / 0 |

recall@5 ce40 không sàn 0,915, recall@1 0,781, MRR 0,852.

## 5. 61 truy vấn ce40 chưa đạt recall@8 = 1

Nhóm:

- **A**: tài liệu không có trong pool 64 (25: 24 user, 1 exact).
- **B**: chỉ ở hạng 41–64 (2).
- **C**: trong 40 đầu nhưng mọi chunk dưới sàn (12, đều là user).
- **D**: CE thấy nhưng xếp ngoài top-8, hoặc chỉ được 1/2 tài liệu (22: 13 user, 9 exact).

### 5.1. A: retrieval không lấy được (25)

d0022:user, d0028:user, d0038:user, d0054:user, d0204:user, d0209:user, d0221:user, d0239:user,
d0278:user, d0300:user, d0310:user, d0311:user, d0314:user, **d0315:exact**, **d0315:user**,
d0326:user, q0001:user, q0004:user, q0008:user, q0057:user, q0082:user, q0106:user, q0140:user,
q0168:user, q0209:user.

- Phần lớn là câu không dấu. Top-1 sau CE thường là "Plan 67-KH-ĐHCN – Anti-terrorism work of IUH in 2025" (văn bản OCR không dấu, khớp BM25 với câu không dấu; nằm trong top-8 RRF của 57 truy vấn).
- d0310/d0311/d0314: câu tính điểm chung chung, nhãn là một sổ tay hoặc programme specification cụ thể.
- d0315: sổ tay tiếng Anh, câu tiếng Việt.

### 5.2. B, C: trong pool nhưng CE không thấy (14)

- B: d0021:user (hạng RRF 46), d0149:user (42).
- C (bị sàn cắt; hạng RRF trước sàn): d0086:user (1), d0089:user (2), d0114:user (23), d0302:user (1), d0324:user (25), d0328:user (23), d0331:user (1), d0336:user (1), q0063:user (11), q0075:user (1), q0125:user (2), q0171:user (1).
- Ở 7/12 truy vấn nhóm C, tài liệu đúng đứng **hạng 1–2 RRF** (nhờ BM25), nhưng cosine dense < 0,45 nên bị bỏ. Bỏ sàn cứu được 11 truy vấn (mục 4).

### 5.3. D: CE xếp sai (22), hạng RRF → hạng ce40

- **Gần trùng / nhãn hẹp:**
  - d0146:exact, d0149:exact (2 → 29), d0146:user (29 → 16): câu hỏi về "Mẫu số 01", nhãn là bản đăng ký của một người cụ thể; CE chọn bản đăng ký của người khác.
  - q0214:exact (2 → 9), q0214:user (1 → 10): "Mạch điện tử" bị nhầm với "Lý thuyết mạch điện".
  - q0133:exact (8 → 12), q0133:user (9 → 10): portfolio của sinh viên khác.
  - q0057:exact (4 → 11): CE chọn chương trình đào tạo, nhãn là kế hoạch đào tạo.
  - q0119:exact (12 → 15): poster PEOs/PLOs bản tiếng Việt so với tiếng Anh.
- **Câu 2 tài liệu, CE giữ 1:** d0220:user, d0224:user, d0229:exact (1 → 8), d0231:user, d0236:user (3 → 5).
- **Khác:**
  - d0056:exact (1 → 9): CE ưu tiên chương trình thạc sĩ QTKD hơn phụ lục ngành gần.
  - d0049:user (27 → 10), d0323:user (15 → 20), q0033:user (9 → 12), q0036:user (11 → 16), q0052:user (20 → 15), q0071:user (19 → 9), q0166:exact (11 → 15).

## 6. 39 truy vấn ce40 khác none ở recall@8

**Cứu được (30)**, hạng none → hạng ce40:

- q0005:user 37→6, q0009:user 13→1, q0012:user 15→1, q0059:user 24→2, q0099:user 9→1, q0126:exact 9→2, q0126:user 16→2, q0131:user 21→4, q0132:exact 9→2, q0160:user 15→1, q0208:user 13→2;
- d0029:user 16→1, d0064:user 9→3, d0071:user 15→1, d0074:user 17→1, d0111:user 14→1, d0281:user 15→1, d0295:user 9→3, d0298:user 10→1, d0309:user 35→2, d0317:user 19→1, d0319:user 14→1, d0335:user 14→6;
- 7 câu 2 tài liệu lấy thêm được tài liệu thứ hai: d0220:exact, d0224:exact, d0230:exact, d0230:user, d0232:exact, d0232:user, d0235:exact.

**Làm rơi (9):** q0057:exact 4→11, q0133:exact 8→12, q0214:exact 2→9, q0214:user 1→10,
d0056:exact 1→9, d0146:exact 2→29, d0149:exact 2→29, d0229:exact (1 tài liệu rơi), d0236:user
(1 tài liệu rơi).

## 7. q0054, d0315

- **q0054** (Kế hoạch đào tạo chuyên ngành Điện tử công nghiệp năm 2022, persona cùng khoa cấp 2):
  - exact: hạng RRF 1 (dense 1, BM25 2), chunk đúng ở hạng 1, 4, 6, 8, 11…;
  - user: hạng RRF 1, 8/9 chunk đầu thuộc tài liệu đúng.
  - Mọi phương án hạng 1. Lần trượt ở smoke test nằm trước retrieval (HyDE / câu hỏi độc lập / tách câu), cần trace ở bước 2.
- **d0315**: tài liệu `5eb66079-…` "Handbook for Student 2024 – School of Finance and Accounting", 69 chunk, is_public true, tiếng Anh (chunk có "Specialized knowledge: 61 credits").
  - Câu tiếng Việt không đưa chunk nào vào pool 64.
  - Top RRF: "Bản mô tả CTĐT ngành Kiểm toán", "Staff Handbook 2024 – SFA".
  - Cần truy vấn song ngữ hoặc nhãn thêm tài liệu tiếng Việt tương đương.

## 8. Rò quyền

204 truy vấn (102 câu × 2), các persona one_level_below / other_department / guest: tài liệu kỳ vọng
không xuất hiện trong pool 64 lần nào.

## 9. Sự cố

- Request CE đầu tiên mất 1 022 ms (khởi động); không loại khỏi thống kê, chỉ ảnh hưởng max.
- Lúc đầu, script chờ (`pgrep -f drive_collect.py`) khớp với chính dòng lệnh chờ của nó nên kẹt; đã dừng script chờ, CE chạy tiếp bình thường, dữ liệu không bị ảnh hưởng.
