# 38 câu calculation: input và response (lần chạy `runs/calc-registry-v2`, 26/10/2026, chế độ `none`)

Mỗi câu: câu hỏi gửi vào hệ thống (input), đáp án kỳ vọng của tập eval, đường xử lý thực tế (node calculation với công thức registry / số học thuần, hay nhánh tư vấn), toàn văn câu trả lời và cảnh báo admin. `pass` theo thang đo cũ (số kỳ vọng xuất hiện và đi qua node calculation).

## d0237

**Input:** Trong chương trình Logistics và Quản lý chuỗi cung ứng của Khoa Quản trị Kinh doanh, tổng số tín chỉ của 4 học phần Quản lý thông tin trong chuỗi cung ứng, Giao nhận vận tải và bảo hiểm trong kinh doanh quốc tế, Phân loại hàng hóa và xuất xứ hàng hóa, Thủ tục hải quan là bao nhiêu?

**Kỳ vọng:** 12 tín chỉ (3 + 4 + 2 + 3 = 12).  
**Số kỳ vọng:** None

**Đường xử lý:** tư vấn + khối calc_request (code tính)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 6.8 s

**Response:**

> Tổng số tín chỉ của 4 học phần nêu là **12 tín chỉ** [1].
> 
> **Phép tính số học: Tổng tín chỉ 4 học phần**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu trích từ tài liệu [1]_
> 
> Thay số:
> 1. Tổng tín chỉ 4 học phần: 3 + 4 + 2 + 3 = 12
> 
> Kết quả: Tổng tín chỉ 4 học phần: **12 tín chỉ**

## d0238

**Input:** Trong bài tập Nhập môn lập trình C của Khoa CNTT, chương trình ở Bài 13 phần bài tập tuần 01-02 gán giá trị cho num1, num2 rồi tính total; giá trị total được in ra là bao nhiêu?

**Kỳ vọng:** total = 60 (num1 = 25, num2 = 35; total = 25 + 35 = 60).  
**Số kỳ vọng:** [60]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 3.4 s

**Response:**

> Câu hỏi này nằm ngoài phạm vi mình hỗ trợ rồi. Mình chuyên về học vụ của Nhà trường, ví dụ quy chế đào tạo, học bổng, học phí hay thủ tục giấy tờ. Bạn có câu hỏi nào về những chủ đề đó không?

## d0241

**Input:** Trên phiếu đăng ký ưu tiên xét tuyển đại học năm 2024 của IUH, thí sinh có chứng chỉ IELTS và thí sinh có TOEIC 4 kỹ năng ghi mã đối tượng ưu tiên xét tuyển là bao nhiêu?

**Kỳ vọng:** Chứng chỉ IELTS ghi mã đối tượng 4; TOEIC 4 kỹ năng ghi mã đối tượng 5.  
**Số kỳ vọng:** None

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 5.7 s

**Response:**

> - Thí sinh có **chứng chỉ IELTS**: mã đối tượng ưu tiên xét tuyển là **4** [1].  
> - Thí sinh có **TOEIC 4 kỹ năng**: mã đối tượng ưu tiên xét tuyển là **5** [1].

## d0242

**Input:** Trong khung chương trình ngành Quản trị nguồn nhân lực (tổng 138 tín chỉ), học phần Thực tập doanh nghiệp và Khóa luận tốt nghiệp cộng lại bao nhiêu tín chỉ, và phần còn lại của chương trình là bao nhiêu tín chỉ?

**Kỳ vọng:** Thực tập doanh nghiệp 5 + Khóa luận tốt nghiệp 5 = 10 tín chỉ; phần còn lại 138 − 10 = 128 tín chỉ.  
**Số kỳ vọng:** None

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 8.9 s

**Response:**

> **Thực tập doanh nghiệp và Khóa luận tốt nghiệp cộng lại là 10 tín chỉ; phần còn lại của chương trình là 128 tín chỉ** [2][8].
> 
> - **Thực tập doanh nghiệp**: 5 tín chỉ [2][8].  
> - **Khóa luận tốt nghiệp / Khóa luận**: 5 tín chỉ (một nơi ghi 5 TC [3], nơi khác ghi 8 TC cho khóa luận đại học theo khung — nhưng trong bảng tổng số tín chỉ toàn chương trình ngành có ghi tổng 139/139 TC và mục Thực tập doanh nghiệp = 5 TC, Khóa luận = 5 TC phù hợp với phần liệt kê học kỳ 7/7. Vì câu hỏi nêu "tổng 138 tín chỉ", ta đối chiếu theo dòng tổng 139/138 trong tài liệu ngành; phần cộng lại theo các mục cụ thể là 5 + 5 = 10 TC) [2][8].
> 
> - **Tổng tín chỉ chương trình**: 139 TC (tài liệu nhiều chỗ ghi tổng 139 TC cho chương trình đại học) — theo yêu cầu câu hỏi dùng mốc "tổng 138" thì phần còn lại = 138 − 10 = 128 TC; theo tài liệu tổng 139 thì phần còn lại = 139 − 10 = 129 TC. Vì nguồn chính liệt kê tổng 139 TC, nhưng bạn nêu 138 TC, mình ghi kết quả đối chiếu theo cả hai giá trị để bạn đối chiếu: phần còn lại là **128 TC (nếu tổng 138)** hoặc **129 TC (nếu tổng 139)** [2][7][8].
> 
> Tham khảo: các mục tín chỉ Thực tập và Khóa luận trong chương trình đào tạo và tổng số tín chỉ của chương trình [2][7][8].

## d0243

**Input:** Chương trình đào tạo ngành Tài chính doanh nghiệp năm 2020 có 139 tín chỉ nhưng chưa tính tiếng Anh 1 và 2. Nếu sinh viên không được miễn tiếng Anh thì tổng số tín chỉ phải học là bao nhiêu? Cần bao nhiêu điểm TOEIC đầu vào để được miễn cả hai môn?

**Kỳ vọng:** 145 tín chỉ = 139 + 3 (Tiếng Anh 1) + 3 (Tiếng Anh 2). Đạt trên 350 điểm TOEIC đầu vào thì được miễn cả Tiếng Anh 1 và Tiếng Anh 2 (trên 250 điểm được miễn Tiếng Anh 1).  
**Số kỳ vọng:** None

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 8.0 s

**Response:**

> **Tổng số tín chỉ phải học nếu sinh viên không được miễn Tiếng Anh 1 và 2 là 145 tín chỉ [1].**  
> **Để được miễn cả hai môn Tiếng Anh 1 và 2, sinh viên phải đạt TOEIC > 350 điểm (theo chuẩn trong chương trình).** [1]
> 
> Giải thích ngắn: mục ghi "Tổng số tín chỉ của chương trình đào tạo là 139 tín chỉ, không tính môn tiếng Anh 1 (3 TC) và tiếng Anh 2 (3 TC)." → nếu không được miễn hai môn này thì phải cộng thêm 3 + 3 = 6 TC vào 139 TC, thành 145 TC [1]. Tiêu chuẩn miễn: "nếu đạt trên 350 điểm được miễn cả tiếng Anh 1 và tiếng Anh 2" [1].

## d0244

**Input:** Chương trình học bổng KHCN du học của Tập đoàn Vingroup năm 2020 cấp tối đa 100 suất toàn phần, trong đó có 10 suất thạc sĩ Toán Bảo mật và Tin Bảo mật tại Đại học Limoges. Vậy số suất học bổng toàn phần còn lại cho các ngành khác tối đa là bao nhiêu?

**Kỳ vọng:** 100 − 10 = 90 suất học bổng toàn phần còn lại.  
**Số kỳ vọng:** None

**Đường xử lý:** node calculation (builtin/computed: Số suất toàn phần còn lại cho các ngành khác tối đa: 90 suất)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 5.1 s

**Response:**

> Số suất học bổng toàn phần còn lại cho các ngành khác tối đa là **90 suất**.
> 
> **Phép tính số học: Số suất toàn phần còn lại cho các ngành khác tối đa**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Số suất toàn phần còn lại cho các ngành khác tối đa: 100 - 10 = 90
> 
> Kết quả: Số suất toàn phần còn lại cho các ngành khác tối đa: **90 suất**

## d0245

**Input:** Theo Quy chế đào tạo tiến sĩ của IUH năm 2017, nghiên cứu sinh có bằng đại học phải học nhiều hơn nghiên cứu sinh có bằng thạc sĩ tối thiểu bao nhiêu tín chỉ?

**Kỳ vọng:** 120 − 90 = 30 tín chỉ.  
**Số kỳ vọng:** None

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 5.0 s

**Response:**

> Nghiên cứu sinh có bằng đại học phải học nhiều hơn nghiên cứu sinh có bằng thạc sĩ tối thiểu 30 tín chỉ. [1]

## d0246

**Input:** Theo quy chế đào tạo tiến sĩ năm 2023 của IUH, một nghiên cứu sinh chưa có bằng thạc sĩ có thời gian đào tạo tiêu chuẩn 4 năm; so với tổng thời gian đào tạo tối đa thì còn tối đa bao nhiêu năm để gia hạn?

**Kỳ vọng:** Tổng thời gian tối đa là 06 năm nên còn tối đa 6 − 4 = 2 năm.  
**Số kỳ vọng:** None

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 23.5 s

**Response:**

> Thời gian gia hạn đào tạo tối đa đối với nghiên cứu sinh có thời gian đào tạo tiêu chuẩn 4 năm là 2 năm (24 tháng) [1][2]. Tổng thời gian đào tạo tối đa tính từ ngày quyết định công nhận nghiên cứu sinh có hiệu lực là 6 năm (72 tháng) [1].

## d0307

**Input:** Theo công thức tính điểm xét tuyển đại học chính quy năm 2026 của IUH, một thí sinh có tổng điểm 3 môn thi tốt nghiệp THPT (ĐTN) trong tổ hợp là 24, không dự thi đánh giá năng lực, tổng điểm học bạ 3 môn (ĐHB) là 27, điểm ưu tiên khu vực/đối tượng là 0,5 và không có điểm cộng. Điểm xét tuyển (ĐXT) của thí sinh là bao nhiêu?

**Kỳ vọng:** ĐXT = 25,4 điểm. Vì không thi ĐGNL nên ĐK = Max(ĐTN, ĐĐGNL) = 24. XT1 = 0,7 × 24 + 0,3 × 27 + 0,5 = 16,8 + 8,1 + 0,5 = 25,4; XT2 = 24 + 0,5 = 24,5; ĐXT = Max(25,4; 24,5) = 25,4.  
**Số kỳ vọng:** [25.4]

**Đường xử lý:** node calculation (builtin/computed: XT1: 25.4; XT2: 24.5; ĐXT: 25.4)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 58.5 s

**Response:**

> Điểm xét tuyển của thí sinh là **25.4**.
> 
> **Điểm xét tuyển đại học chính quy 2026 (thang 30)**
> 
> Công thức:
> - ĐĐGNL = kết quả thi ĐGNL × 30 / ĐTK (ĐTK: điểm thủ khoa kỳ thi ĐGNL năm 2026)
> - ĐK = Max(ĐTN, ĐĐGNL); ĐTN: tổng điểm 3 môn thi tốt nghiệp THPT trong tổ hợp xét tuyển
> - XT1 = 0.7 × ĐK + 0.3 × ĐHB + Đ(Kv;Đt) + ĐC; ĐHB: tổng điểm học bạ 3 môn trong tổ hợp
> - XT2 = ĐTN + Đ(Kv;Đt) + ĐC; XT3 = ĐĐGNL + Đ(Kv;Đt) + ĐC
> - ĐXT = Max(XT1, XT2, XT3), thang điểm 30
> - _Phiên bản công thức 2026.1 · nguồn: Công thức tính điểm xét tuyển đại học chính quy năm 2026 của Trường_
> 
> Thay số:
> 1. Điểm thi dùng để xét: ĐK = ĐTN = 24 (không dự thi ĐGNL)
> 2. XT1: XT1 = 0.7 × 24 + 0.3 × 27 + 0.5 + 0 = 16.8 + 8.1 + 0.5 + 0 = 25.4
> 3. XT2: XT2 = 24 + 0.5 + 0 = 24.5
> 4. Điểm xét tuyển: ĐXT = Max(25.4; 24.5) = 25.4 (XT1)
> 
> Kết quả: ĐXT **25.4** / 30 (lấy theo XT1)
> > Đ(Kv;Đt) là điểm ưu tiên khu vực/đối tượng theo quy định của Bộ GD&ĐT (từ 22,5 điểm trở lên mức ưu tiên giảm dần) - nhập mức đã quy đổi; không nêu thì tính 0.
> > ĐC là điểm cộng (điểm thưởng, điểm xét thưởng, điểm khuyến khích); không nêu thì tính 0.

## d0308

**Input:** Theo công thức điểm xét tuyển năm 2026 của IUH (XT1 = 0,7×ĐK + 0,3×ĐHB + điểm ưu tiên + điểm cộng), một thí sinh có tổng điểm thi tốt nghiệp 3 môn là 23, không thi đánh giá năng lực, điểm ưu tiên 0,25 và không có điểm cộng. Tổng điểm học bạ 3 môn (ĐHB) tối thiểu phải là bao nhiêu để XT1 đạt 24 điểm?

**Kỳ vọng:** ĐHB tối thiểu 25,5 điểm. ĐK = Max(ĐTN, ĐĐGNL) = 23; cần 0,7 × 23 + 0,3 × ĐHB + 0,25 ≥ 24 ⇒ 0,3 × ĐHB ≥ 24 − 16,1 − 0,25 = 7,65 ⇒ ĐHB ≥ 7,65 / 0,3 = 25,5.  
**Số kỳ vọng:** [25.5]

**Đường xử lý:** node calculation (builtin/computed: ĐHB: 25.5; XT1: 24; XT2: 23.25; ĐXT: 24)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 55.5 s

**Response:**

> Tổng điểm học bạ 3 môn tối thiểu phải là **25.5**.
> 
> **Tổng điểm học bạ 3 môn trong tổ hợp (ĐHB) tối thiểu để XT1 ≥ 24**
> 
> Công thức:
> - ĐĐGNL = kết quả thi ĐGNL × 30 / ĐTK (ĐTK: điểm thủ khoa kỳ thi ĐGNL năm 2026)
> - ĐK = Max(ĐTN, ĐĐGNL); ĐTN: tổng điểm 3 môn thi tốt nghiệp THPT trong tổ hợp xét tuyển
> - XT1 = 0.7 × ĐK + 0.3 × ĐHB + Đ(Kv;Đt) + ĐC; ĐHB: tổng điểm học bạ 3 môn trong tổ hợp
> - XT2 = ĐTN + Đ(Kv;Đt) + ĐC; XT3 = ĐĐGNL + Đ(Kv;Đt) + ĐC
> - ĐXT = Max(XT1, XT2, XT3), thang điểm 30
> - Tìm Tổng điểm học bạ 3 môn trong tổ hợp (ĐHB) tối thiểu sao cho XT1 ≥ 24
> - _Phiên bản công thức 2026.1 · nguồn: Công thức tính điểm xét tuyển đại học chính quy năm 2026 của Trường_
> 
> Thay số:
> 1. Giá trị cần tìm: ĐHB = 25.5 (thử lần lượt theo bước 0.01)
> 2. Điểm thi dùng để xét: ĐK = ĐTN = 23 (không dự thi ĐGNL)
> 3. XT1: XT1 = 0.7 × 23 + 0.3 × 25.5 + 0.25 + 0 = 16.1 + 7.65 + 0.25 + 0 = 24
> 4. XT2: XT2 = 23 + 0.25 + 0 = 23.25
> 5. Điểm xét tuyển: ĐXT = Max(24; 23.25) = 24 (XT1)
> 6. Kiểm tra: ĐHB = 25.5 → XT1 24 (đạt); ĐHB = 25.49 → ≈ 23.997 (không đạt)
> 
> Kết quả: Tổng điểm học bạ 3 môn trong tổ hợp (ĐHB) tối thiểu **25.5** → XT1 **24**
> > Đ(Kv;Đt) là điểm ưu tiên khu vực/đối tượng theo quy định của Bộ GD&ĐT (từ 22,5 điểm trở lên mức ưu tiên giảm dần) - nhập mức đã quy đổi; không nêu thì tính 0.
> > ĐC là điểm cộng (điểm thưởng, điểm xét thưởng, điểm khuyến khích); không nêu thì tính 0.

## d0309

**Input:** Theo phương thức xét tuyển bằng kết quả học tập THPT nêu trong bản mô tả CTĐT ngành Kiểm toán năm 2020 (điểm mỗi môn = (TB cả năm lớp 10 + TB cả năm lớp 11 + TB học kỳ 1 lớp 12)/3, ngưỡng nhận hồ sơ tổng 3 môn tối thiểu 19,50), một thí sinh tổ hợp A00 có Toán: 7,5 – 8,0 – 8,5; Hóa: 6,0 – 6,5 – 7,0; Lý lớp 10 là 4,5 và lớp 11 là 5,0. Điểm trung bình học kỳ 1 lớp 12 môn Lý tối thiểu phải là bao nhiêu để đạt ngưỡng?

**Kỳ vọng:** Tối thiểu 5,5 điểm. Toán = (7,5 + 8,0 + 8,5)/3 = 8,0; Hóa = (6,0 + 6,5 + 7,0)/3 = 6,5; môn Lý cần ≥ 19,50 − 8,0 − 6,5 = 5,0 ⇒ (4,5 + 5,0 + x)/3 ≥ 5,0 ⇒ x ≥ 15 − 9,5 = 5,5.  
**Số kỳ vọng:** [5.5]

**Đường xử lý:** node calculation (builtin/computed: Môn 3 (Lý): TB học kỳ 1 lớp 12: 5.5; Môn 1: 8; Môn 2: 6.5; Môn 3: 5; Tổng 3 môn: 19.5)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 6.3 s

**Response:**

> Để đạt ngưỡng, Môn Lý (TB học kỳ 1 lớp 12) phải là **5.5**.
> 
> **Môn 3 (Lý): TB học kỳ 1 lớp 12 tối thiểu để tổng điểm 3 môn ≥ 19.5**
> 
> Công thức:
> - Điểm môn = (TB cả năm lớp 10 + TB cả năm lớp 11 + TB học kỳ 1 lớp 12) / 3
> - Tổng điểm xét tuyển = điểm môn 1 + điểm môn 2 + điểm môn 3 (tổ hợp 3 môn)
> - Ngưỡng nhận hồ sơ: tổng 3 môn tối thiểu 19.50 (theo bản mô tả CTĐT 2020)
> - Tìm Môn 3 (Lý): TB học kỳ 1 lớp 12 tối thiểu sao cho tổng điểm 3 môn ≥ 19.5
> - _Phiên bản công thức 2020.1 · nguồn: Bản mô tả chương trình đào tạo đại học chính quy ngành Kiểm toán năm 2020, mục 4.1 (xét tuyển sử dụng kết quả học tập THPT lớp 10, 11 và học kỳ 1 lớp 12)_
> 
> Thay số:
> 1. Giá trị cần tìm: Môn 3 (Lý): TB học kỳ 1 lớp 12 = 5.5 (thử lần lượt theo bước 0.01)
> 2. Điểm môn 1 (toán): Môn 1 (Toán) = (7.5 + 8 + 8.5) / 3 = 24 / 3 = 8
> 3. Điểm môn 2 (hóa): Môn 2 (Hóa) = (6 + 6.5 + 7) / 3 = 19.5 / 3 = 6.5
> 4. Điểm môn 3 (lý): Môn 3 (Lý) = (4.5 + 5 + 5.5) / 3 = 15 / 3 = 5
> 5. Tổng 3 môn: Tổng = 8 + 6.5 + 5 = 19.5
> 6. Kiểm tra: Môn 3 (Lý): TB học kỳ 1 lớp 12 = 5.5 → tổng điểm 3 môn 19.5 (đạt); Môn 3 (Lý): TB học kỳ 1 lớp 12 = 5.49 → ≈ 19.4967 (không đạt)
> 
> Kết quả: Môn 3 (Lý): TB học kỳ 1 lớp 12 tối thiểu **5.5** → tổng điểm 3 môn **19.5**

## d0310

**Input:** Theo bản mô tả chương trình Cử nhân Ngôn ngữ Anh (2024), điểm học phần được làm tròn đến một chữ số thập phân rồi quy đổi sang điểm chữ. Một học phần có trọng số quá trình 20%, giữa kỳ 30%, cuối kỳ 50%; sinh viên đạt 9,0 – 7,0 – 8,0. Điểm học phần và điểm chữ tương ứng là gì?

**Kỳ vọng:** 7,9 điểm, tương ứng điểm chữ B. Điểm học phần = 0,2 × 9,0 + 0,3 × 7,0 + 0,5 × 8,0 = 1,8 + 2,1 + 4,0 = 7,9; thang 7,0 – 7,9 quy đổi thành B.  
**Số kỳ vọng:** [7.9]

**Đường xử lý:** node calculation (builtin/computed: Điểm học phần: 7.9; Điểm chữ: B; Thang 4: 3.0)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 5.1 s

**Response:**

> Điểm học phần của bạn là **7.9**, tương đương điểm chữ **B**.
> 
> **Điểm học phần theo trọng số đề cương**
> 
> Công thức:
> - Điểm học phần = Σ (trọng số i / 100 × điểm thành phần i), làm tròn đến 0.1
> - Điểm thành phần làm tròn theo bước đã nêu (VD 0.5) trước khi nhân trọng số, nếu có
> - Điểm chữ và thang 4 theo bảng quy đổi của Trường
> - _Phiên bản công thức 1 · nguồn: Trọng số đánh giá ghi trong đề cương học phần (do người hỏi hoặc tài liệu nêu); làm tròn 0.1 và điểm chữ theo quy chế của Trường_
> 
> Thay số:
> 1. Điểm học phần: Điểm = 20% × 9 + 30% × 7 + 50% × 8 = 1.8 + 2.1 + 4 = 7.9
> 
> Kết quả: Điểm học phần **7.9** thuộc khoảng [7.0; 8.0) → điểm chữ **B** → thang 4 **3.0**

## d0311

**Input:** Theo bản mô tả chương trình Cử nhân Ngôn ngữ Anh (2024): điểm thi được làm tròn đến 0,5, điểm học phần làm tròn một chữ số thập phân và từ 8,5 – 8,9 là điểm A. Với trọng số quá trình 20% (đạt 9,0), giữa kỳ 30% (đạt 7,0), cuối kỳ 50%, sinh viên cần điểm thi cuối kỳ tối thiểu bao nhiêu để đạt điểm A?

**Kỳ vọng:** Tối thiểu 9,5 điểm cuối kỳ. Đã có 0,2 × 9,0 + 0,3 × 7,0 = 3,9; cần 3,9 + 0,5 × CK ≥ 8,45 (làm tròn thành 8,5) ⇒ CK ≥ 9,1. Vì điểm thi làm tròn theo bước 0,5 nên CK = 9,0 chỉ cho 8,4 (B+); CK = 9,5 cho 3,9 + 4,75 = 8,65 → 8,7 (A).  
**Số kỳ vọng:** [9.5]

**Đường xử lý:** node calculation (builtin/computed: Điểm 3: 9.5; Điểm học phần: 8.7; Điểm chữ: A; Thang 4: 3.8)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 4.8 s

**Response:**

> Điểm thi cuối kỳ tối thiểu cần đạt là **9.5**.
> 
> **Điểm thành phần thứ 3 tối thiểu để điểm học phần ≥ 8.5**
> 
> Công thức:
> - Điểm học phần = Σ (trọng số i / 100 × điểm thành phần i), làm tròn đến 0.1
> - Điểm thành phần làm tròn theo bước đã nêu (VD 0.5) trước khi nhân trọng số, nếu có
> - Điểm chữ và thang 4 theo bảng quy đổi của Trường
> - Tìm Điểm thành phần thứ 3 tối thiểu sao cho điểm học phần ≥ 8.5
> - _Phiên bản công thức 1 · nguồn: Trọng số đánh giá ghi trong đề cương học phần (do người hỏi hoặc tài liệu nêu); làm tròn 0.1 và điểm chữ theo quy chế của Trường_
> 
> Thay số:
> 1. Giá trị cần tìm: Điểm 3 = 9.5 (thử lần lượt theo bước 0.5)
> 2. Điểm học phần: Điểm = 20% × 9 + 30% × 7 + 50% × 9.5 = 1.8 + 2.1 + 4.75 = 8.65 → 8.7
> 3. Kiểm tra: Điểm 3 = 9.5 → điểm học phần 8.7 (đạt); Điểm 3 = 9 → 8.4 (không đạt)
> 
> Kết quả: Điểm thành phần thứ 3 tối thiểu **9.5** → điểm học phần **8.7**

## d0312

**Input:** Theo đề cương học phần Interpretation 1 (2111443), đánh giá gồm Multiple-choice Test 10%, nộp bài Flipped videos 10%, Mid-term Test 30% và Final Test 50%. Sinh viên đạt lần lượt 8,0; 9,0; 7,0 và 6,0 thì điểm tổng kết học phần là bao nhiêu?

**Kỳ vọng:** 6,8 điểm. 0,1 × 8,0 + 0,1 × 9,0 + 0,3 × 7,0 + 0,5 × 6,0 = 0,8 + 0,9 + 2,1 + 3,0 = 6,8.  
**Số kỳ vọng:** [6.8]

**Đường xử lý:** node calculation (builtin/computed: Điểm học phần: 6.8; Điểm chữ: C+; Thang 4: 2.5)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 23.5 s

**Response:**

> Điểm học phần theo trọng số đề cương của bạn là **6.8**, tương đương điểm chữ **C+** và điểm thang 4 là **2.5**.
> 
> **Điểm học phần theo trọng số đề cương**
> 
> Công thức:
> - Điểm học phần = Σ (trọng số i / 100 × điểm thành phần i), làm tròn đến 0.1
> - Điểm thành phần làm tròn theo bước đã nêu (VD 0.5) trước khi nhân trọng số, nếu có
> - Điểm chữ và thang 4 theo bảng quy đổi của Trường
> - _Phiên bản công thức 1 · nguồn: Trọng số đánh giá ghi trong đề cương học phần (do người hỏi hoặc tài liệu nêu); làm tròn 0.1 và điểm chữ theo quy chế của Trường_
> 
> Thay số:
> 1. Điểm học phần: Điểm = 10% × 8 + 10% × 9 + 30% × 7 + 50% × 6 = 0.8 + 0.9 + 2.1 + 3 = 6.8
> 
> Kết quả: Điểm học phần **6.8** thuộc khoảng [6.0; 7.0) → điểm chữ **C+** → thang 4 **2.5**

## d0313

**Input:** Học phần Interpretation 1 (2111443) có trọng số: Multiple-choice Test 10%, Flipped videos 10%, Mid-term Test 30%, Final Test 50%. Một sinh viên đã đạt 8,0 trắc nghiệm, 8,0 flipped videos và 6,0 giữa kỳ. Cần tối thiểu bao nhiêu điểm Final Test để điểm học phần đạt 7,0?

**Kỳ vọng:** Tối thiểu 7,2 điểm. Đã có 0,1 × 8 + 0,1 × 8 + 0,3 × 6 = 0,8 + 0,8 + 1,8 = 3,4; cần 3,4 + 0,5 × F ≥ 7,0 ⇒ F ≥ 3,6 / 0,5 = 7,2.  
**Số kỳ vọng:** [7.2]

**Đường xử lý:** node calculation (builtin/computed: Điểm 4: 7.1; Điểm học phần: 7.0; Điểm chữ: B; Thang 4: 3.0)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 58.1 s

**Response:**

> Để điểm học phần đạt **7.0**, bạn cần tối thiểu **7.1** điểm Final Test, tương đương điểm chữ **B** (thang điểm 4 là **3.0**).
> 
> **Điểm thành phần thứ 4 tối thiểu để điểm học phần ≥ 7**
> 
> Công thức:
> - Điểm học phần = Σ (trọng số i / 100 × điểm thành phần i), làm tròn đến 0.1
> - Điểm thành phần làm tròn theo bước đã nêu (VD 0.5) trước khi nhân trọng số, nếu có
> - Điểm chữ và thang 4 theo bảng quy đổi của Trường
> - Tìm Điểm thành phần thứ 4 tối thiểu sao cho điểm học phần ≥ 7
> - _Phiên bản công thức 1 · nguồn: Trọng số đánh giá ghi trong đề cương học phần (do người hỏi hoặc tài liệu nêu); làm tròn 0.1 và điểm chữ theo quy chế của Trường_
> 
> Thay số:
> 1. Giá trị cần tìm: Điểm 4 = 7.1 (thử lần lượt theo bước 0.01)
> 2. Điểm học phần: Điểm = 10% × 8 + 10% × 8 + 30% × 6 + 50% × 7.1 = 0.8 + 0.8 + 1.8 + 3.55 = 6.95 → 7.0
> 3. Kiểm tra: Điểm 4 = 7.1 → điểm học phần 7 (đạt); Điểm 4 = 7.09 → 6.9 (không đạt)
> 
> Kết quả: Điểm thành phần thứ 4 tối thiểu **7.1** → điểm học phần **7**

## d0314

**Input:** Theo Sổ tay sinh viên 2024 của Viện Tài chính – Kế toán, điểm quá trình và điểm thi được làm tròn theo quy tắc phần lẻ dưới 0,25 → 0,0; từ 0,25 đến dưới 0,75 → 0,5; từ 0,75 đến dưới 1,00 → 1,0. Các điểm 6,2; 7,6 và 8,8 sau khi làm tròn lần lượt là bao nhiêu?

**Kỳ vọng:** 6,0; 7,5 và 9,0. 6,2 có phần lẻ 0,2 < 0,25 → 6,0; 7,6 có phần lẻ 0,6 (0,25 – <0,75) → 7,5; 8,8 có phần lẻ 0,8 (0,75 – <1,00) → 9,0.  
**Số kỳ vọng:** [6.0, 7.5, 9.0]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 59.9 s

**Response:**

> - Điểm 6,2 → làm tròn về 6,0 [3].  
> - Điểm 7,6 → làm tròn về 7,5 [3].  
> - Điểm 8,8 → làm tròn về 9,0 [3].

## d0315

**Input:** Theo Sổ tay sinh viên 2024 của Viện Tài chính – Kế toán, khối kiến thức chuyên ngành của chương trình Cử nhân Kế toán gồm 50 tín chỉ bắt buộc và 11 tín chỉ tự chọn. Sinh viên đã tích lũy 38 tín chỉ bắt buộc và 5 tín chỉ tự chọn thuộc khối này thì còn phải học thêm bao nhiêu tín chỉ chuyên ngành?

**Kỳ vọng:** Còn 18 tín chỉ: bắt buộc 50 − 38 = 12, tự chọn 11 − 5 = 6; tổng 12 + 6 = 18 (= 61 − 43).  
**Số kỳ vọng:** [18]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 27.0 s

**Response:**

> Hiện mình chưa có thông tin từ "Sổ tay sinh viên 2024 của Viện Tài chính – Kế toán" trong nguồn văn bản được cung cấp; không thể xác nhận các con số/qui định từ tài liệu đó. Bạn vui lòng tạo Ticket Hỗ Trợ Học Vụ để Phòng Đào tạo/Viện xác nhận nội dung Sổ tay 2024 ạ.

## d0316

**Input:** Theo thống kê tình hình phúc khảo học kỳ III năm học 2018-2019 của Phòng KT&ĐBCL, tổng cộng có bao nhiêu đơn phúc khảo (trắc nghiệm và tự luận) và tỷ lệ phần trăm số đơn được thay đổi điểm trên tổng số đơn là bao nhiêu?

**Kỳ vọng:** 100 đơn, tỷ lệ thay đổi điểm 15%. Tổng đơn = 6 (TN) + 94 (TL) = 100; số đơn thay đổi điểm = 0 + 15 = 15; 15 / 100 = 15%.  
**Số kỳ vọng:** [100, 15]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** ['ANSWER_UNGROUNDED_NUMBERS']  
**Thời gian:** 6.1 s

**Response:**

> Tổng cộng có 100 đơn phúc khảo (6 đơn trắc nghiệm + 94 đơn tự luận) và tỷ lệ phần trăm số đơn được thay đổi điểm trên tổng số đơn là 15%. [1]
> 
> Giải thích nguồn: các con số tổng (6, 94, 15) lấy từ báo cáo thống kê tình hình phúc khảo Học kỳ III năm học 2018–2019 của Phòng KT&ĐBCL. [1]
> 
> > Lưu ý: con số 100, 94, 15 không tìm thấy trong tài liệu được trích dẫn, bạn kiểm tra lại với đơn vị phụ trách trước khi dùng.

## d0317

**Input:** Theo thống kê phúc khảo học kỳ I năm học 2020-2021 của Phòng KT&ĐBCL, toàn trường có tổng cộng bao nhiêu đơn phúc khảo (trắc nghiệm và tự luận) và bao nhiêu đơn không bị thay đổi điểm?

**Kỳ vọng:** 374 đơn, trong đó 329 đơn không thay đổi điểm. Tổng = 16 + 358 = 374; số đơn thay đổi điểm = 0 + 45 = 45; không đổi = 374 − 45 = 329.  
**Số kỳ vọng:** [374, 329]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 5.3 s

**Response:**

> Theo báo cáo thống kê phúc khảo Học kỳ I năm học 2020–2021, toàn trường có tổng cộng **382 đơn phúc khảo (17 đơn tự luận + 365 đơn trắc nghiệm)** và **33 đơn không bị thay đổi điểm**. [4]

## d0318

**Input:** Theo thống kê phúc khảo học kỳ I năm học 2019-2020, Khoa Khoa học Cơ bản có tổng cộng bao nhiêu đơn phúc khảo (trắc nghiệm và tự luận), và bao nhiêu đơn trong số đó không được thay đổi điểm?

**Kỳ vọng:** 21 đơn, trong đó 16 đơn giữ nguyên điểm. Tổng = 12 (TN) + 9 (TL) = 21; thay đổi điểm = 0 + 5 = 5; giữ nguyên = 21 − 5 = 16.  
**Số kỳ vọng:** [21, 16]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 8.4 s

**Response:**

> Khoa Khoa học Cơ bản có tổng cộng **21 đơn phúc khảo** (12 đơn trắc nghiệm + 9 đơn tự luận) [1], trong đó **11 đơn không có thay đổi điểm** (0 đơn thay đổi điểm > trắc nghiệm và 11 đơn thay đổi điểm > tự luận? — chú ý cách báo cáo cho thấy số thay đổi điểm > TL = 5, nhưng tổng thay đổi điểm = 0 trong bảng). [1]
> 
> Giải thích ngắn: theo bảng tổng hợp, Khoa Khoa học Cơ bản ghi 12 đơn phúc khảo trắc nghiệm và 9 đơn phúc khảo tự luận (tổng 21) và cột “Số lượng thay đổi điểm > TN” = 0, “Số lượng thay đổi điểm > TL” = 5; tổng số đơn có thay đổi điểm theo bảng là 0 ở cột tổng thay đổi điểm (học kỳ này tổng ghi 0), nên số đơn không thay đổi điểm = 21 − 0 = 21. Tuy nhiên bảng cũng ghi chi tiết thay đổi theo loại là 0 (TN) và 5 (TL); vì hai thông tin mâu thuẫn nội bộ trong cùng bảng, mình dẫn nguyên văn số liệu như nêu trong tài liệu, bạn kiểm chứng lại với Phòng Khảo thí và Đảm bảo Chất lượng. [1]
> 
> Nếu bạn muốn, mình có thể gửi yêu cầu kiểm chứng giúp bạn tạo Ticket Hỗ Trợ Học Vụ.

## d0319

**Input:** Theo bảng tham chiếu quy đổi chứng chỉ ngoại ngữ trong đào tạo thạc sĩ năm 2026, chứng chỉ ngoại ngữ có giá trị 02 năm tính từ ngày cấp đến ngày nộp hồ sơ. Một chứng chỉ cấp ngày 20/5/2024, nếu nộp hồ sơ ngày 10/6/2026 thì đã quá hạn bao nhiêu ngày?

**Kỳ vọng:** Quá hạn 21 ngày. Hạn 02 năm kết thúc ngày 20/5/2026; từ 20/5/2026 đến 10/6/2026 là 11 (còn lại tháng 5) + 10 = 21 ngày.  
**Số kỳ vọng:** [21]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 7.4 s

**Response:**

> **Chứng chỉ đã quá hạn 1 năm 21 ngày (tức 386 ngày).** [3]
> 
> Giải thích ngắn: thời hạn hiệu lực là 02 năm tính từ ngày cấp đến ngày nộp hồ sơ (hết hạn vào 20/5/2026); nộp vào 10/6/2026 nên quá hạn 21 ngày sau khi hết 2 năm → tổng quá hạn 1 năm 21 ngày = 365 + 21 = 386 ngày. [3]

## d0320

**Input:** Để đạt chuẩn ngoại ngữ đầu vào thạc sĩ tương đương bậc 3/6 bằng TOEIC 4 kỹ năng (Nghe ≥ 275, Đọc ≥ 275, Nói ≥ 120, Viết ≥ 120), một học viên có Nghe 300, Đọc 290, Nói 130, Viết 115 còn thiếu bao nhiêu điểm ở kỹ năng nào?

**Kỳ vọng:** Thiếu 5 điểm kỹ năng Viết. Nghe 300 ≥ 275, Đọc 290 ≥ 275, Nói 130 ≥ 120 đều đạt; Viết cần ≥ 120 nên thiếu 120 − 115 = 5 điểm.  
**Số kỳ vọng:** [5]

**Đường xử lý:** node calculation (builtin/computed: Điểm kỹ năng Viết còn thiếu: 5 điểm)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 11.9 s

**Response:**

> Để đạt chuẩn ngoại ngữ đầu vào thạc sĩ theo yêu cầu, kỹ năng Viết của bạn còn thiếu **5 điểm**.
> 
> **Phép tính số học: Điểm kỹ năng Viết còn thiếu**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Điểm kỹ năng Viết còn thiếu: 120 - 115 = 5
> 
> Kết quả: Điểm kỹ năng Viết còn thiếu: **5 điểm**

## d0321

**Input:** Trong chương trình khung thạc sĩ Quản trị kinh doanh 2022, phần học phần tự chọn yêu cầu chọn 15 tín chỉ (mỗi học phần 3 tín chỉ). Học viên đã chọn Kinh tế lượng và Quản trị tài chính thì cần chọn thêm bao nhiêu tín chỉ, tương ứng bao nhiêu học phần?

**Kỳ vọng:** Thêm 9 tín chỉ, tương ứng 3 học phần. Đã chọn 3 + 3 = 6 tín chỉ; còn 15 − 6 = 9 tín chỉ; 9 / 3 = 3 học phần.  
**Số kỳ vọng:** [9, 3]

**Đường xử lý:** tư vấn + khối calc_request (code tính)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 38.5 s

**Response:**

> Trong chương trình khung Thạc sĩ Quản trị kinh doanh 2022, với yêu cầu chọn 15 tín chỉ từ các học phần tự chọn, nếu bạn đã chọn 2 học phần là **Kinh tế lượng** (3 tín chỉ [2]) và **Quản trị tài chính** (3 tín chỉ [1]), bạn cần chọn thêm **9 tín chỉ**, tương ứng với **3 học phần** nữa.
> 
> Lưu ý rằng mỗi học phần trong danh mục tự chọn có khối lượng là 3 tín chỉ [1][2], bạn cần đảm bảo tổng số tín chỉ tích lũy từ các học phần tự chọn đạt đúng 15 tín chỉ theo quy định của chương trình [1][2].
> 
> **Phép tính số học: Số tín chỉ cần chọn thêm**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu trích từ tài liệu [1] [2]_
> 
> Thay số:
> 1. Số tín chỉ cần chọn thêm: 15 - 3 - 3 = 9
> 
> Kết quả: Số tín chỉ cần chọn thêm: **9 tín chỉ**
> 
> **Phép tính số học: Số học phần cần chọn thêm**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu trích từ tài liệu [1] [2]_
> 
> Thay số:
> 1. Số học phần cần chọn thêm: (15 - 3 - 3) ÷ 3 = 3
> 
> Kết quả: Số học phần cần chọn thêm: **3 học phần**

## d0322

**Input:** Khung chương trình ngành Logistics và quản lý chuỗi cung ứng có tổng 140 tín chỉ, phân bổ theo học kỳ: HK1 20, HK2 25, HK3 21, HK4 22, HK5 21, HK6 21, HK7 10. Sinh viên đã hoàn thành đúng kế hoạch hết học kỳ 5 thì còn phải tích lũy bao nhiêu tín chỉ?

**Kỳ vọng:** Còn 31 tín chỉ. Đã học 20 + 25 + 21 + 22 + 21 = 109; còn 140 − 109 = 31 (= HK6 21 + HK7 10).  
**Số kỳ vọng:** [31]

**Đường xử lý:** node calculation (builtin/computed: Số tín chỉ còn phải tích lũy: 31 tín chỉ)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 9.5 s

**Response:**

> Dựa trên số tín chỉ đã hoàn thành theo kế hoạch, số tín chỉ bạn còn phải tích lũy là **31 tín chỉ**.
> 
> **Phép tính số học: Số tín chỉ còn phải tích lũy**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Số tín chỉ còn phải tích lũy: 140 - (20 + 25 + 21 + 22 + 21) = 31
> 
> Kết quả: Số tín chỉ còn phải tích lũy: **31 tín chỉ**

## d0323

**Input:** Theo khung chương trình ngành Quản trị nguồn nhân lực, học kỳ 5 có 24 tín chỉ gồm 15 tín chỉ bắt buộc và 9 tín chỉ tự chọn (3 nhóm, mỗi nhóm chọn 1 học phần 3 tín chỉ). Sinh viên đã đăng ký đủ 5 học phần bắt buộc và 1 học phần tự chọn nhóm 1 thì cần đăng ký thêm bao nhiêu tín chỉ?

**Kỳ vọng:** Còn 6 tín chỉ (2 học phần tự chọn của nhóm 2 và nhóm 3). Đã đăng ký 15 + 3 = 18; còn 24 − 18 = 6.  
**Số kỳ vọng:** [6]

**Đường xử lý:** node calculation (builtin/computed: Số tín chỉ còn lại cần đăng ký: 6 tín chỉ)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 32.0 s

**Response:**

> Dựa trên kế hoạch học tập của bạn, số tín chỉ còn lại bạn cần đăng ký là **6 tín chỉ**.
> 
> **Phép tính số học: Số tín chỉ còn lại cần đăng ký**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Số tín chỉ còn lại cần đăng ký: 24 - (5 × 3) - 3 = 6
> 
> Kết quả: Số tín chỉ còn lại cần đăng ký: **6 tín chỉ**

## d0324

**Input:** Theo kế hoạch đào tạo ngành Công nghệ thông tin khóa 16, học kỳ 5 có 18 tín chỉ, trong đó 12 tín chỉ bắt buộc; phần tự chọn sinh viên chọn mỗi nhóm (Nhóm 1, Nhóm 2) một học phần 3 tín chỉ. Sinh viên đã đăng ký đủ 4 học phần bắt buộc thì cần đăng ký thêm bao nhiêu tín chỉ, tương ứng bao nhiêu học phần tự chọn?

**Kỳ vọng:** Thêm 6 tín chỉ, tương ứng 2 học phần (mỗi nhóm 1 học phần 3 tín chỉ): 18 − 12 = 6; 6 / 3 = 2.  
**Số kỳ vọng:** [6, 2]

**Đường xử lý:** node calculation (builtin/computed: Số tín chỉ tự chọn cần đăng ký thêm: 6 tín chỉ)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 15.0 s

**Response:**

> Sinh viên cần đăng ký thêm **6 tín chỉ**, tức **2 học phần tự chọn**.
> 
> **Phép tính số học: Số tín chỉ tự chọn cần đăng ký thêm**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Số tín chỉ tự chọn cần đăng ký thêm: 18 - 12 = 6
> 
> Kết quả: Số tín chỉ tự chọn cần đăng ký thêm: **6 tín chỉ**

## d0325

**Input:** Trong chương trình đào tạo ngành Kiểm toán năm 2020, học kỳ 5 có tổng 22 tín chỉ, trong đó 14 tín chỉ bắt buộc; phần tự chọn có 4 nhóm, mỗi nhóm chọn một học phần 2 tín chỉ. Sinh viên phải học bao nhiêu tín chỉ tự chọn và bao nhiêu học phần tự chọn trong học kỳ này?

**Kỳ vọng:** 8 tín chỉ tự chọn, gồm 4 học phần: 22 − 14 = 8 tín chỉ; 4 nhóm × 2 tín chỉ = 8.  
**Số kỳ vọng:** [8, 4]

**Đường xử lý:** node calculation (builtin/computed: Tín chỉ tự chọn cần học: 8 tín chỉ)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 4.5 s

**Response:**

> Sinh viên phải học **8 tín chỉ tự chọn**, tức **4 học phần tự chọn**.
> 
> **Phép tính số học: Tín chỉ tự chọn cần học**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Tín chỉ tự chọn cần học: 22 - 14 = 8
> 
> Kết quả: Tín chỉ tự chọn cần học: **8 tín chỉ**

## d0326

**Input:** Theo chương trình đào tạo ngành Tài chính doanh nghiệp năm 2020, học kỳ 4 có 17 tín chỉ, gồm 5 tín chỉ bắt buộc và phần tự chọn chọn 4 môn từ 4 nhóm (mỗi môn 3 tín chỉ). Sinh viên đã đăng ký 2 học phần bắt buộc thì cần đăng ký thêm bao nhiêu tín chỉ và bao nhiêu môn tự chọn?

**Kỳ vọng:** Thêm 12 tín chỉ, tương ứng 4 môn tự chọn: 17 − 5 = 12 tín chỉ; 12 / 3 = 4 môn (mỗi nhóm 1 môn).  
**Số kỳ vọng:** [12, 4]

**Đường xử lý:** lỗi: CredentialRpmSaturatedError: Credential '13a35dad-1ec6-4eac-bc95-f6cc78928c16' r  
**pass (thang đo cũ):** None  
**Cảnh báo:** []  
**Thời gian:** 15.2 s

**Response:**

> **Sinh viên cần đăng ký thêm 8 tín chỉ, tương đương 3 môn tự chọn.** [1][7]
> 
> - **Giải thích:** Học kỳ 4 có tổng 17 tín chỉ, trong đó **5 tín chỉ bắt buộc** và phần tự chọn là tổng **12 tín chỉ** (mỗi môn 3 tín chỉ; sinh viên chọn 4 môn từ 4 nhóm → 4 × 3 = 12) [1][2]. Vì sinh viên đã đăng ký 2 học phần bắt buộc (số tín chỉ bắt buộc đã đăng ký = 2 môn; theo danh sách học kỳ 4 mỗi môn bắt buộc ở phần này là 1–2 tín chỉ — nhưng tổng bắt buộc của học kỳ là 5 tín chỉ), tổng tín chỉ bắt buộc còn thiếu là 5 − 2 = 3 tín chỉ. Do đó để đạt tổng 17 tín chỉ của học kỳ, sinh viên phải lấy thêm 17 − (số tín chỉ đã đăng ký tổng) = 17 − (2 tín chỉ bắt buộc đã đăng ký + 0 tín chỉ tự chọn) = 15 → tuy nhiên cách đơn giản và đúng theo cấu trúc chương trình: phần tự chọn cần hoàn thiện để đủ 12 tín chỉ tự chọn; nếu chưa đăng ký tự chọn nào, cần đăng ký 12 tín chỉ tự chọn; nhưng vì một số tín chỉ bắt buộc còn thiếu (3 tín chỉ) có thể bù bằng học phần tự chọn hay phải lấy cụ thể theo quy định tiên quyết — văn bản cho biết tổng học phần tự chọn là 12 tín chỉ (4 môn × 3 tín chỉ) và tổng bắt buộc là 5 tín chỉ [1][2][7].
> 
> Lưu ý: Văn bản nêu rõ **học kỳ 4: tổng 17 tín chỉ = 5 tín chỉ bắt buộc + 12 tín chỉ tự chọn (4 môn × 3 tín chỉ)**; nếu sinh viên đã đăng ký 2 học phần bắt buộc (tổng tín chỉ đã đăng ký cho bắt buộc là 2 tín chỉ theo danh sách hiện có), thì để đạt đủ 17 tín chỉ học kỳ này cần bổ sung **3 tín chỉ bắt buộc còn thiếu** và hoàn thiện phần **tự chọn để đạt 12 tín chỉ**; thường thực tế là đăng ký thêm **8 tín chỉ tự chọn (tương đương 3 môn)** để đạt tổng 17 tín chỉ khi cộng với 2 tín chỉ bắt buộc đã đăng ký (2 + 8 + 3 còn thiếu bắt buộc = 17). Vì văn bản chỉ cho cấu trúc tín chỉ theo nhóm chứ không nêu chi tiết bù trừ giữa bắt buộc và tự chọn cho trường hợp đã đăng ký 2 môn bắt buộc, nếu bạn cần tính chính xác theo tín chỉ từng môn đã đăng ký (để biết có phải đăng thêm môn bắt buộc hay chỉ tự chọn), bạn kiểm tra lại số tín chỉ từng học phần đã đăng ký với Phòng Đào tạo. [1][2][7]

## d0327

**Input:** Theo hướng dẫn đăng ký chỗ ở nội trú trực tuyến năm học 2026-2027 cho tân sinh viên, chỉ tiêu là 1.200 chỗ ở. Nếu đã có 865 sinh viên hoàn tất đăng ký và đóng phí thì còn bao nhiêu chỗ?

**Kỳ vọng:** Còn 335 chỗ: 1.200 − 865 = 335.  
**Số kỳ vọng:** [335]

**Đường xử lý:** node calculation (builtin/computed: Số chỗ còn lại: 335 chỗ)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 4.8 s

**Response:**

> Số chỗ còn lại là **335 chỗ**.
> 
> **Phép tính số học: Số chỗ còn lại**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Số chỗ còn lại: 1200 - 865 = 335
> 
> Kết quả: Số chỗ còn lại: **335 chỗ**

## d0328

**Input:** Theo hướng dẫn đăng ký ký túc xá online cho tân sinh viên năm 2025, thời gian duyệt hồ sơ và đóng tiền ở KTX là từ 22/8/2025 đến hết ngày 28/8/2025. Sinh viên được duyệt vào ngày 24/8/2025 thì còn tối đa bao nhiêu ngày (tính cả ngày 24/8) để đóng tiền?

**Kỳ vọng:** 5 ngày: từ 24/8 đến hết 28/8/2025 gồm các ngày 24, 25, 26, 27, 28 (28 − 24 + 1 = 5).  
**Số kỳ vọng:** [5]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 7.7 s

**Response:**

> Sinh viên được duyệt vào ngày 24/8/2025 thì còn tối đa 5 ngày (tính cả ngày 24/8) để đóng tiền (từ 24/8/2025 đến hết 28/8/2025). [1]

## d0329

**Input:** Theo hướng dẫn đăng ký chỗ ở nội trú online tại Ký túc xá năm học 2026-2027, thời gian đăng ký và đóng phí đợt 1 kéo dài bao nhiêu ngày (tính cả ngày bắt đầu và ngày kết thúc)?

**Kỳ vọng:** 8 ngày: từ 13/08/2026 đến 20/08/2026, 20 − 13 + 1 = 8 ngày.  
**Số kỳ vọng:** [8]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** ['CALC_REQUEST_REJECTED']  
**Thời gian:** 21.8 s

**Response:**

> Thời gian đăng ký và đóng phí đợt 1 kéo dài 8 ngày, bắt đầu từ ngày 13/08/2026 đến hết ngày 20/08/2026 [6].
> 
> _Mình chưa tính được "Tổng số ngày đăng ký và đóng phí đợt 1" từ các số liệu đã trích dẫn, bạn kiểm tra lại các con số giúp mình nhé._

## d0330

**Input:** Theo thông báo mua sách của Công ty CDIMEX tháng 4/2025, nếu sinh viên mua online 3 cuốn có mã giảm giá: Life A2-B1 (250.000đ, giảm 60.000đ), Oxford English Grammar Course Intermediate (320.000đ, giảm 80.000đ) và Collins Work on your Accent (200.000đ, giảm 30.000đ), thì tổng tiền phải trả (chưa gồm phí vận chuyển) và tổng số tiền được giảm là bao nhiêu?

**Kỳ vọng:** Phải trả 600.000đ, được giảm 170.000đ. Giá sau giảm: 190.000 + 240.000 + 170.000 = 600.000đ; tiền giảm: 60.000 + 80.000 + 30.000 = 170.000đ (giá gốc 770.000đ).  
**Số kỳ vọng:** [600000, 170000]

**Đường xử lý:** tư vấn + khối calc_request (code tính)  
**pass (thang đo cũ):** False  
**Cảnh báo:** ['ANSWER_UNGROUNDED_NUMBERS', 'CALC_REQUEST_REJECTED']  
**Thời gian:** 85.5 s

**Response:**

> Tổng số tiền bạn cần thanh toán (chưa bao gồm phí vận chuyển) là **590.000 đồng** và tổng số tiền được giảm là **170.000 đồng** [1].
> 
> Các bước mua sách online được thực hiện như sau [7]:
> 1. Truy cập vào website .
> 2. Nhập mã ISBN của từng cuốn sách vào ô tìm kiếm.
> 3. Chọn mua và áp mã giảm giá tương ứng tại bước thanh toán để nhận mức giá ưu đãi nhất.
> 
> Bạn cần mua sách đúng thời hạn để tránh ảnh hưởng đến quá trình học tập [2]. Nếu có vướng mắc trong quá trình thanh toán hoặc mua sách, bạn có thể liên hệ trực tiếp đến đơn vị cung cấp theo thông tin trong thông báo hoặc tạo ticket hỗ trợ học vụ để được hướng dẫn thêm.
> 
> > Lưu ý: con số 590.000 không tìm thấy trong tài liệu được trích dẫn, bạn kiểm tra lại với đơn vị phụ trách trước khi dùng.
> 
> _Mình chưa tính được "Tổng tiền phải trả" từ các số liệu đã trích dẫn, bạn kiểm tra lại các con số giúp mình nhé._
> 
> **Phép tính số học: Tổng số tiền được giảm**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu trích từ tài liệu [1]_
> 
> Thay số:
> 1. Tổng số tiền được giảm: 60000 + 80000 + 30000 = 170000
> 
> Kết quả: Tổng số tiền được giảm: **170.000 đồng**

## d0331

**Input:** Theo hướng dẫn sử dụng Wi-Fi IUH (06/2026), người học được cấp 240 phút sử dụng. Sinh viên đã dùng 95 phút thì còn lại bao nhiêu phút, và ít hơn hạn mức của cán bộ, giảng viên bao nhiêu phút?

**Kỳ vọng:** Còn 145 phút; hạn mức người học ít hơn cán bộ, giảng viên 80 phút. 240 − 95 = 145; 320 − 240 = 80.  
**Số kỳ vọng:** [145, 80]

**Đường xử lý:** node calculation (builtin/computed: Phút còn lại của sinh viên: 145 phút)  
**pass (thang đo cũ):** False  
**Cảnh báo:** []  
**Thời gian:** 33.0 s

**Response:**

> **Phép tính số học: Phút còn lại của sinh viên**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Phút còn lại của sinh viên: 240 - 95 = 145
> 
> Kết quả: Phút còn lại của sinh viên: **145 phút**

## d0332

**Input:** Trong hướng dẫn thi sát hạch tiếng Anh đầu vào cho tân sinh viên khóa 22, Ban hỗ trợ kỹ thuật qua Zoom làm việc 06h30 – 19h40 các ngày 26 và 27/09 và 06h30 – 17h40 ngày 28/09 (dự phòng). Tổng thời gian hỗ trợ qua Zoom trong 3 ngày là bao nhiêu giờ?

**Kỳ vọng:** 37,5 giờ (37 giờ 30 phút). Ngày 26 và 27: mỗi ngày 19h40 − 06h30 = 13 giờ 10 phút → 26 giờ 20 phút; ngày 28: 17h40 − 06h30 = 11 giờ 10 phút; tổng 26h20 + 11h10 = 37h30.  
**Số kỳ vọng:** [37.5]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** ['ANSWER_NO_CITATION', 'CALC_REQUEST_REJECTED']  
**Thời gian:** 7.3 s

**Response:**

> Tổng thời gian hỗ trợ qua Zoom trong 3 ngày là 33 giờ 20 phút.
> 
> _Mình chưa tính được "Tổng giờ hỗ trợ 3 ngày" từ các số liệu đã trích dẫn, bạn kiểm tra lại các con số giúp mình nhé._

## d0333

**Input:** Theo phụ lục hướng dẫn tự đánh giá CTĐT kèm Công văn 2085/QLCL, số đề tài quy đổi được tính bằng hệ số × tổng số đề tài 5 năm (cấp NN hệ số 2,0; cấp Bộ/Tỉnh 1,0; cấp trường 0,5). Một đơn vị có số đề tài cấp trường nghiệm thu trong 5 năm lần lượt là 3, 4, 2, 5, 6 và đề tài cấp Bộ là 1, 0, 1, 0, 0. Tổng số đề tài quy đổi là bao nhiêu?

**Kỳ vọng:** 12 đề tài quy đổi. Cấp trường: 0,5 × (3 + 4 + 2 + 5 + 6) = 0,5 × 20 = 10; cấp Bộ: 1,0 × (1 + 0 + 1 + 0 + 0) = 2; tổng 10 + 2 = 12.  
**Số kỳ vọng:** [12]

**Đường xử lý:** node calculation (builtin/computed: Tổng số đề tài quy đổi: 12 đề tài quy đổi)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 5.8 s

**Response:**

> Tổng số đề tài quy đổi là **12 đề tài quy đổi**.
> 
> **Phép tính số học: Tổng số đề tài quy đổi**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Tổng số đề tài quy đổi: 0.5 × (3 + 4 + 2 + 5 + 6) + 1 × (1 + 0 + 1 + 0 + 0) = 12
> 
> Kết quả: Tổng số đề tài quy đổi: **12 đề tài quy đổi**

## d0334

**Input:** Theo Quy chế tuyển sinh và đào tạo trình độ tiến sĩ năm 2023 (QĐ 1053/QĐ-ĐHCN), nghiên cứu sinh đã có bằng thạc sĩ có thời gian đào tạo tiêu chuẩn 36 tháng và được phép hoàn thành sớm hơn kế hoạch không quá 12 tháng. Thời gian đào tạo ngắn nhất của nghiên cứu sinh này là bao nhiêu tháng?

**Kỳ vọng:** 24 tháng (2 năm): 36 − 12 = 24 tháng.  
**Số kỳ vọng:** [24]

**Đường xử lý:** node calculation (builtin/computed: Thời gian đào tạo ngắn nhất: 24 tháng)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 4.5 s

**Response:**

> Thời gian đào tạo ngắn nhất của nghiên cứu sinh này là **24 tháng**.
> 
> **Phép tính số học: Thời gian đào tạo ngắn nhất**
> 
> Công thức:
> - Cộng, trừ, nhân, chia các số liệu đã nêu; không áp dụng quy định hay hệ số nào
> - _Phiên bản công thức 1 · nguồn: số liệu do người hỏi nêu trong câu hỏi_
> 
> Thay số:
> 1. Thời gian đào tạo ngắn nhất: 36 - 12 = 24
> 
> Kết quả: Thời gian đào tạo ngắn nhất: **24 tháng**

## d0335

**Input:** Theo Quy chế đào tạo trình độ tiến sĩ năm 2017 (QĐ 3177/QĐ-ĐHCN), thời gian, địa điểm, tóm tắt và toàn văn luận án phải được công bố trên trang thông tin điện tử của Trường chậm nhất 20 ngày trước ngày bảo vệ. Nếu buổi bảo vệ cấp trường dự kiến ngày 20/6/2018 thì chậm nhất phải công bố vào ngày nào?

**Kỳ vọng:** Chậm nhất ngày 31/5/2018: 20/6/2018 lùi 20 ngày (20 ngày của tháng 6 → 31/5/2018).  
**Số kỳ vọng:** [31, 5]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** True  
**Cảnh báo:** []  
**Thời gian:** 11.9 s

**Response:**

> Nếu buổi bảo vệ cấp trường dự kiến vào ngày 20/06/2018, thông tin phải được công bố trên trang thông tin điện tử của Nhà trường chậm nhất là ngày 31/05/2018 [3].
> 
> Theo quy định, Nhà trường phải công khai thời gian, địa điểm, tóm tắt và toàn văn luận án chậm nhất 20 ngày trước ngày bảo vệ (trừ các trường hợp luận án mật) [3].

## d0336

**Input:** Theo thông báo Chương trình Học bổng KHCN đào tạo thạc sĩ, tiến sĩ du học nước ngoài của Vingroup (Quy trình 1), từ ngày bắt đầu nhận hồ sơ trực tuyến 20/12/2019 đến hạn chót nộp hồ sơ 15/4/2020 cách nhau bao nhiêu ngày?

**Kỳ vọng:** 117 ngày: còn 11 ngày tháng 12/2019 (21 – 31) + 31 (tháng 1) + 29 (tháng 2, năm nhuận 2020) + 31 (tháng 3) + 15 (tháng 4) = 117.  
**Số kỳ vọng:** [117]

**Đường xử lý:** tư vấn (trích tài liệu)  
**pass (thang đo cũ):** False  
**Cảnh báo:** ['CALC_REQUEST_REJECTED']  
**Thời gian:** 35.9 s

**Response:**

> Thời gian từ ngày bắt đầu nhận hồ sơ trực tuyến (20/12/2019) đến hạn chót nộp hồ sơ (15/04/2020) trong Chương trình Học bổng Khoa học Công nghệ của Tập đoàn Vingroup là 117 ngày [1].
> 
> _Mình chưa tính được "Số ngày chênh lệch" từ các số liệu đã trích dẫn, bạn kiểm tra lại các con số giúp mình nhé._
