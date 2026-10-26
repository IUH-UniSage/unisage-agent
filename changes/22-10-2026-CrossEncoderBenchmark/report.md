# Báo cáo benchmark: cross-encoder so với LLM rerank (UNISAGE-106)

Ngày đo: 26/10/2026 · Máy: Linux dev, RTX 4080 SUPER 16 GB, driver 595.91.07 · Nhánh:
`feature/huydh-unisage-106-benchmark-cross-encoder` · Spec: `docs/specs/SPEC-cross-encoder-rerank.md`
· Cách chạy lại: `docs/guide/benchmark-rerank.md`.

Câu hỏi cần trả lời: thay bước 09a (LLM rerank, 3–6 s mỗi lượt) bằng cross-encoder
`BAAI/bge-reranker-v2-m3` chạy local, hoặc đặt cross-encoder trước LLM rerank, có **giảm độ trễ
mà giữ (hoặc tăng) độ chính xác** không.

## 0. Tóm tắt

_(điền sau khi đủ bước C–E)_

## 1. Cách đo

### 1.1. Bộ ứng viên chung

- Nguồn: `runs/hybrid-on` (eval 21/10/2026, bộ demo, category normal + calculation, hybrid
  BM25 + dense bật, cache tắt). Mỗi câu lấy câu hỏi độc lập của sub-query đầu và đúng 16
  `chunk_id` retrieval đã trả về theo thứ tự RRF; nội dung chunk fetch lại từ Qdrant collection
  `unisage_eval_chunks` (collection run đó dùng) với filter quyền của lượt (bộ demo toàn guest →
  chỉ tài liệu public).
- Dùng được **139/154** câu: 15 câu không có retrieval (greeting/off-topic/clarification);
  37/2 224 chunk ghi trong run không còn trong collection (tài liệu đã xoá/ingest lại), câu vẫn
  dùng với các chunk còn lại.
- Ground truth: `expected_document_ids` của câu. Một chunk "đúng" khi `document_id` của nó nằm
  trong tập kỳ vọng.
- **Trần**: tài liệu kỳ vọng có mặt trong 16 ứng viên ở 0,935 (tính theo recall). Không biến thể
  nào vượt được số này; 5 câu (d0040, d0225, d0242, d0296, d0324) không có tài liệu kỳ vọng
  trong ứng viên và 9 câu có 2 tài liệu kỳ vọng nhưng chỉ 1 được retrieval lấy về.
- **Mốc không rerank**: lấy 8 chunk đầu theo thứ tự retrieval → recall@8 0,935, recall@5 0,924,
  MRR 0,854. Đây là mốc mọi bước rerank phải so.

### 1.2. Ba biến thể, một chữ ký

| Biến thể | Làm gì với 16 ứng viên | Số chunk ra |
|---|---|---|
| `cross_encoder` | service chấm 16 cặp (câu hỏi, heading + nội dung), giữ top-8 theo điểm | luôn 8 |
| `llm_rerank` | `llm_rerank` thật (model purpose RERANK của eval), giữ chunk model nói "trả lời câu hỏi" | 0–8 |
| `cross_encoder_then_llm` | cross-encoder cắt 16 → 8, rồi `llm_rerank` trên 8 chunk đó | 0–8 |

Lỗi của biến thể (service/model không trả kết quả) tính là lỗi, **không** fail-open về danh
sách gốc như graph thật làm, để số đo không được "tặng" điểm của retrieval.

### 1.3. Chỉ số

- recall@8, recall@5: phần tài liệu kỳ vọng có trong 8 / 5 chunk giữ đầu.
- NDCG@16 nhị phân theo document_id, lý tưởng = mọi chunk đúng trong 16 ứng viên xếp trước (phạt
  cả thứ tự lẫn việc bỏ bớt chunk đúng).
- MRR: 1/hạng của chunk đúng đầu tiên. Loại hết: câu không còn chunk nào. Giữ TB: số chunk ra.
- Độ ổn định: biến thể LLM chạy 3 lượt → Jaccard trung bình của tập giữ giữa các lượt, tỷ lệ
  câu cùng thứ tự, lệch recall@8; cross-encoder chạy 2 lượt để xác nhận tất định.
- Độ trễ: riêng bước rerank, đo trong tiến trình harness (gồm HTTP tới service / tới API LLM),
  không tính thời gian chờ rate limit.
- -p: warm-up 3, 5 vòng × 139 câu, concurrency 1, thread nền lấy mẫu CPU/RAM/VRAM/GPU mỗi 0,25 s.
- -c: 1 câu cố định (d0002, 16 ứng viên), mức 1 → 2 → 4 → 8, 30 request/mức qua
  `asyncio.Semaphore`.

### 1.4. VRAM

Harness không nạp model; nó spawn service, đo, rồi tắt ở `finally` và chờ `nvidia-smi` về mức
cũ. Mọi lần chạy: trước 494 MiB → khi service nạp model 1 839 MiB → sau khi tắt 494 MiB.

## 2. Bước A — Accuracy cross-encoder (`-a --variant cross_encoder`)

Báo cáo gốc: `benchmark_results/step-a-ce-accuracy/` (gitignore).

| Biến thể | Nhóm | Câu | Lượt | recall@8 | recall@5 | NDCG@16 | MRR | Loại hết | Giữ TB | Lỗi | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mốc không rerank | tất cả | 139 | — | 0.935 | 0.924 | — | 0.854 | 0% | 8 | — | 0 | 0 |
| cross_encoder | tất cả | 139 | 278 | 0.921 | 0.888 | 0.739 | 0.815 | 0.0% | 8.0 | 0 | 111 | 147 |
| cross_encoder | calculation | 26 | 52 | 0.923 | 0.923 | 0.740 | 0.849 | 0.0% | 8.0 | 0 | 103 | 136 |
| cross_encoder | normal | 113 | 226 | 0.920 | 0.881 | 0.739 | 0.807 | 0.0% | 7.9 | 0 | 111 | 149 |

Tất định: 2 lượt cho cùng thứ tự ở 139/139 câu (Jaccard 1,000).

Đọc số:

- Cross-encoder **thấp hơn mốc không rerank** một chút ở recall@8 (0,921 so với 0,935) và
  recall@5 (0,888 so với 0,924), MRR 0,815 so với 0,854. Nghĩa là trên bộ này thứ tự RRF của
  hybrid đã tốt, cross-encoder xếp lại không thêm tài liệu đúng mà còn làm rơi 2 câu.
- 2 câu rơi do cross-encoder (tài liệu đúng nằm trong top-8 retrieval nhưng bị chấm ra ngoài):
  - **d0010** "đề tài triển khai an ninh mạng ... do giảng viên nào hướng dẫn": tài liệu đúng là
    bảng danh sách hội đồng poster (retrieval xếp hạng 1–4, 6); cross-encoder chấm −0,4 cho
    "Danh sách phản biện" và đẩy các chunk bảng tên của tài liệu đúng ra ngoài top-8. LLM rerank
    trong run thật trả lời đúng. Điểm yếu thật của cross-encoder với bảng toàn tên riêng.
  - **d0287** "điều kiện Đạt kỳ thi chứng chỉ ứng dụng CNTT": cross-encoder giữ 5 chunk tiêu đề
    "Kết quả thi chứng chỉ ứng dụng CNTT cơ bản ngày 12-3-2022 ca ..." nhưng là văn bản cùng tên
    khác `document_id` với tài liệu kỳ vọng (các "ca" thi là tài liệu riêng, nội dung giống
    nhau). Metric theo document_id phạt; câu trả lời thực tế không sai. Lỗi này áp lên mọi biến
    thể như nhau.
- 13 câu còn lại dưới 1,0 là do bộ ứng viên (trần 0,935): không phải lỗi rerank.
- Phân phối điểm (logit) của chunk top-1: khi đúng median 5,73 (min −0,83); khi sai median 3,54
  (max 8,28). Hai phân phối chồng nhau nhiều → **không** nên dùng ngưỡng điểm làm gate thay cho
  LLM (đúng với giả định trong spec).

## 3. Bước B — Performance và concurrency cross-encoder (`-p -c --variant cross_encoder`)

Báo cáo gốc: `benchmark_results/step-b-ce-perf/`.

Bảng 2 — warm-up 3, 5 vòng × 139 câu, concurrency 1:

| Biến thể | Phép đo | p50 ms | p95 ms | TB ms | Max ms | req/s | Lỗi | CPU TB % | RAM đỉnh GB | VRAM đỉnh MiB | GPU TB % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_encoder | 695 | 112 | 147 | 107 | 177 | 9.35 | 0 | 13.8 | 8.45 | 2987 | 80.7 |

Bảng 3 — 1 câu cố định `d0002` (16 ứng viên), 30 request mỗi mức:

| Đồng thời | p50 ms | p95 ms | req/s | Lỗi | VRAM đỉnh MiB | GPU TB % |
|---|---|---|---|---|---|---|
| 1 | 146 | 153 | 6.79 | 0% | 2987 | 84.9 |
| 2 | 269 | 272 | 7.36 | 0% | 2987 | 85.2 |
| 4 | 538 | 542 | 7.34 | 0% | 2987 | 86.2 |
| 8 | 1076 | 1085 | 7.26 | 0% | 2987 | 87.2 |

Đọc số:

- Một lượt rerank 16 chunk thật mất **~110 ms** (p95 147 ms) trên 4080, so với 10 ms cho 16
  passage ngắn ở test Task 1: chunk của corpus dài (nhiều chunk chạm `max_length` 1024 token),
  chi phí tỉ lệ với độ dài, không phải số passage.
- Service xử lý **tuần tự**: p50 tăng tuyến tính theo mức đồng thời (146 → 269 → 538 → 1076 ms)
  còn throughput đứng ở ~7,3 req/s. GPU đã bận 85 %, nên không phải nghẽn ở vòng lặp sự kiện mà
  ở compute; muốn hơn phải batch nhiều request vào một lần forward hoặc thêm GPU. Với tải chat
  hiện tại (vài lượt/giây là nhiều) 7 req/s là đủ, nhưng ở 8 người cùng lúc mỗi người đợi ~1 s
  cho bước rerank.
- VRAM đỉnh 2 987 MiB khi chấm (1 839 MiB khi rảnh): activation của 16 cặp × 1024 token FP16
  ~1,1 GB. RTX 3060 12 GB đủ; đặt `RERANKER_BATCH_SIZE=8` nếu muốn giảm đỉnh.
- Nhả VRAM: 494 → 494 MiB sau khi tắt, cả hai lần.

## 4. Bước C — Accuracy hai biến thể LLM (`-a --repeats 3`)

Báo cáo gốc: `benchmark_results/step-c-llm-accuracy/`. Model RERANK của môi trường eval:
`openai:gpt-5-mini`, credential giới hạn 15 lượt/phút → harness giãn cách 14 lượt/phút
(`--llm-rpm`), 834 lượt gọi, 60 phút. Lần chạy đầu không giãn cách hỏng 404/417 lượt vì rate
limit, kết quả đó bị bỏ, không tính.

### 4.1. Bảng tổng (cùng 139 câu với bước A)

| Biến thể | Nhóm | Câu | Lượt | recall@8 | recall@5 | NDCG@16 | MRR | Loại hết | Giữ TB | Lỗi | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mốc không rerank | tất cả | 139 | — | 0.935 | 0.924 | — | 0.854 | 0% | 8 | — | 0 | 0 |
| cross_encoder | tất cả | 139 | 278 | 0.921 | 0.888 | 0.739 | 0.815 | 0.0% | 8.0 | 0 | 111 | 147 |
| llm_rerank | tất cả | 139 | 417 | 0.880 | 0.880 | 0.608 | 0.890 | 2.7% | 2.6 | 2 | 1819 | 3183 |
| llm_rerank | calculation | 26 | 78 | 0.922 | 0.922 | 0.641 | 0.883 | 3.9% | 3.0 | 1 | 1812 | 3415 |
| llm_rerank | normal | 113 | 339 | 0.870 | 0.870 | 0.600 | 0.891 | 2.4% | 2.5 | 1 | 1819 | 2989 |
| cross_encoder_then_llm | tất cả | 139 | 417 | 0.876 | 0.876 | 0.550 | 0.879 | 2.6% | 2.1 | 0 | 1753 | 2764 |
| cross_encoder_then_llm | calculation | 26 | 78 | 0.923 | 0.923 | 0.593 | 0.904 | 2.6% | 2.3 | 0 | 1717 | 3203 |
| cross_encoder_then_llm | normal | 113 | 339 | 0.866 | 0.866 | 0.541 | 0.873 | 2.7% | 2.1 | 0 | 1754 | 2651 |

Độ ổn định giữa 3 lượt:

| Biến thể | Jaccard TB tập giữ | Câu cùng thứ tự cả 3 lượt | Lệch recall@8 TB |
|---|---|---|---|
| cross_encoder (2 lượt) | 1.000 | 100.0% | 0.000 |
| llm_rerank | 0.822 | 49.6% | 0.018 |
| cross_encoder_then_llm | 0.820 | 49.6% | 0.011 |

Phân phối số chunk mỗi biến thể LLM giữ lại (417 lượt): `llm_rerank` 0:11, 1:110, 2:100, 3:120,
4:26, 5:20, 6:8, 7:4, 8:16; `cross_encoder_then_llm` 0:11, 1:150, 2:119, 3:89, 4:25, 5:13,
6:5, 7:2, 8:3. Median 2–3 chunk; cross-encoder luôn 8.

### 4.2. Đọc số

- **recall@8 = recall@5 ở hai biến thể LLM** vì chúng hiếm khi giữ quá 5 chunk. Hai biến thể
  LLM thấp hơn cross-encoder 4 điểm (0,880 / 0,876 so với 0,921) và thấp hơn mốc không rerank
  5,5 điểm: LLM rerank **loại bỏ** tài liệu đúng ở 12 câu mà cross-encoder còn giữ (vì nó chỉ
  giữ cái nó tin là trả lời được). Đó là cái giá của việc "lọc" thay vì "xếp".
- **MRR của LLM cao hơn** (0,890 so với 0,815): khi LLM giữ được tài liệu đúng, nó thường đặt
  đúng chunk lên đầu; cross-encoder giữ đủ 8 nhưng chunk đúng hay nằm hạng 2–4. Với prompt sinh
  câu trả lời đọc cả 8 chunk, MRR ít quan trọng hơn recall@8.
- **NDCG@16 của LLM thấp** (0,61 / 0,55 so với 0,74) chủ yếu vì phạt "giữ ít": lý tưởng đếm mọi
  chunk đúng trong 16 ứng viên, LLM giữ 1–3 nên không bao giờ đạt. Không coi đây là bằng chứng
  LLM xếp sai.
- **11 lượt loại hết** ở mỗi biến thể LLM đều rơi vào 4 câu: d0022, d0221 (tài liệu đúng có
  trong ứng viên nhưng điểm cosine tốt nhất 0,66 / 0,56 < ngưỡng rescue 0,75 → không cứu),
  d0040, d0324 (tài liệu đúng không có trong ứng viên, loại hết là đúng). Trong run thật, 3 câu
  ra "chưa có thông tin", 1 câu ra ticket. Cross-encoder không bao giờ loại hết, nên d0022 và
  d0221 ở chế độ `cross_encoder` sẽ đưa đúng tài liệu vào prompt; câu trả lời có đúng không thì
  phải xem bước E.
- **Không ổn định**: chỉ 49,6 % câu cho cùng tập giữ ở cả 3 lượt (Jaccard 0,82). Lệch recall@8
  trung bình nhỏ (0,018) vì phần lớn dao động là ở chunk phụ, nhưng có câu đổi hẳn kết quả giữa
  các lượt: d0022 (lượt 1–2 loại hết, lượt 3 giữ 4 chunk đúng), d0042 (lượt 1 đúng, lượt 2–3
  sai), d0224. Cross-encoder tất định 100 %.
- **cross_encoder_then_llm không hơn llm_rerank** về độ chính xác (0,876 so với 0,880, trong
  nhiễu) và còn mất 2 câu do cross-encoder đã cắt mất tài liệu trước khi LLM thấy (d0036: chunk
  đúng ở hạng 6–8 sau cross-encoder nhưng LLM 3 lần đều chọn "Quyết định 1035" thay vào; d0022).
  Bù lại nó cứu d0050 (LLM thuần 3/3 lượt chọn sai "Bản mô tả CTĐT", LLM sau cross-encoder 3/3
  chọn đúng "CTĐT Kiểm toán 2020") vì danh sách 8 ít nhiễu hơn 16. Tổng lại hoà.
- 2 lỗi của `llm_rerank` là model trả JSON hỏng (fail-open trong graph thật, ở đây tính lỗi);
  không có lỗi rate limit sau khi giãn cách.

### 4.3. 14 câu mà các biến thể cho kết quả khác nhau

Ký hiệu: recall@8 trung bình qua các lượt; "mốc" = 8 chunk đầu theo thứ tự retrieval.

| Câu | Mốc | CE | LLM | CE→LLM | Chuyện gì xảy ra |
|---|---|---|---|---|---|
| d0010 hội đồng poster, GV hướng dẫn đề tài an ninh mạng | 1.00 | 0.00 | 0.00 | 0.00 | Cả 3 đều chọn "Danh sách phản biện" thay vì "Danh sách hội đồng poster" (retrieval xếp hạng 1–4). Run thật trả lời đúng tên GV nhờ LLM rerank lúc đó giữ đúng, lần này 3/3 lượt sai → câu khó, kết quả LLM dao động giữa các lần chạy. CE sai hẳn (chấm −0,4 cho tài liệu sai, chunk đúng dưới top-8). |
| d0022 số điện thoại vướng mắc đăng ký nguyện vọng 2025 | 1.00 | 1.00 | 0.33 | 0.00 | CE giữ 2 chunk đúng (p3, p7) trong 8. LLM 2/3 lượt loại hết, 1 lượt giữ 4 chunk đúng; CE→LLM 3/3 loại hết. Run thật: "chưa có thông tin". Ứng viên: cosine cao nhất 0,66. |
| d0036 IUH Exchange Handbook 2019, nâng cấp khi nào | 1.00 | 1.00 | 1.00 | 0.00 | LLM thuần 3/3 giữ đúng chunk p4 (hạng 4 retrieval). CE đẩy chunk đó xuống hạng 6–8 (ưu tiên "Quyết định 1035", "Anti-terrorism plan" vì cùng chữ IUH tiếng Anh), LLM sau đó 3/3 chỉ giữ "Quyết định 1035". Trường hợp CE làm hại LLM. |
| d0042 điều kiện đạt chứng chỉ CNTT đợt 12/3/2022 | 1.00 | 1.00 | 0.33 | 0.67 | 4 tài liệu "Kết quả thi / phúc khảo chứng chỉ CNTT" gần giống nhau. LLM lượt 1 giữ đúng, lượt 2–3 chỉ giữ "phúc khảo". Nhiễu giữa tài liệu trùng tên. |
| d0050 học phần Kiểm toán cơ bản: học kỳ, tín chỉ, tiên quyết | 1.00 | 1.00 | 0.00 | 1.00 | Tài liệu đúng ở hạng 6 retrieval. LLM thuần 3/3 chọn "Bản mô tả CTĐT" (p43, p48, p53). CE xếp "CTĐT Kiểm toán 2020" lên top-8, LLM sau đó 3/3 giữ đúng. Trường hợp CE giúp LLM. Run thật: "chưa có thông tin". |
| d0219 NCS có bằng thạc sĩ tích luỹ tối thiểu bao nhiêu tín chỉ | 1.00 | 1.00 | 0.50 | 0.50 | 2 tài liệu kỳ vọng (QĐ 1053 và QĐ 3177). LLM giữ 2 chunk của một tài liệu, bỏ tài liệu kia; CE giữ cả hai. |
| d0221 tổng đài hỗ trợ thí sinh | 0.50 | 0.50 | 0.00 | 0.00 | 2 tài liệu kỳ vọng, retrieval chỉ lấy được 1. LLM 3/3 loại hết (cosine tốt nhất 0,56). CE giữ 3 chunk của tài liệu có. |
| d0224 Kiểm toán cơ bản học trước học phần nào | 0.50 | 0.50 | 0.17 | 0.00 | 2 kỳ vọng, 1 có trong ứng viên (hạng 3). LLM lượt 1 giữ, lượt 2–3 không; CE→LLM 3/3 chọn "Bản mô tả" p43/p46. |
| d0227 pretest KHMT 20/12/2024 GV chấm, số FAIL | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; CE giữ cả hai, LLM chỉ giữ tài liệu khớp ngày. Run thật trả lời cả hai ý đúng. |
| d0230 chuẩn đầu ra TCNH 2020, giao tiếp | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; LLM chỉ giữ 1. |
| d0231 địa chỉ, giờ làm việc CDIMEX | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; LLM chỉ giữ "Thông báo mua sách" (đủ để trả lời đúng, run thật đúng). |
| d0235 số tín chỉ Kiểm toán cơ bản | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; LLM giữ 1. |
| d0236 mật khẩu mặc định thi thử tiếng Anh K22 | 1.00 | 1.00 | 1.00 | 0.67 | CE→LLM 2/3 lượt chỉ giữ tài liệu K22, bỏ K20. |
| d0287 điều kiện Đạt chứng chỉ CNTT | 1.00 | 0.00 | 0.00 | 0.00 | Kỳ vọng là "ca 2", cả 3 biến thể giữ "ca 1"/"phúc khảo" (cùng nội dung điều kiện). Phạt do document_id, câu trả lời thực tế đúng. |

Nhóm 7 câu d0219–d0235 có **2 tài liệu kỳ vọng**: LLM rerank có xu hướng giữ một tài liệu đủ
trả lời rồi bỏ tài liệu còn lại, nên recall theo tài liệu 0,5 dù câu trả lời có thể vẫn đúng.
Cross-encoder giữ 8 nên giữ cả hai. Đây là nguồn chính của chênh lệch 4 điểm recall@8.

## 5. Bước D — Độ trễ hai biến thể LLM (`-p --iterations 1 --warmup 2`)

Báo cáo gốc: `benchmark_results/step-d-llm-perf/`. 139 phép đo mỗi biến thể, concurrency 1,
chỉ tính thời gian của lần gọi thành công (không tính thời gian giãn cách 14 lượt/phút).

| Biến thể | Phép đo | p10 | p25 | p50 | p75 | p90 | p95 | p99 | Max | TB | Lỗi | VRAM đỉnh MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_encoder (bước B) | 695 | 66 | 91 | 112 | 128 | 139 | 147 | 169 | 177 | 107 | 0 | 2987 |
| llm_rerank | 139 | 1290 | 1467 | 1696 | 2002 | 2508 | 3295 | 3745 | 3776 | 1834 | 0 | 1839 |
| cross_encoder_then_llm | 139 | 1290 | 1457 | 1698 | 1935 | 2226 | 2649 | 2934 | 4057 | 1742 | 0 | 4416 (*) |

(*) VRAM đỉnh 4 416 MiB và "sau khi tắt 1 435 MiB" của lần chạy này gồm 936 MiB của tiến trình
`vya-ocr-exp` và 484 MiB của `vya-ocr-staging` (dự án khác) khởi động trong lúc đo; service
cross-encoder đã gỡ model (log "reranker unloaded"), 494 + 936 ≈ 1 435. Không phải rò VRAM.

Đọc số:

- Bước rerank bằng LLM mất **~1,7 s median, 3,3 s p95** (gpt-5-mini, 16 chunk); cross-encoder
  **0,11 s median, 0,15 s p95**: nhanh hơn **15 lần ở median, 22 lần ở p95**. Trên một lượt chat
  p50 ~5,3 s (run hybrid-on), bỏ LLM rerank tiết kiệm ~1,6 s (≈30 %) nếu các bước khác không đổi.
- `cross_encoder_then_llm` không nhanh hơn ở median (LLM vẫn phải gọi) nhưng **p95 giảm từ 3,3 s
  xuống 2,6 s** vì LLM đọc 8 chunk thay vì 16 (prompt ngắn hơn ~một nửa, đuôi phân phối ngắn
  hơn). Cross-encoder thêm ~0,12 s.
- Throughput đo được 0,23 req/s cho cả hai biến thể LLM là do harness giãn cách theo rate
  limit, không phải giới hạn của model; không dùng số này để so với 9,35 req/s của cross-encoder.

## 6. Bước E — Tầng câu trả lời (eval e2e)

_(sau bước D: `runs/ce-on`, `runs/ce-then-llm` so với `runs/hybrid-on`)_

## 7. Khuyến nghị

_(điền sau)_

## 8. Việc cần đo lại trên Windows (RTX 3060 12 GB)

- Bảng 2 và Bảng 3 (độ trễ, throughput, VRAM đỉnh): phụ thuộc GPU.
- Bảng 1 không cần đo lại: cùng model, cùng ứng viên → cùng kết quả (đã xác nhận tất định).
- Lệnh: `task reranker:sync` (torch CUDA wheel cho Windows; không được thì WSL2), rồi
  `task benchmark:run env=eval -- hybrid-on -p -c --variant cross_encoder`.
