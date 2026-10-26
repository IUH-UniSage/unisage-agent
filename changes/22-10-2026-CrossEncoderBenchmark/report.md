# Báo cáo benchmark: cross-encoder so với LLM rerank (UNISAGE-106)

Ngày đo: 26/10/2026 (giờ VN 10:00–12:40) · Máy: Linux dev, RTX 4080 SUPER 16 GB, driver 595.91.07,
CUDA 13 · Nhánh: `feature/huydh-unisage-106-benchmark-cross-encoder` · Spec:
`docs/specs/SPEC-cross-encoder-rerank.md` · Cách chạy lại: `docs/guide/benchmark-rerank.md`.

Câu hỏi của ticket: thay bước 09a (LLM rerank) bằng cross-encoder `BAAI/bge-reranker-v2-m3`
chạy local, hoặc đặt cross-encoder trước LLM rerank, có **giảm độ trễ mà giữ (hoặc tăng) độ
chính xác** không.

Mỗi bảng dưới đây đi kèm ba phần: **đo gì**, **đọc ra gì**, **dùng để quyết định gì**.

---

## 0. Kết luận

| Tiêu chí thành công (spec) | Ngưỡng | Kết quả | Đạt |
|---|---|---|---|
| Độ trễ bước rerank cross-encoder, P95 | < 500 ms (trên 3060) | 147 ms trên 4080 (3060 chậm hơn ~2–2,5 lần → ~300–370 ms, cần đo lại) | Đạt trên 4080, nhiều khả năng đạt trên 3060 |
| recall@8 bước rerank | ≥ LLM rerank − 1 điểm | cross_encoder 0,921 so với llm_rerank 0,880 (+4,1 điểm) | Đạt, vượt |
| Tầng câu trả lời, normal recall@5 | ≥ LLM − 1 điểm | 110/116 (94,8 %) so với 104/116 (89,7 %) | Đạt, vượt |
| Tầng câu trả lời, calculation pass | ≥ LLM − 1 điểm | 20/38 so với 18/38 | Đạt |
| Tầng câu trả lời, calculation recall@5 | ≥ LLM − 1 điểm | 20/38 so với 23/38 (−3) | **Không đạt**, nhưng xem 6.4: 6 câu mất là mất trích dẫn, không mất câu trả lời |
| VRAM được nhả sau benchmark | nvidia-smi về mức trước | 494 → 1 839 → 494 MiB ở mọi lần | Đạt |

**Khuyến nghị: bật `CHAT_RERANK_MODE=cross_encoder` làm mặc định cho môi trường có GPU (dev,
eval, và production nếu máy chủ có GPU ≥ 4 GB trống), giữ `llm` làm fail-over tự động như đã
cài.** Lý do bằng số, trên cùng 154 câu demo:

- Nhanh hơn: bước rerank 112 ms thay vì 1 696 ms (median, −93 %); cả lượt chat p50 4,6 s thay vì
  5,3 s (−13,5 %), p95 7,5 s thay vì 9,5 s (−20 %).
- Đúng hơn ở nhóm normal: thêm 6 câu có tài liệu đúng trong top-5 (110 so với 104), ít câu từ
  chối trả lời hơn (32 so với 41), không còn ticket fallback (0 so với 1), độ chính xác trích
  dẫn 0,910 so với 0,891.
- Rẻ hơn và ổn định hơn: bớt đúng một lượt gọi LLM mỗi lượt (2,33 so với 3,32), chi phí
  −10 % (0,00503 so với 0,00557 USD/lượt), kết quả tất định 100 % (LLM chỉ 49,6 % câu cho cùng
  tập chunk qua 3 lần chạy), và không còn phụ thuộc rate limit 15 lượt/phút của credential
  RERANK, thứ đã làm LLM rerank lỗi ở 70/154 lượt của run gốc.

**Không khuyến nghị `cross_encoder_then_llm`**: không chính xác hơn `llm_rerank` (0,876 so với
0,880 ở bước rerank; 108 so với 104 ở tầng câu trả lời nhưng kém `cross_encoder` 110), vẫn tốn một
lượt LLM, vẫn dính rate limit (32 lượt lỗi), p95 cả lượt 17,9 s trong điều kiện eval.

**Điều kiện và việc cần làm trước khi bật ở production** (chi tiết mục 7):

1. Đo lại Bảng 2 và 3 trên RTX 3060 (Windows); nếu p95 > 500 ms thì giảm `RERANKER_MAX_LENGTH`
   xuống 512 và đo lại recall.
2. Giữ `CHAT_WEB_SEARCH_ENABLED=false` hoặc chấp nhận rằng ở chế độ `cross_encoder` web search
   chỉ chạy khi retrieval trả về 0 chunk (cross-encoder không bao giờ "loại hết").
3. Theo dõi 2 điểm yếu thật của cross-encoder: bảng toàn tên riêng (d0010) và tài liệu trùng
   tiêu đề (bộ "Kết quả thi chứng chỉ CNTT ca 1/ca 2").
4. Sửa nguyên nhân rate limit của credential RERANK/Extraction (maxRpm 15) độc lập với ticket
   này; nó làm méo mọi số đo của biến thể LLM.

---

## 1. Cách đo

### 1.1. Bộ ứng viên chung cho benchmark bước rerank (bước A–D)

- Nguồn: `runs/hybrid-on` (eval 21/10/2026, bộ demo, category normal + calculation, hybrid
  BM25 + dense bật, cache tắt, web off, variant exact). Mỗi câu lấy câu hỏi độc lập của
  sub-query đầu và đúng 16 `chunk_id` retrieval đã trả về theo thứ tự RRF; nội dung chunk fetch
  lại từ Qdrant collection `unisage_eval_chunks` (collection run đó dùng) với filter quyền của
  lượt (bộ demo toàn guest → chỉ tài liệu public).
- Dùng được **139/154** câu: 15 câu không có retrieval (greeting/off-topic/clarification);
  37/2 224 chunk ghi trong run không còn trong collection (tài liệu đã xoá/ingest lại), câu vẫn
  dùng với các chunk còn lại. 117 câu có 16 ứng viên, 19 câu có 8, 3 câu có 6.
- Ground truth: `expected_document_ids` của câu. Một chunk "đúng" khi `document_id` của nó nằm
  trong tập kỳ vọng. 130 câu có 1 tài liệu kỳ vọng, 9 câu có 2.
- **Trần**: tài liệu kỳ vọng có mặt trong 16 ứng viên ở mức recall 0,935. Không biến thể nào
  vượt được số này: 5 câu (d0040, d0225, d0242, d0296, d0324) không có tài liệu kỳ vọng trong
  ứng viên; 9 câu có 2 tài liệu kỳ vọng nhưng retrieval chỉ lấy được 1.
- **Mốc không rerank**: lấy 8 chunk đầu theo thứ tự retrieval → recall@8 0,935 (= trần),
  recall@5 0,924, MRR 0,854. Mọi bước rerank phải so với mốc này: trên bộ này thứ tự RRF của
  hybrid đã đưa tài liệu đúng vào top-8 ở mọi câu có thể.

### 1.2. Ba biến thể, một chữ ký `select(candidate) -> kept`

| Biến thể | Làm gì với 16 ứng viên | Số chunk ra | Gọi gì |
|---|---|---|---|
| `cross_encoder` | service chấm 16 cặp (câu hỏi, heading_path + nội dung), giữ top-8 theo điểm | luôn 8 | 1 HTTP local |
| `llm_rerank` | `llm_rerank` thật của graph (model purpose RERANK = `openai:gpt-5-mini`), giữ chunk model nói "trả lời câu hỏi", có rescue rule | 0–8 | 1 lượt LLM |
| `cross_encoder_then_llm` | cross-encoder cắt 16 → 8, rồi `llm_rerank` trên 8 chunk đó | 0–8 | 1 HTTP + 1 lượt LLM |

Lỗi của biến thể (service/model không trả kết quả) tính là lỗi, **không** fail-open về danh
sách gốc như graph thật, để số đo không được "tặng" điểm của retrieval.

### 1.3. Chỉ số bước rerank

| Chỉ số | Đo gì | Khi nào quan trọng |
|---|---|---|
| recall@8 | phần tài liệu kỳ vọng có trong 8 chunk giữ đầu (8 = `CHAT_CONTEXT_MAX_CHUNKS`, đúng số chunk prompt sinh câu trả lời nhìn thấy) | chỉ số chính: tài liệu không vào prompt thì không có câu trả lời |
| recall@5 | như trên với 5 | so với `recall@5` của eval e2e |
| MRR | 1/hạng của chunk đúng đầu tiên | khi model sinh ưu tiên chunk đầu; ít quan trọng hơn recall@8 vì prompt đọc cả 8 |
| NDCG@16 | nhị phân theo document_id, lý tưởng = mọi chunk đúng trong 16 ứng viên xếp trước | phạt cả xếp sai lẫn bỏ bớt chunk đúng; bất lợi cho biến thể giữ ít chunk |
| Loại hết | câu không còn chunk nào sau rerank | trong graph thật → web search / ticket / "chưa có thông tin" |
| Giữ TB | số chunk ra trung bình | giải thích vì sao NDCG và recall của LLM thấp |
| Ổn định | 3 lượt lặp: Jaccard trung bình của tập giữ, tỷ lệ câu cùng thứ tự cả 3 lượt, lệch recall@8 | có tái lập được kết quả eval không |
| p50/p95 ms | riêng bước rerank, đo trong harness (gồm HTTP tới service hoặc API LLM), **không** tính thời gian chờ rate limit | phần độ trễ ticket này muốn cắt |

### 1.4. Bước E, tầng câu trả lời

Chạy lại toàn bộ graph (`task eval:run env=eval -- --set demo --category normal --category
calculation --variant exact --web off`, cache tắt bằng `RETRIEVAL_CACHE_REDIS_URL=` trống, 3
lượt song song) với `CHAT_RERANK_MODE=cross_encoder` (`runs/ce-on`) và
`cross_encoder_then_llm` (`runs/ce-then-llm`), so với `runs/hybrid-on` (chế độ `llm`, cùng
lệnh, 21/10). Chỉ số của eval: `recall@5` (tài liệu kỳ vọng trong 5 chunk đầu của ngữ cảnh,
hoặc trong trích dẫn với nhánh calculation), `pass` (calculation: số đúng), `refusal_hint`
(câu trả lời chứa dấu hiệu "chưa có thông tin"), ticket fallback, `citation_accuracy`,
`total_ms`/`ttft_ms` toàn lượt, usage (số lượt gọi LLM, token, USD).

### 1.5. Điều kiện môi trường cần biết khi đọc số

- Credential RERANK (và Extraction) của eval giới hạn **15 lượt/phút**. Harness bước C–D giãn
  cách 14 lượt/phút (`--llm-rpm`) nên số đo bước rerank của LLM sạch; eval e2e (bước E và run
  gốc) chạy 3 lượt song song nên LLM rerank **lỗi rate limit và fail-open** ở 70/154 lượt của
  `hybrid-on` và 32/154 lượt của `ce-then-llm`. Nghĩa là "mốc LLM" ở tầng câu trả lời thực tế là
  hỗn hợp 55 % LLM rerank + 45 % xếp theo điểm; và độ trễ e2e của hai chế độ có LLM bị cộng thời
  gian chờ slot.
- VRAM: harness không nạp model; nó spawn service, đo, tắt ở `finally`, chờ `nvidia-smi` về mức
  cũ. Mọi lần: trước 494 MiB → service nạp model 1 839 MiB → sau khi tắt 494 MiB. Riêng lần bước
  D ghi "sau khi tắt 1 435 MiB" vì hai tiến trình `vya-ocr-exp` (936 MiB) và `vya-ocr-staging`
  (484 MiB) của dự án khác khởi động trong lúc đo; service đã gỡ model (log "reranker unloaded").

---

## 2. Bước A — Accuracy cross-encoder (`-a --variant cross_encoder`)

Báo cáo gốc: `benchmark_results/step-a-ce-accuracy/` (gitignore). 2 lượt × 139 câu, 30 giây.

| Biến thể | Nhóm | Câu | Lượt | recall@8 | recall@5 | NDCG@16 | MRR | Loại hết | Giữ TB | Lỗi | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mốc không rerank | tất cả | 139 | — | 0.935 | 0.924 | — | 0.854 | 0% | 8 | — | 0 | 0 |
| cross_encoder | tất cả | 139 | 278 | 0.921 | 0.888 | 0.739 | 0.815 | 0.0% | 8.0 | 0 | 111 | 147 |
| cross_encoder | calculation | 26 | 52 | 0.923 | 0.923 | 0.740 | 0.849 | 0.0% | 8.0 | 0 | 103 | 136 |
| cross_encoder | normal | 113 | 226 | 0.920 | 0.881 | 0.739 | 0.807 | 0.0% | 7.9 | 0 | 111 | 149 |

Tất định: 2 lượt cho cùng thứ tự ở 139/139 câu (Jaccard 1,000, lệch recall 0).

Đường cong recall@k (trung bình 139 câu), để chọn K:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| thứ tự retrieval (RRF) | 0.752 | 0.838 | 0.867 | 0.903 | 0.924 | 0.932 | 0.935 | 0.935 |
| cross_encoder | 0.712 | 0.766 | 0.831 | 0.871 | 0.888 | 0.906 | 0.914 | 0.921 |

**Đọc số**

- Cross-encoder **thấp hơn mốc không rerank** 1,4 điểm recall@8, 3,6 điểm recall@5 và 3,9 điểm
  MRR. Trên bộ này thứ tự RRF của hybrid đã xếp tài liệu đúng cao; cross-encoder xếp lại không
  thêm tài liệu nào (không thể, vì trần = mốc) mà làm rơi 2 câu và đẩy chunk đúng xuống vài
  hạng ở một số câu (recall@1 0,712 so với 0,752).
- 2 câu rơi (tài liệu đúng trong top-8 retrieval nhưng cross-encoder chấm ra ngoài):
  - **d0010** "đề tài triển khai an ninh mạng ... do giảng viên nào hướng dẫn": tài liệu đúng là
    bảng danh sách hội đồng poster (retrieval hạng 1–4, 6, 13, 14); cross-encoder cho điểm cao
    nhất −0,42 cho "Danh sách phản biện khóa luận" và đẩy mọi chunk của tài liệu đúng ra ngoài
    top-8. Bảng toàn tên riêng và mã đề tài không có "ngữ nghĩa" cho cross-encoder bám. Trong
    run gốc LLM rerank giữ đúng và trả lời đúng tên giảng viên, nhưng ở bước C LLM cũng 3/3
    lượt sai (mục 4.3) → câu khó cho cả hai.
  - **d0287** "điều kiện Đạt chứng chỉ ứng dụng CNTT": kỳ vọng là tài liệu "ca 2", cross-encoder
    giữ 5 chunk của "ca 1" và "phúc khảo" (cùng câu điều kiện). Metric theo document_id phạt;
    câu trả lời thực tế đúng. Lỗi áp lên mọi biến thể như nhau.
- 13 câu còn lại dưới 1,0 là do bộ ứng viên (trần 0,935), không phải lỗi rerank.
- Phân phối điểm (logit) của chunk top-1: khi đúng median 5,73 (min −0,83); khi sai median 3,54
  (max 8,28). Hai phân phối chồng nhau nhiều → **không** dùng ngưỡng điểm làm gate thay LLM
  (đúng giả định spec). Có thể dùng điểm để *ưu tiên* nhưng không để *loại*.
- Độ trễ 111 ms median cho 16 chunk thật, 103 ms cho câu calculation (ít ứng viên hơn).

**Dùng để quyết định**

- K = 8 giữ nguyên: từ k=6 (0,906) lên k=8 (0,921) vẫn còn +1,5 điểm; không có lý do giảm K.
- Cross-encoder không phải để "tăng recall" trên corpus này (retrieval đã tốt), mà để **thay LLM
  rerank với chi phí gần 0** (mục 4–6 cho thấy LLM rerank mới là thứ làm giảm recall).

---

## 3. Bước B — Performance và concurrency cross-encoder (`-p -c --variant cross_encoder`)

Báo cáo gốc: `benchmark_results/step-b-ce-perf/`.

Bảng 2 — warm-up 3, 5 vòng × 139 câu, concurrency 1:

| Biến thể | Phép đo | p10 | p25 | p50 | p75 | p90 | p95 | p99 | Max | TB | req/s | Lỗi | CPU TB % | RAM đỉnh GB | VRAM đỉnh MiB | GPU TB % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_encoder | 695 | 66 | 91 | 112 | 128 | 139 | 147 | 169 | 177 | 107 | 9.35 | 0 | 13.8 | 8.45 | 2987 | 80.7 |

Độ trễ theo số ứng viên (bước A, lượt 1): 6 ứng viên p50 35 ms, 8 ứng viên 66 ms, 16 ứng viên
116 ms (max 168 ms). Service báo `truncated=0` ở mọi request: không chunk nào vượt 1024 token.

Bảng 3 — 1 câu cố định `d0002` (16 ứng viên), 30 request mỗi mức:

| Đồng thời | p50 ms | p95 ms | req/s | Lỗi | VRAM đỉnh MiB | GPU TB % |
|---|---|---|---|---|---|---|
| 1 | 146 | 153 | 6.79 | 0% | 2987 | 84.9 |
| 2 | 269 | 272 | 7.36 | 0% | 2987 | 85.2 |
| 4 | 538 | 542 | 7.34 | 0% | 2987 | 86.2 |
| 8 | 1076 | 1085 | 7.26 | 0% | 2987 | 87.2 |

**Đọc số**

- Một lượt rerank 16 chunk thật mất **~110 ms** (p95 147 ms, p99 169 ms) trên 4080, so với 10 ms
  cho 16 passage ngắn ở test Task 1: chi phí tỉ lệ với **độ dài** chunk (corpus này chunk dài,
  nhiều cái sát 1024 token), không phải số passage. Phân phối hẹp (p10 66 → p99 169), không có
  đuôi dài: độ trễ dự đoán được.
- Service xử lý **tuần tự**: p50 tăng tuyến tính theo mức đồng thời (146 → 269 → 538 → 1076 ms)
  còn throughput đứng ở ~7,3 req/s; GPU đã bận 85 % nên nghẽn ở compute, không ở vòng lặp sự
  kiện. Muốn hơn phải gom nhiều request vào một lần forward (dynamic batching) hoặc thêm GPU.
- VRAM đỉnh 2 987 MiB khi chấm so với 1 839 MiB khi rảnh: activation của 16 cặp × 1024 token
  FP16 ≈ 1,1 GB. Tổng cần ~2,5 GB trống. RTX 3060 12 GB dư; `RERANKER_BATCH_SIZE=8` nếu muốn
  đỉnh thấp hơn (đổi lại chậm hơn ~10–20 %).
- Nhả VRAM: 494 → 494 MiB sau khi tắt, cả hai lần.

**Dùng để quyết định**

- Với tải chat hiện tại (dưới 1 lượt/giây) 7 req/s là dư 7 lần. Ngưỡng cần để ý: nếu có **≥ 4
  lượt chat cùng lúc** đến bước rerank, mỗi lượt đợi thêm ~0,5 s; ≥ 8 thì ~1 s, vẫn nhanh hơn LLM
  rerank (1,7 s) nhưng lợi thế co lại. Khi đó làm dynamic batching ở service (gom request trong
  10–20 ms thành một batch) trước khi nghĩ đến GPU thứ hai.
- Trên 3060 (≈ 40–45 % FLOPS FP16 của 4080 SUPER) kỳ vọng p50 ~250–300 ms, p95 ~350 ms: vẫn dưới
  500 ms, nhưng throughput chỉ ~3 req/s → với 4 lượt đồng thời đã ~1,3 s. Phải đo thật.

---

## 4. Bước C — Accuracy hai biến thể LLM (`-a --repeats 3`)

Báo cáo gốc: `benchmark_results/step-c-llm-accuracy/`. Model RERANK: `openai:gpt-5-mini`;
giãn cách 14 lượt/phút, 834 lượt gọi, 60 phút. Lần chạy đầu không giãn cách hỏng 404/417 lượt
vì rate limit, kết quả đó bị bỏ, không tính.

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

Đường cong recall@k:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| thứ tự retrieval | 0.752 | 0.838 | 0.867 | 0.903 | 0.924 | 0.932 | 0.935 | 0.935 |
| cross_encoder | 0.712 | 0.766 | 0.831 | 0.871 | 0.888 | 0.906 | 0.914 | 0.921 |
| llm_rerank | 0.829 | 0.868 | 0.879 | 0.880 | 0.880 | 0.880 | 0.880 | 0.880 |
| cross_encoder_then_llm | 0.811 | 0.860 | 0.872 | 0.874 | 0.876 | 0.876 | 0.876 | 0.876 |

Độ ổn định giữa 3 lượt:

| Biến thể | Jaccard TB tập giữ | Câu cùng thứ tự cả 3 lượt | Lệch recall@8 TB |
|---|---|---|---|
| cross_encoder (2 lượt) | 1.000 | 100.0% | 0.000 |
| llm_rerank | 0.822 | 49.6% | 0.018 |
| cross_encoder_then_llm | 0.820 | 49.6% | 0.011 |

Phân phối số chunk giữ (417 lượt): `llm_rerank` 0:11, 1:110, 2:100, 3:120, 4:26, 5:20, 6:8,
7:4, 8:16 (median 2); `cross_encoder_then_llm` 0:11, 1:150, 2:119, 3:89, 4:25, 5:13, 6:5, 7:2,
8:3 (median 2). Cross-encoder luôn 8.

### 4.2. Đọc số

- **LLM rerank là bộ lọc, không phải bộ xếp**: giữ median 2 chunk trên 16. Vì thế recall@8 =
  recall@5 = recall@3 (0,880), đường cong phẳng từ k=3. Nó thấp hơn cross-encoder 4,1 điểm và
  mốc không rerank 5,5 điểm vì **loại bỏ** tài liệu đúng ở 12 câu mà cross-encoder còn giữ.
- **LLM xếp đúng hơn ở đầu danh sách**: recall@1 0,829 so với 0,712 của cross-encoder và 0,752
  của retrieval; MRR 0,890 so với 0,815. Khi LLM giữ được tài liệu đúng, nó đặt đúng chunk lên
  đầu. Giá trị này chỉ có nghĩa nếu model sinh chịu ảnh hưởng mạnh bởi thứ tự; với prompt 8
  chunk thì recall@8 quyết định nhiều hơn (mục 6 xác nhận: tầng câu trả lời theo cross-encoder
  tốt hơn).
- **NDCG@16 của LLM thấp** (0,61 / 0,55 so với 0,74) chủ yếu vì phạt "giữ ít": lý tưởng đếm mọi
  chunk đúng trong 16 ứng viên, LLM giữ 1–3 nên không bao giờ đạt. Không coi là bằng chứng LLM
  xếp sai.
- **11 lượt loại hết** ở mỗi biến thể LLM đều rơi vào 4 câu: d0022, d0221 (tài liệu đúng có
  trong ứng viên, cosine tốt nhất 0,66 / 0,56 < ngưỡng rescue 0,75 → không cứu), d0040, d0324
  (tài liệu đúng không có trong ứng viên, loại hết là đúng). Trong run gốc 3 câu ra "chưa có
  thông tin", 1 câu ra ticket. Cross-encoder không bao giờ loại hết → d0022, d0221 ở chế độ
  `cross_encoder` đưa tài liệu đúng vào prompt (và ở bước E d0022 vẫn không trả lời được: bản
  thân tài liệu không có số điện thoại; d0221 tương tự).
- **Không ổn định**: chỉ 49,6 % câu cho cùng tập giữ ở cả 3 lượt (Jaccard 0,82). Lệch recall@8
  trung bình nhỏ (0,018) vì phần lớn dao động là ở chunk phụ, nhưng có câu đổi hẳn kết quả giữa
  các lượt: d0022 (lượt 1–2 loại hết, lượt 3 giữ 4 chunk đúng), d0042 (lượt 1 đúng, lượt 2–3
  sai), d0224. Hệ quả thực tế: hai lần chạy eval cùng cấu hình có thể lệch nhau 1–2 câu chỉ vì
  LLM rerank; cross-encoder loại bỏ nguồn nhiễu này.
- **cross_encoder_then_llm không hơn llm_rerank** về độ chính xác (0,876 so với 0,880, trong
  nhiễu) và mất 2 câu do cross-encoder đã đẩy tài liệu xuống trước khi LLM thấy (d0036: chunk
  đúng ở hạng 6–8 sau cross-encoder, LLM 3/3 lượt chọn "Quyết định 1035"; d0022). Bù lại nó cứu
  d0050 (LLM thuần 3/3 lượt chọn sai "Bản mô tả CTĐT", LLM sau cross-encoder 3/3 chọn đúng "CTĐT
  Kiểm toán 2020") vì danh sách 8 ít nhiễu hơn 16. Tổng lại hoà; không đáng thêm một tầng.
- 2 lỗi của `llm_rerank` là model trả JSON hỏng (graph thật fail-open); 0 lỗi rate limit sau
  khi giãn cách.

### 4.3. 14 câu mà các biến thể cho kết quả khác nhau

recall@8 trung bình qua các lượt; "mốc" = 8 chunk đầu theo thứ tự retrieval.

| Câu | Mốc | CE | LLM | CE→LLM | Chuyện gì xảy ra |
|---|---|---|---|---|---|
| d0010 hội đồng poster, GV hướng dẫn đề tài an ninh mạng | 1.00 | 0.00 | 0.00 | 0.00 | Cả 3 chọn "Danh sách phản biện" thay vì "Danh sách hội đồng poster" (retrieval hạng 1–4). Run gốc trả lời đúng tên GV vì LLM rerank lúc đó giữ đúng; lần này 3/3 sai → kết quả LLM dao động giữa các lần. CE chấm −0,4 cho tài liệu sai, chunk đúng dưới top-8. Điểm yếu: bảng toàn tên riêng. |
| d0022 số điện thoại vướng mắc đăng ký nguyện vọng 2025 | 1.00 | 1.00 | 0.33 | 0.00 | CE giữ 2 chunk đúng (p3, p7). LLM 2/3 lượt loại hết, 1 lượt giữ 4 chunk đúng; CE→LLM 3/3 loại hết. Cosine cao nhất 0,66. |
| d0036 IUH Exchange Handbook 2019, nâng cấp khi nào | 1.00 | 1.00 | 1.00 | 0.00 | LLM thuần 3/3 giữ đúng p4 (hạng 4 retrieval). CE đẩy xuống hạng 6–8 (ưu tiên "Quyết định 1035", "Anti-terrorism plan" vì cùng chữ IUH tiếng Anh), LLM sau đó 3/3 chỉ giữ "Quyết định 1035". CE làm hại LLM. Ở tầng câu trả lời (mục 6) chế độ `cross_encoder` lại **đúng** vì prompt đọc cả 8 chunk. |
| d0042 điều kiện đạt chứng chỉ CNTT đợt 12/3/2022 | 1.00 | 1.00 | 0.33 | 0.67 | 4 tài liệu "Kết quả thi / phúc khảo" gần giống nhau. LLM lượt 1 giữ đúng, lượt 2–3 chỉ giữ "phúc khảo". |
| d0050 Kiểm toán cơ bản: học kỳ, tín chỉ, tiên quyết | 1.00 | 1.00 | 0.00 | 1.00 | Tài liệu đúng ở hạng 6 retrieval. LLM thuần 3/3 chọn "Bản mô tả CTĐT" (p43, p48, p53). CE xếp "CTĐT Kiểm toán 2020" lên top-8, LLM sau đó 3/3 giữ đúng. CE giúp LLM. Run gốc: "chưa có thông tin"; bước E chế độ `cross_encoder`: hết từ chối. |
| d0219 NCS có bằng thạc sĩ tích luỹ tối thiểu bao nhiêu tín chỉ | 1.00 | 1.00 | 0.50 | 0.50 | 2 tài liệu kỳ vọng (QĐ 1053 và QĐ 3177). LLM giữ 2 chunk của một tài liệu; CE giữ cả hai. |
| d0221 tổng đài hỗ trợ thí sinh | 0.50 | 0.50 | 0.00 | 0.00 | 2 kỳ vọng, retrieval lấy được 1. LLM 3/3 loại hết (cosine tốt nhất 0,56). CE giữ 3 chunk của tài liệu có. |
| d0224 Kiểm toán cơ bản học trước học phần nào | 0.50 | 0.50 | 0.17 | 0.00 | 2 kỳ vọng, 1 có (hạng 3). LLM lượt 1 giữ, lượt 2–3 không; CE→LLM 3/3 chọn "Bản mô tả" p43/p46. |
| d0227 pretest KHMT 20/12/2024 | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; CE giữ cả hai, LLM chỉ giữ tài liệu khớp ngày. Run gốc trả lời đúng cả hai ý. |
| d0230 chuẩn đầu ra TCNH 2020 | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; LLM chỉ giữ 1. |
| d0231 địa chỉ, giờ làm việc CDIMEX | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; LLM chỉ giữ "Thông báo mua sách" (đủ trả lời đúng). |
| d0235 số tín chỉ Kiểm toán cơ bản | 1.00 | 1.00 | 0.50 | 0.50 | 2 kỳ vọng; LLM giữ 1. |
| d0236 mật khẩu mặc định thi thử tiếng Anh K22 | 1.00 | 1.00 | 1.00 | 0.67 | CE→LLM 2/3 lượt chỉ giữ tài liệu K22, bỏ K20. |
| d0287 điều kiện Đạt chứng chỉ CNTT | 1.00 | 0.00 | 0.00 | 0.00 | Kỳ vọng là "ca 2", cả 3 giữ "ca 1"/"phúc khảo" (cùng nội dung). Phạt do document_id. |

Nhóm 7 câu d0219–d0235 có **2 tài liệu kỳ vọng**: LLM rerank giữ một tài liệu đủ trả lời rồi
bỏ tài liệu còn lại, nên recall theo tài liệu 0,5 dù câu trả lời có thể vẫn đúng. Cross-encoder
giữ 8 nên giữ cả hai. Đây là nguồn chính của chênh lệch 4 điểm recall@8; câu hỏi "so sánh hai
văn bản" là nơi `cross_encoder` hơn hẳn.

**Dùng để quyết định**: nếu giữ LLM rerank thì phải chấp nhận ±1–2 câu nhiễu mỗi lần eval và
mất tài liệu thứ hai ở câu so sánh; nếu bỏ nó thì mất khả năng "không chunk nào trả lời → web
search" (xem 7.2).

---

## 5. Bước D — Độ trễ hai biến thể LLM (`-p --iterations 1 --warmup 2`)

Báo cáo gốc: `benchmark_results/step-d-llm-perf/`. 139 phép đo mỗi biến thể, concurrency 1,
chỉ tính thời gian lần gọi thành công (không tính giãn cách 14 lượt/phút).

| Biến thể | Phép đo | p10 | p25 | p50 | p75 | p90 | p95 | p99 | Max | TB | Lỗi | VRAM đỉnh MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cross_encoder (bước B) | 695 | 66 | 91 | 112 | 128 | 139 | 147 | 169 | 177 | 107 | 0 | 2987 |
| llm_rerank | 139 | 1290 | 1467 | 1696 | 2002 | 2508 | 3295 | 3745 | 3776 | 1834 | 0 | 1839 |
| cross_encoder_then_llm | 139 | 1290 | 1457 | 1698 | 1935 | 2226 | 2649 | 2934 | 4057 | 1742 | 0 | 4416 (*) |

(*) gồm 1 420 MiB của hai tiến trình dự án khác khởi động trong lúc đo (mục 1.5).

**Đọc số**

- Bước rerank bằng LLM mất **1,7 s median, 3,3 s p95** (gpt-5-mini, 16 chunk, prompt ~8–10k
  token); cross-encoder **0,11 s median, 0,15 s p95**: nhanh hơn **15 lần ở median, 22 lần ở
  p95**. Trên lượt chat p50 5,3 s của run gốc, đó là ~30 % thời gian.
- Trong run gốc, lượt mà LLM rerank chạy thật có p50 toàn lượt **6,2 s**, lượt nó lỗi và
  fail-open **3,9 s**: ở tầng người dùng LLM rerank tốn ~2,3 s (hơn 1,7 s đo riêng vì cộng chờ
  slot rate limit và truyền prompt).
- `cross_encoder_then_llm` không nhanh hơn ở median (LLM vẫn phải gọi) nhưng **p95 giảm 3,3 →
  2,6 s** vì LLM đọc 8 chunk thay vì 16 (prompt ngắn một nửa, đuôi ngắn hơn). Cross-encoder thêm
  ~0,12 s.
- 0,23 req/s của hai biến thể LLM là do harness giãn cách theo rate limit, không phải giới hạn
  model; không so với 9,35 req/s của cross-encoder.

**Dùng để quyết định**: nếu mục tiêu là độ trễ, chỉ có `cross_encoder` đạt; `cross_encoder_then_llm`
chỉ cắt được đuôi p95 ~0,6 s mà vẫn giữ toàn bộ chi phí và rủi ro rate limit của LLM.

---

## 6. Bước E — Tầng câu trả lời (eval e2e, 154 câu, cache tắt, web off)

Run gốc `runs/hybrid-on` (chế độ `llm`, 21/10), `runs/ce-on` (`cross_encoder`),
`runs/ce-then-llm` (`cross_encoder_then_llm`), cùng lệnh, cùng collection, cùng credential.

### 6.1. Bảng tổng

| Chỉ số | `llm` (hybrid-on) | `cross_encoder` (ce-on) | `cross_encoder_then_llm` (ce-then-llm) |
|---|---|---|---|
| normal: recall@5 | 104/116 (89,7 %) | **110/116 (94,8 %)** | 108/116 (93,1 %) |
| normal: fail (không có tài liệu đúng) | 12 | **7** | 8 |
| calculation: pass | 18/38 | 20/38 | **21/38** |
| calculation: recall@5 | **23/38** | 20/38 | 20/38 |
| Câu có dấu hiệu từ chối (`refusal_hint`) | 41 (31 normal / 10 calc) | 32 (27 / 5) | **29 (23 / 6)** |
| Ticket fallback | 1 | **0** | 1 |
| Citation accuracy (TB) | 0,891 | **0,910** | 0,905 |
| Lượt có node 09a LLM / CE | 122 / 0 | 0 / 123 | 123 / 123 |
| LLM rerank lỗi (fail-open) | 70 | 0 | 32 (31 rate limit, 1 JSON) |
| Lượt gọi LLM / lượt chat | 3,32 | **2,33** | 3,30 |
| Token vào / lượt (TB) | 28 155 | **25 759** | 27 789 |
| USD / lượt (TB) · cả run | 0,00557 · 0,858 | **0,00503 · 0,774** | 0,00540 · 0,831 |

### 6.2. Độ trễ toàn lượt (ms, 154 lượt, 3 lượt song song)

| Run | p10 | p25 | p50 | p75 | p90 | p95 | max | TTFT p50 | normal p50 / p95 | calculation p50 / p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| `llm` | 3222 | 3790 | 5345 | 6661 | 7900 | 9453 | 12158 | 4963 | 5204 / 9023 | 5737 / 12042 |
| `cross_encoder` | 3404 | 3881 | **4623** | **5505** | **6134** | **7535** | 27803 | **4359** | **4705 / 7535** | **4558 / 8113** |
| `cross_encoder_then_llm` | 3826 | 4899 | 6632 | 8441 | 14255 | 17917 | 34600 | 5906 | 5745 / 9908 | 11486 / 22196 |

**Đọc số**

- `cross_encoder` nhanh hơn `llm` **−722 ms ở p50 (−13,5 %)**, −1 918 ms ở p95 (−20 %), −604 ms
  TTFT. Mức giảm nhỏ hơn 1,7 s đo riêng vì run gốc đã "được lợi" 70 lượt LLM rerank lỗi (fail-open
  nhanh); so với lượt LLM rerank chạy thật (6,2 s) thì giảm ~1,6 s. Max 27,8 s của `ce-on` là
  một lượt chờ credential Chat (không liên quan rerank).
- `cross_encoder_then_llm` **chậm nhất** (p50 6,6 s, p95 17,9 s, calculation p50 11,5 s): nó gọi
  LLM rerank ở mọi lượt (123/123, run gốc chỉ 122 và lỗi 70), và với 3 lượt eval song song
  credential RERANK 15 lượt/phút bão hoà → chờ slot. Đây là độ trễ *trong điều kiện eval*; ở
  production một người dùng không chạm rate limit nhưng vẫn mất ~1,7 s cho LLM.
- Điểm quan trọng: độ trễ của mọi chế độ có LLM rerank **phụ thuộc rate limit của credential**,
  độ trễ của `cross_encoder` chỉ phụ thuộc GPU local.

### 6.3. Câu đổi kết quả, `cross_encoder` so với `llm`

| Nhóm | Chỉ số | Tốt lên | Xấu đi | Ròng |
|---|---|---|---|---|
| normal | recall@5 | 7 | 1 | **+6** |
| normal | refusal_hint | 14 hết từ chối | 10 thêm từ chối | **−4 từ chối** |
| calculation | pass | 5 | 3 | **+2** |
| calculation | recall@5 | 3 | 6 | **−3** |
| calculation | refusal_hint | 7 hết từ chối | 2 thêm | **−5 từ chối** |

7 câu normal từ "không có tài liệu đúng" thành có: d0028 (báo cáo phúc khảo HK III: LLM rerank
đã loại tài liệu đúng, câu trả lời gốc không trích dẫn), d0033 (hồ sơ PGS: run gốc trả lời
"xa chuyên môn", run CE trả lời đúng hướng nghiên cứu), **d0036** (Exchange Handbook: run gốc
"chưa có thông tin", run CE trả lời đúng thời điểm nâng cấp), d0042 (chứng chỉ CNTT), d0217
(học bổng Vingroup: run gốc từ chối, run CE "tối đa 100 suất" đúng), d0286, d0291 (TOEIC Online:
run gốc "chưa có thông tin", run CE trả lời địa điểm). Câu duy nhất xấu đi: d0012 (Interpretation
1): câu trả lời **vẫn đúng** ("3 tín chỉ", trích [7]) nhưng tài liệu đúng tụt xuống vị trí 7 trong
ngữ cảnh nên recall@5 = false. Nghĩa là ở nhóm normal, **7 câu từ không trả lời được thành trả
lời được, 0 câu ngược lại**.

### 6.4. Vì sao recall@5 của calculation giảm 23 → 20 dù pass tăng 18 → 20

Nhánh calculation không ghi `context`; eval tính recall@5 trên **trích dẫn** của câu trả lời. Soi
6 câu xấu đi: d0244, d0311, d0319, d0326, d0327, d0335 đều có `citations: []` ở run CE trong khi
câu trả lời **giống hoặc tương đương** run gốc (d0327 "Còn 335 chỗ" ở cả hai; d0311 cùng câu
"Để đạt điểm A..."; d0335 cả hai "chưa có thông tin"). Tức là model tính toán có đủ tài liệu
nhưng **không gắn số trích dẫn** → recall tính theo trích dẫn rơi. Hai giả thuyết cần kiểm tra
ở ticket sau: (1) ở nhánh calculation, danh sách chunk sau cross-encoder xếp chunk đúng thấp hơn
nên prompt tính toán (giới hạn `LLM_CHUNKS_LIMIT`) cắt mất, model trả lời từ số đã có trong câu
hỏi; (2) nhiễu của model sinh (3 lượt pass flip theo chiều ngược lại: d0315, d0322, d0324, d0333,
d0336 từ sai thành đúng). Vì `pass` (độ đúng con số) tăng +2 và refusal giảm 5, tôi đánh giá
tầng calculation **không xấu đi về chất lượng trả lời**, chỉ xấu đi về trích dẫn; cần theo dõi.

### 6.5. `cross_encoder_then_llm` so với `llm`

Normal +7/−3 recall (108), refusal 23 (ít nhất trong 3 run), calculation pass 21/38 (cao nhất),
recall 20/38, nhưng 1 ticket fallback (d0225), 32 lượt LLM rerank lỗi và độ trễ cao nhất. Chất
lượng không hơn `cross_encoder` một cách có ý nghĩa (±2 câu, trong nhiễu của LLM 49,6 % tái
lập) trong khi trả giá đầy đủ về độ trễ và chi phí.

**Dùng để quyết định**: ở tầng người dùng, bỏ LLM rerank và dùng cross-encoder **không làm mất
câu trả lời nào ở nhóm normal mà thêm 7 câu trả lời được**; nhóm calculation đúng hơn 2 câu,
ít từ chối hơn 5 câu, nhưng cần sửa việc gắn trích dẫn.

---

## 7. Đánh giá tổng hợp và khuyến nghị

### 7.1. Ba biến thể theo bốn trục

| Trục | `llm` (hiện tại) | `cross_encoder` | `cross_encoder_then_llm` |
|---|---|---|---|
| Độ trễ bước rerank p50 / p95 | 1,7 s / 3,3 s | **0,11 s / 0,15 s** | 1,7 s / 2,6 s |
| Độ trễ cả lượt p50 / p95 (eval) | 5,3 s / 9,5 s | **4,6 s / 7,5 s** | 6,6 s / 17,9 s |
| recall@8 bước rerank | 0,880 | **0,921** | 0,876 |
| normal recall@5 e2e | 89,7 % | **94,8 %** | 93,1 % |
| calculation pass e2e | 18/38 | 20/38 | **21/38** |
| Từ chối trả lời | 41 | 32 | **29** |
| Tái lập | 49,6 % | **100 %** | 49,6 % |
| Chi phí LLM / lượt | 0,00557 USD | **0,00503 USD** | 0,00540 USD |
| Phụ thuộc bên ngoài | API + rate limit | GPU local 2,5 GB | cả hai |
| Rủi ro riêng | loại nhầm tài liệu (12 câu), nhiễu | bảng tên riêng, tài liệu trùng tên, mất gate web search | cộng rủi ro của cả hai |

### 7.2. Khuyến nghị và điều kiện

1. **Bật `CHAT_RERANK_MODE=cross_encoder` mặc định** ở dev và eval ngay; ở production khi máy chủ
   có GPU trống ≥ 2,5 GB (3060 12 GB đạt). Fail-over về `llm` đã có sẵn: service tắt → cảnh báo
   `CROSS_ENCODER_FAILED`, câu trả lời vẫn có (đã test ở Task 2).
2. **Không triển khai `cross_encoder_then_llm`**; giữ code vì rẻ và có ích cho thử nghiệm model
   rerank khác.
3. **Web search**: ở chế độ `cross_encoder` sub-query không bao giờ "trống", nên
   `CHAT_WEB_SEARCH_MODE=fallback` chỉ kích hoạt khi retrieval trả 0 chunk; `speculative` không
   đổi. Hiện `CHAT_WEB_SEARCH_ENABLED=false` nên chưa ảnh hưởng; khi bật lại web search cần một
   gate khác (ví dụ: điểm cross-encoder top-1 < 0 **và** cosine top-1 < 0,6 → coi là không có
   tài liệu; phân phối điểm ở mục 2 cho thấy ngưỡng này chỉ bắt được phần đuôi, cần đo riêng).
4. **Trích dẫn ở nhánh calculation** (mục 6.4): mở ticket nhỏ kiểm tra `LLM_CHUNKS_LIMIT` và thứ
   tự chunk đưa vào prompt tính toán khi `CHAT_RERANK_MODE=cross_encoder`.
5. **Rate limit credential RERANK/Extraction**: 70/154 lượt lỗi trong run gốc là lỗi vận hành độc
   lập với ticket này; nếu vẫn dùng LLM ở bất kỳ bước nào, nâng maxRpm hoặc thêm credential.
6. **K giữ nguyên 8**; `RERANKER_MAX_LENGTH` giữ 1024 (0 passage bị cắt). Nếu 3060 không đạt p95
   < 500 ms, thử 512 và đo lại recall@8 (chunk dài sẽ bị cắt phần cuối).
7. **Service**: thêm dynamic batching khi có ≥ 4 lượt chat đồng thời (mục 3). Bind 127.0.0.1,
   không auth: production đặt cùng máy agent hoặc sau mạng nội bộ.
8. **Semantic cache / retrieval cache**: fingerprint đã có `rerank_mode` nên đổi chế độ không
   trả nhầm kết quả cũ; sau khi bật mặc định, L2 sẽ warm lại từ đầu.

### 7.3. Giới hạn của đo đạc này

- Một corpus (demo 106 tài liệu), 154 câu, 1 model LLM rerank (gpt-5-mini), 1 model cross-encoder.
  Kết luận "cross-encoder không thêm recall" gắn với việc retrieval hybrid đã tốt trên corpus này;
  trên corpus khác cross-encoder có thể cải thiện thứ tự nhiều hơn.
- recall theo document_id phạt tài liệu trùng tên (bộ "ca 1 / ca 2", "phúc khảo"); ảnh hưởng đều
  mọi biến thể nhưng làm số tuyệt đối thấp hơn thực tế.
- Mốc `llm` ở tầng câu trả lời bị pha 45 % lượt fail-open; mốc "LLM rerank sạch" ở tầng câu trả
  lời chưa đo được với credential hiện tại.
- Độ trễ e2e đo với 3 lượt song song và rate limit; số tuyệt đối không phải số một người dùng
  thấy, chỉ dùng để so giữa các chế độ.

---

## 8. Việc cần đo lại trên Windows (RTX 3060 12 GB)

- Bảng 2 và 3 (độ trễ, throughput, VRAM đỉnh): phụ thuộc GPU. Kỳ vọng p50 ~250–300 ms, p95
  ~350 ms, throughput ~3 req/s.
- Bảng 1 không cần đo lại: cùng model, cùng ứng viên → cùng kết quả (đã xác nhận tất định).
- Lệnh: `task reranker:sync` (torch CUDA wheel cho Windows; không được thì WSL2 và trỏ
  `CROSS_ENCODER_URL`), rồi `task benchmark:run env=eval -- hybrid-on -p -c --variant cross_encoder`.
- Nếu p95 > 500 ms: thử `RERANKER_MAX_LENGTH=512`, `RERANKER_BATCH_SIZE=8`, đo lại cả `-a`.

---

## 9. Phụ lục

### 9.1. File và lệnh đã dùng

| Bước | Lệnh | Kết quả |
|---|---|---|
| A | `task benchmark:run env=eval -- hybrid-on -a --variant cross_encoder --out benchmark_results/step-a-ce-accuracy` | `rerank_benchmark.{md,json}` |
| B | `... -p -c --variant cross_encoder --out benchmark_results/step-b-ce-perf` | |
| C | `... -a --repeats 3 --variant llm_rerank --variant cross_encoder_then_llm --out benchmark_results/step-c-llm-accuracy` (giãn cách 14/phút) | |
| D | `... -p --iterations 1 --warmup 2 --variant llm_rerank --variant cross_encoder_then_llm --out benchmark_results/step-d-llm-perf` | |
| E | `task reranker:serve` + `CHAT_RERANK_MODE=cross_encoder RETRIEVAL_CACHE_REDIS_URL= task eval:run env=eval -- --set demo --category normal --category calculation --variant exact --web off --out runs/ce-on`; tương tự `cross_encoder_then_llm` → `runs/ce-then-llm` | `summary.json`, `results.jsonl` |

`benchmark_results/` và `runs/` nằm trong gitignore; JSON của từng bước có từng lượt từng câu
(`kept_ids`, `ce_scores`, `took_ms`) để soi lại.

### 9.2. Sự cố trong lúc đo và cách xử lý

- Lần đầu bước C: 404/417 lượt gọi LLM lỗi "credential đã dùng hết 15 lượt gọi/phút" → thêm
  `RpmPacer` (`--llm-rpm 14`), chờ và gọi lại khi rate limit (`--llm-retry-wait 20`,
  `--llm-retries 6`), thời gian đo chỉ tính lần gọi thành công. Chạy lại: 0 lỗi rate limit.
- Biến thể LLM cần snapshot model registry → harness vào `lifespan` của app sau khi đã fetch ứng
  viên (để ứng viên vẫn lấy từ collection gốc của run).
- Bước D báo "VRAM không nhả" là do tiến trình dự án khác (mục 1.5); service đã gỡ model.
- Bước E1 tạm dừng ở 112/154 theo yêu cầu, chạy tiếp cùng `--out`, eval tự bỏ qua câu đã có.
