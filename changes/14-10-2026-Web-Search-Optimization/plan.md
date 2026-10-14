# Plan: Tối ưu web search (WebSearchNode)

> Trạng thái: **đang implement (14-10-2026).** Điều chỉnh so với bản review ở mục "Điều chỉnh khi
> implement" ngay dưới.

## Điều chỉnh khi implement

Số đo sau khi sửa latency retrieval (`changes/14-10-2026-Retrieval-Latency`) làm thay đổi hai quyết định:

- **Retrieval giờ chỉ còn 0.3-0.4s**, nên Tavily (2-5s) không còn nằm gọn sau retrieval. Gộp chunk và
  trang web vào **một** lần rerank (W5 cũ) sẽ bắt rerank chờ Tavily ở **mọi** lượt thuộc topic động,
  kể cả khi KB đã trả lời được. Thay bằng: rerank chunk giữ nguyên; trang web được **một lần chấm
  riêng** (prompt `agents/web_result_judge.yaml`, cùng model RERANK), chỉ chạy khi có câu hỏi con
  không còn chunk. Đường thành công không đổi latency và không tốn lời gọi LLM nào thêm.
- **Tavily bắt đầu ngay đầu nhánh advisory** (trước query transformation, vốn mất 1-2s khi người dùng
  có hồ sơ), song song với query transformation + retrieval + rerank (khoảng 3-5s). Không cần ngưỡng
  `CHAT_WEB_SEARCH_SKIP_IF_CHUNK_SCORE`: kết quả web chỉ được chấm khi rerank để trống một câu hỏi con.
  Lượt không cần web thì để Tavily chạy nốt ở nền (kết quả vào cache cho lượt sau).
- **Bộ chấm web lỗi hoặc không có model RERANK** → quay về luật cũ (`CHAT_WEB_SEARCH_MIN_SCORE`, 0.5),
  kèm cảnh báo admin. Ứng viên lấy với sàn `CHAT_WEB_SEARCH_SCORE_FLOOR` (0.05).
- **W0:** bộ câu hỏi nằm ở repo gateway (`dataset/official/questions.jsonl`, hiện có 6 câu
  `web_search`). Không sửa bộ dữ liệu trong lần này; baseline chạy bằng
  `task eval:run env=eval -- --web on --category web_search`.

## Vấn đề

Câu "học phí ngành cnnt" (log 14-10-2026 09:12) kết thúc ở TicketFallback dù Tavily trả về
đúng văn bản `ptckt.iuh.edu.vn/.../1035-QD-DHCN-11-4-2025.pdf` (QĐ mức học phí 2025-2026).
Nguyên nhân chồng lên nhau:

1. **Ngưỡng `CHAT_WEB_SEARCH_MIN_SCORE=0.5` cố định.** Điểm Tavily phụ thuộc mạnh vào cách viết
   query và không phản ánh độ liên quan:
   - query dạng câu hỏi "…là bao nhiêu?": mọi kết quả 0.04-0.19, file QĐ 0.08 bị loại;
   - query dạng từ khóa: bài *học bổng* 0.90, học phí *Thạc sĩ* 0.72, học phí *cơ sở Quảng
     Ngãi* 0.39, trang `/author/` của chương trình khác 0.71 (đều sai phạm vi nhưng điểm cao).
2. **Query là câu hỏi HyDE** (`extract_standalone_question`): không có "IUH", không có năm học,
   mỗi lần LLM viết một kiểu.
3. **Snippet quá ngắn.** `content` của Tavily là đoạn trích 100-700 ký tự, không phải nguyên
   trang. File QĐ là bản scan (4 trang ảnh, không có lớp text), snippet chỉ có "chi tiết theo
   phụ lục đính kèm".
4. **Tavily không đọc được các trang quan trọng nhất.** `www.iuh.edu.vn`, `pdt.iuh.edu.vn`,
   `tuyensinh.iuh.edu.vn` thiếu chứng chỉ trung gian SSL: `/extract` báo "Failed to fetch url"
   (có lần mất 17.8s), search chỉ trả về tiêu đề (khoảng 140 ký tự). Subdomain có SSL đúng
   (`fit.`) trả về khoảng 1600 ký tự nội dung liên quan.
5. Văn bản học phí chia theo **Khối Kinh tế / Khối Công nghệ × khóa tuyển sinh**, không ghi tên
   ngành, và đã cũ (năm học 2025-2026, hiện là 2026-2027).

Số đo Tavily: basic 2.0-2.6s, advanced (`chunks_per_source=3`) 3.8-7.0s.

## Nguyên tắc

- **Điểm Tavily không dùng làm tiêu chí liên quan.** LLM rerank chấm, điểm chỉ dùng làm sàn lọc
  rác (khoảng 0.05).
- **Không thêm lần gọi LLM.** `web` slots sinh trong lần gọi classify sẵn có; chunk và trang web
  được chấm trong **một** lần rerank.
- **Không thêm latency ở đường thành công** (KB đã trả lời được).
- **KB là nguồn chính.** Web chỉ dùng cho câu hỏi con không còn chunk nào sau rerank.
- **Thuần Tavily.** Không tự viết phần lấy nội dung trang; đo tỷ lệ "chỉ có tiêu đề" (W9) rồi
  mới quyết định có cần không.
- Không bao giờ nêu số liệu khi chỉ có tiêu đề hoặc link.

## Kiến trúc

```
classify (+ web slots) ─┬─ retrieval ──────────────┐
                        └─ Tavily (topic động,     ├─ rerank CHUNG [C..] + [W..] ─ generation
                           có cache, song song) ───┘                             └ fallback mềm
```

| topic | Chế độ | Cache TTL |
|---|---|---|
| `hoc_phi` | song song | 24h |
| `tuyen_sinh` | song song | 6h (tháng 6-8: 1h) |
| `lich_thong_bao` | song song | 3h |
| `lien_he_co_so` | song song | 7 ngày |
| `quy_che_thu_tuc` | fallback | 24h |
| `chuong_trinh_dao_tao` | fallback | 24h |
| `khac` | fallback | 24h |

## Task

### W0 - Bộ eval web search (baseline)

- Khoảng 20 câu trong `evals/` (học phí, tuyển sinh, lịch/thông báo, liên hệ, quy chế), mỗi câu
  có kỳ vọng: trả lời được / fallback mềm / ticket; nguồn đúng phạm vi.
- Metric: tỷ lệ trả lời được, tỷ lệ sai phạm vi (bậc/cơ sở/năm), tỷ lệ trang giữ lại nhưng chỉ
  có tiêu đề, p50/p95 TTFT, cache hit rate, số credit Tavily.
- **AC:** chạy được trên code hiện tại, có số baseline lưu trong `runs/`.
- **Size:** S.

### W1 - Classify sinh `web` slots

- Mỗi task `academic_advisory` có thêm
  `"web": {"topic", "subject", "bac_dao_tao", "he_dao_tao", "nganh_hoc", "co_so", "nam_hoc"}`.
  Tên slot lấy từ `known_metadata_fields.json`. `nam_hoc` chỉ điền khi người dùng nói rõ. MULTI
  thì `topic` là danh sách.
- Chuẩn hóa: topic lạ → `khac`; slot rỗng → `null`.
- **AC:** test normalize; câu "học phí ngành cnnt" cho `topic=hoc_phi`, `nganh_hoc=Công nghệ
  Thông tin`.
- **Files:** `agents/message_classification.yaml`, `app/schemas/intent.py`,
  `app/graph/nodes/message_classification.py`, test.
- **Size:** S.

### W2 - Ghép `web_query` từ slot (code, không LLM)

- Mẫu theo topic, ví dụ `hoc_phi`: `{subject} {bac} {he} năm học {nam_hoc} IUH` (không đưa tên
  ngành vào, vì biểu học phí chia theo khối); `tuyen_sinh`: `{subject} {nganh} {bac} năm {nam}
  IUH`.
- `nam_hoc` mặc định do code tính từ `APP_TIMEZONE` (từ tháng 8 là `Y-(Y+1)`). Khóa nhập học
  không đưa vào query.
- Chuẩn hóa alias ("CNTT"/"IT" → "Công nghệ Thông tin", "ĐHCQ" → Đại học + Chính quy) từ
  `aliases` trong `known_metadata_fields.json`. Slot thiếu hoặc lỗi thì quay về câu hỏi HyDE.
- **AC:** test cho từng topic; ca gốc ra `mức thu học phí Đại học Chính quy năm học 2026-2027 IUH`.
- **Files:** module mới `app/rag/web_query.py`, test.
- **Size:** S.

### W3 - Cache kết quả Tavily

- `CachedWebSearchClient` bọc `TavilyClient` theo Protocol `WebSearchClient`. Key
  `websearch:v1:{sha256(topic, slot đã chuẩn hóa, nam_hoc, tham số Tavily)}`. Value: danh sách
  kết quả sau khi lọc (khoảng 5 trang, content đã cắt), khoảng 5-8 KB.
- TTL theo topic (bảng trên). Kết quả rỗng: 30 phút. Lỗi (timeout/429/hết credit): không cache.
  Redis lỗi: gọi Tavily trực tiếp.
- Redis riêng `WEB_SEARCH_CACHE_REDIS_URL` (mặc định db 6; eval db 7), không dùng `allkeys-lru` trên instance
  dùng chung.
- **AC:** hit/miss, rỗng, lỗi không cache, Redis down đều có test; log `cache=hit|miss`.
- **Files:** `app/integrations/web_search_cache.py`, `app/core/config.py`, `.env.example`, test.
- **Size:** M.

### W4 - Tham số Tavily và lọc theo domain

- `search_depth=advanced`, `chunks_per_source=3`, `max_results=10` (tính tiền theo lượt search,
  không theo số kết quả). `MIN_SCORE` → sàn 0.05.
- `WEB_SEARCH_TOPIC_DOMAINS`: tier 1 theo topic (`hoc_phi` → ptckt, pdt, iuh.edu.vn/vi/thong-bao;
  `tuyen_sinh` → tuyensinh, pdt; `hoc_bong_ctsv` → ctsv, pdt; …). Sắp tier 1 trước rồi mới tới
  điểm; lấy 5 ứng viên.
- Loại: trang chủ (path rỗng), trang danh sách/phân trang (`/page/N`, `@p=`, `/author/`),
  subdomain cơ sở khác (`qn.`…) khi câu hỏi không nêu cơ sở, website khoa khác khi đã nêu ngành.
- **AC:** test sắp xếp/lọc với kết quả thật đã ghi lại trong plan này.
- **Files:** `app/graph/nodes/web_search.py`, `app/integrations/tavily_client.py`, config, test.
- **Size:** M.

### W5 - Rerank chung chunk + web

- Prompt rerank có `[Wn] (Trang: tiêu đề; URL; năm học)` + snippet. Luật web: giữ trang nếu là
  **văn bản chính thức đúng chủ đề và phạm vi**, kể cả khi snippet chưa có số liệu → đánh dấu
  `partial`.
- Parser nhận khóa `Wn`; web chỉ được dùng cho câu hỏi con không còn chunk.
- **AC:** ca gốc giữ file QĐ 1035 (partial), loại bài học bổng / Thạc sĩ / Quảng Ngãi.
- **Files:** `agents/reranker_compressor.yaml`, `app/graph/nodes/llm_rerank.py`,
  `app/graph/streaming_graph.py`, test.
- **Size:** M.

### W6 - Chạy song song với retrieval

- `CHAT_WEB_SEARCH_MODE = fallback | speculative`; `CHAT_WEB_SEARCH_SPECULATIVE_TOPICS`.
- Topic thuộc nhóm song song: Tavily chạy cùng lúc với retrieval. Chunk tốt nhất ≥
  `CHAT_WEB_SEARCH_SKIP_IF_CHUNK_SCORE` thì bỏ kết quả web, không đưa vào rerank. Rerank lại loại
  hết chunk → chấm riêng phần web (đã có sẵn).
- Timeout Tavily khoảng 6s ở chế độ song song.
- **AC:** đường thành công không đổi latency (test thời gian với fake chậm); ca gốc trả lời không
  cần chờ web tuần tự.
- **Files:** `app/graph/streaming_graph.py`, `app/graph/nodes/web_search.py`, config, test.
- **Size:** M.
- **Phụ thuộc:** W1-W5.

### W7 - Fallback mềm

- Trang được giữ nhưng chỉ có tiêu đề hoặc snippet không đủ: TicketFallback nhận
  `<candidate_sources>` (tiêu đề, URL, năm học). Chỉ được dẫn link và năm học, cấm nêu số.
- **AC:** ca gốc trả lời kiểu "Văn bản chính thức là QĐ 1035/QĐ-ĐHCN (năm học 2025-2026), chia
  theo Khối Công nghệ và khóa tuyển sinh: <link>".
- **Files:** `app/graph/nodes/ticket_fallback.py`, `main/chat_ticket_fallback.yaml`,
  `app/rag/prompting/builder.py`, test.
- **Size:** S.

### W8 - Năm học và `WEB_SOURCE_USED`

- Lấy năm học từ tiêu đề/URL/nội dung (`20\d{2}\s*-\s*20\d{2}`, `/files/2026/07/`). Sắp mới lên
  trước; đưa năm học vào prompt rerank và generation, câu trả lời phải ghi năm học áp dụng.
- `AdminWarning` mã `WEB_SOURCE_USED` kèm URL. Log mỗi lượt: `topic`, `web_mode`, `cache`,
  `web_used`.
- **AC:** test parse năm; warning xuất hiện khi dùng trang web.
- **Size:** S.

### W9 - Đo lại và chỉnh

- Chạy W0 lại; so sánh `fallback` vs `speculative`. Điều chỉnh nhóm topic song song, TTL và
  ngưỡng skip theo số liệu.
- Tỷ lệ "chỉ có tiêu đề" cao → cân nhắc tự fetch trang (khoảng 80 dòng: `httpx` qua
  `PinnedNetworkBackend` + CA bundle bổ sung chứng chỉ trung gian của IUH + `bs4`/`pymupdf4llm`,
  chỉ cho trang có content < 300 ký tự).

## Thứ tự

W0 → W1 → W2 → W5 → W7 → **checkpoint** (ca gốc trả lời được, chưa song song) → W4 → W3 → W6 →
W8 → **checkpoint** → W9.

## Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Sau khi sửa latency, retrieval nhanh hơn (khoảng 1-1.5s), Tavily advanced (4-7s) không còn nằm gọn sau retrieval | Cao | Chỉ topic động mới chạy song song, có cache, timeout; đo trong W9, cân nhắc basic cho đường song song |
| Trang quan trọng (www, pdt) chỉ có tiêu đề | Cao | Fallback mềm (W7); báo IT trường sửa chuỗi chứng chỉ SSL; tự fetch (W9) |
| LLM điền slot không ổn định → cache hit thấp | Trung bình | Code ghép query, chuẩn hóa alias, key theo slot |
| Credit Tavily (advanced = 2 credit/search) | Trung bình | Chỉ topic động chạy song song, cache, log số credit |
| Văn bản web cũ (2025-2026) bị trình bày như hiện hành | Trung bình | W8: luôn ghi năm học của nguồn |

## Câu hỏi mở

- Danh sách subdomain IUH đầy đủ (cơ sở Thanh Hóa, sau đại học, các khoa)?
- Gói Tavily dùng cho demo/production, giới hạn credit mỗi tháng?
- Giá trị khởi đầu cho `CHAT_WEB_SEARCH_SKIP_IF_CHUNK_SCORE` (đề xuất 0.65, ca gốc có chunk tốt
  nhất 0.57)?
- Có liên hệ được IT trường để sửa chứng chỉ SSL không?
