# Plan: Chấm web cả khi kho văn bản đã có đoạn (chủ đề động)

## Vấn đề

Hiện tại web chỉ được dùng cho câu hỏi con **không còn đoạn nào** sau LLM rerank (09a)
(`failed_sub_queries` trong `_run_advisory_flow`, `app/graph/streaming_graph.py`). Rerank chấm
"liên quan", không chấm "trả lời được", nên một đoạn cũ vẫn chặn web:

- 14/10/2026, "Trung tâm Ngoại ngữ - Tin học vừa thông báo lịch thi chứng chỉ ứng dụng CNTT đợt gần
  nhất vào ngày nào?": rerank giữ 8 đoạn "Kết quả thi ... ngày 12-3-2022" ("nêu rõ ngày thi"), lượt
  trả lời "chưa tìm thấy lịch 2026". Kết quả Tavily đã tìm sẵn ở nền (speculative) có đúng
  `itc.iuh.edu.vn`: "THÔNG BÁO LỊCH THI VÀ DANH SÁCH THI NGÀY 05,06/11/2026 (12-10-2026)".
- Cùng ngày, "tin tức mới nhất về hội thảo khoa học quốc tế": rerank giữ đoạn cẩm nang 2019 (chỉ nêu
  website phòng ban, nay đã chết), bỏ qua ICATSD 2026, ICTE 2026 Tavily đã tìm được.

## Quyết định (đã chọn 14/10/2026)

Với **chủ đề động** (`CHAT_WEB_SEARCH_SPECULATIVE_TOPICS`: `hoc_phi`, `tuyen_sinh`,
`lich_thong_bao`, `lien_he_co_so`), bộ chấm web (09c) **luôn chạy, song song với retrieval + rerank**,
cho **mọi** câu hỏi con; generation nhận **cả đoạn văn bản lẫn trang web** và tự chọn nguồn theo quy
tắc trong prompt. Chủ đề tĩnh (quy chế, chương trình đào tạo, khác) giữ nguyên luồng cũ (chỉ web khi
câu hỏi con không còn đoạn).

Lý do: kết quả tìm kiếm đã có sẵn (speculative search chạy từ đầu nhánh advisory), bộ chấm chỉ cần
câu hỏi con (có sau node 06) và các trang - không cần đoạn văn bản - nên chạy song song gần như không
thêm độ trễ. Đổi lại: thêm 1 lần gọi model RERANK (gpt-5-mini) cho lượt chủ đề động, prompt
generation dài thêm ~1-2k token khi có trang dùng được.

Đã cân nhắc và không chọn:
- Rerank tự đánh dấu "đủ/không đủ" cho từng câu hỏi con: rẻ nhất (không thêm call) nhưng phụ thuộc
  gpt-5-mini thinking=minimal - chính nó đã ghi "nêu rõ ngày thi" cho đoạn 2022.
- Luật năm bằng code (câu hỏi nêu năm mà đoạn giữ lại không có năm đó → web): chắc chắn nhưng chỉ bắt
  được lệch năm.

## Luồng mới (nhánh advisory)

```
03 classify ──► web slots ──► speculative search (đã có) ─────────────┐
06 query transformation ──► questions ──┬──► 09c judge(questions, pages) ──► answers/sources
                                        └──► 08 retrieval ► 09 ► 09a rerank ──► chunks
                                                                         │
                       generation(chunks + web answers)  ◄───────────────┘ (await cả hai)
                       fallback chỉ khi KHÔNG có chunk VÀ KHÔNG có web answer (như cũ)
```

- Chỉ chạy chấm song song khi `speculative_search is not None` (đã có topic động + search bật).
- Kết quả chấm dùng cho **mọi** câu hỏi con (không chỉ `failed_sub_queries`).
- Không có topic động: giữ nguyên nhánh cũ (`failed_sub_queries` → search → judge).
- Bộ chấm lỗi / thiếu model RERANK:
  - có câu hỏi con không còn đoạn → như cũ (`select_without_judge` + `WEB_JUDGE_FAILED`);
  - mọi câu hỏi con đều có đoạn → **không** thêm trang web (không đưa trang chấm theo điểm vào cạnh
    văn bản), chỉ ghi `WEB_JUDGE_FAILED`.

## Task

### T1 - Config

- `CHAT_WEB_JUDGE_WITH_CHUNKS: bool = True` (tắt được để quay về luồng cũ khi đo/so sánh).
- Dùng lại `CHAT_WEB_SEARCH_SPECULATIVE_TOPICS` làm danh sách chủ đề động; không thêm danh sách mới.
- `.env.example`, `.env.eval.example` (nếu có), mục cấu hình trong docs.
- **AC:** tắt flag → hành vi y như hiện tại (test cũ không đổi).
- **Files:** `app/core/config.py`, `.env*.example`.
- **Size:** XS.

### T2 - Graph: chấm song song

- Trong `_run_advisory_flow`, ngay sau khi có `questions` (sau node 06, trước 08): nếu
  `speculative_search` và flag bật → `judge_task = asyncio.create_task(_search_then_judge(...))`
  (await `speculative_search`, rồi `_judge_web_results(questions, outcome.results)`).
- `questions` hiện được tính sau 09 (`extract_standalone_question` trên `sub_queries`) - dời lên ngay
  sau node 06 (không phụ thuộc rerank).
- Sau 09a: có `judge_task` → `await` lấy `answers`/`sources` cho mọi câu hỏi con; áp dụng quy tắc lỗi
  ở trên; bỏ nhánh `failed_sub_queries` cho lượt này. Không có → nhánh cũ.
- `trace.node("09b_WebSearchNode")` / `09c` vẫn log như cũ; log `Web search turn: mode=parallel ...`
  (thêm mode mới cạnh `speculative`/`fallback`), ghi rõ số câu hỏi con đã có đoạn.
- `WEB_SOURCE_USED` giữ nguyên.
- Hủy `judge_task` nếu nhánh kết thúc sớm (lỗi, ask-user form...) - xem `_finish_in_background`.
- **AC:**
  - lượt `lich_thong_bao` có đoạn được giữ + trang web chấm `answer` → generation nhận cả hai,
    `citations` có cả chunk lẫn trang WEB, không vào fallback;
  - lượt `quy_che_thu_tuc` có đoạn → bộ chấm KHÔNG chạy (như cũ);
  - bộ chấm lỗi khi mọi câu hỏi con có đoạn → không có trang web, có `WEB_JUDGE_FAILED`;
  - bộ chấm bắt đầu trước khi rerank xong (test bằng mock chậm / thứ tự gọi).
- **Files:** `app/graph/streaming_graph.py`, `tests/graph/test_graph_wiring.py`.
- **Size:** M.

### T3 - Trạng thái tiến trình (SSE `status`)

- Hiện `trace.web_page(url)` gửi `web_searching` + URL cho từng trang trước khi chấm. Chạy song song
  thì sự kiện này xen giữa `retrieving` → nhảy qua lại.
- Quyết định khi làm: chỉ gửi `web_searching` (và từng URL) **sau rerank**, khi thật sự dùng trang web
  (có answer/source), hoặc bỏ hẳn từng URL ở chế độ song song. Ghi lại lựa chọn trong
  `contracts/chat-sse.md` nếu đổi ý nghĩa.
- **AC:** test `test_web_branch_reports_each_page_it_reads` / `test_unused_speculative_search_reports_no_web_status`
  cập nhật theo quyết định; không có hai `status` liên tiếp cùng stage.
- **Files:** `app/graph/streaming_graph.py`, `tests/graph/test_graph_wiring.py`, `contracts/chat-sse.md`.
- **Size:** S.

### T4 - Prompt generation: văn bản + web cùng lúc

`common/web_search_context.yaml` hiện viết cho trường hợp web chỉ có khi văn bản không đủ. Sửa:
- Bỏ câu "`<websearch>` chỉ có dữ liệu khi văn bản quy chế ở trên không đủ".
- Quy tắc chọn nguồn:
  - quy định, điều kiện, thủ tục → theo `<academic_context>`; web mâu thuẫn thì theo văn bản (giữ
    quy tắc hiện có);
  - thông tin theo đợt/năm (lịch thi, thông báo, điểm chuẩn, học phí năm mới, liên hệ): đoạn văn bản
    thuộc năm/đợt cũ hơn trang web hoặc không có đúng năm được hỏi → trả lời theo trang web mới hơn
    (kèm `[n]`), có thể nói thêm văn bản trong kho là của năm cũ;
  - không gộp số liệu hai năm thành một.
- Đếm token prompt trước/sau (giữ gọn, xem plan tối ưu token).
- **AC:** test prompt builder: có chunks + web → prompt có cả `<academic_context>` và `<websearch>`,
  chỉ số `[n]` liên tục; snapshot `tests/fixtures/advisory_prompt_snapshot.txt` cập nhật nếu đổi.
- **Files:** `app/rag/prompting/prompt_templates/common/web_search_context.yaml`,
  `tests/rag/test_prompt_builder.py`, snapshot.
- **Size:** S.

### Checkpoint

- `.venv\Scripts\python.exe -m pytest --ignore=tests/e2e` (chỉ `tests/evals/test_judge.py` 503/504 chập
  chờn có sẵn), ruff trên file đã sửa, mypy không thêm lỗi (baseline 437).

### T5 - Chạy thật + eval

- Chạy lại 3 câu của 14/10: lịch thi ITC ("đợt gần nhất"), hội thảo khoa học quốc tế ("mới nhất"),
  nghỉ lễ Quốc khánh "năm nay" - xem log `Web judge SQ1`, `Web search turn: mode=parallel`, thời gian
  `graph_done` so với trước (mục tiêu: không tăng quá ~0,5s).
- Thêm 2 câu đầu vào bộ eval web (`dataset/official` repo gateway, cạnh w0001-w0016), metric
  `web_source`.
- `task eval:run env=eval -- --web on --category web_search` trước/sau (flag tắt/bật).
- **AC:** 2 câu trên trả lời đúng ngày/sự kiện 2026 có trích trang web; các câu quy chế trong eval
  không đổi kết quả.
- **Size:** S.

### T6 - Docs

- Cập nhật `changes/14-10-2026-Web-Search-Optimization/plan.md` (mục speculative/W6: chấm không còn
  chỉ cho câu hỏi con thiếu đoạn) và docs luồng graph nếu có mô tả 09b/09c.
- **Size:** XS.

## Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Thêm chi phí 1 call gpt-5-mini cho mọi lượt chủ đề động | Trung bình | Chỉ chủ đề động; flag tắt được; cache Redis kết quả search đã có |
| Generation trộn số liệu năm cũ (văn bản) với năm mới (web) | Trung bình | Quy tắc T4 + luật năm ở bộ chấm (trang khác năm không là `answer`) đã có |
| Trang web ghi đè quy chế chính thức | Thấp | Quy tắc "quy định → theo văn bản" giữ nguyên |
| Prompt generation dài hơn | Thấp | Tối đa `CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN=2` trang, mỗi trang ≤ `CHAT_WEB_SEARCH_RESULT_MAX_CHARS` |
| Speculative search chậm hơn rerank → chờ | Thấp | Search đã chạy từ đầu nhánh; nếu cần, đặt timeout cho judge_task và bỏ web khi đã có đoạn |
| Status UI nhảy qua lại | Thấp | T3 |

## Câu hỏi mở

- Có mở rộng sang `quy_che_thu_tuc` khi câu hỏi có trạng từ thời gian ("mới nhất", "năm nay") không?
  (VD "quy định mới nhất về học bổng"). Mặc định: chưa.
