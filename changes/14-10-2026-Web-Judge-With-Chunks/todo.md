# Todo: Chấm web cả khi kho văn bản đã có đoạn (chủ đề động)

Chi tiết (AC, files, size) xem `plan.md`.

- [x] T1 Config `CHAT_WEB_JUDGE_WITH_CHUNKS` (mặc định bật), dùng lại `CHAT_WEB_SEARCH_SPECULATIVE_TOPICS`, `.env*.example`
- [x] T2 Graph: `questions` tính ngay sau node 06; `judge_task` (speculative search → 09c) chạy song song 08/09/09a cho MỌI câu hỏi con; generation nhận chunks + web answers; lỗi bộ chấm khi mọi câu có đoạn → không thêm web; log `mode=parallel`; test
- [x] T3 SSE `status`: không để `web_searching`/URL xen giữa `retrieving`; cập nhật test + `contracts/chat-sse.md`
- [x] T4 Prompt `web_search_context.yaml`: văn bản và web cùng lúc - quy định theo văn bản, thông tin theo đợt/năm theo nguồn mới hơn; test prompt builder + snapshot
- [x] Checkpoint: pytest 2206 passed (bỏ `tests/e2e`), ruff sạch trên file đã sửa, mypy 437 (không thêm lỗi)
- [x] T5a Thêm 2 câu vào eval web (repo gateway): w0017 lịch thi ITC, w0018 hội thảo quốc tế (`additions_20261014_web_judge_with_chunks.jsonl`)
- [ ] T5b Chạy thật 3 câu 14/10 (lịch thi ITC, hội thảo quốc tế, nghỉ lễ "năm nay") trên Windows, xem log `Web search turn: mode=parallel wait_ms=...`, so `graph_done`; eval `--category web_search` khi flag tắt/bật
- [x] T6 Docs: cập nhật plan Web-Search-Optimization (W6) và docs luồng graph
- [x] T7 MULTI: classify viết `web` theo từng câu hỏi con; mỗi câu một lần tìm (song song) và một lần chấm trên đúng trang của nó; `CHAT_WEB_SEARCH_MAX_QUERIES=3`; test
- [x] T8 Bộ chấm chọn tối đa 2 trang answer mỗi câu hỏi con theo thứ tự của nó (`CHAT_WEB_SEARCH_MAX_ANSWERS_PER_SUB`), không theo điểm Tavily; test
- [x] T9 Lọc trang thạc sĩ/tiến sĩ/sau đại học khi câu hỏi không nêu bậc (code + prompt bộ chấm); test
- [x] Checkpoint 2: pytest 2213 passed, ruff sạch trên file đã sửa, mypy 437 (không thêm lỗi)
- [ ] T10 Chạy thật câu MULTI 14/10 19:36 ("học phí ngành CNTT năm nay ... lịch thi chứng chỉ CNTT đợt gần nhất"): log phải có 2 query riêng, 2 dòng `Web judge SQ1` (mỗi câu một lần), không có trang thạc sĩ

Ghi chú khi làm:
- Bộ chấm song song được log bằng `trace.step("09c_WebJudgeNode", parent="08_RetrievalFilteringNode")`: không cắt thời gian của node 08/09a, không gửi status.
- Status (T3): chế độ song song chỉ gửi `web_searching` + URL sau rerank, cho các trang câu trả lời dùng (`contracts/chat-sse.md` §2).
- Không có model RERANK mà mọi câu hỏi con đều có chunk: không thêm trang web và không ghi `WEB_JUDGE_FAILED` (đã có `LLM_RERANK_UNAVAILABLE`).
