# Todo: Chấm web cả khi kho văn bản đã có đoạn (chủ đề động)

Chi tiết (AC, files, size) xem `plan.md`.

- [ ] T1 Config `CHAT_WEB_JUDGE_WITH_CHUNKS` (mặc định bật), dùng lại `CHAT_WEB_SEARCH_SPECULATIVE_TOPICS`, `.env*.example`
- [ ] T2 Graph: `questions` tính ngay sau node 06; `judge_task` (speculative search → 09c) chạy song song 08/09/09a cho MỌI câu hỏi con; generation nhận chunks + web answers; lỗi bộ chấm khi mọi câu có đoạn → không thêm web; log `mode=parallel`; test
- [ ] T3 SSE `status`: không để `web_searching`/URL xen giữa `retrieving`; cập nhật test + `contracts/chat-sse.md`
- [ ] T4 Prompt `web_search_context.yaml`: văn bản và web cùng lúc - quy định theo văn bản, thông tin theo đợt/năm theo nguồn mới hơn; test prompt builder + snapshot
- [ ] Checkpoint: pytest (bỏ `tests/e2e`), ruff file đã sửa, mypy không thêm lỗi (437)
- [ ] T5 Chạy thật 3 câu 14/10 (lịch thi ITC, hội thảo quốc tế, nghỉ lễ "năm nay"), so `graph_done`; thêm 2 câu vào eval web; eval trước/sau flag
- [ ] T6 Docs: cập nhật plan Web-Search-Optimization (W6) và docs luồng graph
