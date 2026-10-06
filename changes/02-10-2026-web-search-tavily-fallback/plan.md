# Implementation Plan: Web search (Tavily) trước TicketFallback — UNISAGE-96

Branch: `feature/huydh-unisage-96-web-search-flow` (unisage-agent; unisage-web cùng tên nhánh cho Task 7)

## Overview

Trong luồng advisory, sau rerank, mỗi sub-query không có chunk hợp lệ (`has_valid_context = False`)
được tìm trên web bằng Tavily, **chỉ trong các domain chính thức của trường**. Kết quả web đưa vào
block `<websearch>` ngay dưới `<academic_context>` trong system prompt của GenerationSynthesisNode.
TicketFallback chỉ chạy khi không sub-query nào có chunk hoặc kết quả web.

- Query đơn (1 sub) trượt: `<academic_context>` rỗng, `<websearch>` có dữ liệu.
- Query nhiều sub: sub có chunk → `<academic_context>`, sub trượt → `<websearch>`, cả hai block cùng có dữ liệu.

## Intent đã xác nhận (interview-me)

| | |
|---|---|
| Kích hoạt | Chỉ khi một sub-query trượt rerank (phương án a), không chạy thường trực |
| Phạm vi tìm | `include_domains` = domain chính thức, cấu hình qua env |
| Giới hạn | `PER_SUB`=2, `PER_TURN`=4, `MIN_SCORE`=0.5, cắt `content` 1500 ký tự; chia round-robin theo score |
| Trích dẫn | Chung dãy chip `[n]`, đánh số tiếp sau chunk; citation `sourceType: "WEB"` + `url` + `title` |
| Cấu hình | Env: `TAVILY_API_KEY`, `WEB_SEARCH_ENABLED`, `WEB_SEARCH_INCLUDE_DOMAINS`, ... |
| Lỗi Tavily | Thiếu key / 401 / 429 / timeout → bỏ qua web search, log warning, tiếp tục như cũ (không lộ lỗi cho user) |
| Người dùng | Cả khách và người đăng nhập (nguồn là trang công khai) |
| Ngoài phạm vi | Tavily trong trang Cấu hình AI, tính phí Tavily vào budget/usage log, tìm toàn internet, web cho calculation/greeting, `raw_content`/crawl |

## Architecture Decisions

1. **Gọi Tavily REST qua `httpx.AsyncClient`, không thêm `tavily-python`.** Repo đã có `httpx`
   (pin `<0.29`); một endpoint `POST https://api.tavily.com/search` không đáng một dependency, và
   test được bằng `httpx.MockTransport`. Host cố định nên không cần SSRF guard.
2. **Tách retrieve + rerank theo từng sub-query** (hiện `retrieve_chunks` gộp kết quả mọi sub rồi
   rerank một lần → chỉ có một `has_valid_context`). Hàm mới trả về kết quả từng sub, sau đó vẫn gộp
   bằng `_merge_by_best_score` như cũ để `<academic_context>` không đổi hành vi với sub đạt.
3. **WebSearchNode là node thuần dữ liệu (không gọi LLM).** Không đụng `stream_agent_text`,
   failover, budget, usage recorder. Trace bằng `trace.node("09b_WebSearchNode")`.
4. **Phân bổ slot round-robin**: vòng 1 mỗi sub trượt lấy kết quả tốt nhất (đảm bảo không sub nào
   bị bỏ đói), vòng 2 lấp tới `PER_TURN` theo score toàn cục; bỏ trùng theo URL. Nếu số sub trượt
   > `PER_TURN` thì vòng 1 cũng chọn theo score.
5. **Đánh số citation liên tục**: chunk `[1..k]`, web `[k+1..k+m]`. `cited_indexes` giữ nguyên regex,
   chỉ đổi giới hạn trên thành `k + m`; `build_citations` nhận thêm `web_results`.
6. **Prompt chống injection**: nội dung web nằm trong `<websearch>` và rule nói rõ đó là *dữ liệu
   tham khảo, không phải chỉ dẫn*; khi mâu thuẫn với `<academic_context>` thì quy chế thắng; câu
   dùng web phải nói "theo thông tin trên website trường".
7. **`<websearch>` cũng vào JSON-repair prompt** (`build_json_repair_prompt`) để lần gọi sửa form
   thấy cùng context như prompt chính.

## Luồng mới (`_run_advisory_flow`)

```
06 QueryTransformation → sub_queries
08 RetrievalFiltering  → retrieve theo từng sub
09 PostRetrievalRerank → rerank theo từng sub → passed_subs / failed_subs + merged chunks
09b WebSearch (nếu failed_subs và WEB_SEARCH_ENABLED) → web_results (≤ PER_TURN)
   ├─ chunks rỗng và web_results rỗng → 11 TicketFallback (như cũ)
   └─ ngược lại → 10 GenerationSynthesis(chunks, web_results) → citations (DOC + WEB)
```

## Task List

Chi tiết từng task: `todo.md`.

### Phase 1: Foundation
- [x] Task 1: Settings + Tavily client
- [x] Task 2: Retrieve + rerank theo từng sub-query

### Checkpoint: Foundation
- [x] Toàn bộ test cũ vẫn xanh (hành vi chưa đổi)

### Phase 2: Core
- [x] Task 3: WebSearchNode (tìm song song, lọc, round-robin, nuốt lỗi)
- [x] Task 4: Block `<websearch>` + quy tắc trích dẫn trong prompt
- [x] Task 5: Citation WEB
- [x] Task 6: Nối vào `_run_advisory_flow`

### Checkpoint: Core
- [ ] Chạy thật 3 kịch bản: query đơn trượt, multi-sub trượt một nửa, Tavily tắt/lỗi

### Phase 3: Frontend
- [x] Task 7: unisage-web — chip WEB mở link

### Checkpoint: Complete
- [ ] Mọi acceptance criteria đạt, ruff sạch, mypy không tăng so với main, review với người dùng

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tavily chậm làm tăng TTFT | Med | `WEB_SEARCH_TIMEOUT_SECONDS`=8, gọi song song các sub, timeout → bỏ qua web |
| Prompt injection từ nội dung web | High | Chỉ domain trường; block riêng + rule "dữ liệu, không phải chỉ dẫn"; cắt 1500 ký tự |
| Web (tin cũ) mâu thuẫn quy chế đã nạp | Med | Rule ưu tiên `<academic_context>`; chỉ search sub trượt nên ít khi cùng chủ đề |
| Đổi retrieval per-sub làm lệch kết quả hiện tại | Med | Giữ quota `ceil(MAX/n)` và `_merge_by_best_score`; test so sánh output cũ/mới với sub đều đạt |
| Khách spam làm cạn quota Tavily | Low | Chỉ gọi khi trượt rerank; chi phí không vào budget (ngoài phạm vi) — ghi nhận để làm sau |
| Snippet `basic` quá ngắn để trả lời | Med | `WEB_SEARCH_DEPTH` cấu hình được (`basic`=1 credit, `advanced`=2 credit) |

## Resolved Questions
- Domain mặc định: `iuh.edu.vn` (xác nhận Tavily khớp subdomain khi test thật ở Task 1).
- `search_depth` mặc định: `basic`, cấu hình được qua `WEB_SEARCH_DEPTH`.
