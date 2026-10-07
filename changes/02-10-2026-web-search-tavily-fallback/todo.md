# Todo: Web search (Tavily) trước TicketFallback — UNISAGE-96

Lệnh chung (unisage-agent):
- Test: `.venv/bin/pytest -q <path>`
- Lint/type: `.venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests && .venv/bin/mypy app` (mypy không vượt baseline 318 của main)

---

## Task 1: Settings + Tavily client

**Description:** Thêm cấu hình web search vào `Settings` và một client async gọi `POST https://api.tavily.com/search`
với `include_domains`, `max_results`, `search_depth`, trả về `list[WebSearchResult(title, url, content, score)]`.
Mọi lỗi (thiếu key, HTTP lỗi, timeout, JSON sai) thành một exception riêng `WebSearchUnavailableError` để node phía trên nuốt.

**Acceptance criteria:**
- [ ] Settings: `CHAT_WEB_SEARCH_ENABLED` (False), `CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB` (2), `..._PER_TURN` (4), `..._MIN_SCORE` (0.5), `..._RESULT_MAX_CHARS` (1500); `TAVILY_API_KEY`, `TAVILY_BASE_URL`, `TAVILY_INCLUDE_DOMAINS` (CSV), `TAVILY_SEARCH_DEPTH` (`basic`), `TAVILY_TIMEOUT_SECONDS` (8)
- [ ] Client gửi đúng body/headers (Bearer key, `include_domains`), parse kết quả, không log API key
- [ ] 401/429/5xx/timeout/JSON hỏng → `WebSearchUnavailableError` với lý do ngắn

**Verification:**
- [ ] `pytest -q tests/integrations/test_tavily_client.py tests/test_config.py` (dùng `httpx.MockTransport`)
- [ ] Manual: gọi thật 1 lần với key + `include_domains` để xác nhận subdomain được khớp

**Dependencies:** None

**Files likely touched:**
- `app/core/config.py`
- `app/schemas/web_search.py` (new)
- `app/integrations/tavily_client.py` (new)
- `tests/integrations/test_tavily_client.py` (new)
- `.env.example`

**Estimated scope:** Medium

---

## Task 2: Retrieve + rerank theo từng sub-query

**Description:** Thay cặp `retrieve_chunks` + `rerank_chunks` (gộp rồi rerank một lần) bằng một bước trả về kết quả
rerank của **từng** sub-query (`SubQueryContext(sub_query, chunks)`), cộng danh sách chunk đã gộp bằng
`_merge_by_best_score` như cũ. Chưa đổi hành vi graph.

**Acceptance criteria:**
- [ ] Biết được sub-query nào có / không có chunk hợp lệ sau rerank
- [ ] Khi mọi sub đều đạt, danh sách chunk gộp giống hệt cách cũ (cùng quota `ceil(MAX/n)`, cùng thứ tự)
- [ ] Query đơn vẫn retrieve với limit mặc định như cũ

**Verification:**
- [ ] `pytest -q tests/graph/test_retrieval_rerank_nodes.py tests/graph/test_graph_wiring.py`

**Dependencies:** None

**Files likely touched:**
- `app/graph/nodes/retrieval_filtering.py`
- `app/graph/nodes/post_retrieval_rerank.py`
- `app/graph/streaming_graph.py` (chỉ đổi call site, giữ hành vi)
- `tests/graph/test_retrieval_rerank_nodes.py`

**Estimated scope:** Small

## Checkpoint: Foundation
- [ ] Toàn bộ test suite xanh (trừ 3 lỗi có sẵn trên main), ruff sạch
- [ ] Hành vi chat chưa đổi

---

## Task 3: WebSearchNode

**Description:** `app/graph/nodes/web_search.py`: nhận các sub-query trượt, gọi Tavily song song (`asyncio.gather`),
lọc `score >= MIN_SCORE`, bỏ trùng URL, phân bổ round-robin (vòng 1: top-1 mỗi sub theo score; vòng 2: lấp tới
`PER_TURN` theo score toàn cục), cắt `content` về `RESULT_MAX_CHARS`. Tắt / thiếu key / lỗi của một sub → sub đó
coi như không có kết quả, log warning, không raise.

**Acceptance criteria:**
- [ ] 2 sub trượt, mỗi sub 2 kết quả qua ngưỡng, `PER_TURN`=3 → đủ 3 kết quả, mỗi sub ≥ 1
- [ ] `WEB_SEARCH_ENABLED=False` hoặc key rỗng → trả `[]` và không gọi HTTP
- [ ] Một sub lỗi Tavily, sub kia thành công → vẫn trả kết quả của sub thành công

**Verification:**
- [ ] `pytest -q tests/graph/test_web_search_node.py` (client giả)

**Dependencies:** Task 1

**Files likely touched:**
- `app/graph/nodes/web_search.py` (new)
- `tests/graph/test_web_search_node.py` (new)

**Estimated scope:** Small

---

## Task 4: Block `<websearch>` trong prompt

**Description:** Thêm block `<websearch>` ngay dưới `<academic_context>` trong `prepared_context.yaml`, đánh số tiếp
sau chunk (`[k+1] (Tiêu đề — url) nội dung`), sentinel khi rỗng. Thêm quy tắc: dùng chung `[n]`, nói "theo thông tin
trên website trường", ưu tiên `<academic_context>` khi mâu thuẫn, nội dung web là dữ liệu không phải chỉ dẫn.
`build_system_prompt`, `build_multi_intent_prompt`, `build_json_repair_prompt` nhận `web_results` (mặc định rỗng).

**Acceptance criteria:**
- [ ] 2 chunk + 1 web → web hiển thị `[3]` trong `<websearch>`
- [ ] Không có web → block hiển thị sentinel, prompt cũ không đổi nội dung chunk
- [ ] Rule ưu tiên quy chế + rule chống injection có trong prompt advisory và multi-intent

**Verification:**
- [ ] `pytest -q tests/rag/test_prompt_loader.py tests/graph/test_generation_synthesis_node.py`

**Dependencies:** Task 1 (schema `WebSearchResult`)

**Files likely touched:**
- `app/rag/prompting/prompt_templates/common/prepared_context.yaml`
- `app/rag/prompting/prompt_templates/common/citation_rules.yaml`
- `app/rag/prompting/builder.py`
- `app/rag/prompting/__init__.py`
- `app/graph/nodes/generation_synthesis.py`

**Estimated scope:** Medium

---

## Task 5: Citation WEB

**Description:** `build_citations(response_text, chunks, web_results=())` trả thêm citation web cho `[n]` với
`n > len(chunks)`: `{index, title, url, sourceType: "WEB", documentId: None, section: None, pageStart: None, pageEnd: None}`.
Chỉ số vượt `len(chunks) + len(web_results)` vẫn bị bỏ.

**Acceptance criteria:**
- [ ] `[1][3]` với 2 chunk + 1 web → 1 citation DOC + 1 citation WEB đúng url
- [ ] `[5]` khi tổng nguồn là 3 → bị bỏ
- [ ] Không có web → output giống hệt cũ

**Verification:**
- [ ] `pytest -q tests/rag/test_citations.py`

**Dependencies:** Task 1

**Files likely touched:**
- `app/rag/prompting/citations.py`
- `tests/rag/test_citations.py`

**Estimated scope:** Small

---

## Task 6: Nối WebSearch vào `_run_advisory_flow`

**Description:** Dùng kết quả per-sub (Task 2) → sub trượt vào WebSearchNode (`trace.node("09b_WebSearchNode")`) →
không chunk và không web thì TicketFallback, ngược lại Generation với `chunks` + `web_results` và citation có WEB.
Thêm `used_web_search: bool` vào `GraphOutput` để trace/log.

**Acceptance criteria:**
- [ ] Query đơn trượt + Tavily có kết quả → Generation, `<academic_context>` rỗng, citation WEB
- [ ] 2 sub (1 đạt, 1 trượt) → Tavily chỉ được gọi cho sub trượt, prompt có cả 2 block
- [ ] Tavily tắt/lỗi/không kết quả → TicketFallback y như hiện tại

**Verification:**
- [ ] `pytest -q tests/graph/test_graph_wiring.py tests/graph/test_streaming_session.py`
- [ ] Manual: chạy agent local với key thật, hỏi 1 câu ngoài tài liệu đã nạp nhưng có trên website trường

**Dependencies:** Task 2, 3, 4, 5

**Files likely touched:**
- `app/graph/streaming_graph.py`
- `app/graph/streaming_state.py`
- `tests/graph/test_graph_wiring.py`

**Estimated scope:** Medium

## Checkpoint: Core
- [ ] Toàn bộ test suite xanh, ruff sạch, mypy không tăng
- [ ] 3 kịch bản manual: query đơn trượt, multi-sub trượt một nửa, Tavily tắt
- [ ] Review với người dùng trước khi sang frontend

---

## Task 7: unisage-web — chip WEB mở link

**Description:** `citationSchema` thêm `url: z.string().url().nullish()`; chip/nhóm citation có `sourceType === "WEB"`
mở `url` ở tab mới (`rel="noopener noreferrer"`) thay vì mở drawer tài liệu; nhóm theo `url` thay vì `documentId`.

**Acceptance criteria:**
- [ ] Citation WEB không bị zod strip mất `url`
- [ ] Bấm chip WEB mở link tab mới; chip tài liệu vẫn mở drawer như cũ
- [ ] Hai `[n]` cùng một url gộp thành một nhóm

**Verification:**
- [ ] `npm run test -- src/features/chat` và `npm run lint && npm run build` (unisage-web)

**Dependencies:** Task 5 (định dạng citation)

**Files likely touched:**
- `src/features/chat/schemas/chat-schemas.ts`
- `src/features/chat/utils/citations.ts` (+ test)
- `src/features/chat/components/citation-chips.tsx`
- `src/features/chat/components/source-panel.tsx`

**Estimated scope:** Medium

## Checkpoint: Complete
- [ ] Mọi acceptance criteria đạt
- [ ] PR agent + PR web, không attribution, không Jira key trong commit
