# Plan: Đồng hồ và trạng thái tiến trình của lượt chat

Hai repo: `unisage-agent` (SSE) và `unisage-web` (giao diện). Gateway chỉ chuyển tiếp
`/api/v1/ai/chat/stream` → agent, nên backend Java không đổi.

## Mục tiêu

1. **Đồng hồ:** từ lúc tin nhắn của người dùng vào khung chat, bong bóng trả lời hiện thời gian
   chạy tăng dần dạng `X,Xs` (dấu phẩy thập phân, cập nhật mỗi 0,1s) cho tới khi lượt kết thúc
   (`done` / `error` / người dùng bấm dừng).
2. **Trạng thái:** agent gửi event SSE `status` cho biết đang làm gì, web hiện thay cho dòng
   "Đang suy nghĩ..." cố định hiện nay:

   | `stage` | Hiển thị | Node |
   |---|---|---|
   | `analyzing` | Đang phân tích | 03 classify, 04 routing, 06 query transformation |
   | `retrieving` | Đang tìm thông tin | 08 retrieval, 09 rerank điểm, 09a LLM rerank |
   | `calculating` | Đang tính toán | 07 calculation, 07a calculation LLM |
   | `web_searching` | Đang tìm kiếm web / Đang tìm kiếm trên `<URL>` | 09b web search, 09c web judge |
   | `generating` | Đang tạo câu trả lời | 10 generation |
   | `thinking` | Đang suy nghĩ | mọi node khác (01, 02, 04 social chat, 05, 11) |

## Quyết định

- **Một event mới `status`**, `data` = `{"stage": "<stage>"}` hoặc
  `{"stage": "web_searching", "url": "<url>"}`. Agent gửi mã `stage`, web tự đặt chữ hiển thị (chữ
  là việc của giao diện; stage lạ → "Đang suy nghĩ"). Client cũ bỏ qua event lạ, nên không phá
  tương thích.
- **Nguồn sự kiện là `GraphTrace.node()`**, nơi mọi node đã được gọi khi bắt đầu (log `node=...`).
  `GraphTrace` nhận thêm callback `on_status`; `run_and_persist` (chủ của queue SSE) ánh xạ tên node
  → `stage` (`app/graph/progress.py`) và đẩy `StatusItem` vào queue. Không phải sửa từng node.
  Stage trùng với stage vừa gửi thì không gửi lại (08 → 09 → 09a chỉ một event).
- **Từng trang web:** Tavily trả tất cả trang trong **một** lần gọi, không có bước "đang đọc trang
  X" thật. Sau khi có kết quả (trước khi bộ chấm 09c đọc), agent gửi một event `web_searching` có
  `url` cho mỗi trang ứng viên; web hiện lần lượt từng URL, mỗi URL ít nhất 0,8s, và bỏ phần còn lại
  ngay khi có stage khác (VD `generating`). Tìm web song song ở nền (speculative) không gửi status:
  người dùng chỉ thấy "Đang tìm kiếm web" khi lượt thật sự đi sang nhánh web.
- **Đồng hồ hoàn toàn ở web**, tính từ lúc bấm gửi (không chờ server). Chỉ hiện trong lượt đang
  chạy: trước token đầu nằm cạnh trạng thái ("Đang tìm thông tin · 3,2s"), khi đang stream thì
  hiện nhỏ dưới câu trả lời; lượt xong thì ẩn (lịch sử được tải lại từ server sau `done`).

## Task

### T1 - Agent: event `status` theo node

- `StatusItem(stage, url)` trong `queue_items.py`; `_sse_token_generator` phát
  `event: status`.
- `app/graph/progress.py`: `stage_for_node(name)` theo bảng trên.
- `GraphTrace(..., on_status=...)`: `node()` gọi callback; thêm `trace.web_page(url)`.
- `run_and_persist`: callback ánh xạ + bỏ trùng + `queue.put_nowait`.
- **AC:** một lượt advisory phát `status` theo thứ tự analyzing → retrieving → generating trước
  token đầu; không có hai `status` liên tiếp cùng stage và không url; test SSE generator.
- **Files:** `app/graph/queue_items.py`, `app/services/chat_stream_service.py`,
  `app/graph/streaming_session.py`, `app/core/observability/graph_trace.py`,
  `app/graph/progress.py` (mới), test.
- **Size:** M.

### T2 - Agent: từng trang web

- Trong nhánh 09b của `streaming_graph.py`, sau khi có `web_outcome.results`: `trace.web_page(url)`
  cho mỗi trang ứng viên.
- **AC:** lượt đi nhánh web phát `{"stage": "web_searching"}` rồi một event có `url` cho mỗi trang;
  lượt có speculative search nhưng không cần web không phát `web_searching`.
- **Files:** `app/graph/streaming_graph.py`, `tests/graph/test_graph_wiring.py`.
- **Size:** S.

### T3 - Contract

- `contracts/chat-sse.md` §2: thêm `status`.
- **Size:** XS.

### Checkpoint agent

- pytest (bỏ `tests/e2e`, lỗi import có sẵn), ruff, mypy không thêm lỗi.

### T4 - Web: đọc event `status`

- Schema zod `chatStatusSchema` (`stage`: string, `url`: string tùy chọn); `useChatStream` gọi
  `onStatus`.
- **AC:** test `use-chat-stream.test.ts`: `status` hợp lệ → `onStatus`; sai dạng → bỏ qua.
- **Files:** `src/features/chat/hooks/use-chat-stream.ts`,
  `src/features/chat/schemas/chat-progress-schemas.ts` (mới), test.
- **Size:** S.

### T5 - Web: tiến trình của lượt

- `useTurnProgress()`: `startedAt`, trạng thái đang hiện, hàng đợi URL (mỗi URL ≥ 0,8s, stage
  khác thì bỏ hàng đợi); `start()` / `push(status)` / `stop()`.
- `useChatWorkspace` gắn vào lượt đang chạy (`assistantMessageId`), `stop()` khi `done`/`error`/
  dừng/đổi hội thoại.
- **AC:** test hook bằng fake timers: hàng đợi URL, stage khác cắt hàng đợi, `stop` xoá.
- **Files:** `src/features/chat/hooks/use-turn-progress.ts` (mới), `use-chat-workspace.ts`, test.
- **Size:** M.

### T6 - Web: hiển thị

- `TurnProgressIndicator` (nhãn theo stage, URL rút gọn host + path, đồng hồ `X,Xs`) thay dòng
  "Đang suy nghĩ..."; `ElapsedTime` nhỏ dưới câu trả lời đang stream.
- `formatElapsed(ms)` → `"3,2s"`; `progressLabel(status)`.
- **AC:** test định dạng/nhãn; `pnpm lint && pnpm typecheck && pnpm test && pnpm build` sạch; e2e
  chụp màn hình lượt đang chạy ở 375px và 1280px (`ui-rules.md`).
- **Files:** `src/features/chat/components/turn-progress.tsx` (mới),
  `chat-conversation-content.tsx`, `chat-workspace.tsx`, `e2e/support/chat.ts`, e2e spec.
- **Size:** M.

### Checkpoint cuối

- Chạy thật với agent: thấy trạng thái đổi theo node, URL khi đi nhánh web, đồng hồ chạy.

## Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Lượt vừa tính toán vừa advisory chạy song song → stage nhảy qua lại | Thấp | Chấp nhận: hiện node bắt đầu gần nhất |
| URL dài làm vỡ layout ở 375px | Trung bình | Rút gọn host + path, `truncate`, kiểm tra ảnh chụp 375px |
| Proxy/gateway gộp buffer SSE → status đến cùng lúc | Thấp | Token hiện đã stream được qua gateway, status đi cùng đường |

## Câu hỏi mở

- Có muốn giữ thời gian trả lời ("Trả lời trong 7,6s") trên tin nhắn sau khi xong không? Hiện
  plan chỉ hiện trong lúc chạy vì sau `done` lịch sử được tải lại từ server (cần lưu vào
  `messages.metadata` nếu muốn giữ).
