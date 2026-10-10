# Contract: `POST /api/v1/chat/stream` + clarification panel

Đây là nguồn tham chiếu chuẩn (canonical) giữa `unisage-agent` và `unisage-web`. Nếu spec khác với file
này thì file này đúng.

Spec chi tiết: `docs/specs/SPEC-clarification-panel.md`. Schema Pydantic tương ứng: `app/schemas/clarification.py`,
`app/schemas/chat.py`. Mọi object dưới đây đều **strict**: field lạ bị từ chối.

Kiểu `Decimal` được serialize thành **string** (`"7.5"`) ở cả hai chiều. Server cũng nhận số JSON khi
client gửi lên.

Từ gateway, đường dẫn là `POST /api/v1/ai/chat/stream`; gateway rewrite thành `/api/v1/chat/stream`
của agent.

## 1. Request

Body có **đúng một** trong hai field `message` hoặc `clarification`. Body tối đa 16 KB (vượt quá trả
`413`).

```jsonc
// Lượt thường
{"conversation_id": "c-1", "message": "GPA của em bao nhiêu?"}

// Gửi câu trả lời panel
{"conversation_id": "c-1",
 "clarification": {"action": "submit", "panel_id": "7f1c…", "answers": [ /* Answer[] */ ]}}

// Huỷ panel
{"conversation_id": "c-1", "clarification": {"action": "cancel", "panel_id": "7f1c…"}}
```

### `Answer`

Mỗi answer có `question_id` và **đúng một** field giá trị. Field nào hợp lệ tuỳ theo `kind` của câu hỏi:

| kind | Field giá trị | Ví dụ |
|---|---|---|
| `choice` | `option_id` hoặc `other_text` (chỉ khi `allow_other`) | `{"question_id": "q1", "option_id": "k20"}` |
| `number` | `number` | `{"question_id": "q2", "number": "6.5"}` |
| `number_list` | `numbers` (1..`max_items`) | `{"question_id": "q3", "numbers": ["9", "8"]}` |
| `number_or_list` | `number` (giá trị đã tổng hợp sẵn) **hoặc** `numbers` (từng cột, 1..`max_items`) | `{"question_id": "q6", "number": "7.3"}` hoặc `{"question_id": "q6", "numbers": ["8", "7", "7"]}` |
| `text` | `text` | `{"question_id": "q4", "text": "Chất lượng cao"}` |
| `course_table` | `rows` (1..`max_items`) | `{"question_id": "q5", "rows": [{"name": "Toán", "credits": 3, "score": "8.5"}, {"name": null, "credits": 2, "score": "B+"}]}` |

Mỗi câu hỏi của panel phải có đúng một answer: không thiếu, không trùng, không có id lạ.

## 2. Response: SSE (`text/event-stream`)

| Event | `data` | Ghi chú |
|---|---|---|
| `token` | chuỗi JSON, ví dụ `"Điểm "` | Đã được lọc: không bao giờ chứa khối `ask_user_form`/`confirmed_metadata` |
| `clarification` | `ClarificationPanel` (§3) | Tối đa 1 lần, sau token cuối, trước `done`. **Khi đã nhận event này, server bảo đảm panel đã được lưu ở cả state lẫn `messages.metadata`** |
| `clarification_closed` | `{"panel_id": "7f1c…", "status": "cancelled"}` | Chỉ có trong response của lệnh huỷ |
| `warning` | `{"code", "message"}` | Như cũ (chỉ AI admin) |
| `error` | `{"code", "message", "retryable"}` | Như cũ, tối đa 1 lần, ngay trước `done` |
| `done` | `{}` | Luôn là event cuối |

Response của lệnh huỷ chỉ có `clarification_closed` rồi `done`, không có token nào.

## 3. `ClarificationPanel` (bản client thấy)

```json
{
  "schema_version": 1,
  "panel_id": "7f1c2a9e-…",
  "questions": [
    {
      "id": "q1", "tab_label": "Khoá", "prompt": "Bạn thuộc khoá nào?", "kind": "choice",
      "options": [
        {"id": "k19", "label": "K19", "description": null, "recommended": false},
        {"id": "k20", "label": "K20", "description": "Nhập học 2020", "recommended": true}
      ],
      "allow_other": true, "number": null, "max_items": null, "max_length": null
    },
    {
      "id": "q2", "tab_label": "Điểm CK", "prompt": "Điểm cuối kỳ (thang 10)", "kind": "number",
      "options": [], "allow_other": false,
      "number": {"min": "0", "max": "10", "step": "0.01", "unit": null},
      "max_items": null, "max_length": null
    },
    {
      "id": "q3", "tab_label": "Điểm TH", "prompt": "Các cột điểm thực hành", "kind": "number_list",
      "options": [], "allow_other": false,
      "number": {"min": "0", "max": "10", "step": "0.01", "unit": null},
      "max_items": 20, "max_length": null
    },
    {
      "id": "q4", "tab_label": "Các môn", "prompt": "Nhập các môn để tính GPA", "kind": "course_table",
      "options": [], "allow_other": false, "number": null, "max_items": 30, "max_length": null
    }
  ]
}
```

- `id` có dạng `q1`..`q99`. Mỗi câu hỏi là một tab, thứ tự trong mảng là thứ tự tab. **Panel hiện đủ mọi câu hỏi của lượt** (không cắt); thanh tab phải cuộn ngang được. Server chặn ở 50 câu chỉ để từ chối payload bất thường.
- Bất biến theo `kind`:

  | kind | `options` | `allow_other` | `number` | `max_items` | `max_length` |
  |---|---|---|---|---|---|
  | `choice` | 2..12 | có thể true | null | null | null |
  | `number` | [] | false | bắt buộc | null | null |
  | `number_list` | [] | false | bắt buộc (cho từng phần tử) | 1..20 | null |
  | `number_or_list` | [] | false | bắt buộc (cho giá trị tổng hợp và từng cột) | 1..20 | null |
  | `text` | [] | false | null | null | 1..200 |
  | `course_table` | [] | false | null | 1..30 | null |

- `number_or_list`: mặc định hiện danh sách ô cho từng cột (gửi `numbers`); một dòng link bên dưới cho
  đổi sang một ô nhập sẵn giá trị tổng hợp (gửi `number`) và ngược lại. `prompt` ngắn, VD
  "Điểm thường xuyên (các cột TX), thang 10".
- Cột của `course_table` cố định: `name` (string ≤ 80 hoặc null), `credits` (nguyên 1..10), `score`
  (`"0"`..`"10"` step 0.01, hoặc một trong `A+ A B+ B C+ C D+ D F`).
- Client **không** nhận và **không** gửi `origin`, `task_id`, `field`.

## 4. Lỗi HTTP (trước khi stream, envelope `{code, message}` như hiện nay)

| HTTP | `code` | Khi nào | Web xử lý |
|---|---|---|---|
| 400 | 4010 `CLARIFICATION_INVALID` | Câu trả lời sai; `errors` là map `{"q2": "lý do", ...}` theo `question_id` (cùng dạng `errors` của mọi lỗi khác) | Hiện lỗi dưới đúng tab, panel vẫn mở |
| 409 | 4091 `CLARIFICATION_STALE` | Panel đã được trả lời hoặc huỷ, `panel_id` cũ, hoặc đang có request khác xử lý | Đóng panel, toast, refetch messages |
| 409 | 4092 `CLARIFICATION_PENDING` | Gửi `message` khi panel đang mở. Body **không** kèm `panel_id`: lỗi này trả về trước khi Java kiểm tra quyền sở hữu, mà `panel_id` là khoá bí mật để huỷ panel | Refetch messages, panel tự hiện lại |
| 409 | 4093 `CLARIFICATION_PROCESSING` | Gửi `message` khi câu trả lời trước đang được xử lý | Toast "đang xử lý", giữ composer khoá |
| 413 | 4131 `REQUEST_TOO_LARGE` | Body > 16 KB | Toast lỗi chung |
| 502 | 5004 `BACKEND_JAVA_UNAVAILABLE` | Huỷ không ghi được trạng thái (mã lỗi sẵn có trong `error_codes.py`) | Giữ panel, toast "thử lại" |

## 5. `messages.metadata` (đọc qua `GET /messages/conversation/{id}` của backend)

### Message ASSISTANT đã hiện panel

```json
{"clarification": {"schema_version": 1, "status": "open", "panel": { /* ClarificationPanel */ }}}
```

`status` chỉ có hai giá trị: `open` hoặc `cancelled`. Panel đang mở **khi và chỉ khi** message cuối cùng
của cuộc chat là ASSISTANT, `COMPLETED`, và có `clarification.status == "open"`.

### Message USER của lượt submit (dữ liệu cho card có border)

```json
{"clarification_answers": {
  "schema_version": 1,
  "panel_id": "7f1c2a9e-…",
  "items": [
    {"question_id": "q1", "tab_label": "Khoá", "prompt": "Bạn thuộc khoá nào?", "kind": "choice",
     "option_id": "k20", "display": "K20"},
    {"question_id": "q3", "tab_label": "Điểm TH", "prompt": "Các cột điểm thực hành", "kind": "number_list",
     "display": "9, 8"},
    {"question_id": "q4", "tab_label": "Các môn", "prompt": "Nhập các môn để tính GPA", "kind": "course_table",
     "display": null,
     "rows": [{"name": "Toán", "credits": 3, "score": "8.5"}, {"name": null, "credits": 2, "score": "B+"}]}
  ]
}}
```

- Agent dựng object này từ câu trả lời đã validate và gửi kèm `start_turn`, nên nó được lưu cùng message
  USER trong một transaction. `content` của message USER là bản tóm tắt text cùng nội dung, dùng cho
  history và cho client cũ.
- `display` là text đã định dạng cho mọi kind trừ `course_table`; với `course_table` thì `display = null`
  và có `rows`.
- `option_id`: chỉ có với `kind == "choice"`. Là id của option được chọn, hoặc `null` khi sinh viên chọn
  "Khác" (khi đó `display` là text đã nhập). Các kind khác không có field này.
- Message USER có `clarification_answers` thì web hiện thành card có border thay cho bong bóng.

### Cách hiển thị card (theo mẫu UI của Claude)

- **Header thu/mở được:** `Đã trả lời · N câu hỏi ⌃`, mặc định mở.
- **Thân card**, mỗi câu hỏi gồm `prompt` và:
  - `choice`: **toàn bộ options** của câu đó, mỗi option có label, description và "(Đề xuất)". Option đã
    chọn có radio đầy; các option khác mờ đi. Chọn "Khác" thì có thêm một dòng "Khác: <display>" ở trạng
    thái đã chọn. Options lấy từ `metadata.clarification.panel` của message ASSISTANT **ngay trước**
    message USER này (đối chiếu bằng `panel_id` và `question_id`). Không tìm thấy panel thì chỉ hiện
    `display`.
  - `number` / `number_list` / `number_or_list` / `text`: hiện `display` (với `number_or_list`, `display` ghi rõ cách nhập: `"Nhập sẵn: 7.3"` hoặc `"Từng cột: 8, 7, 7"`).
  - `course_table`: bảng `rows`.
- **Panel bị huỷ** (message ASSISTANT có `metadata.clarification.status == "cancelled"`): ngay dưới câu
  trả lời của message đó, hiện card thu gọn `Đã huỷ · N câu hỏi ⌄`. Mở ra thì thấy các câu hỏi và
  options, không có lựa chọn nào được đánh dấu.

## 5b. Trace tính toán và phản hồi Đúng/Sai

### `metadata.calculation` trên message ASSISTANT

Message **chỉ** chứa các field dưới đây. Trace đầy đủ (câu hỏi gốc, điểm số, biểu thức, câu trích, model,
prompt version) nằm ở bảng `calculation_traces` bên backend, chỉ staff xem được qua ticket. Trace không bao
giờ xuất hiện trong response của message hay trong SSE.

```json
{"calculation": {"schema_version": 1, "items": [
  {"item_id": "T1", "run_id": "…", "mode": "llm", "status": "computed",
   "result_summary": null,
   "source_summary": {"title": "QĐ-123.pdf", "heading": "Chương II › Điều 8"}}
]}}
```

- `mode` là `builtin` (Python tính xuôi 3 công thức cài sẵn) hoặc `llm` (LLM tự tính: câu hỏi ngược,
  công thức trong tài liệu...). `result_summary` chỉ có với `builtin`.
- `status` là một trong `computed`, `needs_input`, `unresolved`.
- Web hiện nút **Đúng / Sai** cho item có `mode == "llm"` và `status == "computed"`. Khối markdown tương
  ứng trong `content` bắt đầu bằng dòng `**Kết quả do AI tự tính, có thể sai - bạn kiểm tra lại giúp mình
  nhé**`. Nút được gắn sau khối, theo thứ tự `items`.

### `metadata.calculation_feedback` (do backend ghi)

```json
{"calculation_feedback": {"T1": {"verdict": "WRONG", "reason": "WRONG_RESULT", "at": "2026-10-09T…Z"}}}
```

### `POST /api/v1/master/messages/{messageId}/calculation-feedback` (backend, qua gateway)

```json
// request (≤ 2 KB)
{"itemId": "T1", "verdict": "CORRECT", "reason": null, "note": null}
{"itemId": "T1", "verdict": "WRONG", "reason": "WRONG_FORMULA", "note": "Quy chế 2024 đã đổi"}

// response: envelope {code: 1000, data, message}
{"itemId": "T1", "verdict": "WRONG", "reason": "WRONG_FORMULA", "ticketCreated": true}
```

- `reason`: phải là `null` khi `CORRECT`. Khi `WRONG` thì bắt buộc là một trong `WRONG_FORMULA`,
  `WRONG_RESULT`, `WRONG_SOURCE`, `MISSING_INFO`, `OTHER`.
- `note`: tối đa 500 ký tự; bắt buộc khi `reason == "OTHER"`.
- `400`: sai validation. `404`: message không thuộc người gọi, hoặc không có phần tử `llm` đã tính.
  `409`: ticket của phần tử đã được xử lý xong, không đổi phản hồi được nữa.
- Guest gửi được, nhưng `ticketCreated` luôn là `false`.
- Mỗi item sai (của user đã đăng nhập) có **một ticket riêng**. T1 và T2 cùng sai thì có 2 ticket. Report
  thường của message vẫn tạo được song song.
- `note` chỉ được lưu trong ticket, không ghi vào metadata của message.

## 6. Thay đổi so với contract cũ

- Bỏ khối ```` ```json ask_user_form ```` trong `content`/token. Web chỉ còn đọc khối này trong message
  cũ, ở dạng read-only.
- Bỏ quy ước trả lời form bằng text `"<label>: <option>. …"`.
- `message` không còn bắt buộc; khi không có `message` thì phải có `clarification`.
