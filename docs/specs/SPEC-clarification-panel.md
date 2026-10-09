# Spec: clarification-panel (unisage-agent + endpoint nội bộ unisage-backend)

> **Status: Implemented** (UNISAGE-99, nhánh `feature/huydh-unisage-99-calculation-flow`). Rủi ro còn
> lại nằm ở `unisage-agent/docs/specs/known-gaps.md`.

Gồm hai module trong [capability map](../../changes/09-10-2026-calculation-flow/capability-map.md):
`clarification-panel` (agent) và `message-metadata` (backend, mục 7). Spec này thay thế cơ chế
hỏi lại cũ: form JSON nằm trong text + Clarification Guard so khớp text.

## Objective

Khi một lượt chat cần thêm thông tin, dù từ luồng tính toán, luồng tư vấn hay cả hai, agent gửi về
**đúng một panel có cấu trúc**. Sinh viên trả lời hết rồi gửi một lần, hoặc huỷ. Panel phải:

- là contract chặt: server tự validate mọi câu trả lời, không tin bất cứ thông tin nào về câu hỏi
  (origin, kind, options) do client gửi lên;
- có đúng một source of truth và chống được việc submit panel cũ hoặc submit hai lần;
- không bao giờ để JSON nội bộ lộ ra token stream hay nằm trong `content` được lưu;
- dựng lại được sau khi reload trang;
- triển khai được mà không làm hỏng lịch sử chat cũ.

## 1. Data model

Tất cả model Pydantic trong `app/schemas/clarification.py` đều để `model_config = ConfigDict(extra="forbid")`.

### 1.1 Câu hỏi (server-side, đầy đủ)

```python
QuestionKind = Literal["choice", "number", "number_list", "number_or_list", "text", "course_table"]
# number_or_list: sinh viên chọn "Nhập sẵn" (gửi `number`) hoặc "Nhập từng cột" (gửi `numbers`)

class ChoiceOption(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    label: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=200)
    recommended: bool = False

class NumberConstraint(BaseModel):
    min: Decimal
    max: Decimal
    step: Decimal                       # 0.01 (điểm) hoặc 1 (tín chỉ)
    unit: str | None = Field(default=None, max_length=20)

class Question(BaseModel):
    id: str = Field(pattern=r"^q[1-9][0-9]?$")  # do graph gán: q1, q2, ...
    origin: Literal["advisory", "calculation"]
    task_id: str = Field(pattern=r"^T[1-3]$")     # task của lượt gốc
    field: str = Field(max_length=64)             # tên metadata (advisory) hoặc tên param (calculation)
    tab_label: str = Field(min_length=1, max_length=24)
    prompt: str = Field(min_length=1, max_length=200)
    kind: QuestionKind
    options: list[ChoiceOption] = []              # choice: 2..12; các kind khác: []
    allow_other: bool = False                     # chỉ cho choice
    number: NumberConstraint | None = None        # number, number_list (ràng buộc cho từng phần tử)
    max_items: int | None = Field(default=None, ge=1, le=30)   # number_list ≤ 20, course_table ≤ 30
    max_length: int | None = Field(default=None, ge=1, le=200) # text
```

Mỗi `kind` có một model validator riêng để bắt buộc đúng tổ hợp field. Ví dụ `choice` phải có 2..12
options và không có `number`; `number` phải có `number` và `options` rỗng.

Cột của `course_table` cố định, không cấu hình được: `name` (text, ≤ 80, không bắt buộc), `credits`
(nguyên 1..10), `score` (0..10, step 0.01, hoặc điểm chữ `A+..F`).

### 1.2 Panel

```python
class ClarificationPanel(BaseModel):
    schema_version: Literal[1] = 1
    panel_id: UUID                       # uuid4, sinh lại cho mỗi panel mới
    questions: list[Question] = Field(min_length=1, max_length=50)   # mỗi câu hỏi = 1 tab; 50 chỉ chặn payload bất thường
```

**Bản chiếu cho client** (`PublicClarificationPanel`) là panel bỏ đi `origin`, `task_id`, `field`. Client
chỉ cần biết hiển thị cái gì, nên không có lý do gì để thấy hay gửi lại các thông tin định tuyến.

### 1.3 Pending round (state server-side)

```python
class PendingAdvisoryTask(BaseModel):
    kind: Literal["advisory"]
    task_id: str
    origin_task: ClassifiedTask
    sub_query_id: str | None = None

class PendingCalculationTask(BaseModel):
    kind: Literal["calculation"]
    task_id: str
    query: str
    plan: CalculationPlan                # formula_id cài sẵn, hoặc RetrievedFormula + provenance (xem SPEC-calculation-node)
    known_params: dict[str, JsonValue]   # tham số đã trích được ở lượt gốc

class PendingRound(BaseModel):
    schema_version: Literal[2] = 2
    panel: ClarificationPanel
    assistant_message_id: UUID           # message ASSISTANT đã hiện panel, dùng để PATCH metadata
    original_query: str
    tasks: list[Annotated[PendingAdvisoryTask | PendingCalculationTask, Field(discriminator="kind")]]
    chain_depth: int = Field(default=1, ge=1)   # số panel liên tiếp của cùng một câu hỏi gốc (không giới hạn)
    created_at: datetime
```

Bất biến được kiểm tra bằng validator: mỗi `question.task_id` phải trỏ tới đúng một phần tử trong
`tasks`, và mỗi task trong `tasks` phải có ít nhất một câu hỏi.

### 1.4 Câu trả lời (client gửi lên)

```python
class CourseRow(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    credits: int
    score: Decimal | str                 # số hoặc điểm chữ

class Answer(BaseModel):
    question_id: str = Field(pattern=r"^q([1-9]|1[0-2])$")
    option_id: str | None = None
    other_text: str | None = Field(default=None, max_length=200)
    number: Decimal | None = None
    numbers: list[Decimal] | None = Field(default=None, max_length=20)
    text: str | None = Field(default=None, max_length=200)
    rows: list[CourseRow] | None = Field(default=None, max_length=30)
    # validator: đúng một trong các field giá trị khác None

class ClarificationSubmit(BaseModel):
    action: Literal["submit"]
    panel_id: UUID
    answers: list[Answer] = Field(min_length=1, max_length=50)

class ClarificationCancel(BaseModel):
    action: Literal["cancel"]
    panel_id: UUID

class ChatStreamRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=100)
    message: str | None = Field(default=None, min_length=1, max_length=2000)
    clarification: Annotated[ClarificationSubmit | ClarificationCancel, Field(discriminator="action")] | None = None
    # validator: phải có đúng một trong hai, message hoặc clarification
```

Body request bị giới hạn **16 KB** (kiểm tra `Content-Length` và số byte đọc thật). Vượt quá thì trả
`413`.

### 1.5 Validate câu trả lời (`validate_answers(round, submit)`)

Mọi kiểm tra đều đối chiếu với `round.panel` đã lưu ở server, không dùng bất cứ thông tin gì về câu
hỏi từ client:

1. Mỗi `question_id` của panel có **đúng một** câu trả lời. Không được thiếu, trùng hay có id lạ.
2. Field giá trị phải khớp `kind` của câu hỏi:
   - `choice`: dùng `option_id` (phải có trong options) hoặc `other_text` (chỉ khi `allow_other`;
     không được rỗng sau khi trim).
   - `number`: giá trị nằm trong `[min, max]` và là bội của `step`.
   - `number_list`: có 1..`max_items` phần tử, mỗi phần tử thoả ràng buộc.
   - `text`: không rỗng sau khi trim, không dài quá `max_length`.
   - `course_table`: có 1..`max_items` dòng, mỗi dòng có `credits` hợp lệ, `score` là số hợp lệ hoặc
     điểm chữ hợp lệ.
3. Kết quả trả về là `dict[question_id, NormalizedAnswer]`, hoặc ném `ClarificationInvalid` kèm
   `errors: [{question_id, reason}]`.

## 2. Source of truth và vòng đời

### 2.1 Một nguồn duy nhất: `conversation_clarification_states` (agent)

- Giữ bảng hiện có và thêm 4 cột bằng một Alembic migration mới:

  | Cột | Kiểu | Ý nghĩa |
  |---|---|---|
  | `pending_status` | `VARCHAR(16) NULL`, CHECK `IN ('OPEN','PROCESSING')` | `NULL` = không có round |
  | `pending_panel_id` | `UUID NULL`, index | `PendingRound.panel.panel_id` |
  | `claim_token` | `UUID NULL` | chỉ có khi `PROCESSING` |
  | `claim_expires_at` | `TIMESTAMPTZ NULL` | lease của lần xử lý, mặc định `now() + 120s` (`CHAT_CLARIFICATION_LEASE_SECONDS`) |

  `pending_clarification` (JSON) chứa `PendingRound`. **`NULL` không bao giờ được dùng để biểu thị
  "đang xử lý"**: trạng thái đang xử lý là `PROCESSING` cùng một `claim_token` riêng.
- **`messages.metadata.clarification` bên Java là bản chiếu để hiển thị.** Agent không đọc nó để ra quyết
  định. Bản chiếu chỉ được coi là đã có khi PATCH **thành công**, và event tương ứng chỉ được gửi sau đó
  (mục 2.6, 2.4).
- `confirmed_metadata` vẫn nằm trong bảng này như cũ.

### 2.2 State machine

```
            tạo panel (2.6)                    claim (submit/cancel)
  (none) ───────────────────► OPEN ─────────────────────────────► PROCESSING(token, lease)
    ▲                          ▲                                     │
    │                          └──── restore(token) ◄── lỗi trước khi có tác dụng phụ
    │                                                                │
    └──────────── complete(token, new_round | None) ◄────────────────┘
                  (new_round != None → OPEN với panel_id mới)
```

- Mọi chuyển trạng thái từ `PROCESSING` đều có điều kiện `WHERE claim_token = :token`. Token lạ (của
  request khác, hoặc lease đã hết và bị thu hồi) thì không ghi được gì: `rowcount = 0` → log
  `clarification.claim_lost`.
- **Lease và fencing.** Mục tiêu: khi lease bị thu hồi, request cũ **chắc chắn đã dừng**, không còn ghi
  gì sang Java, SSE hay state. Có ba lớp:
  1. **Deadline cứng cho lượt đang giữ claim.** Toàn bộ phần chạy sau claim (start_turn → graph → PATCH
     finalize → complete) nằm trong `asyncio.timeout(CHAT_CLAIMED_TURN_DEADLINE_SECONDS)`, mặc định
     150 s. Hết hạn thì task bị cancel; nhánh xử lý cancel chỉ được gửi `event: error` cho chính client
     đó, không được gọi Java hay ghi state. Hiện chưa có timeout tổng cho một lượt (chỉ có timeout cho
     token đầu và cho lời gọi phụ), nên đây là setting mới.
  2. **Lease luôn dài hơn deadline.** `CHAT_CLARIFICATION_LEASE_SECONDS` mặc định 210 s. Settings bị
     từ chối lúc khởi động nếu `lease < deadline + 60`. Biên 60 s bù cho độ trễ giữa đồng hồ DB
     (`now()`) và đồng hồ process. Vì vậy, khi một request khác thấy lease đã hết hạn thì request cũ đã
     bị cancel từ ít nhất 60 s trước.
  3. **Fencing token trên mọi lần ghi state**, gồm `restore`, `complete` và `revoke_open`, đều có
     `WHERE claim_token = :token`. Trước PATCH finalize của lượt submit còn kiểm tra
     `still_owner(token)` một lần nữa; mất quyền sở hữu thì bỏ PATCH và log `clarification.claim_lost`.

  Lease hết hạn chỉ xảy ra khi process chết hoặc bị treo lâu hơn deadline + 60 s. Lần đọc kế tiếp thấy
  `PROCESSING` mà `claim_expires_at < now()` thì coi như round đã dùng, chạy
  `UPDATE ... SET pending_status = NULL ... WHERE claim_token = :old AND claim_expires_at < now()`, rồi
  mới xử lý request mới.

  **Rủi ro còn lại (chấp nhận):** process bị đóng băng hẳn (SIGSTOP, VM pause) lâu hơn lease, rồi chạy
  tiếp. Khi đó `asyncio.timeout` chưa kịp kích hoạt. Lớp 3 vẫn chặn được mọi lần ghi state và PATCH
  finalize; thứ duy nhất có thể lọt là token SSE gửi cho **chính client cũ**, vốn đã ngắt từ lâu.
- Lượt bình thường (không có round) ghi round mới bằng `upsert_open(round) WHERE pending_status IS NULL`.
  Nếu lúc đó đã có round khác (không xảy ra được trong luồng hợp lệ, vì có round thì `message` bị chặn
  ở 2.3) thì log và không ghi đè.

### 2.3 Định tuyến ở đầu lượt (thay cho `chat.py:279-295`)

State luôn được đọc **trước** `start_turn` (bỏ cách đọc song song hiện tại; chỉ tốn thêm một query theo
khoá unique).

| State | Request | Kết quả |
|---|---|---|
| không có | `message` | Lượt bình thường |
| không có | `clarification` | `409 CLARIFICATION_STALE` |
| `OPEN` | `message` | `409 CLARIFICATION_PENDING` (không kèm `panel_id`, vì đây là khoá bí mật để huỷ panel) |
| `PROCESSING` | `message` | `409 CLARIFICATION_PROCESSING` |
| `OPEN`/`PROCESSING`, `panel_id` khác | `clarification` | `409 CLARIFICATION_STALE` |
| `PROCESSING`, `panel_id` khớp | `clarification` | `409 CLARIFICATION_STALE` (đang có request khác xử lý) |
| `OPEN`, `panel_id` khớp | `submit` sai | `400 CLARIFICATION_INVALID`, `errors` là map `{question_id: reason}`, state không đổi |
| `OPEN`, `panel_id` khớp | `submit` hợp lệ | Claim, rồi chạy luồng submit (2.4) |
| `OPEN`, `panel_id` khớp | `cancel` | Claim, rồi chạy luồng huỷ (2.5) |

Mọi `4xx` ở trên đều trả về **trước** `start_turn`: không tạo message, không trừ quota.

Mã lỗi mới trong `app/core/errors/error_codes.py`:

- `CLARIFICATION_INVALID = (400, 4010, "Câu trả lời chưa hợp lệ, kiểm tra lại giúp mình.")`
- `CLARIFICATION_STALE = (409, 4091, "Câu hỏi này đã được trả lời hoặc đã huỷ.")`
- `CLARIFICATION_PENDING = (409, 4092, "Bạn cần trả lời hoặc huỷ câu hỏi bổ sung trước.")`
- `CLARIFICATION_PROCESSING = (409, 4093, "Mình đang xử lý câu trả lời trước của bạn, chờ chút nhé.")`

Row **legacy** (có `pending_clarification` mà không có `schema_version: 2`, hoặc `pending_status IS NULL`)
được coi như không có round, ghi log `clarification.legacy_dropped`, và bị ghi đè ở lần ghi kế tiếp.

### 2.4 Luồng submit

```sql
-- claim: OPEN → PROCESSING
UPDATE conversation_clarification_states
SET pending_status = 'PROCESSING', claim_token = :token, claim_expires_at = now() + :lease, updated_at = now()
WHERE conversation_id = :cid AND pending_status = 'OPEN' AND pending_panel_id = :panel_id
```

`rowcount = 1` thì request này là bên duy nhất đang giữ round. `rowcount = 0` thì trả
`409 CLARIFICATION_STALE`.

1. Claim (round vẫn còn trong row, chỉ đổi trạng thái).
2. Gọi `start_turn` với:
   - `content` là **bản tóm tắt do agent tự dựng** từ câu trả lời đã chuẩn hoá, ví dụ
     `"Khoá: K20 · Số tín chỉ LT: 2 · Điểm TH: 9, 8"`;
   - `metadata = {"clarification_answers": {...}}`, gồm `panel_id` và các cặp câu hỏi → câu trả lời đã
     có label (mục 7.1).

   Như vậy message USER và dữ liệu của card có border được tạo **trong cùng một transaction Java**, nên
   không cần PATCH nào nữa cho luồng submit.
   - `start_turn` lỗi (429, 5xx, timeout) → `restore(token)`: `PROCESSING → OPEN WHERE claim_token = :token`,
     rồi trả lỗi như hiện nay. Panel vẫn mở để sinh viên thử lại.
3. Resume: chỉ chạy các task trong `round.tasks`, với câu trả lời đã phân về đúng task theo `task_id`
   (SPEC-calculation-node §5).
4. Kết thúc lượt:
   - Graph thành công → `complete(token, new_round)`. Có `new_round` thì về `OPEN` với `panel_id` mới
     (theo thứ tự ở 2.6); không có thì về none.
   - Graph lỗi sau bước 2 → `complete(token, None)`. Round coi như đã dùng (đã chốt 09-10-2026). Message
     USER chứa bản tóm tắt đã nằm trong history, nên LLM có đủ context khi sinh viên hỏi lại.

Không có lúc nào state bị `NULL` trong khi request vẫn đang xử lý. Vì vậy một `message` gửi song song
luôn nhận `4093`, không thể lọt qua thành lượt bình thường.

### 2.5 Luồng huỷ

1. Claim (`OPEN → PROCESSING`).
2. Gọi `PATCH /internal/messages/{assistant_message_id}/clarification` với `status = "cancelled"`. Bước
   này **bắt buộc thành công**: thử lại tối đa 3 lần, backoff 200/400/800 ms.
   - Thành công → `complete(token, None)` → SSE gồm `clarification_closed` rồi `done`.
   - Vẫn lỗi → `restore(token)` (về `OPEN`) → trả `503 BACKEND_JAVA_UNAVAILABLE`. Web giữ panel và báo
     lỗi. State và bản chiếu vẫn khớp nhau (cả hai đều `open`).

Huỷ **không** gọi `start_turn`, không tạo message, không trừ quota, không gọi LLM.
`confirmed_metadata` giữ nguyên.

### 2.6 Tạo panel mới (cuối lượt)

Thứ tự trong `streaming_session.run_and_persist`. **Bản chiếu phải có trước, event mới được gửi**:

1. Graph trả về `GraphOutput` có `pending_round` (có thể None).
2. Ghi state `OPEN`: bằng `upsert_open` cho lượt thường, hoặc `complete(token, round)` cho lượt
   submit. Bước này lỗi → không có panel; PATCH finalize **không** kèm metadata panel; gửi `event: error`.
3. Gọi `PATCH /messages/{id}` (finalize hiện có, thử lại tối đa 3 lần) với `content` là text **đã redact**
   và `metadata = {"clarification": {"schema_version": 1, "status": "open", "panel": public_panel}}`.
   - Vẫn lỗi → **thu hồi round** (`DELETE` có điều kiện theo `panel_id`, về none), rồi gửi `event: error`
     như mọi lượt finalize lỗi hiện nay. Không có panel nào tồn tại ở một phía mà phía kia không có.
4. Đẩy `ClarificationItem(public_panel)` vào queue, ra `event: clarification`.
5. Gửi `done`.

Hệ quả: nếu client nhận được `event: clarification` thì reload chắc chắn thấy panel, và submit chắc chắn
không bị `409` vì lý do lệch state.

Không có giới hạn chuỗi panel (đã chốt 09-10-2026): còn thiếu gì thì panel kế tiếp hỏi tiếp, với
`chain_depth + 1`. Không thể hỏi vòng: thuộc tính advisory sinh viên đã trả lời (kể cả "Khác") nằm trong
`confirmed_metadata` và không bao giờ được hỏi lại; câu hỏi tính toán do code dựng nên luôn hữu hạn.
Mọi câu hỏi của một lượt nằm trong **một** panel (không cắt ở 12 tab nữa).

### 2.7 Sửa hai bug của luồng cũ

- Lượt chào không xoá round đang mở nữa, vì khi có round, mọi `message` đều bị chặn bằng `409` từ trước
  khi vào graph.
- Khi panel đang mở, tin nhắn không còn bị ép đi luồng advisory, vì lý do tương tự.

## 3. SSE contract

| Event | Data | Khi nào |
|---|---|---|
| `token` | chuỗi JSON (như cũ, **đã redact**) | trong khi stream |
| `clarification` | `PublicClarificationPanel` | tối đa 1 lần, sau token cuối, trước `done` |
| `clarification_closed` | `{"panel_id": "...", "status": "cancelled"}` | chỉ có trong response của lệnh huỷ |
| `warning`, `error`, `done` | như cũ | như cũ |

Thêm `ClarificationItem` và `ClarificationClosedItem` vào `app/graph/queue_items.py`. Contract được ghi
thành `contracts/chat-sse.md` để web dùng làm nguồn tham chiếu.

## 4. Lọc fence khỏi token stream (`app/graph/fence_redactor.py`)

Luồng advisory vẫn để LLM báo thiếu thông tin bằng ```` ```json {"type":"ask_user_form",...} ``` ````
và ```` ```json {"type":"confirmed_metadata",...} ``` ````. Hai khối này **không bao giờ** được tới
client hay nằm trong `content` được lưu.

```python
class FenceRedactor:
    def feed(self, chunk: str) -> str: ...           # trả phần được phép hiện
    def finish(self) -> tuple[str, list[dict]]: ...  # (phần còn lại được phép hiện, các khối đã bắt)
```

- Máy trạng thái gồm `TEXT → MAYBE_OPEN → IN_FENCE → TEXT`.
  - Ở `TEXT`: nếu cuối chunk là tiền tố có thể của ```` ```json ```` (1–7 ký tự) thì **giữ lại**,
    chưa xả ra, để xử lý trường hợp fence bị chia giữa hai chunk.
  - Mở fence bằng ```` ```json ```` (không phân biệt hoa thường, cho phép khoảng trắng) thì bắt nội
    dung cho tới ```` ``` ```` đóng.
  - Khi đóng: parse JSON. Nếu `type ∈ {ask_user_form, confirmed_metadata}` thì nuốt khối đó và
    ghi lại; các khối khác (code block thật, JSON khác) được xả ra nguyên văn.
  - Fence không phải `json` thì xả ra ngay.
- `finish()`: một fence chưa đóng mà có chứa `"ask_user_form"` hoặc `"confirmed_metadata"` thì nuốt;
  còn lại thì xả ra.
- Được bọc quanh `token_sink` của mọi lần gọi generation (`run_generation_synthesis`, multi-intent).
  `response_text` được ghép từ phần đã redact. Khối do `_repair_missing_ask_form` sinh ra đi thẳng
  vào phần đã bắt, không bao giờ qua sink.
- Các khối đã bắt được chuyển thành câu hỏi `origin="advisory"`: `ask_user_form.fields[]` thành
  `choice` có `allow_other=True`. Field thiếu options, hoặc có dưới 2 options, thì bị bỏ (giữ nguyên
  hành vi hiện nay). Khối `confirmed_metadata` vẫn đi qua `collect_confirmed_metadata_updates`.
- `build_missing_metadata_block` gửi cả **label thật** của option cho LLM (sửa lỗi hiện tại là label bị
  thay bằng id).

## 5. Gỡ bỏ

Các phần sau bị xoá kèm test tương ứng:

- `resolve_clarification_guard`, `_match_reply`, `ClarificationGuardResult`
- `retry_count`, `CHAT_CLARIFICATION_MAX_RETRY`
- `_carry_forward_unanswered`, `_resume_retrieval_query`
- PendingClarification v1

Logic resume thật sự được viết lại theo `PendingRound` (SPEC-calculation-node).

## 6. Tương thích ngược và thứ tự triển khai

- **Message cũ** chứa fence trong `content`: web giữ một đường đọc legacy chỉ để hiển thị (xem
  SPEC-clarification-panel-ui). Không migrate dữ liệu Java.
- **State cũ** (v1): bị bỏ qua như mục 2.2.
- **Thứ tự triển khai:** backend → web → agent.
  - Web mới chạy với agent cũ: không có event `clarification`, form cũ vẫn hiển thị nhờ đường legacy.
  - Không được triển khai agent trước web. Web cũ không hiện được panel, nên sinh viên sẽ bị `409` mãi.

## 7. Backend (`message-metadata`)

### 7.1 `POST /messages/turn` nhận thêm `metadata` (cho message USER)

- `StartTurnRequest` có thêm `Map<String, Object> metadata` (nullable). Lưu vào message USER trong cùng
  transaction.
- Chỉ chấp nhận đúng một key `clarification_answers`; key khác trả `400`. Phần serialize tối đa
  **32 KB**.
- Shape của `clarification_answers` nằm ở `contracts/chat-sse.md` §metadata. Java chỉ kiểm tra kích
  thước và tên key, không kiểm tra nội dung (nội dung đã được agent validate).
- Endpoint này đi qua route master nên client có thể tự gọi. Tệ nhất thì sinh viên tự làm giả card của
  **chính mình** (chỉ là hiển thị, vẫn trừ quota như mọi lượt). Không có cờ nào ở đây ảnh hưởng tới quota
  hay quyền.

### 7.2 `PATCH /internal/messages/{id}/clarification` (chỉ dùng cho huỷ)

- Nằm dưới `/internal/**`. `InternalSecretFilter` đã bảo vệ namespace này, và gateway chặn nó từ bên
  ngoài (`InternalPathBlockFilter`).
- Body: `{"conversationId": "uuid", "status": "cancelled"}`, có `@Valid`.
- Hành vi:
  - Message phải tồn tại, là `ASSISTANT`, thuộc `conversationId`, và có `metadata.clarification`.
    Không thoả thì trả `404`.
  - Chỉ cho chuyển `open → cancelled`. Gửi lại khi đã `cancelled` là idempotent (`200`). Mọi chuyển
    trạng thái khác trả `409`.
  - Chỉ đổi `metadata.clarification.status`, giữ nguyên các key khác. Không đụng tới `content`,
    `status` hay usage.

## Commands

```
Agent:   .venv/bin/pytest -q --ignore=tests/e2e
         .venv/bin/ruff check app tests && .venv/bin/ruff format --check app tests && .venv/bin/mypy app
         .venv/bin/alembic upgrade head
Backend: ./mvnw test -Dtest='MessageControllerTest,MessageServiceImplTest,*Clarification*'
```

## Testing Strategy

- **Schema:** mỗi `kind` có tổ hợp hợp lệ và không hợp lệ; `extra="forbid"`; giới hạn độ dài và số
  lượng.
- **`validate_answers`:** thiếu, trùng, id lạ, sai kind, ngoài khoảng, sai step, chọn "khác" khi
  `allow_other=False`, bảng 31 dòng.
- **Deadline và fencing:** lượt submit chạy quá deadline thì bị cancel, không có PATCH nào, state không đổi bởi request đó; settings có `lease < deadline + 60` thì bị từ chối khi khởi động; mất quyền sở hữu trước finalize thì bỏ PATCH.
- **Repository (state machine):**
  - Claim `OPEN → PROCESSING`; claim sai `panel_id`; hai claim đồng thời chỉ một thắng (SQLite
    file-backed).
  - `restore` và `complete` với token sai không ghi được gì; lease hết hạn thì bị dọn ở lần đọc kế
    tiếp.
  - `upsert_open` không ghi đè round đang có; row legacy bị bỏ qua.
- **API** (`tests/api/test_chat_stream_clarification.py`):
  - Đủ 9 dòng của bảng 2.3.
  - Một `message` gửi trong lúc đang `PROCESSING` nhận `4093` (mô phỏng `start_turn` chậm bằng một
    event chặn).
  - `start_turn` lỗi thì state về `OPEN` với đúng token.
  - Huỷ không gọi `start_turn` hay LLM; PATCH huỷ lỗi 3 lần thì trả `503` và state về `OPEN`.
  - Body 17 KB bị trả `413`.
- **FenceRedactor:** chia một câu trả lời mẫu có fence ở **mọi vị trí ký tự** thành 2 chunk, và cả
  từng ký tự một chunk; output hiện ra luôn giống nhau và không chứa `ask_user_form`. Thêm các ca:
  code block thật được giữ lại, fence chưa đóng, hai fence liên tiếp.
- **Session:**
  - Thứ tự đúng là ghi state → PATCH finalize kèm metadata → event `clarification`.
  - PATCH finalize lỗi thì round bị thu hồi và không có event `clarification`.
  - Ghi state lỗi thì không có metadata panel.
  - `content` lưu sang Java không có fence.
- **Backend:**
  - `StartTurnRequest.metadata`: lưu vào message USER; key lạ trả `400`; quá 32 KB trả `400`.
  - Endpoint huỷ: thiếu secret bị chặn; sai conversation, message USER hoặc không có panel đều trả
    `404`; `cancelled` hai lần là idempotent; chuyển trạng thái khác trả `409`.

## Boundaries

- **Always:** validate theo panel đã lưu; mọi `4xx` của panel trả về trước khi gọi `start_turn`; chỉ gửi
  event `clarification` sau khi cả state lẫn bản chiếu đều đã ghi thành công; mọi chuyển trạng thái từ
  `PROCESSING` đều kèm điều kiện `claim_token`.
- **Ask first:** thêm `kind` câu hỏi mới; nâng giới hạn 50 câu hỏi hoặc 30 dòng; đổi thứ tự triển khai.
- **Never:** dùng `NULL` để biểu thị "đang xử lý"; đọc `origin`/`kind`/`options` từ client; để fence `ask_user_form` lọt vào token hoặc
  `content`; cho lệnh huỷ đi qua `start_turn`; thêm endpoint public mới cho panel.

## Success Criteria

- [ ] Một lượt có cả câu hỏi calculation và advisory chỉ sinh ra **một** event `clarification`, mỗi
      câu hỏi trỏ đúng task.
- [ ] Submit hai lần liên tiếp: lần đầu chạy, lần sau nhận `409 CLARIFICATION_STALE`. Chỉ có đúng một
      message USER được tạo.
- [ ] Một `message` gửi trong lúc submit đang xử lý nhận `4093`, không chạy thành lượt bình thường.
- [ ] Huỷ không tạo message và không đổi usage của người dùng; huỷ lỗi thì panel vẫn mở ở cả hai phía.
- [ ] Đã nhận `event: clarification` thì reload luôn thấy panel (PATCH lỗi thì không có event).
- [ ] Không có test nào thấy chuỗi `ask_user_form` trong token gửi ra hay trong `content` gửi sang Java.
- [ ] Reload sau khi có panel: metadata của message ASSISTANT cuối có `status="open"` và đúng `panel_id`
      đang lưu trong state.
- [ ] Toàn bộ test agent và backend xanh, ruff/mypy không vượt baseline.

## Decisions (09-10-2026)

- Mỗi câu hỏi là một tab; một panel chứa mọi câu hỏi của lượt đó (không cắt; 50 chỉ chặn payload bất thường).
- Graph lỗi sau khi claim thì không khôi phục round; history đã đủ context.
- ~~Chuỗi panel tối đa 3~~ - đã bỏ ngày 09-10-2026: không giới hạn số panel nối tiếp, không cắt ở 12 tab; chống hỏi vòng nhờ `confirmed_metadata`.
- (Review lần 2) State machine `OPEN/PROCESSING` + `claim_token` + lease thay cho việc dùng `NULL`. Bản
  chiếu không còn best-effort: dữ liệu card đi cùng `start_turn`; panel mới và lệnh huỷ đều phải PATCH
  thành công trước khi gửi event.
