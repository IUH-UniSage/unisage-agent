# Implementation Plan: Structural Chunking Metadata + Table-Safe Chunkers (unisage-agent)

**v5 (2026-09-14)** — chỉnh sau review lần 4 của người dùng trên v4. Người dùng kết luận v4 "đúng hướng,
6 thay đổi đều hợp lý, có thể bắt đầu implement theo kiến trúc hiện tại" nhưng CHƯA khuyến nghị archive —
6 điểm bổ sung, đánh dấu `[v5]`: (1) chính sách rõ ràng cho draft legacy thiếu `source_type`/`block_index`
khi bấm Embed — **chọn reject 409, yêu cầu chunk lại** (không cố embed thiếu metadata) vì không thể khôi
phục chính xác heading/page/table từ dữ liệu cũ — người dùng xác nhận **không cần quan tâm dữ liệu cũ
trong Qdrant** nên không cần migration/backfill, chỉ cần chặn đường EMBED MỚI từ 1 draft cũ; (2) canonical
merge ở Task 4.5 siết chặt hơn "so tập hợp chunk_index" — thêm kiểm tra số lượng khớp draft, không trùng
index, không thiếu/thừa, xác nhận draft thuộc đúng `document_id`/`department_id`/`object_key`, CHỈ
`content` lấy từ client; (3) validator bảng (Task 4.2) bắt buộc `row_start`/`row_end`/`row_count` đều
KHÔNG `None` cho TABLE/EXCEL_ROW (khác v4: chỉ kiểm tra IF có `row_count`) + rule ngược của header:
`header_source != MISSING` ⇒ `has_header=True`; `has_header=True` ⇒ `column_names` không rỗng;
`is_partial_row=True` ⇒ `row_part_count >= 2`; (4) `TableRowChunker` (Task 2.2) phải escape `|` và chuẩn
hoá newline trong cell khi render markdown pipe-table, kèm test round-trip cell có `|`/newline; (5) ghi rõ
chính sách race condition khi người dùng re-chunk cùng document trong lúc đang ở bước review — chọn
**last-write-wins đơn giản cho phase này**, ghi vào known-gaps thay vì thêm draft version/hash (tránh mở
rộng scope); (6) dọn plan: sửa "SourceLocator (5 field)" thành đúng "toàn bộ field"; đảo thứ tự Task 4.5
trước Task 4.4 (merge canonical PHẢI xong trước khi validate); xác nhận `unisage-web/package.json` có
`pnpm typecheck`/`pnpm test`/`pnpm lint` thật (đóng Open Question cũ); làm rõ `TableStructureError` khi lọt
tới API nên log kèm context đầy đủ, coi là lỗi invariant nội bộ (bug chunker), không phải lỗi input người
dùng.

**v4 (2026-09-14)** — chỉnh sau review lần 3 của người dùng trên v3. 2 điểm P1 + 4 điểm P2, đánh dấu
`[v4]` ở chỗ thay đổi: (1) đã stage + đọc thật `unisage-web/.../ingestion-schemas.ts` và
`use-ingest-wizard.ts`, xác nhận đúng claim: `chunkSchema` (dòng 18-22) chỉ giữ
`chunk_index`/`content`/`region_type`, mọi field cấu trúc mới sẽ bị Zod loại bỏ ngay khi
`chunkMutation`/`jobQuery` parse response, và `embeddingRequestSchema` (dòng 45-51) tái dùng đúng
`chunkSchema` đó nên `runEmbed()` (`use-ingest-wizard.ts:221-238`) gửi lên backend đúng cái đã bị cắt —
thêm **Phase 7 mới** (frontend `unisage-web`) mở rộng `chunkSchema`; (2) `POST /ingestion/embedding`
(`app/api/v1/ingestion.py:129`) nhận thẳng `request.chunks` từ client và dispatch Celery, không gọi
`validate_chunks()` — client (hoặc ai gọi API trực tiếp, bỏ qua UI wizard) có thể tự bịa
`page_start`/`source_locator`/bất kỳ field nào; **làm CẢ HAI theo đề nghị của người dùng**: gọi
`validate_chunks()` ngay trong endpoint này (Task 4.4) VÀ đổi endpoint để lấy metadata GỐC từ draft đã
lưu ở DB, chỉ nhận `content` client sửa tay từ payload (Task 4.5) — canonical-metadata-from-draft là lớp
phòng thủ chính, validate_chunks() là lớp kiểm tra thứ 2 trên chunk đã merge; (3) `block_index` đổi
default từ `0` (v3) sang **`None`** để phân biệt "chưa được chunker gán" với "region 0 hợp lệ" — validator
(Task 4.2) giờ bắt buộc `block_index is not None and block_index >= 0` cho mọi chunk qua `dispatch()`,
cùng hàng với rule bắt buộc `source_type is not None` đã có; (4) thêm chính sách rõ ràng cho edge case
`tokens(heading_prefix) >= max_tokens` (hoặc `effective_chunk_size <= overlap` ở `RecursiveChunker`):
**từ chối config bằng exception rõ ràng** (không âm thầm giảm overlap, không âm thầm vượt hard cap) — nhất
quán với văn hoá fail-fast của dự án; (5) mở rộng rule nhất quán header ở validator (Task 4.2): thêm 5 rule
con (e)-(i) theo đúng yêu cầu người dùng; (6) thay `assert` trong self-check của `TableRowChunker` (Task
2.2) bằng exception rõ ràng `TableStructureError` (không bị strip bởi `python -O`), test assert trên
exception này thay vì dựa vào `AssertionError`.

**v3 (2026-09-14)** — chỉnh sau review lần 2 của người dùng trên v2. 6 điểm đã sửa, đánh dấu `[v3]` ở chỗ
thay đổi: (1) Phase 0 không được phá test/code cũ — field mới cần default an toàn; (2) validator không thể
tái xác minh row từ `Chunk` không lưu dữ liệu thô — làm rõ trách nhiệm: `TableRowChunker` tự kiểm tra bằng
assertion + test kiệt (exhaustive), validator chỉ kiểm tra field nhất quán, không phải nguồn sự thật thứ
2; (3) prepend heading SAU khi chunk có thể làm content vượt token budget — phải trừ ngân sách TRƯỚC khi
cắt; (4) đã đọc thật `app/api/v1/chat.py` xác nhận citation hiện 100% là text tự do LLM viết, không có
structured mapping — hỏi người dùng và **đã chốt: chấp nhận citation dạng text, KHÔNG thêm structured
citation mapping trong plan này** (đó là 1 thay đổi cross-repo Java+frontend riêng, ngoài scope "chỉ
ingestion"); (5) `source_locator` (table_id/row range) chưa lên `RetrievedChunk` — bổ sung; (6) làm rõ
`header_source=EXPLICIT` của XLSX là hợp đồng sản phẩm "dòng đầu luôn là header" (behavior không đổi, chỉ
label đúng ý nghĩa), không phải tín hiệu cấu trúc verified như HTML/DOCX. Cộng 1 lưu ý known-gap không
block (bảng PDF trải nhiều trang).

**v2 (2026-09-14)** — chỉnh sau review lần 1. 7 điểm đã sửa, đánh dấu `[v2]`. Thay đổi lớn nhất: metadata
giờ phải đi tới tận citation trong prompt (Phase 6 mới), không dừng ở Qdrant; và bảng được parse thành cấu
trúc `TableBlock` (`list[list[str]]`) làm nguồn dữ liệu chính thay vì round-trip qua markdown.

## Overview

`unisage-agent`'s ingestion pipeline (`preview` → `chunking` → `embedding`) hiện chunk văn bản mà không giữ
lại cấu trúc tài liệu. Đọc code thật xác nhận đúng 3 bug người dùng nêu ban đầu (mất heading PDF/DOCX, cắt
sai hàng bảng ở 3/5 strategy, thiếu metadata cấu trúc trên `Chunk`/`DocumentChunk`/`ChunkPoint`) — xem chi
tiết ở bản v1 (không lặp lại ở đây). Sau khi review v1, người dùng chỉ ra: mục tiêu chính là **citation
theo trang**, nhưng v1 dừng lại ở Qdrant payload — `RetrievalService._to_retrieved_chunk`,
`RetrievedChunk.metadata`, và `build_prepared_context_section` (prompt) chưa đọc `page_start`/`page_end`
nào cả, nên dù Qdrant có lưu số trang, LLM vẫn không nhìn thấy. Cộng thêm 6 lỗ hổng khác (heading không
chắc xuất hiện ở mọi chunk con, header_source suy luận ngược từ markdown thay vì xác định trước, validator
không phân biệt PDF/TXT, oversized-row fallback thiếu metadata để nhận diện các phần cùng 1 hàng,
`chunking_version` default sai làm lẫn chunk cũ/mới, thiếu `table_id`/`block_index` để phân biệt 2 bảng).
Plan v2 sửa cả 7 điểm, thêm 1 phase mới (Phase 6 — citation propagation), và đổi kiến trúc parse bảng.

**Vẫn giữ nguyên từ v1 (đã chốt qua interview, không đổi):** refactor có mục tiêu, không rewrite hybrid
Document→Section→Block đầy đủ; TABLE luôn qua `TableRowChunker` chung; `page_start`/`page_end` chỉ có giá
trị thật ở PDF; `column_names` áp dụng 4 nguồn nhưng không bịa khi header không rõ; validator bắt buộc
trong pipeline thật; hybrid retrieval (BM25+vector)/parent-child retrieval vẫn ngoài scope (đồng ý với
người dùng: KHÔNG thêm ở plan này — chỉ thêm phần propagate metadata để citation *dùng được*, không phải
tự động cải thiện thuật toán retrieval).

## 7 điểm đã sửa so với v1

1. **[P1] Metadata chưa tới citation** → thêm **Phase 6**: mở rộng `RetrievedChunk`
   (`app/schemas/retrieval.py`), `_to_retrieved_chunk` (`app/rag/retrieval/service.py:45`), và
   `build_prepared_context_section` (`app/rag/prompting/builder.py:102`) để citation hiển thị được
   `page_start`/`page_end`/heading. Cũng thêm 1 task audit `app/api/v1/chat.py` (chưa đọc code này —
   không giả định tên hàm cụ thể) tìm logic map citation index `[1][2]` → nguồn trả về frontend, mở rộng
   nếu có.
2. **[P1] Heading không chắc ở mọi chunk con** → đổi kiến trúc: **bỏ việc prefix heading vào
   `region.content`** ở Phase 1 (trước đây HTML làm vậy). Thay vào đó, **mọi chunker cuối cùng
   (`RecursiveChunker`/`TokenBasedChunker`/`SemanticChunker`/`TableRowChunker`) tự prepend
   `heading_path` vào `content` của TỪNG chunk con nó tạo ra**, ngay tại điểm construct `Chunk` — không
   phụ thuộc region gốc đã có heading ở đầu hay chưa. 1 region dài bị cắt thành 3 chunk → cả 3 chunk đều
   có heading. Không dùng phương án `content`/`embedding_text` tách đôi (people đề xuất) vì phương án
   prepend-tại-chunker đạt cùng mục tiêu mà không cần field mới / không có nguy cơ 2 field lệch nhau khi 1
   trong 2 quên cập nhật.
3. **[P1] `header_source="explicit"` gán sai cho HTML/DOCX** → **đổi hẳn cách xác định header: KHÔNG suy
   luận ngược từ markdown nữa.** Bảng được parse thành `TableBlock` (dataclass nội bộ:
   `header_row: list[str] | None`, `data_rows: list[list[str]]`, `header_source`, `header_confidence`)
   **trực tiếp từ nguồn** trước khi có markdown: HTML dựa vào `<th>` có mặt hay không; DOCX dựa vào
   `<w:trPr><w:tblHeader/>` (thuộc tính "repeat as header row" thật của Word, đọc qua XML của
   `python-docx`, không phải style tên); PDF vẫn `inferred`/`0.6` (không có tín hiệu structural nào từ
   pymupdf4llm). Markdown giờ chỉ là **format hiển thị sinh ra TỪ `TableBlock`** để build `content`, không
   phải nguồn cho validator đọc ngược — giải quyết luôn điểm 6.
4. **[P1] Validator không biết chunk đến từ định dạng nào** → thêm field `source_type` (enum
   `pdf|docx|html|txt|xlsx`) vào `Chunk`, set tại `split_regions()`. Validator dùng field này để bắt buộc
   `page_start`/`page_end` non-null cho **mọi chunk có `source_type == "pdf"`** (cả TEXT lẫn TABLE), không
   chỉ TABLE như v1.
5. **[P1] Oversized-row fallback mâu thuẫn "row nguyên tử"** → làm rõ: nguyên tắc "không cắt giữa row" áp
   dụng cho **row bình thường** (luôn đúng, đây là bug chính cần fix). Row vượt cả ngưỡng cứng là **ngoại
   lệ hiếm, được đánh dấu rõ ràng**, không giả vờ là row nguyên vẹn: thêm
   `is_partial_row: bool`, `row_part: int | None`, `row_part_count: int | None` vào `SourceLocator`, cộng
   `table_id` (điểm 7) để mọi phần của cùng 1 row nối lại được. Acceptance criteria của Task 2.2 sửa lại:
   "row bình thường không bao giờ bị chia; row vượt ngưỡng cứng bị chia CHỈ KHI có đủ 3 field trên".
6. Gộp vào điểm 3 (TableBlock).
7. **[P2] Thiếu `table_id`/`block_index`** → thêm `block_index: int` vào `ParsedRegion` VÀ `Chunk` (thứ tự
   region gốc trong tài liệu — dùng làm sort key khi merge `table_chunks`/`text_chunks` ở Task 3.1, thay
   cho "region_id" nói suông ở v1 mà model không có field). `table_id = f"table-{block_index}"` (scope
   trong 1 document, không cần global unique vì luôn đi kèm `document_id` ở tầng DB/Qdrant giống cách
   `chunk_id` đang làm).
8. **[P2] `chunking_version` default che chunk cũ** → default đổi thành **`"legacy"`** (literal string, không
   phải `settings.CHUNKING_VERSION`). Mọi chunker MỚI (Task 3.2, 2.2, 2.3) phải **set tường minh**
   `chunking_version=settings.CHUNKING_VERSION` lúc construct `Chunk` — không dựa vào Pydantic default,
   vì default chạy cả khi deserialize row cũ thiếu field, gán nhầm version mới cho data cũ.

## 6 điểm đã sửa ở v3 so với v2

1. **[P1] Phase 0 sẽ làm vỡ code/test hiện tại** → `Chunk.source_type`/`block_index` và
   `ParsedRegion.block_index`/`source_type`/`heading_path`/`page_start`/`page_end`/`table` đều có default
   an toàn (`None`/`0`/`[]`), đặt SAU field gốc trong `ParsedRegion` để giữ tương thích positional-call.
   Trách nhiệm gán giá trị THẬT chuyển hết sang Phase 1 (region)/Phase 3 (chunk); validator Phase 4 mới là
   nơi bắt buộc.
2. **[P1] Validator không thể verify row thật từ `Chunk`** → làm rõ 2 lớp trách nhiệm KHÁC NHAU:
   `TableRowChunker` (nơi duy nhất còn giữ `list[list[str]]` gốc) tự assert + có test kiệt (exhaustive,
   nối lại mọi chunk con phải khớp 100% dữ liệu gốc) — đây là nơi PHÁT HIỆN lỗi cắt row. Validator ở Phase
   4 chỉ kiểm tra các field tóm tắt tự khai báo (`row_start`/`row_end`/`row_count`) có nhất quán với nhau
   — đây là nơi bắt lỗi ROUND-TRIP (DB/Qdrant), không phải nơi verify lại cấu trúc bảng. Thêm field
   `source_locator.row_count` để có 1 con số đối chiếu độc lập với `row_start`/`row_end`.
3. **[P1] Prepend heading SAU khi chunk có thể vượt token budget** → mọi chunker (`TableRowChunker` lẫn
   `RecursiveChunker`/`TokenBasedChunker`/`SemanticChunker`) phải tính token/ký tự của
   `heading_prefix` (+ `header_line` với bảng) TRƯỚC, trừ khỏi ngân sách cắt, rồi mới gom nội dung — không
   phải cắt theo ngân sách gốc rồi cộng thêm heading vào sau.
4. **[P1] Citation "theo trang" chưa chắc là citation có cấu trúc** → đã đọc thật `app/api/v1/chat.py`
   (dòng 146-263), xác nhận không có structured citation mapping nào. Hỏi người dùng, **chốt: chấp nhận
   citation dạng text**, KHÔNG mở rộng thành structured citation API trong plan này (đó là 1 thay đổi
   cross-repo Java+frontend riêng).
5. **[P2] `source_locator` chưa lên `RetrievedChunk`** → thêm field `source_locator: SourceLocator | None`
   vào `RetrievedChunk`, đọc từ Qdrant payload giống các field khác.
6. **[P2] XLSX `header_source=EXPLICIT` dễ hiểu nhầm là "verified"** → làm rõ đây là HỢP ĐỒNG sản phẩm có
   sẵn ("dòng đầu luôn là header"), không phải tín hiệu cấu trúc xác minh được như HTML `<th>`/DOCX
   `<w:tblHeader>` — không đổi hành vi, chỉ sửa docstring + ghi known-gap.

**Điểm bổ sung không block**: bảng PDF trải dài qua nhiều trang hiện có `table_id` khác nhau theo từng
trang (vì `block_index` đổi theo trang) dù thực tế là 1 bảng logic — ghi vào known-gaps, cần
`table_group_id` nếu muốn merge chính xác ở Phase sau (không làm ở đây, không block Phase 1-6 hiện tại).

## Data Model v5 (thay thế toàn bộ Data Model ở v4 — không đổi field/type nào thêm, chỉ sửa chính sách
đọc/ghi ở Task 4.2/4.5, xem điểm sửa `[v5]` ở phần Task List)

```python
# app/schemas/ingestion.py

class HeaderSource(StrEnum):
    EXPLICIT = "explicit"   # tín hiệu header THẬT từ nguồn: HTML <th>, DOCX <w:tblHeader>, XLSX header row
    INFERRED = "inferred"   # không có tín hiệu structural, suy luận row đầu là header (PDF luôn rơi đây)
    MISSING = "missing"     # không xác định được -> column_names = None

class SourceType(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"
    XLSX = "xlsx"

class SourceLocator(BaseModel):
    section: str | None = None        # DOCX/HTML/TXT: heading_path nối " > ", None nếu không có heading
    sheet_name: str | None = None     # XLSX
    row_start: int | None = None      # XLSX hoặc TABLE chunk: dòng dữ liệu đầu (1-indexed, không tính header)
    row_end: int | None = None        # XLSX hoặc TABLE chunk: dòng dữ liệu cuối
    row_count: int | None = None      # [v3] số row dữ liệu THẬT trong chunk này — validator (Phase 4)
                                       #      đối chiếu row_end - row_start + 1 == row_count làm phép kiểm
                                       #      tra nhất quán (KHÔNG phải xác minh nội dung từng cell — xem
                                       #      Task 2.2/4.2 về giới hạn thật của việc "verify" ở tầng Chunk)
    table_id: str | None = None       # [v2] "table-{block_index}" — phân biệt 2 bảng cùng page/heading
    row_part: int | None = None       # [v2] 1-indexed, chỉ set khi 1 row bị chia (oversized fallback)
    row_part_count: int | None = None # [v2]
    is_partial_row: bool = False      # [v2]

class Chunk(BaseModel):
    chunk_index: int
    content: str                            # [v2] LUÔN có heading prepend ở MỌI chunk, do chunker tự làm
    region_type: RegionType
    source_type: SourceType | None = None   # [v3] default None (khác v2: required) — mọi constructor cũ
                                             #      (test, code hiện có) không set field này vẫn chạy được;
                                             #      chunker MỚI (Phase 3) luôn set tường minh; validator
                                             #      (Phase 4) coi None là lỗi CHO CHUNK ĐI QUA dispatch(),
                                             #      không áp cho test đơn vị dựng Chunk tay ở Phase 0-2
    block_index: int | None = None          # [v4] default None (khác v3: 0) — 0 là giá trị HỢP LỆ cho
                                             #      region đầu tiên, nên không thể dùng làm "sentinel chưa
                                             #      gán" được; None mới phân biệt được rạch ròi "chunker
                                             #      chưa gán" (Phase 0-2, trước Phase 3) với "gán rồi và
                                             #      đúng là region 0" (sau Phase 3) — validator (Task 4.2)
                                             #      bắt buộc `block_index is not None and >= 0` cho MỌI
                                             #      chunk qua `dispatch()`, cùng hàng với `source_type`
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_locator: SourceLocator | None = None
    column_names: list[str] | None = None
    has_header: bool = False
    header_source: HeaderSource = HeaderSource.MISSING
    header_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    chunking_version: str = "legacy"        # [v2] KHÔNG dùng settings.CHUNKING_VERSION làm default nữa
```

`TableBlock` (nội bộ, KHÔNG phải API schema — sống trong `app/rag/ingestion/table_aware_parser.py`):

```python
@dataclass(frozen=True)
class TableBlock:
    header_row: list[str] | None
    data_rows: list[list[str]]
    header_source: HeaderSource
    header_confidence: float
```

`ParsedRegion` mở rộng — **[v3] mọi field mới đặt SAU 2 field gốc (`region_type`, `content`) và đều có
default**, để `ParsedRegion(RegionType.TEXT, text)` (cách gọi positional đang có trong test hiện tại) vẫn
chạy được nguyên vẹn qua hết Phase 0-2, chỉ Phase 1 mới thực sự gán giá trị thật:
`heading_path: list[str] = field(default_factory=list)`, `page_start`/`page_end: int | None = None`,
`block_index: int | None = None` **[v4, khác v3: `= 0`]**, `source_type: SourceType | None = None`, và
(chỉ TABLE)
`table: TableBlock | None = None`. **Không còn prefix heading vào `content`** (khác v1) — `content` của
TEXT region là text thô, heading chỉ nằm ở field `heading_path`.

## Task List

### Phase 0: Schema nền tảng (không đổi hành vi chunk, chỉ thêm field + migration)

- [ ] Task 0.1 **[v3: sửa để không phá test/code cũ; v4: `block_index` đổi default]**: Thêm `HeaderSource`,
      `SourceType`, `SourceLocator` (**[v5, sửa lại cho đúng]** toàn bộ 9 field theo Data Model:
      `section`/`sheet_name`/`row_start`/`row_end`/`row_count`/`table_id`/`row_part`/`row_part_count`/
      `is_partial_row` — v3/v4 ghi nhầm "5 field", con số đó chỉ đúng ở thời điểm v2 trước khi thêm
      `row_count` `[v3]`, không tính đủ), mở rộng `Chunk` theo
      đúng Data Model v4 — **chú ý `source_type: SourceType | None = None` và
      `block_index: int | None = None`** (v4: đổi từ `= 0` ở v3 — 0 là giá trị hợp lệ cho region đầu nên
      không dùng làm sentinel được, `None` mới phân biệt "chưa gán") **đều có default an toàn** (khác
      cách viết ở v2 coi 2 field này là required), để mọi nơi đang gọi
      `Chunk(chunk_index=0, content="...", region_type=...)` (vd `tests/test_chunking_semantic.py:34` và
      các chunker hiện tại) tiếp tục chạy được qua Phase 0-2 mà không cần sửa. Trách nhiệm gán giá trị
      THẬT cho 2 field này thuộc về Phase 1 (`block_index`/`source_type` trên `ParsedRegion`) và Phase 3
      (copy sang `Chunk`) — validator ở Phase 4 mới là nơi bắt buộc chúng khác giá trị mặc định, không
      phải Pydantic model. Thêm `CHUNKING_VERSION: str = "2026-09-structural-v1"` vào `app/core/config.py`.
- [ ] Task 0.2: Thêm cột `metadata: JSONB not null default '{}'` vào `DocumentChunk`
      (`app/database/models.py`) + Alembic migration mới trong `migrations/versions/`. Cập nhật
      `ChunkRepository.get_page` (`app/database/repositories/chunk.py`) và
      `upsert_chunking_draft`/`get_draft` (`app/database/repositories/ingestion_job.py`) để
      serialize/deserialize toàn bộ field mới của `Chunk` (trừ `chunk_index`/`content`/`region_type`) vào
      cột `metadata`. Test riêng: đọc 1 row Postgres KHÔNG có `chunking_version` trong `metadata` (giả lập
      data trước migration) → `Chunk.chunking_version == "legacy"`, không phải version mới.

**Acceptance criteria:**
- [ ] `Chunk` có đủ field theo Data Model v3, `mypy` không báo lỗi type.
- [ ] **[v3]** Mọi lệnh gọi `Chunk(chunk_index=.., content=.., region_type=..)` hiện có trong test suite
      (không sửa gì thêm) vẫn construct thành công, không raise `ValidationError` do thiếu field bắt buộc.
- [ ] **[v3]** Mọi lệnh gọi `ParsedRegion(RegionType.TEXT, text)` (positional, 2 arg) hiện có trong test
      suite vẫn construct thành công.
- [ ] Migration mới chạy `alembic upgrade head` sạch trên DB dev, có `downgrade()` tương ứng.
- [ ] Round-trip test: tạo `Chunk` với đủ field mới → `upsert_chunking_draft` → `get_draft` → khớp 100%.
- [ ] Row Postgres cũ (JSON `metadata` không có `chunking_version`) đọc lên → `chunking_version == "legacy"`.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] `pytest tests/test_ingestion_schemas.py tests/test_ingestion_job_repository.py
      tests/test_document_chunks.py` xanh.
- [ ] Manual: `alembic upgrade head` rồi `alembic downgrade -1` trên DB dev không lỗi.

**Dependencies:** None
**Files:** `app/schemas/ingestion.py`, `app/core/config.py`, `app/database/models.py`,
`migrations/versions/<new>.py`, `app/database/repositories/chunk.py`,
`app/database/repositories/ingestion_job.py`, `tests/test_ingestion_schemas.py`,
`tests/test_ingestion_job_repository.py`, `tests/test_document_chunks.py`
**Estimated scope:** Large — cân nhắc tách Task 0.2 thành migration riêng + repository mapping riêng.

### Checkpoint: Phase 0
- [ ] `document_chunks` có cột `metadata`; chunk cũ đọc lên không crash, `chunking_version=="legacy"`.
- [ ] Chưa có thay đổi hành vi chunk nào — mọi test chunking cũ vẫn xanh y nguyên.

### Phase 1: Heading path (metadata thuần, KHÔNG prefix vào content) + block_index cho mọi region

- [ ] Task 1.1: Refactor `_regions_from_html`: **bỏ việc prefix heading vào `content`** (khác v1) —
      `heading_path: list[str]` giờ CHỈ là field riêng của `ParsedRegion`. Gán `block_index` tăng dần theo
      thứ tự region xuất hiện. `source_type = SourceType.HTML`.
- [ ] Task 1.2: `_regions_from_docx`: dựng heading_stack từ `paragraph.style.name` (regex
      `^Heading (\d)$`) — giữ nguyên cách nhận diện heading của v1 (đây là chuyện khác với việc detect
      header CỦA BẢNG ở Task 2.1, đừng nhầm 2 khái niệm "heading" ngữ nghĩa văn bản và "header row" của
      bảng). `block_index`/`source_type = SourceType.DOCX` tương ứng.
- [ ] Task 1.3: PDF: đổi sang `pymupdf4llm.to_markdown(document, page_chunks=True)`, duyệt per-page,
      heading_stack chạy xuyên suốt các trang (không reset khi sang trang), gắn `page_start = page_end =
      <trang thật>`, `block_index` tăng dần, `source_type = SourceType.PDF`.
- [ ] Task 1.4: TXT: `region.heading_path = []`, `block_index = 0` (1 region duy nhất),
      `source_type = SourceType.TXT`, `page_start = page_end = None` — không đổi logic, chỉ gắn field mới.

**Acceptance criteria:**
- [ ] HTML/DOCX/PDF: `heading_path` đúng theo section chứa nó, nhưng `region.content` KHÔNG chứa dòng
      heading nào (khác v1 — test phải assert `content` không bắt đầu bằng heading text, chỉ
      `heading_path` field mới có).
- [ ] `block_index` tăng dần đúng thứ tự xuất hiện trong tài liệu gốc, không trùng trong cùng 1 document.
- [ ] PDF: heading ở trang 1 áp dụng cho text ở trang 2 (heading_path không rỗng ở trang 2); mọi region có
      `page_start == page_end == <trang thật>`.
- [ ] Không có heading nào → `heading_path == []`.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] `pytest tests/test_table_aware_parser.py tests/test_parser.py` xanh (fixture PDF 2 trang có heading,
      fixture DOCX có Heading style — cập nhật assertion cũ nào đang check heading nằm trong content).
- [ ] Manual: `POST /ingestion/preview` + `POST /ingestion/chunking` (strategy `recursive`) với 1 PDF thật
      nhiều trang có mục lục.

**Dependencies:** Task 0.1
**Files:** `app/rag/ingestion/table_aware_parser.py`, `tests/test_table_aware_parser.py`,
`tests/test_parser.py`, `tests/fixtures/`
**Estimated scope:** Large — tách theo 4 task con, review độc lập từng định dạng.

### Checkpoint: Phase 1
- [ ] `ruff check`, `mypy` sạch. Cả 4 định dạng trả `heading_path`/`block_index`/`source_type` đúng.
- [ ] `region.content` không còn heading text lẫn vào (đây là thay đổi hành vi so với `main` hiện tại —
      ghi rõ trong PR description vì HTML trước đây CÓ prefix, giờ bỏ).

### Phase 2: TableBlock (parse header thật từ nguồn) + TableRowChunker (chưa nối vào dispatch)

- [ ] Task 2.1: Viết hàm parse trực tiếp thành `TableBlock` cho từng nguồn (thay cho `parse_table_header`
      đọc markdown ngược của v1):
      - HTML (`_html_table_to_block`): duyệt `<tr>` trong `<table>`; nếu hàng đầu toàn `<th>` →
        `header_source=EXPLICIT, confidence=1.0, header_row=<cells>`; nếu không có `<th>` nào trong bảng →
        `header_source=MISSING, header_row=None`, mọi hàng vào `data_rows`.
      - DOCX (`_docx_table_to_block`): đọc `row._tr.trPr` tìm `<w:tblHeader/>` (namespace `w:`, dùng
        `qn("w:tblHeader")` như code hiện đang dùng `qn("w:tbl")`/`qn("w:p")`) trên từng `<w:tr>` — hàng có
        thuộc tính này là header → `EXPLICIT, 1.0`; không có hàng nào có → `MISSING, header_row=None`.
      - PDF: bảng do `pymupdf4llm` tự phát hiện, không có tín hiệu structural nào để kiểm chứng →
        LUÔN `header_row = <hàng đầu>, header_source=INFERRED, confidence=0.6` (hằng số tạm, ghi rõ vào
        known-gaps — xem Risks).
      - XLSX **[v3: làm rõ ý nghĩa]**: không qua hàm này — `ExcelRowChunker` hiện tại (`header =
        next(rows, None)`) LUÔN coi dòng đầu tiên là header, không có tín hiệu nào từ `openpyxl` xác minh
        điều đó (khác HTML `<th>`/DOCX `<w:tblHeader>` — 2 tín hiệu đó THẬT SỰ verified từ nguồn). Đây là
        **hợp đồng sản phẩm có sẵn từ trước** ("upload XLSX thì dòng đầu luôn là header"), không phải phát
        hiện mới của Task 2.1 — giữ nguyên hành vi (KHÔNG thêm `header_row` config được — ngoài scope đã
        chốt ở interview ban đầu), chỉ gán `EXPLICIT, 1.0` với docstring ghi rõ ý nghĩa khác HTML/DOCX:
        "chắc chắn theo HỢP ĐỒNG, không phải theo XÁC MINH cấu trúc". Ghi vào known-gaps: nếu người dùng
        upload 1 file XLSX mà dòng đầu không phải header thật, hệ thống vẫn coi là header — giới hạn đã
        biết, chấp nhận được vì đây là hành vi cũ không đổi.
      `ParsedRegion.table: TableBlock` gắn ngay tại thời điểm parse (Phase 1 các hàm), KHÔNG còn bước
      "sinh markdown rồi parse ngược" như v1.
- [ ] Task 2.2: Viết `TableRowChunker` (`app/rag/chunking/table_row.py`) hoạt động trực tiếp trên
      `TableBlock.data_rows: list[list[str]]` (không re-parse markdown). **[v3] Trừ ngân sách token TRƯỚC
      khi gom row, không phải sau**: tính `heading_prefix = " > ".join(heading_path)` và
      `header_line` (dòng header markdown) trước, rồi
      `usable_budget = max_tokens - tokens(heading_prefix + "\n\n") - tokens(header_line)`; gom row cho
      tới khi tổng token các row ĐÃ gom vượt `usable_budget` (không phải `max_tokens` thô — nếu trừ
      ngân sách SAU khi gom xong rồi mới prepend heading, `content` cuối cùng có thể vượt `max_tokens` gốc,
      đúng lỗi người dùng chỉ ra). **[v4] Chính sách edge case `usable_budget <= 0`** (heading_prefix quá
      dài do nhiều cấp `heading_path`, hoặc `header_line` của bảng quá rộng, chiếm hết/vượt `max_tokens`):
      **từ chối config, raise ngay 1 exception rõ ràng** — thêm mới `ChunkingConfigException` +
      `ErrorCode.CHUNKING_CONFIG_INVALID` (`app/core/error_codes.py`, `app/core/exceptions.py`, theo đúng
      pattern `UniSageException(ErrorCode, message, errors)` mà `ChunkValidationException` ở Task 4.1 cũng
      dùng — định nghĩa NGAY Ở ĐÂY vì Task 2.2 (Phase 2) chạy trước Task 4.1 (Phase 4) nên cần exception
      này sớm hơn; KHÔNG âm thầm giảm overlap, KHÔNG âm thầm cho phép `content` vượt hard cap) với
      message nêu rõ heading path/table header dài bao nhiêu token, `max_tokens` đang cấu hình là bao
      nhiêu, gợi ý tăng `max_tokens` hoặc rút gọn heading — nhất quán với các quyết định fail-fast khác
      trong plan này (vd điểm sửa #5 v2 với oversized row, và [v4] điểm 6 dưới đây); KHÔNG chọn 2 phương
      án còn lại đã cân nhắc (giảm overlap tự động — có thể che giấu vấn đề thật; cho phép vượt hard cap
      kèm warning log — dễ bị bỏ qua trong pipeline chạy nền, và "hard cap" mất ý nghĩa nếu có thể vượt).
      KHÔNG BAO GIỜ cắt giữa 1 row bình thường; mỗi chunk con
      **tự prepend `heading_prefix` vào đầu `content`** (điểm sửa #2) rồi render
      `header_row + rows-trong-chunk` thành markdown pipe-table để làm `content` hiển thị. **[v5] Escape
      khi render markdown pipe-table**: mọi cell (header lẫn data) PHẢI escape ký tự `|` thành `\|` trước
      khi ghép vào dòng markdown (nếu không, 1 cell chứa `|` sẽ tự tạo thêm cột giả, làm lệch số cột của
      cả hàng — đúng loại lỗi "cắt sai cấu trúc bảng" mà plan này đang sửa, chỉ khác là xảy ra ở bước RENDER
      thay vì bước CẮT); newline trong cell (`\n`/`\r\n`) PHẢI chuẩn hoá thành `<br>` (quy ước markdown
      pipe-table phổ biến, giữ được ý nghĩa "xuống dòng trong 1 cell" mà không phá cấu trúc hàng — pipe-
      table thật sự không cho phép newline sống trong 1 dòng bảng). Việc escape/chuẩn hoá này CHỈ áp dụng
      lúc render `content` hiển thị — `TableBlock.data_rows` (dữ liệu gốc dùng cho self-check nội bộ, Task
      2.2 phần trên) và `source_locator.row_start`/`row_end`/`row_count` vẫn tính trên dữ liệu THẬT trước
      escape, không bị ảnh hưởng. Set `column_names = header_row` (nếu có), `has_header`, `header_source`,
      `header_confidence` kế thừa từ `TableBlock`; `source_locator.table_id = f"table-{region.block_index}"`,
      `row_start`/`row_end` = range hàng thật trong chunk đó; `chunking_version =
      settings.CHUNKING_VERSION` (set tường minh, điểm sửa #8).
      **Oversized row** (1 row vượt token budget): giữ nguyên nếu dưới ngưỡng cứng (2x budget); vượt cả
      ngưỡng cứng → chia theo cell dài nhất thành nhiều chunk con "Phần k/n", MỌI chunk con set
      `is_partial_row=True, row_part=k, row_part_count=n` cộng `table_id`/`row_start=row_end=<số dòng
      row đó>` giống nhau ở cả n phần (điểm sửa #5 — đây là NGOẠI LỆ có đánh dấu rõ, không phải vi phạm
      ngầm nguyên tắc atomic). Set `source_locator.row_count` = số row dữ liệu thật trong chunk (đếm trực
      tiếp từ `data_rows` đưa vào chunk đó — KHÔNG đếm lại từ `content` markdown sau khi render).
      **[v3] `TableRowChunker` TỰ CHỊU TRÁCH NHIỆM đảm bảo row nguyên vẹn** — đây là nơi DUY NHẤT còn giữ
      `list[list[str]]` gốc của bảng (từ `TableBlock`), nên tự kiểm tra (fail-fast, không chỉ hy vọng test
      bắt được) mỗi khi gom 1 row vào 1 chunk: số cell của row == `len(header_row)` khi có header, row
      không bị cắt ngang trừ nhánh oversized đã đánh dấu. **[v4] Dùng exception rõ ràng, KHÔNG dùng
      `assert`**: định nghĩa `TableStructureError` (subclass `RuntimeError`, đặt trong `table_row.py`) và
      raise nó thay vì `assert` — lý do: `assert` bị Python strip hoàn toàn khi chạy với cờ `-O`
      (`PYTHONOPTIMIZE`), nên nếu production/CI vô tình bật cờ này, lớp phòng thủ "DUY NHẤT" nói trên biến
      mất không dấu vết, chunk lỗi vẫn lọt qua như không có gì xảy ra — dùng exception thường thì không có
      rủi ro đó. Test tương ứng (Task 2.2 dưới) phải `pytest.raises(TableStructureError)`, không dựa vào
      `AssertionError`. `Chunk`/`document_chunks`/Qdrant sau đó CHỈ lưu
      `content` (markdown đã render) + field tóm tắt (`row_start`/`row_end`/`row_count`/`column_names`) —
      **không lưu lại `list[list[str]]` thô** (tránh trùng lặp dữ liệu + phình payload). Hệ quả: validator
      ở Phase 4 KHÔNG THỂ và KHÔNG CÓ NHIỆM VỤ tái xác minh nội dung cell thật từ 1 `Chunk` đã persist —
      nó chỉ kiểm tra các field tóm tắt có nhất quán với nhau (xem Task 4.2). Độ tin cậy "row không bị cắt"
      đến từ test kiệt (exhaustive) của `TableRowChunker` ở Task 2.2 (nối lại mọi chunk con → khớp 100%
      row gốc), không phải từ validator chạy sau — 2 lớp phòng thủ khác nhau, không phải 1 lớp làm 2 lần.
- [ ] Task 2.3: `ExcelRowChunker`: thêm `column_names`/`source_locator.sheet_name`/`row_start`/`row_end`/
      `table_id="table-0"` (1 sheet = 1 table_id, XLSX hiện chỉ đọc `workbook.active`), `chunking_version`
      set tường minh, `block_index=0`, `source_type=SourceType.XLSX`. Heading không áp dụng cho XLSX
      (`heading_path=[]`).

**Acceptance criteria:**
- [ ] `TableRowChunker` trên 1 bảng dài: mọi chunk con có header đầy đủ (nếu `has_header`), `column_names`
      giống nhau, nối lại toàn bộ row của mọi chunk con → khớp 100% row gốc (không thiếu không lặp), MỌI
      chunk con có heading_path prepend trong `content`.
- [ ] HTML không có `<th>` nào / DOCX không có `<w:tblHeader>` nào → `has_header=False`,
      `column_names=None`, `header_source=MISSING`, dữ liệu row vẫn giữ nguyên theo vị trí cột.
- [ ] Row bình thường: KHÔNG BAO GIỜ bị chia (test âm tính rõ ràng). Row vượt ngưỡng cứng: chia "Phần
      k/n" ĐÚNG KHI VÀ CHỈ KHI có đủ `is_partial_row=True`+`row_part`+`row_part_count`+`table_id` giống
      nhau ở mọi phần.
- [ ] **[v3]** Test với `heading_path` dài (nhiều cấp) trên 1 bảng có `max_tokens` nhỏ: token count thật
      của `content` cuối cùng (đo bằng `tiktoken`, đúng cách đo lúc chunk) không vượt `max_tokens`.
- [ ] **[v4]** Test edge case `usable_budget <= 0` (heading_path cực dài hoặc bảng có `header_line` cực
      rộng trên `max_tokens` nhỏ): `TableRowChunker` raise `ChunkingConfigException` rõ ràng, KHÔNG tạo ra
      chunk rỗng/âm/vô hạn loop.
- [ ] **[v4]** Test self-check nội bộ: giả lập 1 row bị đưa vào chunk sai số cell (mock/monkeypatch dữ liệu
      hỏng) → raise đúng `TableStructureError`, test dùng `pytest.raises(TableStructureError)` (không phải
      `AssertionError`).
- [ ] **[v5]** Test escape/chuẩn hoá: 1 cell chứa ký tự `|` → `content` render ra `\|` (không tạo thêm cột
      giả — parse lại `content` bằng chính markdown table parser của dự án nếu có, hoặc đếm số `|` chưa
      escape trên mỗi dòng phải khớp `len(header_row) + 1`); 1 cell chứa `\n` → `content` render ra `<br>`,
      không có dòng markdown nào bị vỡ thành 2 dòng bảng. Test round-trip: `source_locator.row_start`/
      `row_end`/`row_count` tính từ `TableBlock.data_rows` GỐC (trước escape) phải khớp đúng số row thật,
      không bị ảnh hưởng bởi việc render.
- [ ] `ExcelRowChunker` có `column_names`/`source_locator` đầy đủ.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] Test mới: `tests/test_chunking_table_row.py`, `tests/test_table_block_parsing.py` (thay
      `test_table_metadata.py` của v1 — đổi tên vì giờ test parse `TableBlock`, không phải
      `parse_table_header` đọc markdown).
- [ ] `pytest tests/test_chunking_excel_rows.py tests/test_chunking_table_row.py
      tests/test_table_block_parsing.py` xanh.

**Dependencies:** Task 0.1, Task 1.1-1.4
**Files:** `app/rag/chunking/table_row.py` (mới), `app/rag/chunking/excel_rows.py`,
`app/rag/ingestion/table_aware_parser.py`, `app/core/error_codes.py`, `app/core/exceptions.py` **[v4:
thêm `ChunkingConfigException` — dùng pattern `UniSageException`, khác với `TableStructureError` định
nghĩa cục bộ ngay trong `table_row.py` vì đó chỉ là lỗi bất biến nội bộ, không phải lỗi API trả về
client]**, `tests/test_chunking_table_row.py` (mới), `tests/test_table_block_parsing.py` (mới),
`tests/test_chunking_excel_rows.py`
**Estimated scope:** Large — tách row merge cơ bản (2.2 phần đầu) và oversized-row fallback (2.2 phần
"Phần k/n") thành 2 commit review riêng; tách 3 hàm `_x_table_to_block` (2.1) theo định dạng.

### Checkpoint: Phase 2
- [ ] `TableRowChunker` đứng độc lập, có test đầy đủ, **chưa được `strategy.dispatch` gọi tới**.
- [ ] Review với người dùng: xác nhận hướng xử lý `ChunkingStrategyName.MARKDOWN_AWARE` (trùng
      `RECURSIVE` sau Phase 3 — xem Architecture Decisions ở v1, không đổi ở v2), trước khi làm Phase 3.

### Phase 3: Nối TableRowChunker vào `strategy.dispatch` + mọi chunker text tự prepend heading

- [ ] Task 3.1: Viết lại `strategy.dispatch()`: partition `regions` thành `table_regions`/`text_regions`
      theo `region_type`; `table_chunks = TableRowChunker().split(table_regions)`; `text_chunks =
      chunker.split(text_regions)`; **merge lại theo `block_index` của region gốc** (điểm sửa #7 — dùng
      field thật thay vì "region_id" nói suông ở v1) rồi renumber `chunk_index` liên tục 0..n-1. Mỗi
      `Chunk` giữ `block_index` của region cha (không đổi khi renumber `chunk_index`).
- [ ] Task 3.2: `RecursiveChunker`/`TokenBasedChunker`/`SemanticChunker`: copy `heading_path`/
      `page_start`/`page_end`/`block_index`/`source_type` từ `region` sang mọi `Chunk` con; **tự prepend
      `heading_prefix = " > ".join(heading_path) + "\n\n"` vào `content` của TỪNG chunk con** (điểm sửa #2
      — không phụ thuộc region đã prefix sẵn, vì Phase 1 đã bỏ việc đó). **[v3] Trừ ngân sách TRƯỚC khi
      cắt, không phải sau**: `TokenBasedChunker`/`SemanticChunker` tính `tokens(heading_prefix)` 1 lần cho
      cả region rồi dùng `usable_max_tokens = max_tokens - tokens(heading_prefix)` làm giới hạn cắt thật
      (thay vì cắt theo `max_tokens` gốc rồi mới prepend, khiến `content` cuối vượt giới hạn — đúng lỗi
      người dùng chỉ ra); `RecursiveChunker` (char-based) tương tự trừ `len(heading_prefix)` khỏi
      `chunk_size` hiệu dụng, tính `effective_chunk_size = chunk_size - len(heading_prefix)`. **[v4]
      Cùng chính sách edge case như Task 2.2**: nếu `usable_max_tokens <= 0`
      (`TokenBasedChunker`/`SemanticChunker`) hoặc `effective_chunk_size <= overlap`
      (`RecursiveChunker` — `overlap` bằng hoặc lớn hơn ngân sách còn lại sau khi trừ heading nghĩa là
      bước cắt tiếp theo không tiến được, có thể sinh `step <= 0` hoặc vô hạn loop) → raise
      `ChunkingConfigException` ngay, KHÔNG âm thầm giảm `overlap` xuống 0, KHÔNG âm thầm bỏ qua giới hạn
      — cùng lý do đã nêu ở Task 2.2 (fail-fast, không che giấu heading quá dài so với `chunk_size` đã
      chọn). Set `source_locator.section = " > ".join(heading_path) or None`;
      `chunking_version = settings.CHUNKING_VERSION` tường minh.
- [ ] Task 3.3: `MarkdownAwareChunker` — theo quyết định checkpoint Phase 2 (mặc định: alias
      `RecursiveChunker`, đánh dấu deprecated trong docstring, không breaking `ChunkingStrategyName`
      enum).

**Acceptance criteria:**
- [ ] **[v3]** Với `heading_path` dài trên 1 region gần đầy `chunk_size`/`max_tokens`: `content` cuối cùng
      của mọi chunk con (đã gồm heading prefix) không vượt giới hạn gốc — test riêng cho cả 3 chunker.
- [ ] **[v4]** Edge case `usable_max_tokens <= 0` / `effective_chunk_size <= overlap`: cả 3 chunker raise
      `ChunkingConfigException` rõ ràng thay vì tạo chunk rỗng, âm, hoặc treo vô hạn loop — test riêng cho
      từng chunker.
- [ ] TABLE region không bị cắt giữa hàng ở bất kỳ strategy nào (`recursive`/`token_based`/`semantic`).
- [ ] `chunk_index` đúng thứ tự xuất hiện gốc (TEXT/TABLE xen kẽ không xáo trộn) — test bằng file
      TEXT-TABLE-TEXT-TABLE, kiểm tra thứ tự `block_index` không giảm qua các `chunk_index` liên tiếp.
- [ ] Mọi chunk (TEXT lẫn TABLE) có `heading_path` đúng VÀ `content` bắt đầu bằng heading text tương ứng
      (khi `heading_path` không rỗng) — test riêng cho trường hợp 1 region dài bị cắt thành ≥3 chunk con,
      assert CẢ 3 chunk con đều có heading trong `content`, không chỉ chunk đầu.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] `pytest tests/test_chunking_strategy.py tests/test_chunking_recursive.py
      tests/test_chunking_token_based.py tests/test_chunking_semantic.py
      tests/test_chunking_markdown_aware.py tests/test_ingestion_chunking.py` xanh.
- [ ] Manual: `POST /ingestion/chunking` với `strategy=semantic` trên 1 DOCX có bảng dài + nhiều section,
      xác nhận không bảng nào bị cắt giữa dòng, và mọi chunk hiển thị đúng heading của nó.

**Dependencies:** Task 1.1-1.4, Task 2.1-2.3
**Files:** `app/rag/chunking/strategy.py`, `app/rag/chunking/recursive.py`, `app/rag/chunking/token_based.py`,
`app/rag/chunking/semantic.py`, `app/rag/chunking/markdown_aware.py`, test tương ứng
**Estimated scope:** Medium

### Checkpoint: Phase 3
- [ ] Toàn bộ `pytest` xanh, `ruff`/`mypy` sạch.
- [ ] Bug gốc (mất heading, cắt sai hàng bảng) fix được, xác nhận bằng test end-to-end qua
      `strategy.dispatch()`. Heading xuất hiện ở MỌI chunk con, không chỉ chunk đầu của region.

### Phase 4: Validator bắt buộc trong pipeline (dùng field, không reparse content)

- [ ] Task 4.1: Thêm `ErrorCode.CHUNK_VALIDATION_FAILED` + `ChunkValidationException`
      (`app/core/error_codes.py`, `app/core/exceptions.py`, theo pattern
      `UniSageException(ErrorCode, message, errors)` hiện có). **[v4] Phân biệt với `ChunkingConfigException`
      dùng ở Task 2.2/3.2** (Phase 2-3, làm TRƯỚC Phase 4): 2 exception phục vụ 2 giai đoạn khác nhau —
      `ChunkValidationException` cho chunk ĐÃ TẠO XONG nhưng field không nhất quán (validator, Phase 4);
      `ChunkingConfigException` cho lỗi CẤU HÌNH phát hiện TRƯỚC khi tạo chunk (budget âm, Phase 2-3) — do
      cần dùng sớm hơn Phase 4, `ChunkingConfigException` + `ErrorCode.CHUNKING_CONFIG_INVALID` được định
      nghĩa ngay ở Task 2.2 (xem Phase 2), không đợi tới task này; Task 4.1 chỉ thêm
      `ChunkValidationException`, không định nghĩa lại.
- [ ] Task 4.2 **[v3: làm rõ giới hạn thật của validator; v5: siết rule bảng + thêm rule ngược header]**:
      Viết `validate_chunks(chunks: list[Chunk]) -> None` (`app/rag/chunking/validation.py`). Đây là
      **validator kiểm tra NHẤT QUÁN GIỮA CÁC FIELD của 1 `Chunk` đã persist, KHÔNG PHẢI nơi phát hiện
      "row có bị cắt hay không"** (việc đó là trách nhiệm của `TableRowChunker` tự đảm bảo + test kiệt,
      xem Task 2.2) — mục đích thật của validator này là bắt lỗi round-trip/mapping (DB, Qdrant,
      `chunking_version` cũ-mới lẫn lộn), không phải re-derive sự thật cấu trúc bảng từ text. Rule:
      (a) `heading_path` luôn là `list`;
      (b) **[v5, khác v4: bắt buộc chứ không còn "nếu có"]** `region_type in (TABLE, EXCEL_ROW)` →
      `source_locator.row_start`/`row_end`/`row_count` đều KHÔNG `None` (v4 chỉ kiểm tra
      `row_end - row_start + 1 == row_count` KHI `row_count` có giá trị — để optional; người dùng chỉ ra
      với chunk bảng, 3 field này phải LUÔN có, không phải optional, vì chúng là căn cứ duy nhất để định vị
      hàng dữ liệu của 1 chunk bảng); `row_start <= row_end`; `row_end - row_start + 1 == row_count`;
      `has_header=True` → `column_names` không rỗng — đây là kiểm tra field tự khai báo khớp nhau, không
      phải đếm cell thật trong `content`;
      (c) `source_type == "pdf"` → `page_start`/`page_end` không `None` và `page_end >= page_start`, áp
      dụng cho MỌI `region_type` (kể cả TEXT), không chỉ TABLE (điểm sửa #4 v2);
      (d) `is_partial_row=True` → phải có `row_part`/`row_part_count`/`table_id` không `None` và
      `1 <= row_part <= row_part_count`;
      (e) MỌI chunk (không phân biệt `region_type`) phải có `source_type is not None` VÀ
      `block_index is not None and block_index >= 0` — đây là điều kiện "chunk đã thật sự đi qua
      `dispatch()`" (khác chunk dựng tay trong unit test Phase 0-2, không qua validator này);
      (f) `header_source == HeaderSource.MISSING` → `has_header is False` VÀ `column_names is None`;
      (g) `has_header is False` → `column_names is None` (không có header thì không thể có tên cột, kể cả
      khi `header_source` khác `MISSING`);
      (h) `header_source == HeaderSource.MISSING` → `header_confidence == 0.0`;
      (i) `is_partial_row is False` → `row_part is None` VÀ `row_part_count is None` (chiều ngược của rule
      (d) — 1 row bình thường không được mang dấu vết của field chỉ dành cho oversized-row fallback);
      (j) `region_type in (TABLE, EXCEL_ROW)` → `source_locator is not None` VÀ
      `source_locator.table_id is not None` (chunk bảng luôn phải định vị được thuộc bảng nào).
      **[v5] Rule ngược bổ sung (theo yêu cầu người dùng — hoàn thiện tính "2 chiều" của rule (f)/(g)/(d)):**
      (k) `header_source != HeaderSource.MISSING` (tức `EXPLICIT` hoặc `INFERRED`) → `has_header is True`
      (nếu chunker đã xác định được nguồn header rõ ràng mà `has_header` vẫn `False` là mâu thuẫn — chiều
      ngược của rule (f));
      (l) `is_partial_row is True` → `row_part_count >= 2` (1 row chỉ tách "1 phần trong 1 phần" là vô
      nghĩa — nếu chỉ có 1 phần thì không nên đánh dấu `is_partial_row=True` ngay từ đầu, đây là rule kiểm
      tra chunker không tự tạo fallback thừa cho row không thật sự cần chia).
      (m) raise `ChunkValidationException` gộp TẤT CẢ lỗi tìm được ở (a)-(l) (không dừng ở lỗi đầu).
      "Không cắt giữa câu" vẫn chỉ là heuristic yếu (log warning, không raise cứng) — giữ nguyên, lý do ở
      Risks.
- [ ] Task 4.3: Gọi `validate_chunks(chunks)` trong `chunk_document` (`app/api/v1/ingestion.py`) ngay sau
      `strategy.dispatch(...)`, trước `upsert_chunking_draft`.
- [ ] Task 4.4 **[v4 mới, [v5] đổi thành BƯỚC 2 — chạy SAU Task 4.5, không phải trước]**: `POST
      /ingestion/embedding` (`app/api/v1/ingestion.py:129`) hiện nhận thẳng `request.chunks` từ client và
      gọi `embed_chunks.delay(...)` ngay, KHÔNG qua `validate_chunks()` — khác `/ingestion/chunking` đã có
      validator từ Task 4.3. Thêm lệnh gọi `validate_chunks(merged_chunks)` — **input là kết quả ĐÃ MERGE
      xong từ Task 4.5** (KHÔNG phải `request.chunks` thô từ client — validate trước khi merge là vô
      nghĩa vì lúc đó field vẫn còn là dữ liệu chưa đáng tin) — ngay trước khi dispatch Celery, cùng
      pattern với Task 4.3 — raise `ChunkValidationException` (→ HTTP 4xx qua exception handler hiện có)
      nếu bất kỳ chunk nào sai field. Numbering giữ nguyên `4.4`/`4.5` để không phải đổi số task đã review
      qua nhiều vòng, nhưng **thứ tự THỰC THI trong code là 4.5 trước, 4.4 sau** — ghi rõ trong code
      comment tại endpoint để tránh nhầm.
- [ ] Task 4.5 **[v4 mới, lớp phòng thủ chính; [v5] siết chặt điều kiện merge + chính sách draft legacy]**:
      Đổi endpoint `/ingestion/embedding` để KHÔNG tin tưởng metadata client gửi lên cho từng chunk (client
      chỉ nên sửa được `content`, không phải `page_start`/`source_locator`/bất kỳ field cấu trúc nào —
      hiện tại client gửi gì backend lưu nấy). Cách làm: đọc draft đã lưu qua `ChunkRepository`/`get_draft`
      làm **nguồn metadata chính thức (canonical)** — điều kiện tra draft **[v5]**: draft phải tồn tại VÀ
      thuộc đúng `document_id`+`department_id`+`object_key` của request (không chỉ tra theo
      `document_id` — nếu `department_id`/`object_key` trong request lệch với draft đã lưu, coi là request
      không hợp lệ, raise lỗi, không âm thầm dùng draft của context khác). **[v5] Điều kiện merge chunk
      siết chặt hơn v4** (khác v4 chỉ "so tập hợp `chunk_index`"): (1) `len(request.chunks) ==
      len(canonical_chunks)`; (2) `chunk_index` trong `request.chunks` không được trùng lặp (dùng `set()`
      rồi so độ dài với list gốc để phát hiện); (3) tập `chunk_index` của `request.chunks` phải khớp
      CHÍNH XÁC (không thiếu, không thừa) với tập `chunk_index` của canonical — bất kỳ sai lệch nào ở
      (1)-(3) đều raise lỗi rõ ràng yêu cầu chunk lại, KHÔNG cố merge một phần. Khi đủ điều kiện: với mỗi
      `chunk_index`, chỉ lấy `content` từ payload client gửi (đúng luồng UI hiện tại — người dùng chỉnh
      sửa `content` ở bước review, xem `updateChunkContent`/`use-ingest-wizard.ts:213-219`), MỌI field còn
      lại (heading_path, page_start, source_locator, column_names, has_header, header_source, ...) lấy
      nguyên từ canonical, KHÔNG lấy từ client. **[v5] Chính sách draft legacy (`chunking_version ==
      "legacy"`, hoặc bất kỳ chunk canonical nào có `source_type is None`/`block_index is None`)**: đây là
      draft được tạo TRƯỚC khi hệ thống này triển khai (hoặc đọc lên từ Postgres cũ theo default
      `"legacy"` của Task 0.2) — **reject với HTTP 409**, message yêu cầu người dùng chunk lại tài liệu
      (không cố embed thiếu metadata, không cố suy luận lại heading/page/table từ dữ liệu cũ — không thể
      khôi phục chính xác). Không cần xử lý migrate/backfill dữ liệu ĐÃ NẰM trong Qdrant từ trước (người
      dùng xác nhận không cần quan tâm dữ liệu cũ trong Qdrant) — rule 409 này chỉ chặn việc TẠO EMBED MỚI
      từ 1 draft cũ, không đụng tới point đã có sẵn trong Qdrant. Danh sách chunk đã merge (đã qua hết các
      điều kiện trên) mới là input cho `validate_chunks()` (Task 4.4) và cho `embed_chunks.delay(...)`.

**Acceptance criteria:**
- [ ] Chunk hợp lệ (từ Phase 3) qua validator không lỗi.
- [ ] Chunk PDF loại TEXT (không phải TABLE) thiếu `page_start` → validator raise đúng lỗi (case v1
      KHÔNG bắt được).
- [ ] Chunk `is_partial_row=True` nhưng thiếu `row_part_count` → validator raise đúng lỗi.
- [ ] **[v5]** Chunk TABLE có `row_count=None` (dù `row_start`/`row_end` đều có) → validator raise lỗi
      (khác v4 — v4 coi field này optional).
- [ ] **[v5]** Chunk có `header_source=INFERRED` nhưng `has_header=False` → validator raise lỗi (rule (k)).
- [ ] **[v5]** Chunk `is_partial_row=True` nhưng `row_part_count=1` → validator raise lỗi (rule (l)).
- [ ] Response lỗi có đủ thông tin debug (chunk_index + lý do từng chunk).
- [ ] **[v4]** Gọi `POST /ingestion/embedding` trực tiếp (bỏ qua UI wizard, giả lập client/script gọi API
      thẳng) với 1 payload cố tình sửa `page_start`/`source_locator` của 1 chunk khác với draft đã lưu →
      backend BỎ QUA giá trị đó, dùng đúng metadata từ draft; chỉ `content` client gửi được áp dụng.
- [ ] **[v4, mở rộng ở v5]** `POST /ingestion/embedding` với `chunks` thiếu 1 `chunk_index` có trong draft,
      có thêm 1 `chunk_index` lạ, hoặc có `chunk_index` TRÙNG LẶP trong payload → raise lỗi rõ ràng, KHÔNG
      dispatch Celery, KHÔNG cố merge một phần.
- [ ] **[v5]** `POST /ingestion/embedding` với `department_id`/`object_key` trong request lệch với draft
      đã lưu (dù `document_id` đúng) → raise lỗi, không dùng nhầm draft.
- [ ] **[v5]** `POST /ingestion/embedding` trên 1 draft `chunking_version == "legacy"` → raise HTTP 409,
      không dispatch Celery.
- [ ] **[v4]** Chunk hợp lệ merge từ draft (không có gì bất thường) qua `validate_chunks()` ở endpoint này
      không lỗi, dispatch Celery bình thường — không regress luồng hiện có.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] Test mới: `tests/test_chunk_validation.py`.
- [ ] **[v4]** Test mới/mở rộng: `tests/test_ingestion_embedding_endpoint.py` (hoặc file test endpoint
      `/ingestion/embedding` hiện có nếu đã tồn tại) cho toàn bộ case acceptance criteria `[v4]`/`[v5]` ở
      trên, gồm case trùng `chunk_index`, case lệch `department_id`/`object_key`, case draft legacy 409.
- [ ] `pytest tests/test_chunk_validation.py tests/test_ingestion_chunking.py
      tests/test_ingestion_embedding_endpoint.py` xanh.

**Dependencies:** Task 3.1-3.3, Task 0.2 (đọc draft qua `ChunkRepository`)
**Files:** `app/core/error_codes.py`, `app/core/exceptions.py`, `app/rag/chunking/validation.py` (mới),
`app/api/v1/ingestion.py`, `app/database/repositories/chunk.py`, `tests/test_chunk_validation.py` (mới),
`tests/test_ingestion_chunking.py`, `tests/test_ingestion_embedding_endpoint.py` (mới hoặc mở rộng)
**Estimated scope:** Medium — Task 4.4/4.5 cộng thêm so với v3 nhưng tái dùng hạ tầng draft đã có từ
Phase 0, không cần thiết kế mới; `[v5]` siết thêm điều kiện merge không đổi kiến trúc, chỉ thêm check.

### Checkpoint: Phase 4
- [ ] Pipeline thật (`parse → chunk → validate → lưu draft`) chạy đúng thứ tự, test end-to-end xác nhận
      validator chặn được chunk lỗi PDF thiếu trang (không chỉ chunk bảng như v1).
- [ ] **[v5]** Draft legacy không thể embed được nữa (409), draft mới luôn đủ điều kiện merge chặt trước
      khi tới validator — không còn đường nào để 1 chunk thiếu/sai metadata tới được Qdrant qua endpoint
      embedding.
- [ ] **[v4]** `POST /ingestion/embedding` không còn là đường tắt bỏ qua validator/metadata thật — mọi
      chunk được embed đều mang metadata từ draft đã qua `dispatch()` + `validate_chunks()`, `content` là
      phần DUY NHẤT client kiểm soát được.

### Phase 5: Propagate metadata xuống Qdrant + embedding task

- [ ] Task 5.1: Mở rộng `ChunkPoint` (`app/rag/vectorstore/qdrant_store.py`) thêm: `source_type`,
      `block_index`, `heading_path`, `page_start`, `page_end`, `source_locator` (dict), `column_names`,
      `has_header`, `header_source`, `chunking_version`. Cập nhật `upsert_chunk` payload tương ứng.
- [ ] Task 5.2: Cập nhật `embed_chunks` (`app/worker/celery_app.py`) truyền đủ field mới từ `chunk` sang
      `ChunkPoint`.

**Acceptance criteria:**
- [ ] Point Qdrant mới upsert có đủ field, đọc lại khớp 100% `Chunk` gốc.
- [ ] Point cũ (thiếu field mới) vẫn query/search được bình thường.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] `pytest tests/test_qdrant_store.py tests/test_embed_chunks_task.py` xanh.

**Dependencies:** Task 3.1-3.3
**Files:** `app/rag/vectorstore/qdrant_store.py`, `app/worker/celery_app.py`,
`tests/test_qdrant_store.py`, `tests/test_embed_chunks_task.py`
**Estimated scope:** Small

### Checkpoint: Phase 5
- [ ] Metadata đi tới Qdrant payload đầy đủ — nhưng **CHƯA đủ để citation dùng được** (đó là lý do có
      Phase 6, khác v1 coi đây là bước cuối).

### Phase 6 [MỚI trong v2]: Propagate metadata tới citation thật (retrieval + prompt)

- [ ] Task 6.1 **[v3: thêm source_locator]**: Mở rộng `RetrievedChunk` (`app/schemas/retrieval.py`): thêm
      field tường minh `heading_path: list[str] = []`, `page_start: int | None = None`,
      `page_end: int | None = None`, `source_type: str | None = None`, **`source_locator: SourceLocator |
      None = None`** (tái dùng model `SourceLocator` từ `app/schemas/ingestion.py` — thiếu field này thì
      citation 1 chunk bảng chỉ biết "trang mấy" mà không biết "hàng nào/bảng nào trong trang đó", không
      đủ để debug hoặc mở rộng UI sau này hiển thị đúng vùng dữ liệu). Giữ `metadata: dict[str, Any]` cho
      phần còn lại (department/access_level/category/region_type như hiện tại, không đổi).
- [ ] Task 6.2: Cập nhật `_to_retrieved_chunk` (`app/rag/retrieval/service.py:45`) đọc các field mới —
      **gồm cả `source_locator`** — từ Qdrant `payload` (đã có từ Task 5.1) vào `RetrievedChunk` field mới
      ở Task 6.1.
- [ ] Task 6.3: Cập nhật `build_prepared_context_section` (`app/rag/prompting/builder.py:102`): đổi format
      dòng citation từ `f"  [{index}] ({chunk.source}) {chunk.content}"` sang có số trang khi có, ví dụ
      `f"  [{index}] ({chunk.source}{_page_suffix(chunk)}) {chunk.content}"` với `_page_suffix` trả
      `f", tr. {page_start}"` (hoặc `f", tr. {page_start}-{page_end}"` nếu khác nhau) khi
      `chunk.page_start is not None`, else chuỗi rỗng. `content` đã tự mang heading từ Phase 3 (Task 3.2)
      nên không cần builder tự thêm heading nữa — chỉ cần đảm bảo không bị cắt/escape sai khi render vào
      template YAML.
- [x] Task 6.4 **[đã đọc code thật xác nhận, 2026-09-14, quyết định chốt]**: Đọc `app/api/v1/chat.py`
      (dòng 146-263) xác nhận: `POST /chat/stream` stream token thô từ LLM qua SSE
      (`_sse_token_generator`), lưu nguyên văn câu trả lời vào Java qua `create_message` — **không có bước
      nào parse `[1]`/`[2]` thành citation object có cấu trúc**. Citation hiện tại 100% là text tự do LLM
      tự viết theo hướng dẫn ở `citation_rules.yaml`. Đã hỏi người dùng, **chốt: KHÔNG thêm structured
      citation mapping trong plan này** — chấp nhận citation dạng text (độ tin cậy phụ thuộc LLM tuân thủ
      prompt, không có validation/guarantee ở tầng API). Structured citation
      (`citation_index -> {chunk_id, source, page}` trả kèm response, đổi SSE event schema, có thể đổi
      schema message Java lưu, frontend phải đọc field mới) là 1 thay đổi cross-repo (Java + frontend)
      NGOÀI SCOPE plan này — nếu cần, đó là 1 plan/đề xuất riêng, không mở rộng ở đây. Task này coi như
      hoàn thành ở việc audit + chốt quyết định, không có code thay đổi ở `chat.py`.
- [ ] Task 6.5 **[đọc code thật xác nhận, 2026-09-14]**: `app/rag/prompting/prompt_templates/common/
      citation_rules.yaml` — prompt hiện tại dạy LLM tạo footer trích dẫn dạng cố định `- [1] <Tên văn
      bản> (<Số Quyết định/Phiên bản>) - <Điều/Khoản/Mục>` (dòng 10), **hoàn toàn không có chỗ cho số
      trang** dù Task 6.3 đã đưa `tr. X` vào ngay cạnh mỗi chunk trong `<academic_context>`
      (`prepared_context.yaml`). Không sửa `citation_rules.yaml` thì LLM có nhìn thấy `tr. X` trong context
      cũng không có hướng dẫn phải đưa nó vào câu trả lời — Task 6.3 (đổi `prepared_context`) là điều kiện
      CẦN, Task 6.5 (đổi `citation_rules`) là điều kiện ĐỦ. Sửa mục 2 của `citation_rules.yaml` thành:
      ```markdown
      ---
      📌 **Văn bản tham chiếu chính thức:**
      - [1] <Tên văn bản> (<Số Quyết định/Phiên bản>) - <Điều/Khoản/Mục>[, Trang <X> hoặc <X>-<Y> nếu
        `<academic_context>` có ghi kèm "tr. X" cho chunk đó; bỏ qua phần này nếu chunk không có số trang
        (HTML/DOCX/TXT/XLSX)]
      ```
      và thêm 1 dòng rule mới (mục 4) yêu cầu LLM copy đúng số trang xuất hiện trong ngoặc `(nguồn, tr. X)`
      của `<academic_context>` cho chunk mình trích, KHÔNG tự suy đoán/làm tròn số trang khi context không
      ghi. Test: `build_prepared_context_section` trả context có `tr. 5`, gọi generation thật (hoặc mock
      LLM theo `tests/llm_mocks.py`) → response chứa "Trang 5" ở phần văn bản tham chiếu.

**Acceptance criteria:**
- [ ] `RetrievalService.retrieve(...)` trả `RetrievedChunk` có `page_start`/`page_end`/`heading_path`/
      `source_locator` khớp
      với chunk đã ingest (test bằng cách ingest 1 chunk có `page_start=5` qua `embed_chunks`, sau đó
      `retrieve` và assert field khớp).
- [ ] `build_prepared_context_section` với 1 chunk có `page_start=5, page_end=6` → dòng citation chứa
      `"tr. 5-6"` (hoặc format tương đương đã chọn); chunk không có trang (HTML/DOCX/TXT) → không có suffix
      trang, không hiển thị `"tr. None"` hay chuỗi rác nào.
- [x] Task 6.4 **[đã xác nhận]**: `chat.py` không có structured citation mapping — ghi finding này vào
      `docs/specs/known-gaps.md` (Checkpoint Phase 6) làm tài liệu cho quyết định "chấp nhận citation dạng
      text" đã chốt, để người đọc sau không tưởng nhầm đây là thiếu sót chưa phát hiện.
- [ ] Task 6.5: `citation_rules.yaml` sau khi sửa yêu cầu LLM đưa "Trang X" (hoặc "Trang X-Y") vào phần
      "📌 Văn bản tham chiếu chính thức" khi `<academic_context>` có ghi số trang cho chunk đó, và KHÔNG
      thêm phần này khi chunk không có trang (HTML/DOCX/TXT/XLSX) — test bằng mock LLM
      (`tests/llm_mocks.py`) xác nhận template render ra đúng rule mới, không chỉ test chuỗi tĩnh.

**Verification:**
- [ ] `ruff check app tests`, `mypy app tests` sạch.
- [ ] Test mới/cập nhật: `tests/test_retrieval.py`, thêm test cho `build_prepared_context_section` trong
      test file prompting hiện có (nếu chưa có, tạo `tests/test_prompting_builder.py`).
- [ ] Manual: chạy 1 câu hỏi thật qua graph (nếu môi trường cho phép) với tài liệu PDF đã ingest ở Phase
      0-5, xác nhận `<academic_context>` trong prompt log có `tr. X` đúng.

**Dependencies:** Task 5.1-5.2
**Files:** `app/schemas/retrieval.py`, `app/rag/retrieval/service.py`, `app/rag/prompting/builder.py`,
`app/rag/prompting/prompt_templates/common/citation_rules.yaml`, `docs/specs/known-gaps.md`,
`tests/test_retrieval.py`, `tests/test_prompting_builder.py` (mới nếu chưa có)
**Estimated scope:** Medium. Task 6.5 rủi ro thấp (sửa text prompt YAML, không đổi code path) nhưng LÀ MẮT
XÍCH BẮT BUỘC — thiếu nó thì Task 6.1-6.3 chỉ đưa số trang tới prompt mà LLM không có lệnh dùng nó, mục
tiêu chính (citation theo trang, dạng text — theo quyết định đã chốt ở Task 6.4) coi như chưa đạt dù mọi
phase trước đã đúng.

### Checkpoint: Phase 6 (Hoàn thành)
- [ ] Metadata đi xuyên suốt đủ `parser → Chunk → document_chunks (Postgres) → Qdrant payload →
      RetrievedChunk → prompt citation` — mục tiêu chính (citation theo trang) đã có đường đi trọn vẹn,
      không dừng ở Qdrant như v1.
- [ ] Toàn bộ `pytest`, `ruff check app tests`, `mypy app tests` xanh trên toàn repo.
- [ ] Cập nhật `docs/specs/SPEC-ingestion.md` (structural metadata) và `docs/specs/known-gaps.md`: DOCX
      heading không chuẩn style "Heading N" không nhận diện được; PDF `header_confidence` hằng số 0.6 chưa
      hiệu chỉnh bằng dữ liệu thật; validator "không cắt giữa câu" là heuristic yếu; citation là text tự
      do LLM viết, không có structured mapping (quyết định đã chốt ở Task 6.4, không phải thiếu sót);
      XLSX `header_source=EXPLICIT` là hợp đồng sản phẩm "dòng đầu luôn là header", không phải xác minh
      cấu trúc — sai nếu file thật không có header ở dòng đầu; **[v3, không block]** 1 bảng PDF trải dài
      qua nhiều trang hiện sinh ra `table_id` khác nhau theo từng trang (`block_index` đổi theo trang) dù
      về logic là 1 bảng — cần `table_group_id` nếu muốn merge citation của bảng xuyên trang chính xác,
      để Phase sau; **[v5, không block]** draft race condition khi re-chunk cùng lúc đang embed — chính
      sách hiện tại là last-write-wins đơn giản, chưa có `draft_version`/optimistic lock (xem Risks); draft
      `chunking_version == "legacy"` không thể embed được nữa (HTTP 409, Task 4.5) — dữ liệu ĐÃ NẰM trong
      Qdrant từ trước không bị đụng tới, đây chỉ là chặn embed MỚI từ draft cũ.
- [ ] Review với người dùng trước khi archive plan này.

### Phase 7 [MỚI trong v4]: Đồng bộ Zod schema frontend (`unisage-web`) — không để metadata bị Zod cắt

**Đã đọc thật code xác nhận (2026-09-14):** `ingestion-schemas.ts:18-22` — `chunkSchema` hiện chỉ có
`chunk_index`/`content`/`region_type`; mọi response backend trả về (từ Phase 0-6) đi qua
`chunkingResponseSchema`/`ingestionJobResponseSchema` (cả 2 đều tái dùng `chunkSchema`, dòng 41-43 và
59-70) sẽ bị Zod `.parse()` **âm thầm loại bỏ** hết field mới ngay khi request/response chạm frontend —
kể cả khi Phase 0-6 backend đã đúng 100%, UI/state (`use-ingest-wizard.ts`) không bao giờ thấy được
`heading_path`/`page_start`/`source_locator`/... Và vì `embeddingRequestSchema` (dòng 45-51) cũng tái
dùng đúng `chunkSchema` đó, `runEmbed()` (`use-ingest-wizard.ts:221-238`) gửi lên
`POST /ingestion/embedding` đúng cái đã bị cắt — dù Task 4.5 (Phase 4) đã chặn phía server bằng cách lấy
canonical metadata từ draft thay vì tin client, việc Zod cắt field vẫn là bug thật ở tầng hiển thị/state
frontend (chunk hiển thị trong bước review sẽ thiếu heading/trang/tên cột dù backend đã trả về đủ).

- [ ] Task 7.1: Thêm `sourceLocatorSchema` (`ingestion-schemas.ts`) khớp 1-1 với `SourceLocator`
      (`app/schemas/ingestion.py`, Data Model v4): `section`/`sheet_name`: `z.string().nullable()`;
      `row_start`/`row_end`/`row_count`/`row_part`/`row_part_count`: `z.number().int().nullable()`;
      `table_id`: `z.string().nullable()`; `is_partial_row`: `z.boolean()`.
      Thêm `headerSourceSchema = z.enum(["explicit", "inferred", "missing"])` và
      `sourceTypeSchema = z.enum(["pdf", "docx", "html", "txt", "xlsx"])` khớp `HeaderSource`/`SourceType`.
- [ ] Task 7.2: Mở rộng `chunkSchema` (dòng 18-22) thêm toàn bộ field mới của `Chunk` (Data Model v4),
      **cùng độ "optional/nullable" với backend Pydantic** (field nào có default ở backend thì
      `.nullable().optional()` hoặc `.default(...)` tương ứng ở Zod — không đặt required nếu backend không
      required, tránh chunk cũ/legacy thiếu field làm Zod parse fail toàn bộ response):
      `source_type: sourceTypeSchema.nullable().optional()`,
      `block_index: z.number().int().nullable().optional()` **(v4: nullable, khớp `block_index: int | None`
      backend)**, `heading_path: z.array(z.string()).default([])`,
      `page_start`/`page_end: z.number().int().nullable().optional()`,
      `source_locator: sourceLocatorSchema.nullable().optional()`,
      `column_names: z.array(z.string()).nullable().optional()`,
      `has_header: z.boolean().default(false)`, `header_source: headerSourceSchema.optional()`,
      `header_confidence: z.number().min(0).max(1).optional()`,
      `chunking_version: z.string().optional()`.
- [ ] Task 7.3: Xác nhận không cần sửa `chunkingResponseSchema`/`ingestionJobResponseSchema`/
      `embeddingRequestSchema` (đều tái dùng `chunkSchema`, tự động thừa hưởng field mới) — chỉ cần review
      lại chú thích dòng 91-94 (`chunkingFormSchema` — schema khác, phía input form, không liên quan chunk
      output) để tránh nhầm 2 khái niệm.
- [ ] Task 7.4: Xác nhận `use-ingest-wizard.ts` không cần sửa logic: `updateChunkContent` (dòng 213-219)
      dùng object spread `{ ...chunk, content }` nên tự động giữ nguyên mọi field mới của `Chunk` type
      (không allowlist field nào) một khi Task 7.2 mở rộng type `Chunk`; `runEmbed` (dòng 221-238) gửi
      thẳng state `chunks` nên cũng tự động mang đủ field khi state đã có đủ (từ `chunkMutation`/`jobQuery`
      parse đúng response nhờ Task 7.2). Chỉ cần review lại sau khi Task 7.2 xong để xác nhận không có chỗ
      nào khác trong `unisage-web` đang tự liệt kê field tường minh của `Chunk` (destructure 3 field cũ)
      mà bỏ sót field mới — nếu tìm thấy, sửa tại chỗ đó, không phải sửa hook này.

**Acceptance criteria:**
- [ ] `chunkSchema.parse(...)` trên 1 fixture JSON response `/ingestion/chunking` đầy đủ field mới (lấy từ
      backend thật sau Phase 0-6, hoặc fixture tay khớp đúng shape `Chunk` model) → object trả về giữ
      NGUYÊN mọi field, không field nào bị Zod cắt.
- [ ] Round-trip: parse response `/ingestion/chunking` bằng `chunkingResponseSchema` → build request body
      cho `/ingestion/embedding` từ đúng `chunks` đã parse (mô phỏng `runEmbed`) → parse lại bằng
      `embeddingRequestSchema` → field vẫn còn nguyên ở cả 2 lượt parse.
- [ ] Chunk cũ/legacy (fixture thiếu toàn bộ field mới, chỉ có `chunk_index`/`content`/`region_type` —
      mô phỏng response trước migration) vẫn parse thành công, không throw `ZodError`.
- [ ] **[v5, đã xác nhận đúng tên lệnh từ `package.json`]** `pnpm typecheck` không báo lỗi type ở chỗ nào
      dùng `Chunk`/`ChunkingResponse`/`EmbeddingRequest`.

**Verification:**
- [ ] Test mới: `ingestion-schemas.test.ts` (hoặc theo convention test hiện có của `unisage-web` — kiểm
      tra file test framework đang dùng, vd Vitest/Jest, trước khi viết test thật) cho 3 acceptance
      criteria đầu.
- [ ] **[v5]** Chạy `pnpm test`, `pnpm typecheck`, `pnpm lint` (3 lệnh thật đã xác nhận có trong
      `package.json` của `unisage-web`) — cả 3 phải sạch.
- [ ] Manual: chạy wizard thật (preview → chunking → review → embedding) trên 1 tài liệu có bảng + heading
      sau khi backend Phase 0-6 đã deploy, mở DevTools Network xác nhận request `/ingestion/embedding`
      mang đủ `heading_path`/`page_start`/`source_locator` giống response `/ingestion/chunking` nhận được.

**Dependencies:** Phase 0 (Data Model v4 đã chốt field/type), không phụ thuộc Phase 1-6 phải xong trước để
BẮT ĐẦU (có thể làm song song bằng fixture tay), nhưng cần Phase 0-6 xong để chạy được test manual E2E
thật.
**Files:** `unisage-web/src/features/ingestion/schemas/ingestion-schemas.ts`,
`unisage-web/src/features/ingestion/hooks/use-ingest-wizard.ts` (chỉ review, khả năng không cần sửa —
Task 7.4), test file mới tương ứng.
**Estimated scope:** Small-Medium — chủ yếu mở rộng schema (Task 7.1-7.2), Task 7.3-7.4 là review/xác
nhận không cần sửa code, không phải viết thêm.

### Checkpoint: Phase 7
- [ ] Metadata cấu trúc đi được TRỌN VẸN tới tận UI wizard (`review` step) và quay lại đúng trong request
      `/ingestion/embedding` — khép kín toàn bộ chuỗi `parser → Chunk → Postgres → Qdrant → RetrievedChunk
      → prompt citation` (Phase 0-6) VÀ `backend response → frontend state → backend request` (Phase 7),
      không còn điểm nào âm thầm làm rơi field.
- [ ] Review với người dùng trước khi archive plan này (gộp chung với review Checkpoint Phase 6, vì Phase
      7 là điều kiện để Phase 0-6 "dùng được" trọn vẹn qua UI thật, không phải 1 nhánh phụ độc lập).

## Phase sau (out of scope — chỉ liệt kê để biết việc cần làm tiếp)

- Hybrid retrieval: BM25/keyword search trên `heading_path`/`column_names`/identifier chính xác kết hợp
  vector search (đồng ý với người dùng ở review: KHÔNG làm trong plan này).
- Parent-child retrieval: trúng 1 chunk con của bảng dài → tự lấy thêm chunk con cùng `table_id` để mở
  rộng context.
- Dùng `chunking_version` để tự động phát hiện + trigger re-chunk/re-embed hàng loạt khi logic đổi.
- Hiệu chỉnh `header_confidence` PDF bằng dữ liệu thật (hiện là hằng số 0.6).
- Validator "không cắt giữa câu" chính xác hơn.
- Structured citation API (`citation_index -> {chunk_id, source, page}` có cấu trúc, đổi SSE + có thể đổi
  schema message Java + frontend đọc field mới) — **đã hỏi người dùng và chốt KHÔNG làm ở plan này**
  (2026-09-14); nếu cần sau, đó là 1 plan/đề xuất cross-repo riêng (Java + Python + frontend).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `pymupdf4llm.to_markdown(..., page_chunks=True)` có thể trả shape khác giữa các version | High | Task 1.3 đọc thật docstring/source version đang cài trước khi code, viết fixture dựa trên output thật |
| DOCX `<w:tblHeader>` có thể không được Word luôn set (nhiều bảng do người dùng tạo thủ công không bật "Repeat as header row") → phần lớn bảng DOCX thực tế rơi vào `MISSING` dù "trực quan" có header | Medium | Đây là đánh đổi có chủ đích của điểm sửa #3 (không bịa header khi không chắc) — ghi rõ vào known-gaps, KHÔNG thêm heuristic fallback (ví dụ "hàng đầu toàn text, các hàng sau có số" ) vì đó lại là suy luận ngược đã bị phê phán ở review |
| Citation dạng text (quyết định đã chốt) phụ thuộc hoàn toàn vào LLM tuân thủ `citation_rules.yaml` — không có gì đảm bảo LLM luôn ghi đúng "Trang X" kể cả khi context có sẵn | Medium | Chấp nhận có chủ đích (đây là bản chất của prompt-based citation, không phải bug của plan này); nếu độ chính xác không đủ trong thực tế, nâng cấp lên structured citation là hướng đi tiếp theo, đã ghi ở Phase sau |
| **[v3, đã giải quyết ở v4]** Field mới (`source_type`/`block_index`) có default an toàn ở Phase 0, nhưng nếu 1 dev sau này quên cập nhật 1 trong các chunker ở Phase 3 để set giá trị thật, chunk đó âm thầm mang `block_index=0`/`source_type=None` mà không lỗi ngay — ở v3, `block_index=0` không bắt được vì 0 là giá trị hợp lệ cho region đầu | Medium | **[v4]** `block_index` default đổi sang `None` (khác v3), validator (Task 4.2 rule (e)) bắt buộc `block_index is not None and >= 0` cùng hàng với `source_type is not None` — không còn ambiguity 0-vs-chưa-gán; Task 3.1-3.3 vẫn cần test riêng khẳng định MỌI chunk output từ `dispatch()` có cả 2 field hợp lệ |
| **[v4]** `usable_budget`/`usable_max_tokens`/`effective_chunk_size` có thể `<= 0` khi `heading_path` quá sâu hoặc `max_tokens`/`chunk_size` cấu hình quá nhỏ — nếu không xử lý, có thể tạo `step <= 0` (vô hạn loop) hoặc chunk rỗng | Medium | Task 2.2/3.2 raise `ChunkingConfigException` fail-fast thay vì âm thầm xử lý sai; test edge case riêng cho cả 4 chunker (`TableRowChunker`/`RecursiveChunker`/`TokenBasedChunker`/`SemanticChunker`) |
| **[v4]** Task 4.5 (canonical metadata từ draft) thêm 1 bước đọc DB (`get_draft`) vào `/ingestion/embedding` — nếu draft bị xoá/hết hạn giữa lúc người dùng ở bước review và lúc bấm "Embed", endpoint sẽ lỗi thay vì embed được (khác hành vi hiện tại, vốn chấp nhận bất kỳ `chunks` nào client gửi) | Low-Medium | Đánh đổi có chủ đích — đây chính là mục đích của Task 4.5 (không tin metadata client tự gửi); thông báo lỗi cần rõ ràng ("draft đã hết hạn, vui lòng chunk lại") để không gây khó hiểu cho người dùng cuối |
| **[v4, đã xác nhận ở v5]** Phase 7 (frontend) phụ thuộc đúng tên script test/typecheck thật của `unisage-web` | Low | **[v5]** Đã xác nhận `package.json` thật có `pnpm typecheck`/`pnpm test`/`pnpm lint` — dùng đúng 3 lệnh này trong Verification của Phase 7, không cần đoán/mở là Open Question nữa |
| Bỏ heading-prefix-trong-content ở HTML (Phase 1) là thay đổi hành vi so với `main` hiện tại, có thể ảnh hưởng chunk đã embed trước đó nếu có A/B so sánh | Low | `chunking_version` phân biệt rõ chunk cũ/mới; ghi trong PR description đây là breaking change có chủ đích |
| `header_confidence=0.6` cho PDF vẫn là số hằng | Low-Medium | Không đổi so với v1 — để Phase sau hiệu chỉnh bằng dữ liệu thật |
| Phase 1-2 (parse 4 định dạng + TableBlock + TableRowChunker) vẫn là phần rủi ro kỹ thuật cao nhất | High | Giữ nguyên thứ tự phase sớm + checkpoint riêng như v1 |
| **[v5]** Race condition: người dùng bấm "chunk lại" (tạo draft mới, ghi đè draft cũ cùng `document_id`) trong lúc TAB/request khác đang ở giữa `POST /ingestion/embedding` (đã đọc xong draft ở Task 4.5, chưa kịp dispatch Celery) — 2 luồng có thể dùng 2 phiên bản draft khác nhau, kết quả embed không xác định phiên bản nào "thắng" | Low-Medium | **[v5]** Chấp nhận có chủ đích: chính sách **last-write-wins đơn giản** cho phase này (không thêm draft version/hash + optimistic-lock check trước dispatch — tránh mở rộng scope ngoài mục tiêu structural metadata) — ghi rõ vào `docs/specs/known-gaps.md` (Checkpoint Phase 4/6) làm giới hạn đã biết; nếu thực tế phát sinh vấn đề (nhiều người dùng cùng sửa 1 document), thêm `draft_version`/optimistic lock là hướng nâng cấp tiếp theo, để Phase sau |

## Open Questions

- `ChunkingStrategyName.MARKDOWN_AWARE`: alias `RECURSIVE` hay xoá hẳn? (như v1, cần xác nhận Java/frontend
  không hard-code giá trị này trước khi xoá).
- Fixture PDF/DOCX nhiều trang, có heading, có bảng dài, có bảng KHÔNG có header rõ ràng (để test
  `header_source=MISSING` thật) — cần người dùng cung cấp tài liệu thật hay tự tạo fixture?
- Task 6.4: nếu audit phát hiện `chat.py` hiện không có structured citation mapping nào (citation chỉ là
  text LLM tự sinh trong câu trả lời), người dùng có muốn mở rộng scope plan này để thêm structured
  citation API, hay để đó làm 1 plan riêng sau khi thấy rõ hiện trạng?
- Format hiển thị số trang trong citation (`"tr. 5"` / `"trang 5"` / `"p.5"` / khác) — plan tạm chọn
  `"tr. X"` (viết tắt tiếng Việt, khớp văn phong hệ thống hiện có), có thể đổi theo yêu cầu UI thật.
- **[v4, đã chốt ở v5]** ~~Task 4.5: khi `request.chunks` gửi lên `/ingestion/embedding` có tập
  `chunk_index` không khớp hoàn toàn với draft đã lưu, nên raise lỗi hay merge một phần?~~ Người dùng đã
  chốt ở review lần 4: **raise lỗi yêu cầu chunk lại, KHÔNG merge một phần** — bắt buộc số lượng khớp
  draft, không trùng/thiếu/thừa `chunk_index` (xem Task 4.5 `[v5]`). Không còn là câu hỏi mở.
- **[v4]** `ChunkingConfigException` (Task 2.2/3.2) khi raise trong lúc người dùng đang thao tác UI wizard
  (`POST /ingestion/chunking`) sẽ hiển thị lỗi thế nào ở frontend — cần message người dùng hiểu được (gợi
  ý tăng `chunk_size`/`max_tokens`), hay chỉ cần lỗi kỹ thuật chung chung vì case này hiếm (heading quá
  sâu) và thường chỉ xảy ra với tài liệu có cấu trúc bất thường?
- **[v5, đã làm rõ — không còn là câu hỏi mở, ghi lại để không quên khi code]** `TableStructureError`
  (Task 2.2) là lỗi BẤT BIẾN NỘI BỘ của `TableRowChunker` (bug chunker, không phải lỗi input người dùng) —
  nếu nó lọt được tới tầng API (`POST /ingestion/chunking`), phải log kèm ĐẦY ĐỦ context (document_id,
  block_index, table_id, nội dung row gây lỗi) ở mức `ERROR`/`CRITICAL` trước khi propagate thành HTTP 5xx
  chung chung — KHÔNG log như 1 validation error bình thường (khác `ChunkValidationException`/
  `ChunkingConfigException`, vốn là lỗi có thể do input/cấu hình người dùng, hợp lý để trả 4xx với message
  rõ ràng). Ghi rõ điều này trong docstring của `TableStructureError` và trong Task 2.2/4.3 khi code thật.
