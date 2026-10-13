# Todo: Structural Chunking Metadata + Table-Safe Chunkers (v5)

CHƯA IMPLEMENT CODE — danh sách task theo `plan.md` v5 (chỉnh sau review lần 4 của người dùng, 2026-09-14).
Người dùng kết luận: **v4 đúng hướng, có thể bắt đầu implement**, nhưng CHƯA archive — còn 6 điểm bổ sung.
Chi tiết đầy đủ (AC, verification, files, rationale từng điểm `[v5]`) xem `plan.md` cùng thư mục.

## Phase 0: Schema nền tảng — field mới có default an toàn; `block_index` default `None` ([v4])
- [ ] T0.1 Thêm `HeaderSource`, `SourceType`, `SourceLocator` (**[v5]** toàn bộ 9 field, không phải "5
      field" như v3/v4 ghi nhầm); mở rộng `Chunk` với `source_type: SourceType | None = None` và
      `block_index: int | None = None`; `chunking_version` default `"legacy"`; `settings.CHUNKING_VERSION`
- [ ] T0.2 Cột `metadata: JSONB` + Alembic migration + repository round-trip; test row cũ thiếu
      `chunking_version` → đọc lên `"legacy"`
- [ ] Checkpoint Phase 0: mọi `Chunk(...)`/`ParsedRegion(RegionType.TEXT, text)` (positional) hiện có
      trong test suite vẫn construct thành công KHÔNG SỬA GÌ THÊM; test chunking cũ vẫn xanh y nguyên

## Phase 1: Heading path (KHÔNG prefix vào content) + block_index/source_type mọi region
- [ ] T1.1 `_regions_from_html`: bỏ prefix heading vào `content`; gán `block_index`, `source_type=HTML`
- [ ] T1.2 `_regions_from_docx`: heading_stack từ `paragraph.style.name`; gán `block_index`,
      `source_type=DOCX`
- [ ] T1.3 PDF: `pymupdf4llm.to_markdown(document, page_chunks=True)`, heading_stack xuyên trang,
      `page_start=page_end=<trang thật>`, `block_index`, `source_type=PDF`
- [ ] T1.4 TXT: `block_index=0`, `source_type=TXT`, `heading_path=[]`, `page_start=page_end=None`
- [ ] Checkpoint Phase 1: cả 4 định dạng có field đúng; `content` KHÔNG còn heading text

## Phase 2: TableBlock (header thật từ nguồn) + TableRowChunker (chưa nối vào dispatch)
- [ ] T2.1 Parse trực tiếp thành `TableBlock`: HTML theo `<th>`, DOCX theo `<w:tblHeader>` thật trong XML,
      PDF luôn `INFERRED/0.6`; XLSX `EXPLICIT/1.0` là HỢP ĐỒNG sản phẩm có sẵn, ghi rõ docstring +
      known-gap, không đổi hành vi
- [ ] T2.2 `TableRowChunker`: hoạt động trên `TableBlock.data_rows`; trừ ngân sách token cho
      heading_prefix + header_line TRƯỚC khi gom row; mỗi chunk con tự prepend heading; set
      `source_locator.row_count`; oversized row → "Phần k/n" đầy đủ metadata.
      - `ChunkingConfigException` (+ `ErrorCode.CHUNKING_CONFIG_INVALID`, định nghĩa NGAY Ở TASK NÀY vì
        chạy trước Phase 4) khi `usable_budget <= 0`, KHÔNG âm thầm giảm overlap/vượt hard cap.
      - Self-check nội bộ dùng `TableStructureError` (subclass `RuntimeError`, định nghĩa trong
        `table_row.py`) THAY VÌ `assert` (bị strip bởi `python -O`). **[v5]** Đây là lỗi BẤT BIẾN NỘI BỘ
        (bug chunker) — nếu lọt tới API, phải log ERROR/CRITICAL kèm đủ context (document_id, block_index,
        table_id, row gây lỗi) trước khi trả 5xx, KHÔNG log như validation error bình thường.
      - **[v5]** Escape `|` → `\|` và chuẩn hoá newline → `<br>` khi render markdown pipe-table (chỉ ảnh
        hưởng `content` hiển thị, KHÔNG ảnh hưởng `TableBlock.data_rows` gốc dùng cho self-check/
        `row_start`/`row_end`/`row_count`). Test round-trip cell có `|`/newline.
- [ ] T2.3 `ExcelRowChunker`: `column_names`/`source_locator` đầy đủ, `chunking_version` tường minh
- [ ] Checkpoint Phase 2: `TableRowChunker` có test đầy đủ (token budget edge case, `TableStructureError`,
      escape `|`/newline) nhưng CHƯA nối vào dispatch; review hướng xử lý `MARKDOWN_AWARE` trước Phase 3

## Phase 3: Nối TableRowChunker vào dispatch + mọi chunker text tự prepend heading (trừ ngân sách trước)
- [ ] T3.1 `strategy.dispatch()`: TABLE luôn qua `TableRowChunker`; merge theo `block_index` thật; renumber
      `chunk_index`
- [ ] T3.2 `RecursiveChunker`/`TokenBasedChunker`/`SemanticChunker`: copy field mới sang mọi chunk con; trừ
      `tokens(heading_prefix)`/`len(heading_prefix)` khỏi ngân sách cắt TRƯỚC khi cắt; `chunking_version`
      set tường minh. Cùng chính sách edge case như T2.2: `usable_max_tokens <= 0` hoặc
      `effective_chunk_size <= overlap` → raise `ChunkingConfigException` (tái dùng exception đã định
      nghĩa ở T2.2).
- [ ] T3.3 `MarkdownAwareChunker`: alias `RecursiveChunker` (mặc định theo checkpoint Phase 2)
- [ ] Checkpoint Phase 3: bug gốc đã fix; heading ở MỌI chunk con; content không vượt budget; edge case
      config xấu bị chặn rõ ràng; toàn bộ pytest/ruff/mypy xanh

## Phase 4: Validator (siết chặt [v5]) + chặn đường tắt bypass ở `/ingestion/embedding`
- [ ] T4.1 `ErrorCode.CHUNK_VALIDATION_FAILED` + `ChunkValidationException` (khác `ChunkingConfigException`
      đã định nghĩa ở T2.2)
- [ ] T4.2 `validate_chunks()`: rule (a)-(d) như v3/v4 (heading_path là list; PDF mọi region_type phải có
      page_start/page_end; is_partial_row đủ field). Rule (e)-(j) [v4]: block_index/source_type bắt buộc;
      header_source=MISSING ⇒ has_header=False + column_names=None; has_header=False ⇒ column_names=None;
      header_confidence=0 khi MISSING; is_partial_row=False ⇒ row_part/row_part_count=None;
      TABLE/EXCEL_ROW ⇒ source_locator + table_id không None.
      **[v5] Siết + rule ngược mới:**
      - (b) đổi thành BẮT BUỘC (không còn optional): TABLE/EXCEL_ROW phải có `row_start`/`row_end`/
        `row_count` đều KHÔNG `None` (khác v4 chỉ check IF có `row_count`).
      - (k) `header_source != MISSING` ⇒ `has_header=True` (chiều ngược của rule f).
      - (l) `is_partial_row=True` ⇒ `row_part_count >= 2` (chặn fallback thừa cho row không cần chia).
- [ ] T4.3 Gọi `validate_chunks()` trong `chunk_document` ngay sau `dispatch()`, trước `upsert_chunking_draft`
- [ ] T4.4 **[thứ tự thực thi: SAU T4.5]** Gọi `validate_chunks(merged_chunks)` (input đã merge từ T4.5,
      KHÔNG phải `request.chunks` thô) trong `POST /ingestion/embedding` trước khi dispatch Celery.
- [ ] T4.5 **[lớp phòng thủ chính, [v5] siết điều kiện merge + chính sách draft legacy]** Đổi
      `/ingestion/embedding` lấy metadata GỐC từ draft đã lưu, CHỈ nhận `content` client sửa tay.
      - Tra draft: phải khớp ĐÚNG `document_id` + `department_id` + `object_key` (không chỉ document_id).
      - **[v5]** Điều kiện merge siết: (1) số lượng `request.chunks` == số lượng canonical; (2) không có
        `chunk_index` trùng lặp trong request; (3) tập `chunk_index` khớp CHÍNH XÁC (không thiếu/thừa) —
        sai bất kỳ điều nào → raise lỗi yêu cầu chunk lại, KHÔNG merge một phần.
      - **[v5]** Draft `chunking_version=="legacy"` (hoặc canonical chunk có `source_type`/`block_index`
        None) → **reject HTTP 409**, yêu cầu chunk lại — không cố embed thiếu metadata (không thể khôi
        phục chính xác từ dữ liệu cũ). Không cần migrate dữ liệu ĐÃ CÓ trong Qdrant (người dùng xác nhận
        không cần quan tâm data cũ trong Qdrant) — chỉ chặn EMBED MỚI từ draft cũ.
- [ ] Checkpoint Phase 4: validator chặn PDF text chunk thiếu trang; `/ingestion/embedding` không còn là
      đường tắt; draft legacy không thể embed được nữa (409); draft mới luôn qua merge chặt + validator
      trước khi tới Qdrant.

## Phase 5: Propagate metadata xuống Qdrant + embedding task
- [ ] T5.1 Mở rộng `ChunkPoint`/`upsert_chunk` payload thêm mọi field structural mới
- [ ] T5.2 `embed_chunks`: truyền đủ field mới từ `chunk` sang `ChunkPoint`
- [ ] Checkpoint Phase 5: metadata tới Qdrant đầy đủ — CHƯA đủ để citation dùng được, xem Phase 6

## Phase 6: Propagate metadata tới citation thật (retrieval + prompt) — mục tiêu chính
- [ ] T6.1 Mở rộng `RetrievedChunk`: `heading_path`/`page_start`/`page_end`/`source_type` +
      `source_locator: SourceLocator | None`
- [ ] T6.2 `_to_retrieved_chunk` (`service.py:45`): đọc field mới — gồm cả `source_locator` — từ payload
- [ ] T6.3 `build_prepared_context_section` (`builder.py:102`): hiển thị "tr. X" khi có `page_start`
- [x] T6.4 **[đã chốt, 2026-09-14]** Đọc thật `chat.py`: xác nhận KHÔNG có structured citation mapping —
      chốt chấp nhận citation dạng text, KHÔNG thêm structured mapping trong plan này.
- [ ] T6.5 **[mắt xích bắt buộc]** `citation_rules.yaml`: thêm "Trang X"/"Trang X-Y" vào footer khi context
      có ghi trang; rule "copy đúng số trang từ context, không tự đoán"
- [ ] Checkpoint Phase 6: metadata đi xuyên suốt `parser → Chunk → Postgres → Qdrant → RetrievedChunk →
      prompt citation` (dạng text, theo quyết định đã chốt); cập nhật `docs/specs/SPEC-ingestion.md` +
      `known-gaps.md` (gồm cả `[v5]`: race condition last-write-wins, draft legacy 409)

## Phase 7: Đồng bộ Zod schema frontend (`unisage-web`)
**Đã đọc thật `ingestion-schemas.ts`/`use-ingest-wizard.ts` xác nhận claim của người dùng**: `chunkSchema`
chỉ có `chunk_index`/`content`/`region_type`; các schema response/request khác đều tái dùng nó nên field
mới bị Zod cắt ngay khi chạm frontend, kể cả khi Task 4.5 đã chặn phía server.
- [ ] T7.1 Thêm `sourceLocatorSchema`, `headerSourceSchema`, `sourceTypeSchema` khớp 1-1 backend.
- [ ] T7.2 Mở rộng `chunkSchema` thêm toàn bộ field mới của `Chunk`, đúng độ nullable/optional với backend
      (`block_index: z.number().int().nullable().optional()` khớp `int | None`).
- [ ] T7.3 Xác nhận `chunkingResponseSchema`/`ingestionJobResponseSchema`/`embeddingRequestSchema` tự động
      thừa hưởng field mới — không cần sửa riêng.
- [ ] T7.4 Xác nhận `use-ingest-wizard.ts` không cần sửa logic — chỉ review không có chỗ khác destructure
      field cũ mà bỏ sót field mới.
- [ ] Checkpoint Phase 7: metadata đi trọn vẹn `backend response → frontend state → backend request`;
      Verification dùng đúng 3 lệnh đã xác nhận: `pnpm test`, `pnpm typecheck`, `pnpm lint`; review chung
      với Checkpoint Phase 6 trước khi archive plan.

## Ghi chú
- `tasks/plan.md`/`tasks/todo.md` hiện tại của repo giữ 1 kế hoạch khác dở dang ("UniSage Academic RAG
  Graph") — plan/todo này đặt ở `changes/13-09-2026-Chunking-Structural-Metadata/` để không ghi đè.
- **v5 khác v4 ở 6 điểm** sau review lần 4 (người dùng kết luận v4 đúng hướng, có thể implement, nhưng
  chưa archive): (1) chính sách draft legacy → reject 409 khi embed, không cố suy luận lại metadata cũ,
  không cần migrate Qdrant cũ; (2) canonical merge ở T4.5 siết chặt (số lượng, không trùng/thiếu/thừa
  index, khớp document_id+department_id+object_key); (3) validator bảng bắt buộc row_start/row_end/
  row_count (không còn optional) + 2 rule ngược mới (k)/(l); (4) escape `|`/newline khi render markdown
  table; (5) chính sách race condition draft: last-write-wins, ghi known-gap, không thêm version/lock;
  (6) dọn plan: sửa "SourceLocator 5 field" thành đúng "9 field", đảo thứ tự thực thi T4.5 trước T4.4, xác
  nhận `pnpm typecheck`/`test`/`lint` thật, làm rõ log `TableStructureError` khi lọt API.
- Vẫn CHƯA implement — người dùng đã xác nhận có thể BẮT ĐẦU implement theo kiến trúc v5, nhưng chưa
  archive plan (còn có thể có thêm 1 vòng review sau khi implement 1 phần, hoặc review cuối trước archive
  theo Checkpoint Phase 6/7).
