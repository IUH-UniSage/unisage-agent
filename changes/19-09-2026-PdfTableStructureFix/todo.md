# Todo: PDF Table Structure Fix

Repo: unisage-agent, nhánh `feature/huydh-unisage-84-chat-citation`. Mẫu: `_to_delete/Quyet dinh 1035 QD DHCN Hoc phi 2025-2026.pdf`. Thiết kế chi tiết: `plan.md` (AD1-AD8).

Lệnh test chung: `.venv/Scripts/python -m pytest tests/<file>`; lint: `.venv/Scripts/ruff check .`

## Task 1: Canonical table model + invariants

**Description:** Mở rộng `TableBlock` thành canonical table (AD1): `table_id`, `header_levels`, `column_names`, `rows: list[TableRow]` (cells, raw_text, row_index, page_start/page_end, ancestors, confidence, warnings). Giữ `header_row`/`data_rows` là thuộc tính đọc. Thêm sổ cái nguồn `source_rows` (source_row_id, page, raw_text, source_cell_count, disposition, canonical_row_ids), `normalization_events[]`, và `source_row_ids`/`source_cell_count`/`canonical_cell_count` trên `TableRow` (AD1). Sổ cái được ghi trước mọi bước chuẩn hóa; các bước sau chỉ đổi disposition và thêm sự kiện. Thêm hàm kiểm tra invariants I1-I5 (AD2) dựa trên sổ cái. HTML/DOCX điền giá trị mặc định (sổ cái dựng từ các hàng đã đọc).

**Acceptance criteria:**

- [x] HTML/DOCX/PDF hiện có vẫn ra kết quả cũ (test cũ không đổi vẫn xanh)
- [x] Invariant kiểm được dữ liệu **nguồn**, không chỉ canonical: test dựng sổ cái cố tình vi phạm và bắt được từng loại — hàng nguồn không có disposition/không sinh dòng canonical; hàng `repeated_header` không có sự kiện `drop_repeated_header`; dòng canonical trỏ tới hàng header; chênh số ô không có sự kiện giải thích; token của `raw_text` biến mất không có `recover_text`; row_index không liên tục; trang giảm dần
- [x] Đẳng thức đối chiếu số hàng nguồn ↔ số dòng canonical của I1 có trong log khi lệch
- [x] Không thêm hook/lớp nào ngoài model này

**Verification:**

- [x] `pytest tests/test_table_block_parsing.py tests/test_table_aware_parser.py tests/test_chunking_table_row.py`
- [x] Test đơn vị cho từng invariant (cố tình vi phạm → bắt được)

**Dependencies:** None

**Files likely touched:** `app/rag/ingestion/table_aware_parser.py`, `tests/test_table_block_parsing.py`

**Estimated scope:** Medium

## Task 2: Đưa `find_tables()` evidence vào parse + spike căn hàng

**Description:** `split_regions` hiện đóng PDF ngay sau `to_markdown`. Đổi để khi PDF còn mở thu `TableEvidence` theo trang (bbox, ranh giới cột, ô gộp/số tầng header, x0 nhãn từng hàng, font/bold) và truyền vào `_regions_from_pdf_pages`. Spike trên file mẫu: chọn cách căn hàng markdown ↔ `find_tables()` (theo thứ tự + so khớp text mờ), ghi kết quả vào plan.md.

**Acceptance criteria:**

- [x] `document.close()` chỉ chạy sau khi evidence đã thu và parse xong
- [x] Evidence có: số tầng header, span ô gộp của "Năm học 2025 - 2026", ranh giới cột từng trang
- [x] Spike result ghi vào plan.md: tỷ lệ hàng căn được trên trang 2/3/4, các hàng không căn được và lý do

**Verification:**

- [x] `pytest tests/test_table_aware_parser.py` (không đổi hành vi khi chưa dùng evidence)
- [x] Script spike in ra evidence trang 2, 3, 4

**Dependencies:** Task 1

**Files likely touched:** `app/rag/ingestion/table_aware_parser.py`, `tests/test_table_aware_parser.py`

**Estimated scope:** Medium

## Task 3: Căn hàng, chuẩn hóa cột, header nhiều tầng, bỏ header lặp, khôi phục text rác

**Description:** Căn từng hàng markdown với evidence; chuẩn hóa mọi trang về cùng số cột (trang 3: tách `TT<br>HỆ ĐÀO TẠO`, tách `3.2<br>Khóa tuyển sinh…` thành TT + nhãn); dựng `header_levels` và `column_names` phẳng, điền ô gộp cho cột con, ghép token bị cắt (`Năm học 2|025 - 2026`); nhận diện và bỏ header lặp; hàng rác (`~~ô~~<br>~~h~~`) khôi phục từ `find_tables()`/`get_text(clip=...)`, nếu không được thì giữ raw text + `confidence` thấp + warning.

**Acceptance criteria:**

- [x] `column_names` gồm `TT`, `HỆ ĐÀO TẠO`, `Năm học 2025 - 2026 > Mức thu 01 tín chỉ`, `Năm học 2025 - 2026 > Mức thu theo năm học`
- [x] Trang 2/3/4 cùng số cột; không dòng header nào còn trong `rows`
- [x] Dòng "Môn thực hành 1.130.000" của mục 3.4 được khôi phục hoặc giữ raw + warning; không bị bỏ
- [x] Mỗi thao tác tách/ghép/điền/bỏ/khôi phục/đệm ô ghi một `NormalizationEvent` với `source_row_ids`; sổ cái `source_rows` không bị xóa hay ghi đè
- [x] I1, I2, I3 xanh trên sổ cái; số dòng nguồn ↔ canonical được giải thích bằng disposition/sự kiện

**Verification:**

- [x] `pytest tests/test_table_block_parsing.py tests/test_table_aware_parser.py`
- [x] Test fixture markdown: header 1/2/3 tầng, ô gộp bị cắt, thiếu cell, cột lệch giữa trang

**Dependencies:** Task 2

**Files likely touched:** `app/rag/ingestion/table_aware_parser.py`, `tests/test_table_block_parsing.py`, `tests/test_table_aware_parser.py`

**Estimated scope:** Medium

## Checkpoint: After Tasks 1-3

- [x] Toàn bộ test liên quan xanh
- [x] Trang 2/3/4 file mẫu: cùng số cột, header 2 tầng, không header lặp, invariants I1-I3 xanh
- [ ] Review với người dùng trước khi sang ghép bảng (bỏ qua theo chỉ đạo "implement hết")

## Task 4: Ghép bảng liên trang + provenance

**Description:** Ghép bảng theo điểm số AD4 với trọng số/ngưỡng của AD9 (header 0.35, số cột 0.20, ranh giới cột 0.20, liên tục vị trí 0.25; ngưỡng 0.75, nâng 0.80 khi không có header lặp; chặn cứng khi khác số cột/có nội dung chen giữa/không liền trang; `[0.65,0.75)` → không ghép + `merge_borderline`). Header chuẩn hóa: NFC, casefold, bỏ `<br>`, ghép token. Gán `table_id = table-{thứ tự bảng}`, `row_index` liên tục, `page_start/page_end` từng dòng; hàng header lặp chuyển disposition `repeated_header` + sự kiện, không xóa khỏi sổ cái.

**Acceptance criteria:**

- [x] Hằng số nằm một chỗ; test ghim giá trị, test đúng biên (0.7499 không ghép, 0.75 ghép; 0.7999/0.80 khi không header lặp), test tín hiệu `None` được chuẩn hóa lại và dưới `MIN_PRESENT_WEIGHT=0.50` không ghép
- [x] Cùng input chạy hai lần ra cùng kết quả
- [x] File mẫu ra đúng 1 bảng logic, `row_index` 1..N liên tục qua trang
- [x] Hai bảng khác nhau có cùng header ở hai trang liên tiếp **không** bị ghép khi vị trí/hình học không liên tục
- [x] Bảng tiếp tục sang trang không lặp header vẫn được ghép
- [x] Header khác nhẹ (xuống dòng, khoảng trắng, token bị cắt) vẫn ghép
- [x] Có heading/footer chen giữa → không ghép
- [x] I4 xanh

**Verification:**

- [x] `pytest tests/test_table_aware_parser.py tests/test_chunking_table_row.py`
- [x] Fixture PDF tổng hợp tạo bằng `pymupdf` (vẽ đường kẻ + chữ) cho các tình huống trên

**Dependencies:** Task 3

**Files likely touched:** `app/rag/ingestion/table_aware_parser.py`, `app/rag/chunking/table_row.py` (table_id/row range), `tests/test_table_aware_parser.py`, `tests/test_chunking_table_row.py`

**Estimated scope:** Medium

## Task 5: Suy luận tổ tiên từ tín hiệu văn bản + confidence

**Description:** Chấm điểm phân cấp (AD5, tín hiệu 1, 5): nhận dạng cột thứ tự tổng quát (số, dotted nhiều cấp, chữ hoa/thường, La Mã, `a)`), số ô có dữ liệu, mặt nạ ô số; dòng cha có thể có số tổng. Xuất `ancestors` (ngăn xếp đầy đủ) và `confidence` từng dòng; dưới ngưỡng thì để trống + `hierarchy_uncertain`. Không dùng từ khóa ngôn ngữ. Dùng đúng cơ chế AD9: mỗi tín hiệu bỏ phiếu depth kèm trọng số (đánh số 0.35, mặt nạ ô số 0.10 ở task này; các tín hiệu còn lại là `None` cho tới Task 6), chuẩn hóa lại theo tổng trọng số có mặt; `ASSIGN_THRESHOLD=0.60`, `MIN_PRESENT_WEIGHT=0.30`; tie-break: hai ứng viên chênh `< 0.10` (hoặc bằng nhau) → không gán; `depth` không tăng quá +1 so với dòng trước (kẹp + `depth_clamped`).

**Acceptance criteria:**

- [x] Hằng số một chỗ, có test ghim và test biên ngưỡng; test tie-break (hai ứng viên bằng nhau → không gán, kết quả không phụ thuộc thứ tự duyệt)
- [x] Khi thiếu tín hiệu, điểm chuẩn hóa lại; tổng có mặt `< 0.30` → không gán

- [x] File mẫu: dòng `Môn lý thuyết` ở đầu trang 3 có ancestors đủ `[A/B…, Đại học chính quy, Khóa tuyển sinh…, Khối…]`
- [x] Parent có số tổng (`Khối Công nghệ | | 38.350.000`) được nhận là cha; `Môn QPAN` (lá đơn lẻ) không bị nhận nhầm là cha
- [x] Bảng phẳng và bảng không có cột thứ tự: không gán ancestors sai (test âm)
- [x] Không có từ khóa tiếng Việt hardcode trong logic

**Verification:**

- [x] `pytest tests/test_table_aware_parser.py` (test đánh số: 1.1.1, A/I/a, La Mã)
- [x] Ít nhất 2 bảng phân cấp không phải học phí (vd danh mục môn học theo chương/mục, ngân sách theo I/II/a/b)

**Dependencies:** Task 4

**Files likely touched:** `app/rag/ingestion/table_aware_parser.py` (hoặc module nhỏ cạnh đó nếu quá dài), `tests/test_table_aware_parser.py`

**Estimated scope:** Medium

## Task 6: Tín hiệu hình học cho phân cấp

**Description:** Thêm tín hiệu 2, 3, 4, 6 (AD5): thụt đầu dòng (cụm x0 của nhãn), ô gộp, font đậm/lớn, chiều cao dòng/khoảng cách. Cho phép bảng không có cột thứ tự (chỉ thụt lề) vẫn suy ra được tổ tiên; confidence phản ánh đồng thuận. Trọng số theo AD9 (thụt lề 0.25, ô gộp 0.15, font 0.10, chiều cao dòng 0.05); hiệu chỉnh một lần trên fixture rồi đóng băng bằng test.

**Acceptance criteria:**

- [x] Sau hiệu chỉnh, trọng số/ngưỡng được ghim bằng test; đổi giá trị phải sửa test
- [x] Geometry thiếu (không có evidence) → tín hiệu `None`, điểm chuẩn hóa lại, kết quả xác định

- [x] Fixture PDF chỉ phân cấp bằng thụt lề (không số thứ tự) → ancestors đúng
- [x] Tín hiệu xung đột → confidence giảm, dưới ngưỡng thì không gán
- [x] PDF borderless không làm crash; degrade về "không gán"

**Verification:**

- [x] `pytest tests/test_table_aware_parser.py`

**Dependencies:** Task 5

**Files likely touched:** `app/rag/ingestion/table_aware_parser.py`, `tests/test_table_aware_parser.py`

**Estimated scope:** Medium

## Checkpoint: After Tasks 4-6

- [x] Ancestors đầy đủ ≥ 3 cấp trên file mẫu; bảng không nhận diện được giữ nguyên
- [ ] Review kết quả chunk mẫu với người dùng

## Task 7: Render chunk + page range từng chunk

**Description:** `TableRowChunker` render theo AD6: `heading_path` + header phẳng (mọi tên cột có trong `content`) + các nhóm dòng. Mỗi dòng luôn có ancestors **đầy đủ của chính nó**; các dòng liên tiếp có ancestors giống hệt được gom thành một nhóm in breadcrumb đầy đủ một lần; ancestors đổi thì mở nhóm mới với breadcrumb đầy đủ (không dùng breadcrumb chung/rút gọn tiền tố). `page_start/page_end` của chunk tính từ các dòng thuộc chunk; `row_start/row_end/row_count` theo bảng logic.

**Acceptance criteria:**

- [x] Một chunk chứa hai cha liên tiếp (`Khối Công nghệ > Môn lý thuyết` rồi `Khối Kinh tế > Môn lý thuyết`): `chunk.content` phân biệt được cả hai, mỗi dòng gắn đúng cha của nó, không cha nào bị mất
- [x] Nhóm vượt ngân sách token tách chunk, chunk mới mở lại nhóm với breadcrumb đầy đủ
- [x] `chunk.content` (không chỉ metadata) chứa toàn bộ tên cột phẳng và toàn bộ tổ tiên của các dòng
- [x] Chunk vượt ranh giới trang có `page_start != page_end` đúng
- [x] Breadcrumb dài + dòng oversized vẫn trong `max_tokens`; lỗi cấu hình có exception rõ như hiện tại
- [x] Dòng warning được giữ (raw text) trong chunk

**Verification:**

- [x] `pytest tests/test_chunking_table_row.py tests/test_chunk_validation.py`

**Dependencies:** Task 6

**Files likely touched:** `app/rag/chunking/table_row.py`, `app/rag/chunking/validation.py`, `tests/test_chunking_table_row.py`

**Estimated scope:** Medium

## Task 8: Metadata `structure_confidence` / `parse_warnings`

**Description:** Thêm hai trường vào `Chunk` (default an toàn) và đi qua schema → DB (alembic migration) → `qdrant_store.ChunkPoint`/payload → `celery_app.py`.

**Acceptance criteria:**

- [x] Chunk cũ (không có trường) vẫn đọc được, mặc định `None`/`[]`
- [x] Round-trip DB và Qdrant giữ nguyên giá trị
- [x] Không cần migration: DB lưu mọi trường ngoài `chunk_index/content/region_type` vào cột JSON `metadata` qua `model_dump`, hai trường mới tự round-trip (test repository)

**Verification:**

- [x] `pytest tests/test_ingestion_schemas.py tests/test_qdrant_store.py tests/test_embed_chunks_task.py tests/test_ingestion_job_repository.py`
- [x] (N/A vì không có migration) round-trip DB được kiểm bằng `tests/test_ingestion_job_repository.py`

**Dependencies:** Task 7

**Files likely touched:** `app/schemas/ingestion.py`, `migrations/versions/*`, `app/rag/vectorstore/qdrant_store.py`, `app/worker/celery_app.py`, `app/database/models.py`

**Estimated scope:** Medium

## Task 9: Version v2 + từ chối bản nháp version cũ

**Description:** (AD8) Bump `CHUNKING_VERSION` lên `2026-09-structural-v2`; bản nháp có `chunking_version != settings.CHUNKING_VERSION` bị từ chối 409 khi embed (mở rộng kiểm tra `legacy` ở `ingestion.py:199`). **Không** làm gì với chunk đã embed vào Qdrant: không xóa điểm, không dọn stale, không script reindex, không re-embed (đã chốt ngoài phạm vi). Xóa chunk trong wizard trước lần embed đầu tiên giữ nguyên.

**Acceptance criteria:**

- [x] Draft `structural-v1` bị từ chối khi embed; draft `v2` embed được như cũ
- [x] Không có thay đổi nào ở `qdrant_store.py` ngoài việc tải thêm trường Task 8
- [x] Hành vi xóa chunk trong wizard trước embed không đổi (test cũ vẫn xanh)
- [ ] Mô tả PR ghi known limitation: tài liệu PDF có bảng đã embed bằng v1 giữ chunk cũ; xử lý re-embed thuộc plan riêng (làm khi tạo PR)

**Verification:**

- [x] `pytest tests/test_ingestion_embedding.py tests/test_embed_chunks_task.py tests/test_chunk_validation.py`

**Dependencies:** Task 8

**Files likely touched:** `app/core/config.py`, `app/api/v1/ingestion.py`, `app/core/exceptions.py` (nếu tách exception), `tests/test_ingestion_embedding.py`

**Estimated scope:** Small

## Task 10: Test matrix + fixture + regression PDF thật

**Description:** Hoàn thiện test theo ma trận, không dùng mỗi file `_to_delete` làm regression. Fixture: markdown nhỏ, PDF tổng hợp tạo bằng `pymupdf`, PDF thật, ≥2 bảng phân cấp không liên quan học phí.

**Acceptance criteria (ma trận):**

- [x] Header 1/2/3 tầng; ô gộp và header bị cắt
- [x] 3/4/5 cột; thiếu cell; cell rỗng
- [x] Có/không cột TT; đánh số số/chữ/La Mã/nhiều cấp; chỉ thụt lề
- [x] Parent có số tổng; nhiều ancestor; child bắt đầu ở trang sau
- [x] Hai bảng khác nhau cùng header; bảng tiếp tục không lặp header; header đổi nhẹ; footer/heading chen giữa
- [x] Dòng lỗi/thiếu ký tự; chunk vượt ranh giới trang; breadcrumb dài + dòng oversized
- [x] Kiểm tra `chunk.content` chứa mọi tên cột và mọi tổ tiên
- [x] Hai cha liên tiếp trong cùng một chunk: content phân biệt được cả hai
- [x] Ngưỡng/trọng số AD9: test biên, tie-break, thiếu geometry chuẩn hóa lại, chạy hai lần cùng kết quả
- [x] Invariants I1-I5 chạy trên mọi fixture, dựa trên sổ cái `source_rows`
- [x] Regression file mẫu: đúng số bảng, header, table_id, row range, page range, ancestors, mọi dòng nguồn có mặt

**Verification:**

- [x] `.venv/Scripts/python -m pytest` (toàn bộ) và `.venv/Scripts/ruff check .`

**Dependencies:** Task 9

**Files likely touched:** `tests/test_table_aware_parser.py`, `tests/test_table_block_parsing.py`, `tests/test_chunking_table_row.py`, `tests/fixtures/*`

**Estimated scope:** Medium

## Checkpoint: Complete

- [x] Toàn bộ test xanh, ruff sạch
- [x] Bốn vấn đề trong Overview đã hết trên file mẫu
- [ ] Known limitation về tài liệu đã embed được ghi trong mô tả PR (làm khi tạo PR)
- [ ] Người dùng review và duyệt

## Implementation notes (khác với plan ban đầu)

- Module mới: `canonical_table.py` (model + invariants), `table_evidence.py`, `table_normalizer.py`, `table_merger.py`, `table_hierarchy.py` (tách khỏi `table_aware_parser.py` cho gọn).
- AD9 hierarchy: `min_present_weight` hạ 0.30 → 0.25 (thụt lề đơn lẻ 0.25 là bằng chứng đủ dùng); dòng có số thứ tự chỉ do tín hiệu đánh số quyết định (các tín hiệu khác dùng thang khác, bỏ phiếu sẽ mâu thuẫn); dòng không đánh số trong bảng có đánh số nhận thêm "prior" đánh số (0.35); tín hiệu nhị phân (span/font/height/mask) chỉ hoạt động khi phạm vi có cả hai lớp; dòng có text hỏng bỏ tín hiệu hình học.
- AD9 merge: thêm chặn cứng `position_not_continuous` khi cả hai bảng có hình học mà bảng đầu không sát đáy trang/bảng sau không sát đầu trang.
- Chân trang kiểu `Trang 2/4` khác nhau mỗi trang được coi là "furniture" (dòng lặp lại sau khi che chữ số) và không chặn ghép; region đó vẫn giữ trong output.
- Task 8: không cần migration Alembic (xem trên). `parse_warnings` chưa hiển thị trên UI wizard, chỉ lưu metadata.
- Task 9: dùng lại `EmbeddingDraftLegacyException` (409) với thông điệp nêu cả "phiên bản chia đoạn cũ" (trả lời câu hỏi mở).
- `tests/test_config.py::test_new_settings_have_sane_defaults` fail từ trước khi làm (kỳ vọng `RERANK_SCORE_THRESHOLD == 0.70`, cấu hình thực là 0.4), không liên quan.
