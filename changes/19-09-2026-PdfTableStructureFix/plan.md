# Implementation Plan: PDF Table Structure Fix

## Overview

Bảng PDF (mẫu: `Quyet dinh 1035 QD DHCN Hoc phi 2025-2026.pdf`, 4 trang, 1 bảng logic ~118 dòng) hiện được ingest sai:

1. Header 2 tầng bị đọc thành 1 dòng: `column_names = ["TT","HỆĐÀO TẠO","Năm học 2","025 - 2026"]`; dòng `Mức thu 01 tín chỉ | Mức thu theo năm học` bị coi là data row.
2. Bảng bị cắt theo từng trang (parser flush TABLE ở cuối mỗi trang), header lặp ở trang sau; `table_id = table-{block_index}` (không phải thứ tự bảng); `row_start/row_end/row_count` đánh lại từ 1 mỗi trang.
3. Dòng con (`Môn lý thuyết | 980.000`) mất toàn bộ tổ tiên (`Khóa tuyển sinh > Khối > Ngành`) khi chunk theo dòng.
4. Trang 3 bị `pymupdf4llm` parse thành 3 cột (`TT<br>HỆ ĐÀO TẠO` dính nhau, `3.2<br>Khóa tuyển sinh…`), có dòng rác `~~ô~~<br>~~h~~` (mất chữ).

Nhánh `feature/huydh-unisage-84-chat-citation` dùng `page_start/page_end` và `source_locator` để trích dẫn, nên trang và số dòng phải đúng tới từng chunk.

## Nguyên tắc thiết kế (rút ra từ review)

- **Không mất nội dung, không sửa âm thầm.** Mọi dòng không rỗng của nguồn phải nằm trong bảng canonical; nếu không chắc thì giữ raw text và gắn cảnh báo, không bỏ.
- **Không hardcode từ khóa theo tài liệu mẫu** (`Khối`, `Ngành`…); chỉ dùng tín hiệu cấu trúc/hình học.
- **Không tự tin thì không gán**: mọi suy luận (header, merge, tổ tiên) có confidence; dưới ngưỡng thì giữ nguyên dòng, không gán breadcrumb.
- **Có model canonical nhỏ** (xem dưới) thay cho `list[list[str]]`. Đây là abstraction duy nhất được thêm; không thêm hook/lớp khác.

## Architecture Decisions

### AD1. Canonical table model (tiến hóa từ `TableBlock`, không thay tên)

`TableBlock` (`table_aware_parser.py:21`) mở rộng để mang provenance từng dòng; `header_row` và `data_rows` giữ dạng thuộc tính đọc để HTML/DOCX và test cũ không vỡ.

```
TableBlock (canonical table)
  table_id            "table-{thứ tự bảng trong tài liệu}"
  header_levels       list[list[str]]        # từng tầng header, ô gộp đã điền cho mọi cột con
  column_names        list[str]              # phẳng: "Năm học 2025 - 2026 > Mức thu 01 tín chỉ"
  header_source / header_confidence          # giữ nguyên, nay dựa trên bằng chứng thật
  rows: list[TableRow]
    cells            list[str]
    raw_text         str                     # text nguồn trước chuẩn hóa (để đối chiếu/khôi phục)
    row_index        int                     # 1-based trong bảng logic, không tính header
    page_start/page_end  int | None
    ancestors        list[str]               # toàn bộ tổ tiên, gốc → gần nhất
    confidence       float                   # độ tin cậy cấu trúc của dòng (0..1)
    warnings         list[str]               # "cell_recovered_from_geometry", "garbled_text_raw_kept", ...
  warnings           list[str]               # cảnh báo mức bảng (cột không khớp giữa trang...)
  source_rows: list[SourceRow]               # sổ cái nguồn: MỌI hàng nguồn, kể cả hàng không thành data row
    source_row_id    str                     # "p{page}-r{thứ tự hàng nguồn trên trang}" (ổn định, do parser gán)
    page             int | None
    raw_text         str                     # text hàng nguồn nguyên bản (từ markdown)
    source_cell_count int                    # số ô nguồn (từ markdown) và, nếu có, số ô hình học từ find_tables
    disposition      "data" | "header" | "repeated_header" | "merged" | "split" | "dropped_empty"
    canonical_row_ids list[int]              # row_index của các dòng canonical sinh ra từ hàng nguồn này ([] nếu header/lặp/rỗng)
  normalization_events: list[NormalizationEvent]
    kind             "split_cell" | "merge_cells" | "fill_merged_header" | "drop_repeated_header"
                     | "recover_text" | "pad_missing_cell" | "reorder"
    source_row_ids   list[str]
    detail           str                     # mô tả ngắn, ví dụ "split 'TT<br>HỆ ĐÀO TẠO' -> 2 cells"

TableRow còn có: source_row_ids list[str]    # hàng nguồn tạo ra dòng này (1 hoặc nhiều nếu ghép/tách)
                 source_cell_count int, canonical_cell_count int
```

HTML/DOCX điền `page_* = None`, `ancestors = []`, `confidence = 1.0`; `source_rows` được dựng từ chính các hàng đọc (disposition `data`/`header`) để invariant chạy đồng nhất cho mọi nguồn.

**Vì sao cần sổ cái nguồn:** invariant phải chứng minh dữ liệu nguồn không mất, không chỉ kiểm dữ liệu canonical đã tạo. `source_rows` được ghi **trước** mọi bước chuẩn hóa (ngay khi đọc markdown/evidence), các bước sau chỉ được đổi `disposition`/`canonical_row_ids` và thêm `normalization_events`; không bước nào được xóa `source_rows`.

### AD2. Invariants (kiểm bằng test và một hàm kiểm tra chạy trong parser, vi phạm → cảnh báo bảng + log)

- I1 (bảo toàn hàng): mọi `SourceRow` không rỗng có `disposition` hợp lệ; `data`/`merged`/`split` phải có `canonical_row_ids` khác rỗng và các id đó tồn tại trong `rows`; `header` phải khớp một tầng trong `header_levels`; `repeated_header` phải có `NormalizationEvent(drop_repeated_header)`. Đẳng thức đối chiếu: `len(source_rows không rỗng) = #data + #header + #repeated_header + #merged + #split` và `len(rows) = Σ canonical_row_ids` (ghi log cả hai vế khi lệch).
- I2: không dòng nào trong `rows` có `source_row_ids` trỏ tới hàng `header`/`repeated_header`.
- I3 (bảo toàn ô): với mỗi dòng, `canonical_cell_count` khớp `source_cell_count` ± các sự kiện giải thích (`split_cell`, `merge_cells`, `pad_missing_cell`, `recover_text`); chênh lệch không có sự kiện → vi phạm. Nội dung: mọi token (tách theo khoảng trắng, bỏ dấu câu/`<br>`/`~~`) của `raw_text` phải xuất hiện trong `cells` của dòng canonical, trừ khi có sự kiện `recover_text` thay thế.
- I4: `row_index` liên tục 1..N; `page_start <= page_end`; trang không giảm dần theo dòng.
- I5: dòng không phục hồi được giữ `raw_text` trong chunk + `confidence` thấp + warning. **Không quarantine/không bỏ** ở phase này (chưa có UI/retry để hiển thị); quarantine chỉ xét sau nếu có cơ chế đó.

### AD3. Nguồn dữ liệu: markdown cho text, `find_tables()` cho cấu trúc

- Hiện `split_regions` đóng PDF ngay sau `to_markdown`, nên không còn cơ hội lấy bằng chứng hình học. Đổi: khi tài liệu còn mở, thu thập `TableEvidence` theo từng trang (bbox bảng, ranh giới cột, cell/span gộp, số hàng header, x0 nhãn từng dòng, font size/bold nếu cần) và truyền vào bước dựng bảng (`_regions_from_pdf_pages(pages, evidence)`); `document.close()` chỉ sau đó.
- Text ô luôn ưu tiên markdown (đúng dấu/khoảng trắng); `find_tables()`/`page.get_text(clip=...)` dùng để (a) biết ô gộp/số tầng header, (b) căn hàng với markdown, (c) khôi phục text ô bị rác.

### AD4. Ghép bảng liên trang theo bằng chứng, không chỉ so chuỗi header

Điều kiện ghép bảng B (trang p+1) vào bảng A (trang p), tính điểm và có ngưỡng:

- chuỗi header **đã chuẩn hóa** khớp: NFC, casefold, bỏ `<br>`/xuống dòng/dấu câu thừa, ghép token bị tách (`2|025 - 2026`), so sánh theo từng tầng;
- cùng số cột và ranh giới cột tương đối khớp trong dung sai (`x` chuẩn hóa theo bề rộng bảng);
- liên tục về vị trí: A kết thúc gần lề dưới trang, B bắt đầu gần lề trên trang kế, không có heading/đoạn văn (không phải boilerplate) chen giữa;
- header lặp là **tùy chọn**: nếu B không lặp header nhưng số cột + ranh giới khớp + liên tục vị trí thì vẫn ghép (bảng tiếp tục không lặp header).
- Nếu điểm dưới ngưỡng → không ghép, hai bảng riêng, `warnings` ghi lý do.

### AD5. Suy luận tổ tiên bằng chấm điểm nhiều bằng chứng

Mỗi dòng ước lượng `depth` và `is_parent` từ các tín hiệu, mỗi tín hiệu bỏ phiếu kèm trọng số; `ancestors` là **ngăn xếp đầy đủ** (không chỉ cha gần nhất):

1. Đánh số: nhận dạng cột thứ tự tổng quát (số `1`, dotted `1.1.1`, chữ `A`/`a`, La Mã `I`/`iv`, `a)`, `-`) → độ sâu.
2. Thụt đầu dòng: x0 của nhãn (gom cụm các mức thụt) khi không có cột thứ tự.
3. Hình học ô gộp: ô nhãn trải nhiều cột / dòng có ô gộp là dòng nhóm.
4. Font: đậm/lớn hơn dòng kế tiếp.
5. Số ô có dữ liệu và **mặt nạ ô số** (dòng cha có thể có số tổng, nên không coi "có số" là lá).
6. Chiều cao dòng/khoảng cách so với dòng trước/sau.

`confidence` = mức đồng thuận giữa các tín hiệu có mặt. Dưới ngưỡng → dòng không có `ancestors` (giữ nguyên), cảnh báo `hierarchy_uncertain`. Không tín hiệu nào dùng từ khóa ngôn ngữ.

### AD6. Render chunk

Nội dung chunk = `heading_path` + header phẳng (mọi tên cột xuất hiện trong content) + các **nhóm dòng**. Quy tắc bắt buộc, theo từng dòng:

- Mỗi dòng luôn được render cùng **ancestors đầy đủ của chính nó**; không dùng "breadcrumb chung của chunk" và không rút gọn theo tiền tố chung, vì một chunk có thể chứa dòng thuộc hai cha khác nhau (`Khối Công nghệ > Môn lý thuyết` rồi `Khối Kinh tế > Môn lý thuyết`).
- Để đỡ lặp, các dòng **liên tiếp có ancestors giống hệt nhau** được gom thành một nhóm, nhóm đó in dòng `ancestors` đầy đủ một lần rồi liệt kê dòng. Khi ancestors đổi, mở nhóm mới với breadcrumb đầy đủ của nhóm mới. Chunk có thể chứa nhiều nhóm.
- Breadcrumb đầy đủ cộng vào `usable_budget` theo nhóm; nhóm lớn hơn ngân sách thì tách chunk như hiện tại (mỗi chunk mới mở lại nhóm với breadcrumb đầy đủ).

Chunk vượt ranh giới trang có `page_start/page_end` tính từ các dòng thuộc chunk.

### AD7. Metadata

`Chunk` thêm `structure_confidence: float | None` (nhỏ nhất của các dòng) và `parse_warnings: list[str]`; đi qua schema → DB (migration alembic) → Qdrant payload → `celery_app.py`. Tổ tiên nằm trong `content`, không thêm trường payload riêng.

### AD8. Phiên bản, dữ liệu cũ, Qdrant (đã đọc code, không suy đoán)

Hiện trạng:

- `CHUNKING_VERSION = "2026-09-structural-v1"` (`app/core/config.py:41`). Bản nháp chỉ bị từ chối khi `chunking_version == "legacy"` (`app/api/v1/ingestion.py:199`) → bản nháp `structural-v1` cũ vẫn embed được sau khi bump.
- `qdrant_store` chỉ có `upsert_chunk`, point id = `uuid5(f"{document_id}:{chunk_index}")` (`celery_app.py:77-78`), không có delete; retrieval không lọc theo `chunking_version` (chỉ ghi nhận, xem "Ngoài phạm vi" bên dưới).

Quyết định đề xuất:

- Bump `CHUNKING_VERSION` lên `2026-09-structural-v2`.
- Bản nháp có `chunking_version != settings.CHUNKING_VERSION` bị từ chối (409, phải chunk lại), mở rộng kiểm tra hiện chỉ áp cho `legacy`.
**Ngoài phạm vi (đã chốt với người dùng):** không làm gì liên quan đến sửa/xóa chunk đã embed vào Qdrant. Việc xóa chunk hiện có chỉ là chỉnh sửa trước lần embed đầu tiên (bản nháp trong wizard) và giữ nguyên. Do đó **không** thêm `delete_stale_points`, không thêm script reindex, không dọn điểm cũ, không re-embed tài liệu đã index.

Hệ quả cần ghi rõ (known limitation, không xử lý trong plan này): tài liệu PDF có bảng đã embed bằng `structural-v1` giữ nguyên chunk cũ; nếu sau này có luồng re-embed cùng `document_id` thì điểm cũ chỉ số cao còn dư (do chỉ có `upsert`, id = `uuid5(document_id:chunk_index)`), và retrieval không lọc theo `chunking_version`. Ghi nhận trong PR; xử lý thuộc một plan riêng khi cần re-embed.

Phạm vi thực làm: bump version + chặn embed bản nháp version cũ (bản nháp là dữ liệu trước embed, nên thuộc phạm vi).

## AD9. Chấm điểm deterministic: trọng số, ngưỡng, tie-break

Mọi hằng số nằm ở **một chỗ** (một `dataclass` frozen/dict hằng ở đầu module parser), có test ghim giá trị và test đúng tại biên ngưỡng. Giá trị dưới đây là **khởi điểm** để hiệu chỉnh một lần bằng fixture ở Task 5/6, sau đó đóng băng (đổi phải sửa test). Không có bước học/ngẫu nhiên; cùng input luôn ra cùng output.

**Chuẩn chung (áp cho merge và hierarchy):** mỗi tín hiệu trả `(vote, weight)` hoặc `None` nếu không có dữ liệu (vd không có geometry). Tín hiệu `None` bị loại, điểm được **chuẩn hóa lại** trên tổng trọng số của các tín hiệu có mặt: `score = Σ(w_i · s_i) / Σ(w_i có mặt)`. Có thêm `MIN_PRESENT_WEIGHT`: nếu tổng trọng số có mặt thấp hơn ngưỡng thì kết quả là **không đủ bằng chứng** (không merge / không gán ancestors), không đoán.

### Ghép bảng (AD4)

| Tín hiệu | Trọng số | Giá trị `s` |
| --- | --- | --- |
| Header đã chuẩn hóa khớp (từng tầng) | 0.35 | tỷ lệ tầng/cột khớp, 0..1 |
| Cùng số cột | 0.20 | 1 hoặc 0 |
| Ranh giới cột tương đối khớp | 0.20 | 1 − (sai lệch trung bình / dung sai), kẹp 0..1; dung sai 3% bề rộng bảng |
| Liên tục vị trí (A sát lề dưới, B sát lề trên, không nội dung chen giữa) | 0.25 | 1 hoặc 0; có heading/đoạn văn không phải boilerplate chen giữa → **chặn cứng** |

- Chặn cứng (không ghép bất kể điểm): khác số cột; có heading/đoạn văn chen giữa; B không ở trang liền sau A.
- `MERGE_THRESHOLD = 0.75`; header lặp vắng (bảng tiếp tục không lặp header) thì tín hiệu header là `None`, ngưỡng nâng lên `0.80`.
- `MIN_PRESENT_WEIGHT = 0.50` (không có geometry và không có header → không ghép).
- Điểm trong `[0.65, 0.75)` → không ghép, ghi warning `merge_borderline` (để soi bằng mắt), không ghép ngầm.

### Phân cấp (AD5)

Mỗi tín hiệu bỏ phiếu cho `depth` ứng viên của dòng (số nguyên ≥ 0) hoặc `None`.

| Tín hiệu | Trọng số |
| --- | --- |
| Đánh số (cột thứ tự, độ sâu theo kiểu ký tự/số chấm) | 0.35 |
| Thụt đầu dòng (cụm x0 nhãn, xếp hạng theo thứ tự tăng dần) | 0.25 |
| Ô gộp/nhãn trải nhiều cột | 0.15 |
| Font đậm/lớn hơn dòng kế | 0.10 |
| Mặt nạ ô số/số ô có dữ liệu (dòng khác mặt nạ với hàng xóm) | 0.10 |
| Chiều cao/khoảng cách dòng | 0.05 |

- `depth` thắng = ứng viên có tổng trọng số cao nhất. `confidence = trọng_số_thắng / tổng_trọng_số_có_mặt`.
- `MIN_PRESENT_WEIGHT = 0.25` (đã hạ từ 0.30 khi implement: thụt lề đơn lẻ có trọng số 0.25 và là bằng chứng đủ dùng); `ASSIGN_THRESHOLD = 0.60`: dưới ngưỡng → không gán `ancestors` cho dòng (giữ nguyên), warning `hierarchy_uncertain`.
- **Tie-break:** nếu tổng trọng số của ứng viên nhất và nhì chênh nhau `< 0.10` (theo trọng số tuyệt đối, sau chuẩn hóa) → coi là không xác định (không gán), **không** chọn ứng viên sâu hơn/nông hơn. Trường hợp hai ứng viên bằng nhau tuyệt đối cũng xử như vậy, nên kết quả không phụ thuộc thứ tự duyệt.
- Ràng buộc nhất quán: `depth` một dòng không tăng quá +1 so với dòng trước trong cùng phạm vi (nhảy cấp bị kẹp về +1 và ghi warning `depth_clamped`); ancestors dựng từ ngăn xếp `depth` nên luôn đầy đủ.
- Khi hình học thiếu (PDF không có evidence, hoặc HTML/DOCX): các tín hiệu 2, 3, 4, 6 là `None`, điểm chuẩn hóa lại trên tín hiệu còn lại; nếu tổng có mặt < 0.25 thì không gán.
- `structure_confidence` của chunk = `min(confidence)` các dòng trong chunk có gán ancestors; dòng không gán không kéo `min` về 0 nhưng thêm warning vào chunk.

## Task List

Thứ tự theo phụ thuộc: contract → bằng chứng → căn chỉnh → ghép → suy luận → render → metadata → version → test.

### Phase 1: Contract

- [x] Task 1: Canonical table model + invariants (M)

### Phase 2: Bằng chứng cấu trúc

- [x] Task 2: Truyền `find_tables()` evidence vào parse khi PDF còn mở + spike chốt cách căn hàng (M)
- [x] Task 3: Căn hàng markdown ↔ evidence, chuẩn hóa số cột, header nhiều tầng, bỏ header lặp, khôi phục text rác (M)

### Checkpoint: Structure

- [x] Trang 2/3/4 của file mẫu ra cùng số cột; `header_levels` 2 tầng; không dòng header lặp; dòng rác được khôi phục hoặc giữ raw + warning; invariants I1-I3 xanh

### Phase 3: Bảng logic

- [x] Task 4: Ghép bảng liên trang theo điểm số + provenance (page/row_index/table_id) (M)

### Phase 4: Phân cấp

- [x] Task 5: Suy luận tổ tiên từ tín hiệu văn bản (đánh số tổng quát, mặt nạ ô số, số ô có dữ liệu) + confidence (M)
- [x] Task 6: Bổ sung tín hiệu hình học (thụt lề, ô gộp, font, chiều cao dòng) vào chấm điểm (M)

### Checkpoint: Hierarchy

- [x] Ancestors đầy đủ ≥ 3 cấp trên file mẫu; bảng không nhận diện được giữ nguyên, không breadcrumb sai

### Phase 5: Chunk và lưu trữ

- [x] Task 7: Render chunk (ancestor breadcrumb + header phẳng) + page range từng chunk (M)
- [x] Task 8: `structure_confidence`/`parse_warnings` qua schema → DB migration → Qdrant → celery (M)
- [x] Task 9: Version v2 + từ chối bản nháp version cũ (S). Không đụng chunk đã embed vào Qdrant

### Phase 6: Xác nhận

- [x] Task 10: Test matrix + fixture PDF tổng hợp (không phải học phí) + regression PDF thật (M)

### Checkpoint: Complete

- [x] Toàn bộ test xanh, ruff sạch, 4 vấn đề trong Overview đã hết trên file mẫu
- [ ] Known limitation về tài liệu đã embed được ghi trong mô tả PR

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Không căn được hàng markdown ↔ `find_tables()` (số hàng khác nhau vì gộp/tách) | High | Task 2 spike trước khi code Task 3; căn theo thứ tự + so khớp text mờ; không căn được thì giữ hàng markdown + warning, không bỏ |
| Ghép nhầm hai bảng khác nhau có cùng header | High | AD4 dùng điểm số + liên tục vị trí; test riêng hai bảng cùng header ở hai trang liên tiếp |
| Bản nháp `structural-v1` cũ vẫn embed được | High | AD8: từ chối khi version ≠ hiện tại |
| Tài liệu đã embed bằng v1 giữ chunk cũ; re-embed cùng `document_id` sau này sẽ để lại điểm dư | Low (ngoài phạm vi) | Đã chốt không làm sửa/xóa chunk đã embed; ghi known limitation trong PR |
| Chunk chứa hai cha khác nhau bị mất ngữ cảnh | High | AD6: ancestors đầy đủ theo từng dòng/nhóm; test hai cha liên tiếp trong một chunk |
| Ngưỡng/trọng số cảm tính, test không lặp lại được | Med | AD9: hằng số một chỗ, tie-break xác định, test ghim giá trị và biên ngưỡng |
| Suy luận tổ tiên sai ở bảng borderless/OCR/thụt lề/dòng cha có số tổng | Med | AD5 nhiều tín hiệu + confidence + ngưỡng; test trên ≥2 bảng phân cấp không phải học phí; không chắc thì không gán |
| Mất/đổi nội dung khi chuẩn hóa | Med | Invariants I1-I5; `raw_text` giữ nguyên; đếm dòng trước/sau được ghi log và test |
| Schema mở rộng làm vỡ dữ liệu/test hiện có | Med | Trường mới có default; migration alembic; `TableBlock.header_row/data_rows` giữ dạng view |
| Chunk dài hơn do breadcrumb | Low | `usable_budget` đã tính prefix; test breadcrumb dài + dòng oversized |

## Open Questions

- Từ chối bản nháp version cũ: dùng lại `EmbeddingDraftLegacyException` hay tách exception/message riêng?
- Có muốn `parse_warnings` hiển thị trên UI wizard chunk (unisage-web) ở phase này không, hay chỉ lưu metadata?

## Spike result

Chạy trên file mẫu (trang 2/3/4):

- Số hàng `find_tables()` = số hàng markdown trên cả 3 trang (33/43/43) → **căn hàng theo chỉ số 1:1** là đủ cho mẫu; khi số bảng hoặc số hàng trên một trang không khớp thì bảng đó rơi về markdown-only + warning `evidence_unaligned`.
- Text từng ô lấy bằng `page.get_text("dict", clip=cell)` sạch dấu/khoảng trắng hơn markdown (`Trụ sở chính` vs `Trụsởchính`, `trở về trước` vs `trởvềtrước`); đây là nguồn text canonical. Markdown vẫn là sổ cái nguồn (`raw_text`); so khớp bằng multiset ký tự chữ-số (`normalize_for_compare`).
- Cột TT của trang 3 **có** trong hình học (x=51) dù markdown gộp `TT<br>HỆ ĐÀO TẠO` và `3.2<br>Khóa tuyển sinh…` → chuẩn hóa 3→4 cột lấy từ hình học, ghi `split_cell`.
- Ô gộp header: hàng 0 có `None` ở cột 4 (span ngang của `Năm học 2025 - 2026`); số tầng header suy từ hình học: hàng bắt đầu trong phạm vi dọc của ô cao nhất hàng 0 (`TT` cao 34pt phủ 2 hàng) → `header_row_count = 2`. Không dùng từ khóa.
- Dòng rác `~~ô~~<br>~~h~~<br>~~hà h~~ | ~~1 130 000~~` hỏng **ngay trong PDF** (geometry cũng ra `ô / h / hà h`) → không khôi phục được; giữ raw + `garbled_text_raw_kept` + confidence thấp, không bỏ.
- Evidence cũng có x0 chữ (thụt lề), bold, font size, bbox ô cho Task 6.
