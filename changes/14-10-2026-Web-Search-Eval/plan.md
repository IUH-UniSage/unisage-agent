# Plan: Mở rộng bộ eval web search

Hai repo: `unisage-gateway/dataset/official` (câu hỏi) và `unisage-agent/evals` (chạy và chấm).

## Vấn đề

6 câu `web_search` hiện có (d0271-d0276, bộ demo) chỉ kiểm tra "node web search có chạy"
(`expected_answer` = "Node web search được kích hoạt (không đánh giá nội dung kết quả)"). Thiết kế
mới (`changes/14-10-2026-Web-Search-Optimization`) cần đo thêm: trả lời từ web hay chỉ dẫn link
(fallback mềm), có lấy nhầm trang sai phạm vi (thạc sĩ, cơ sở khác) không, câu kho văn bản trả lời
được có bị dùng web nhầm không, cache trúng bao nhiêu.

## Quyết định

- **Kết quả kỳ vọng theo hành vi, không theo con số.** Không kiểm chứng được số liệu đang có trên
  website trường, nên mỗi câu mới có `expected_web_outcome`:
  - `web_any`: phải dùng website (trả lời từ trang web **hoặc** dẫn link văn bản chính thức);
  - `kb`: kho văn bản trả lời được, không được dùng web.
  Kèm `expected_source_hosts` (trang được phép, VD `ptckt.iuh.edu.vn`) và
  `forbidden_source_patterns` (dấu hiệu sai phạm vi trong URL/tiêu đề, VD `thac-si`,
  `qn.iuh.edu.vn`). `reviewed=false` cho người duyệt.
- **Outcome mới `web_source`:** lượt kết thúc ở ticket fallback nhưng có dẫn link văn bản chính
  thức (`web_sources`). Runner ghi `web_sources` (hook `run_ticket_fallback`) và mã cảnh báo admin.
- **Câu cũ d0271-d0276** đạt khi có `web` hoặc `web_source` (thiết kế mới coi dẫn link là dùng web).
- **Nơi ghi câu mới:** theo tiền lệ `additions_20261013.jsonl`: file nguồn
  `official/question_work/additions_20261014_web_search.jsonl`, nối vào `official/questions.jsonl`
  và `questions_review.csv` với id `w0001`..., `source_set=official`, không tài liệu nguồn (chạy
  được ở cả `--set demo` lẫn `official`).

## Task

### E1 - Runner ghi đủ dữ liệu web
- `recording.py`: hook `run_ticket_fallback` lấy `web_sources`; hook `search_web` ghi query (chuỗi)
  và cộng `cache_hits` từ kết quả.
- `runner.py`: row thêm `web_sources`, `warning_codes`, `web_cache_hits`, `web_queries`.
- **Size:** S.

### E2 - Chấm
- `metrics.py`: `outcome_of` thêm `web_source`; `score_row` cho `web_search`: có
  `expected_web_outcome` thì chấm theo nó + kiểm `expected_source_hosts` /
  `forbidden_source_patterns` trên mọi trang đã dùng; không có (câu cũ) thì `web` hoặc
  `web_source` là đạt. `summarize` thêm số lượt dùng web, tỷ lệ cache hit.
- **AC:** test cho từng nhánh (web, web_source, kb bị dùng web nhầm, trang sai phạm vi).
- **Size:** M.

### E3 - Câu hỏi
- 15 câu: học phí (4, gồm 1 bẫy thạc sĩ và 1 câu cơ sở Quảng Ngãi), tuyển sinh (3), lịch/thông
  báo (3), liên hệ (2), câu kho văn bản trả lời được (3, `kb`). Có `question_user` (văn nói, viết
  tắt, thiếu dấu).
- **Size:** S.

### Checkpoint
- pytest `tests/evals`, ruff; `python -m evals.run --dry-run --category web_search` liệt kê đủ 21 câu.
- Chạy thật: `task eval:run env=eval -- --web on --category web_search` (cần môi trường eval).

## Rủi ro

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| Câu `kb` thật ra không có trong kho eval → bị chấm sai | Trung bình | Chọn từ tài liệu đã thấy trong log (hướng dẫn mật khẩu GV, KTX); `reviewed=false` |
| Website đổi nội dung → câu `web_any` không còn trang | Thấp | Chấm hành vi, không chấm con số |
| `official/questions.jsonl` đang có thay đổi chưa commit của người dùng | Trung bình | Chỉ nối thêm dòng cuối, không sửa dòng cũ; không commit repo gateway |
