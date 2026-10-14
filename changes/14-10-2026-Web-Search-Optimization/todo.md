# Todo: Tối ưu web search

Chi tiết (AC, files, size) xem `plan.md`, nhất là mục "Điều chỉnh khi implement".

- [ ] W0 Baseline: `task eval:run env=eval -- --web on --category web_search` (6 câu hiện có ở `dataset/official`, repo gateway) - chạy trước và sau khi bật, cần môi trường eval
- [x] W1 Classify sinh `web` slots (`topics` + `subject` + bậc/hệ/ngành/cơ sở/năm học; không lấy từ `<student_profile>`), normalize, test
- [x] W2 `web_query` ghép bằng code theo mẫu topic (`app/rag/web_query.py`), năm học do code tính, chuẩn hóa alias, không slot thì dùng câu hỏi con, test
- [x] W5' Bộ chấm trang web riêng (`09c_WebJudgeNode`, `agents/web_result_judge.yaml`): `answer` → `<websearch>`, `source` → link; lỗi/thiếu model → luật điểm cũ + `WEB_JUDGE_FAILED`, test
- [x] W7 Fallback mềm `<web_sources>` (`common/web_candidate_sources.yaml`: link + năm học, cấm số), test
- [x] W4 Tavily `advanced` + `chunks_per_source=3` + `max_results=10`, sàn 0.05, xếp/lọc domain theo topic (bỏ trang chủ, trang danh sách, cơ sở khác), test
- [x] W3 Cache Redis (`app/integrations/web_search_cache.py`, db 3, key theo câu hỏi chuẩn hóa + tham số, TTL theo topic, rỗng 30 phút, lỗi không cache, fail-open), test
- [x] W6 Tìm song song ngay đầu nhánh advisory cho topic động; chỉ chấm khi có câu hỏi con không còn chunk, không cần thì chạy nốt ở nền cho cache, test
- [x] W8 Năm học từ title/URL/nội dung, xếp mới trước, ghi trong `<websearch>`/`<web_sources>`; `WEB_SOURCE_USED` + log `Web search turn: mode/topics/cache_hits/...`, test
- [x] Checkpoint: pytest 2152 passed (bỏ `tests/e2e`, lỗi import có sẵn); ruff sạch trên file đã sửa; mypy không thêm lỗi
- [ ] Checkpoint: chạy thật câu "học phí ngành cntt" trên Windows (Tavily + model RERANK thật), xem log `Web search query=` / `Web judge SQ1` / `Web search turn:`
- [ ] W9 Chỉnh nhóm topic / TTL / domain theo số liệu; quyết định có tự fetch trang (www, pdt, tuyensinh lỗi chứng chỉ SSL) không

Chưa làm: lọc website của khoa khác khi câu hỏi đã nêu ngành (cần bảng ngành → khoa); danh sách subdomain cơ sở mới có `qn.iuh.edu.vn`.
