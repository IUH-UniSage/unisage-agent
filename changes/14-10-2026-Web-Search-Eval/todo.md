# Todo: Mở rộng bộ eval web search

Chi tiết xem `plan.md`.

- [x] E1 Runner: `web_sources`, `warning_codes`, `web_cache_hits`, `web_queries`, các kỳ vọng web trong row
- [x] E2 Metrics: outcome `web_source`, chấm theo `expected_web_outcome` + `expected_source_hosts` / `forbidden_source_patterns`, `summary["web"]`, test
- [x] E3 16 câu `w0001`-`w0016` (13 `web_any`, 3 `kb`): `official/question_work/additions_20261014_web_search.jsonl`, nối vào `questions.jsonl` + `questions_review.csv` (repo gateway, chưa commit)
- [x] Checkpoint: pytest `tests/evals` (chỉ `test_judge` 503/504 chập chờn có sẵn), ruff; runner chọn 22 câu / 38 lượt `web_search`
- [ ] Duyệt 16 câu (`reviewed=false`), nhất là 3 câu `kb` có nằm trong kho eval không
- [ ] Chạy thật: `task eval:run env=eval -- --web on --category web_search`
