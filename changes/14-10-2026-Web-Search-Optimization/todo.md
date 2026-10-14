# Todo: Tối ưu web search

> Chờ review, chưa implement. Chi tiết (AC, files, size) xem `plan.md`.

- [ ] W0 Bộ eval web search ~20 câu + baseline (answer rate, sai phạm vi, title-only, TTFT, cache, credit)
- [ ] W1 Classify sinh `web` slots (topic + slot theo `known_metadata_fields.json`), normalize, test
- [ ] W2 `web_query` ghép bằng code theo mẫu topic, năm học do code tính, chuẩn hóa alias, fallback HyDE, test
- [ ] W5 Rerank chung `[C..]` + `[W..]`, luật web `partial`, parser `Wn`, test ca gốc
- [ ] W7 Fallback mềm `<candidate_sources>` (link + năm học, cấm số), test
- [ ] Checkpoint: ca "học phí ngành cnnt" trả lời được (chưa song song), pytest/ruff/mypy pass, review
- [ ] W4 Tavily advanced + `chunks_per_source=3` + `max_results=10`, sàn 0.05, sắp/lọc domain theo topic, test
- [ ] W3 `CachedWebSearchClient` (Redis riêng, key theo slot, TTL theo topic, negative cache, fail-open), test
- [ ] W6 Chạy song song với retrieval cho topic động, skip khi KB mạnh, timeout, test latency
- [ ] W8 Năm học từ title/URL, sắp mới trước, `WEB_SOURCE_USED` + log topic/web_mode/cache/web_used, test
- [ ] Checkpoint: chạy eval, so sánh fallback vs speculative, review
- [ ] W9 Chỉnh nhóm topic / TTL / ngưỡng theo số liệu; quyết định có tự fetch trang không
