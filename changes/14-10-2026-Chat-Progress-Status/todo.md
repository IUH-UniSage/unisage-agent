# Todo: Đồng hồ và trạng thái tiến trình của lượt chat

Chi tiết (AC, files, size) xem `plan.md`.

- [x] T1 Agent: `StatusItem` + `event: status`, `GraphTrace(on_status)`, `stage_for_node` (`app/graph/progress.py`), bỏ trùng, test
- [x] T2 Agent: `web_searching` + `url` cho từng trang ứng viên ở 09b (không cho speculative không dùng), test
- [x] T3 `contracts/chat-sse.md` thêm `status`
- [x] Checkpoint agent: pytest 2172 passed (chỉ `tests/evals/test_judge.py` 503/504 chập chờn có sẵn), ruff sạch, mypy không thêm lỗi
- [x] T4 Web: `chatStatusSchema` + `onStatus` trong `useChatStream`, test
- [x] T5 Web: `useTurnProgress` (đồng hồ, hàng đợi URL ≥ 0,8s), gắn vào `useChatWorkspace`, test
- [x] T6 Web: `TurnProgressIndicator` + `ElapsedTime`; lint/typecheck/format/build sạch; vitest 210/211 (`axios-client.test.ts` hỏng sẵn); e2e `chat-progress.spec.ts` + ảnh chụp 375px/1280px; e2e chat cũ 14 passed (`user-workspace.spec.ts` hỏng sẵn 6/10)
- [ ] Checkpoint cuối: chạy thật với agent
