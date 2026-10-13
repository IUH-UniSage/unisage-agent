# Todo: Hồ sơ sinh viên theo người dùng, tên field thống nhất, sửa khi "tôi nhầm"

Chi tiết quyết định và lý do: `plan.md`.

## T1 Sổ đăng ký field - XONG (`bfb60b3`)

- [x] `known_metadata_fields.json`: `scope` (`profile` | `calculation`), `key_aliases`, field `khoi_nganh`,
      field `calculation` cho mọi `ParamSpec` + `so_tien_da_nop`, `tong_hoc_phi`
- [x] `app/graph/metadata_fields.py`: `canonical_field`, `field_scope`, `is_profile_field`,
      `normalize_value`, `apply_updates`, `canonical_params`, `concept`
- [x] `khoa_tuyen_sinh` → `khoa_nhap_hoc` (năm nhập học) trong `formulas.py`, extractor, value extractor
- [x] Bảng field cho node 10 chỉ render `scope = profile`
- [x] `tests/graph/test_metadata_fields.py` (mọi `ParamSpec` có trong sổ, alias không trùng)

## T2 DB - ĐANG LÀM (commit WIP, test chưa sửa)

Đã viết:
- [x] `StudentProfile` (`app/database/models.py`): `user_id` | `conversation_id` (đúng một), `attributes`
- [x] `ConversationClarificationState`: bỏ `confirmed_metadata`, `last_calculation` → `recent_calculations`
      (list, tối đa `RECENT_CALCULATIONS_MAX = 5`)
- [x] Migration `b5c6d7e8f9a0`: tạo `student_profiles`, chép `confirmed_metadata` cũ thành hồ sơ theo
      hội thoại, `last_calculation` → `[last_calculation]`, drop hai cột cũ (phương án b)
- [x] `app/database/repositories/student_profile.py`: `ProfileOwner.of(security, conversation_id)`,
      `StudentProfileRepository.get/save` (upsert theo cột chủ sở hữu)
- [x] `app/database/repositories/dialect.py`: `insert_builder` (tách từ `clarification_state.py`)
- [x] `ClarificationRoundRepository`: `RoundState.recent_calculations`; `save_confirmed_metadata` →
      `save_calculations(conversation_id, recent_calculations)`; `upsert_open(conv, round, recent)`;
      `complete(conv, token, new_round, recent)` - không còn tham số `confirmed_metadata`
- [x] `chat_stream_service.open_stream`: nạp hồ sơ theo `ProfileOwner`, `canonical_profile(...)` →
      `GraphInput.confirmed_metadata`; `GraphInput.recent_calculations`
- [x] `GraphInput.recent_calculations` (+ `last_calculation` tự lấy phần tử cuối)
- [x] `streaming_session._store_round_state`: ghi hồ sơ (khi lượt làm đổi `confirmed_metadata`) cùng
      transaction; nối `graph_output.last_calculation` vào `recent_calculations`
- [x] `cancel_panel` bỏ tham số `confirmed_metadata`
- [x] `metadata_fields.canonical_profile`

Còn lại:
- [ ] Sửa 27 test đang đỏ - đều do gọi API cũ (truyền `confirmed_metadata` vào vị trí
      `recent_calculations` nên `KeyError: slice(-5, None)`; đọc `round_state.confirmed_metadata`):
  - `tests/database/test_clarification_round_repository.py` (11): đổi sang `save_calculations`,
    `recent_calculations`; test `test_last_calculation_is_kept_until_a_new_one_is_written` → giữ list,
    cắt còn 5
  - `tests/api/test_chat_stream_clarification.py` (13): seed hồ sơ qua `StudentProfileRepository`
    thay vì `confirmed_metadata` của round
  - `tests/graph/test_session_clarification.py` (1), `tests/graph/test_streaming_session.py` (2):
    hồ sơ đọc từ `student_profiles`
- [ ] Test mới: `StudentProfileRepository` (user vs khách, upsert, check constraint);
      `canonical_profile`; hồ sơ người dùng đi qua 2 hội thoại; `recent_calculations` cắt 5
- [ ] `streaming_graph._last_calculation`: lưu cả `params` (= `known_params`) của `LlmAnswered`
- [ ] Chạy migration thật trên Postgres local (`alembic upgrade head` rồi `downgrade -1`)
- [ ] mypy: `chat_stream_service.py:152` (`model_name` có thể None) - kiểm tra có từ trước chưa
      (`git stash` rồi chạy lại)

## T3 Câu hỏi và câu trả lời

- [ ] `_run_resume` (`streaming_graph.py`): mỗi câu trả lời → `canonical_field`; `field_scope(..., default
      theo origin: advisory → profile, calculation → calculation)`; field `profile` →
      `normalize_value` vào `confirmed_metadata` (thay khối riêng cho `khoa_nhap_hoc` hiện có) VÀ vẫn vào
      `answers_by_task` nếu task cần; field `calculation` chỉ vào task
- [ ] `advisory_questions` (`clarification_round.py`): `canonical_field` cho `field`, bỏ câu có field nằm
      trong hồ sơ
- [ ] `llm_questions` (`nodes/calculation.py`): `canonical_field`; bỏ câu có field trong `known` (đã chuẩn
      hoá bằng `canonical_params`) HOẶC trong hồ sơ (`deps.confirmed_metadata`)
- [ ] `builtin_task`: tham số phạm vi hồ sơ (`is_profile_field`) lấy từ hồ sơ, GHI ĐÈ số cũ của
      `previous`/extractor
- [ ] Prompt LLM tự tính (`chat_calculation_llm.yaml`): khối `<recent_calculations>` (tiêu đề + params tên
      chuẩn của `deps` recent) + bảng tên field chuẩn (mọi scope) + quy tắc "đọc
      `<student_declared_attributes>` và `<recent_calculations>` trước, chỉ hỏi field còn thiếu, dùng đúng
      tên trong bảng"; `CalculationDeps` cần thêm `recent_calculations`
- [ ] Node 10: thêm danh sách TÊN field đã có từ phép tính (không kèm số) vào `{calculation_results}`
- [ ] Test: câu hỏi trùng hồ sơ bị bỏ; alias (`so_tc_lt`) bị bỏ khi đã có `tc_lt`; câu trả lời
      `khoa_tuyen_sinh` vào hồ sơ dưới `khoa_nhap_hoc`; test cũ dùng field `training_type` giờ thành
      `he_dao_tao` (`test_mixed_turn_asks_both_parts_on_one_panel`)

## T4 Node 03 `metadata_updates`

- [ ] `IntentClassification.metadata_updates: dict[str, str | None]` + parse trong
      `message_classification.py`
- [ ] Prompt `message_classification.yaml`: thấy hồ sơ hiện tại + bảng field `profile`; chỉ cập nhật khi
      người dùng tự nói về bản thân hoặc sửa ("tôi nhầm, khoá 2023 mới đúng"); `null` = xoá
- [ ] `_run_graph`: `apply_updates` ngay sau node 03 → nếu đổi: ghi DB NGAY (callback từ
      `streaming_session` qua `run_graph`, vì graph không tự gọi DB) → dùng dict mới cho mọi bước sau
- [ ] Dòng báo đầu câu trả lời (Python, không LLM): "Mình đã cập nhật năm nhập học: 2023."
      (`metadata_fields.concept`)
- [ ] Test: sửa khoá → hồ sơ đổi, ghi ngay kể cả khi phần sau của lượt lỗi, phép tính trước dùng giá trị
      mới; field không có trong sổ / scope `calculation` bị bỏ qua

## T5 Spec, commit, push

- [ ] `SPEC-calculation-node.md`, `SPEC-clarification-panel.md` (nơi lưu câu trả lời), `known-gaps.md`
- [ ] Bỏ nhánh chết `confirmed_metadata` trong `FenceRedactor.CAPTURED_TYPES` (tuỳ chọn)
- [ ] Full test + ruff + mypy, commit từng task, push
