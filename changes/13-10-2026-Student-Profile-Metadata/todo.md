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

## T2 DB - XONG (`6acc639`, `2ffde23`, `c4947b3`, `8b84e73`)

- [x] `StudentProfile` (`app/database/models.py`): `user_id` | `conversation_id` (đúng một), `attributes`
- [x] `ConversationClarificationState`: bỏ `confirmed_metadata`, `last_calculation` → `recent_calculations`
      (list, tối đa `RECENT_CALCULATIONS_MAX = 5`)
- [x] Migration `b5c6d7e8f9a0`: tạo `student_profiles`, chép `confirmed_metadata` cũ thành hồ sơ theo
      hội thoại, `last_calculation` → `[last_calculation]`, drop hai cột cũ (phương án b). Downgrade chép
      ngược (hồ sơ theo hội thoại → `confirmed_metadata`, phần tử cuối → `last_calculation`; hồ sơ theo
      người dùng mất). Đã chạy upgrade → downgrade → upgrade trên DB Postgres tạm (`unisage_migcheck`,
      đã xoá), kiểm tra upsert + check constraint trên Postgres
- [x] `StudentProfileRepository` (`ProfileOwner.of`, `get`, `save` upsert), `insert_builder` (`dialect.py`)
- [x] `ClarificationRoundRepository`: `save_calculations`, `upsert_open(conv, round, recent)`,
      `complete(conv, token, new_round, recent)`
- [x] `chat_stream_service.open_stream` nạp hồ sơ (`canonical_profile`); `streaming_session` ghi hồ sơ
      cùng transaction khi lượt làm đổi
- [x] `GraphOutput.calculations` (thay `last_calculation`, giữ property): MỌI phép tính của lượt nối vào
      `recent_calculations`; nhánh LLM lưu `known_params` (`registered_name`: tên chuẩn nếu có trong sổ,
      giữ nhãn nếu không; bỏ `REVIEW_ISSUE_KEY`)
- [x] Test: repository vòng hỏi (cắt 5, dòng hỏng bị bỏ), `StudentProfileRepository` (user/khách, upsert,
      check constraint), `canonical_profile`, `registered_name`, hồ sơ người dùng qua 2 hội thoại (API)
- [x] mypy: 3 lỗi có từ trước (`chat_stream_service.py:152` `model_name`,
      `test_calculation_document_values.py:78`, `test_calculation_node.py` `extraction`) - không sửa
- Ghi chú: `tests/evals/test_judge.py::test_503_and_504_...` chập chờn (fail ~5/6 khi chạy riêng), không
  liên quan; `tests/e2e/test_advisory_flow_e2e.py` import `ClarificationStateRepository` không còn (có từ
  trước, test integration)

## T3 Câu hỏi và câu trả lời - XONG (`5cd4d8b`)

- [x] `_run_resume`: `canonical_field` cho mọi câu trả lời; `field_scope(..., advisory → profile,
      calculation → calculation)`; field `profile` → `normalize_value` vào `confirmed_metadata` (thay khối
      riêng cho `khoa_nhap_hoc`) và vẫn vào `answers_by_task`
- [x] `advisory_questions`: `canonical_field`, bỏ câu có field trong hồ sơ HOẶC trong `known_fields(outcomes)`
      (giá trị phép tính của lượt đã có)
- [x] `llm_questions(forms, known, profile)`: `canonical_field`; bỏ câu có field trong
      `canonical_params(known)` hoặc trong hồ sơ
- [x] `profile_params` + `run_calculation_task`/`resume_calculation_task`: tham số phạm vi hồ sơ lấy từ hồ
      sơ. Thứ tự ưu tiên: số gõ ở lượt này (extractor) > hồ sơ > số của phép tính trước (`previous`) - câu
      giả định "khoá 2021 thì sao" không bị hồ sơ ghi đè; sửa hồ sơ thật là việc của T4
- [x] Prompt LLM tự tính: `<recent_calculations>` (`build_recent_calculations_section`) + bảng tên field
      mọi scope (`build_known_metadata_fields_section(every_scope=True)`) + quy tắc "ĐỌC TRƯỚC KHI HỎI" và
      dùng tên chuẩn; `CalculationDeps.recent_calculations`
- [x] Node 10: `calculation_titles` kèm `[đã có: tên field...]` (không số)
- [x] `slug` chuyển sang `metadata_fields` (tránh import vòng với `clarification_round`)
- [x] Test (`tests/graph/test_profile_fields.py`, `test_prompt_loader.py`): alias/hồ sơ bị bỏ, câu trả lời
      `khoa_tuyen_sinh` vào hồ sơ dưới `khoa_nhap_hoc`, ưu tiên extractor > hồ sơ > previous, prompt có
      `<recent_calculations>`; test cũ đổi sang tên chuẩn (`he_dao_tao`, `nganh_hoc`, `khoa_nhap_hoc`)

## T4 Node 03 `metadata_updates` - XONG (`ce46138`)

- [x] `IntentClassification.metadata_updates: dict[str, str | None]`, parse dễ dãi (số → chuỗi, rác bị bỏ)
- [x] Prompt `message_classification.yaml`: mục "Cập nhật hồ sơ sinh viên" + bảng field `profile`
      (`{profile_fields}`, thay bằng `.replace`) + 2 ví dụ (sửa khoá → cập nhật; hỏi khoá khác → `{}`);
      `classify_intent(profile=...)` nối khối `<student_profile>` vào prompt
- [x] `_run_graph`: `apply_updates` ngay sau node 03 → `on_profile_change` (từ `streaming_session`,
      `_profile_saver`: ghi DB ngay, lỗi chỉ log) → dict mới cho mọi bước sau (`_classified_turn`)
- [x] Dòng báo đầu câu trả lời (`metadata_fields.update_notice`, Python): "Mình đã cập nhật thông tin của
      bạn - khóa nhập học: 2023."
- [x] Test (`tests/graph/test_profile_updates.py`): parse; prompt có hồ sơ + bảng; ghi trước khi hiện gì;
      giữ dù phần sau của lượt lỗi; giá trị trùng không tính là thay đổi; alias/field `calculation`/field lạ
      bị bỏ; câu nối tiếp sau khi sửa tính lại với khoá mới; `_profile_saver` ghi DB, lỗi DB không làm hỏng lượt
- [x] `tests/llm_mocks.py`: mock classifier cắt khối `<student_profile>` khỏi `query`

## T5 Spec, commit, push

- [x] `SPEC-clarification-panel.md` §2.8 (hồ sơ, tên field chuẩn, nơi lưu câu trả lời, sửa bằng tin nhắn),
      `SPEC-calculation-node.md` §5, §8 (`recent_calculations`, thứ tự ưu tiên tham số), §9.5, `known-gaps.md`
      (hồ sơ khách không theo sang khi đăng nhập; classifier quyết định thế nào là sửa hồ sơ)
- [x] `FenceRedactor.CAPTURED_TYPES` giữ `confirmed_metadata`: không phải nhánh chết - vẫn nuốt khối JSON lạc
      để không lộ ra câu trả lời (spec đã sửa câu nhắc tới `collect_confirmed_metadata_updates` không còn)
- [x] Full test + ruff + mypy trên file đã sửa (lỗi còn lại đều có từ trước), commit từng task
- [ ] Push
