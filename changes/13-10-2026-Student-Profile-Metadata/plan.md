# Plan: Hồ sơ sinh viên theo người dùng, tên field thống nhất, sửa khi người dùng nói "tôi nhầm"

## Vấn đề

1. `confirmed_metadata` lưu **theo hội thoại** (`conversation_clarification_states.confirmed_metadata`):
   mở hội thoại mới là phải khai lại khoá, ngành, hệ.
2. Ba nguồn đặt tên field khác nhau: node 10 (theo `known_metadata_fields.json`), công thức cài sẵn
   (`ParamSpec`: `khoa_tuyen_sinh` trùng nghĩa `khoa_nhap_hoc`), LLM tự tính (tên tự đặt). Cùng một khái
   niệm có thể nằm dưới hai khoá.
3. Chỉ câu trả lời form của node 10 vào `confirmed_metadata`; form tính toán không. Số của nhánh LLM tự
   tính không được lưu; `last_calculation` chỉ giữ một phép tính.
4. Giá trị đã xác nhận không bao giờ được hỏi lại, và không có cách sửa: khai sai một lần là sai cả hội
   thoại.

## Quyết định (13-10-2026, đã chốt với người dùng)

- **Tách hai tập:**
  - **Hồ sơ** (thuộc tính ổn định của sinh viên: khoá nhập học, ngành, hệ, bậc, khối ngành, cơ sở...):
    bảng mới `student_profiles`, theo `user_id`; khách (không có `user_id`) theo `conversation_id`.
  - **Số liệu phép tính** (tín chỉ, điểm, đơn giá, số tiền đã nộp...): theo **từng phép tính**, danh sách
    5 phép tính gần nhất của hội thoại (`recent_calculations`, thay `last_calculation`), gồm cả nhánh
    LLM tự tính. Không trộn số giữa các phép tính, không lưu theo người dùng.
- **Một nơi cho hồ sơ:** bỏ cột `confirmed_metadata` của `conversation_clarification_states`. Migration
  chép dữ liệu cũ thành hồ sơ theo hội thoại rồi drop cột (phương án b).
- **Tên field chuẩn:** `known_metadata_fields.json` là sổ đăng ký duy nhất: mỗi field có `scope`
  (`profile` | `calculation`) và `key_aliases`. Mọi tên field (câu hỏi form, câu trả lời, `ParamSpec`,
  cập nhật hồ sơ) đi qua `canonical_field` trước khi dùng. `ParamSpec` dùng đúng tên chuẩn
  (`khoa_tuyen_sinh` → `khoa_nhap_hoc`); test bắt buộc mọi `ParamSpec` có trong sổ.
- **Đọc hai tập trước khi hỏi:**
  - LLM tự tính nhận `<student_declared_attributes>` (hồ sơ) và `<recent_calculations>` (số của các phép
    tính gần đây, tên chuẩn) cùng bảng tên field chuẩn; chỉ hỏi field còn thiếu.
  - Node 10 nhận hồ sơ như cũ và danh sách tên field đã có từ phép tính (không kèm số - node 10 không
    nhận số của phép tính, SPEC-calculation-node §3).
  - Code chặn lại lần nữa: câu hỏi có field (đã chuẩn hoá) nằm trong hồ sơ hoặc trong số đã biết của task
    thì bỏ. Công thức cài sẵn lấy tham số phạm vi hồ sơ từ hồ sơ trước khi hỏi.
- **Câu trả lời form:** chuẩn hoá tên → field `profile` vào hồ sơ (giá trị chuẩn hoá, VD
  `khoa_nhap_hoc` = năm), và vẫn vào tham số của task nếu task cần; field `calculation` chỉ vào task.
- **Sửa "tôi nhầm, XX mới đúng":** node 03 trả thêm `metadata_updates: {field: giá_trị | null}` (thấy
  hồ sơ hiện tại và bảng field hồ sơ). Chỉ nhận field `profile` có trong sổ, giá trị chuẩn hoá. Áp ngay
  sau node 03, **ghi DB ngay** (không đợi cuối lượt), mọi bước sau dùng giá trị mới, giá trị hồ sơ ghi đè
  số cũ của phép tính trước; câu trả lời mở đầu bằng một dòng báo đã cập nhật.

## Task

- T1 Sổ đăng ký field: `scope`, `key_aliases`, field `khoi_nganh`, các field `calculation` cho mọi
  `ParamSpec`; `app/graph/metadata_fields.py` (`canonical_field`, `field_scope`, `normalize_value`,
  `apply_updates`); `khoa_tuyen_sinh` → `khoa_nhap_hoc`; bảng cho node 10 chỉ field `profile`.
- T2 DB: `student_profiles` + repository; `recent_calculations`; migration (chép, drop cột, đổi cột);
  nạp hồ sơ đầu lượt, ghi cuối lượt; `RoundState` bỏ `confirmed_metadata`.
- T3 Câu hỏi và câu trả lời: chuẩn hoá tên, chặn field đã có ở hai tập, định tuyến câu trả lời theo
  `scope`, LLM tự tính đọc `<recent_calculations>` + bảng field; lưu số của nhánh LLM.
- T4 Node 03 `metadata_updates`, ghi ngay, dòng báo cập nhật, hồ sơ ghi đè số cũ.
- T5 Spec + commit + push.

## Rủi ro

- Node 03 cập nhật nhầm: chỉ field `profile` trong sổ, chỉ khi người dùng tự nói về bản thân; câu trả lời
  luôn báo lại để người dùng thấy và sửa.
- Mất hồ sơ theo người dùng cũ: không có (trước đây chỉ có theo hội thoại); hồ sơ theo hội thoại của
  người dùng đã đăng nhập không được gộp sang hồ sơ người dùng (phương án b).
