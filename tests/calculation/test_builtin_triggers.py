import pytest

from app.calculation.formulas import route_builtin

GPA = [
    "Tính GPA giúp em",
    "gpa học kỳ này của em là bao nhiêu",
    "Điểm trung bình tích lũy của em bao nhiêu?",
    "điểm trung bình học kỳ 1 tính sao",
    "Em muốn biết điểm trung bình chung tích lũy",
    "ĐTBTL của mình được bao nhiêu",
    "trung bình tích luỹ thang 4 của em",
    "Tính giúp mình GPA với 5 môn",
    "GPA tích lũy sau 3 kỳ",
    "điểm trung bình học kì này",
]
COURSE = [
    "Điểm tổng kết học phần được bao nhiêu?",
    "tổng kết môn này nếu TX 8 GK 7 CK 6",
    "Môn có 2 tín lý thuyết 1 tín thực hành thì tính sao",
    "điểm học phần tích hợp tính thế nào",
    "giữa kỳ 6 cuối kỳ 7 thì qua môn không",
    "thực hành 9 với lý thuyết 7 thì tổng kết bao nhiêu",
    "tbtx 8 gk 7 ck 6.5",
    "điểm tổng kết môn lập trình",
    "Tính điểm học phần giúp em",
    "giữa kì 5 cuối kì 8",
]
CONVERSION = [
    "8.45 thì được điểm chữ gì",
    "quy đổi 7.5 sang thang 4",
    "7 điểm là điểm chữ gì?",
    "điểm chữ B+ tương đương bao nhiêu",
    "đổi sang thang 4 giúp em",
    "6.9 thì ra c+ hả",
    "qui đổi điểm 9 ra hệ 4",
    "8.5 là A phải không",
    "điểm chữ của 5.4",
    "đổi ra điểm chữ 7.8",
]
TUITION = {
    "Học phí học kỳ này bao nhiêu?": "tuition_semester",
    "tính học phí học kì 2 giúp em": "tuition_semester",
    "học phí 1 kỳ của em là bao nhiêu": "tuition_semester",
    "học phí một năm học ngành CNTT": "tuition_year",
    "Tính học phí mỗi năm cho em": "tuition_year",
    "tính giúp tôi học phí cả ngành công nghệ thông tin đi": "tuition_program",
    "học phí toàn khoá học là bao nhiêu": "tuition_program",
    "học phí cả khóa ngành Kế toán": "tuition_program",
    "tổng học phí đến khi ra trường": "tuition_program",
}
NONE = [
    "Học phí khóa 2025 ngành CNTT là bao nhiêu?",
    "Điều kiện tốt nghiệp là gì",
    "Em còn thiếu bao nhiêu tín chỉ để tốt nghiệp",
    "điểm rèn luyện tính như thế nào",
    "Khi nào đăng ký học phần",
    "học bổng khuyến khích cần điều kiện gì",
    "Em bị cảnh báo học vụ thì sao",
    "lịch thi cuối kỳ khi nào có",
    "Phòng đào tạo ở đâu",
    "Được học vượt bao nhiêu tín",
]


@pytest.mark.parametrize("question", GPA)
def test_gpa_questions(question: str) -> None:
    assert route_builtin(question) == ["gpa"]


@pytest.mark.parametrize("question", COURSE)
def test_course_score_questions(question: str) -> None:
    assert route_builtin(question) == ["course_score"]


@pytest.mark.parametrize("question", CONVERSION)
def test_conversion_questions(question: str) -> None:
    assert route_builtin(question) == ["grade_conversion"]


@pytest.mark.parametrize(("question", "formula_id"), TUITION.items())
def test_tuition_questions(question: str, formula_id: str) -> None:
    assert route_builtin(question) == [formula_id]


@pytest.mark.parametrize("question", NONE)
def test_other_questions_are_not_routed(question: str) -> None:
    assert route_builtin(question) == []
