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
    "kỳ này em học 15 tín lý thuyết 4 tín thực hành, học phí bao nhiêu?": "tuition_semester",
    "mỗi năm em đóng học phí bao nhiêu": "tuition_year",
}
NONE = [
    "15 tín lý thuyết với 4 tín thực hành thì học phí bao nhiêu?",
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


ADMISSION = [
    "Điểm xét tuyển của em là bao nhiêu nếu thi tốt nghiệp được 24 điểm, học bạ 27?",
    "thi ĐGNL 900 điểm thì quy đổi về thang 30 được bao nhiêu",
    "XT1 của em bao nhiêu khi điểm thi 23, học bạ 25,5, ưu tiên 0,25",
    "ĐXT năm 2026 tính sao nếu không thi đánh giá năng lực",
]
TRANSCRIPT = [
    "xét học bạ: Toán lớp 10 là 7,5, lớp 11 là 8, học kỳ 1 lớp 12 là 8,5 thì Toán bao nhiêu",
    "Lý HK1 lớp 12 cần tối thiểu bao nhiêu để tổng 3 môn đạt ngưỡng nhận hồ sơ 19,5?",
]
WEIGHTED = [
    "Môn này trọng số trắc nghiệm 10%, giữa kỳ 30%, cuối kỳ 50% thì tổng kết bao nhiêu",
    "theo đề cương Interpretation 1, Mid-term Test 30% Final Test 50%, cuối kỳ cần bao nhiêu",
]


@pytest.mark.parametrize("question", ADMISSION)
def test_admission_questions_route_to_the_2026_formula(question: str) -> None:
    assert "admission_score_2026" in route_builtin(question)


@pytest.mark.parametrize("question", TRANSCRIPT)
def test_transcript_questions_route_to_the_three_term_formula(question: str) -> None:
    assert "transcript_score_3hk" in route_builtin(question)


@pytest.mark.parametrize("question", WEIGHTED)
def test_syllabus_weight_questions_offer_the_weighted_formula(question: str) -> None:
    assert "weighted_score" in route_builtin(question)
