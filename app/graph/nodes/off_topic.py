"""Off-topic rejection node - deterministic, static templates picked at random
so a user who drifts off-topic twice does not get the same sentence back."""

import random

OFF_TOPIC_TEMPLATES: list[str] = [
    (
        "Xin lỗi, mình chỉ có thể hỗ trợ các câu hỏi liên quan đến học vụ của Nhà trường.\n"
        "Ví dụ mình có thể giúp bạn:\n"
        "• Tra cứu điều kiện học bổng, cảnh báo học vụ\n"
        "• Giải thích quy chế thi cử, đăng ký môn học\n"
        "• Tính toán GPA, học phí theo ngành\n"
        "• Hướng dẫn thủ tục xin giấy tờ học vụ"
    ),
    (
        "Câu hỏi này nằm ngoài phạm vi mình hỗ trợ rồi. Mình chuyên về học vụ của Nhà trường, "
        "ví dụ quy chế đào tạo, học bổng, học phí hay thủ tục giấy tờ. "
        "Bạn có câu hỏi nào về những chủ đề đó không?"
    ),
    (
        "Tiếc quá, nội dung này mình chưa giúp được vì mình chỉ tư vấn học vụ. "
        "Bạn có thể hỏi mình về:\n"
        "• Điều kiện tốt nghiệp, cảnh báo học vụ\n"
        "• Đăng ký học phần, lịch thi\n"
        "• Học phí, học bổng\n"
        "• Thủ tục chuyển ngành, bảo lưu, xin giấy tờ"
    ),
    (
        "Mình chỉ trả lời được các câu hỏi về học vụ của trường thôi bạn ạ. "
        "Nếu bạn cần tra cứu quy chế, học bổng, học phí hay thủ tục hành chính, cứ hỏi mình nhé!"
    ),
    (
        "Chủ đề này hơi xa chuyên môn của mình rồi. Mình là trợ lý học vụ, nên mạnh nhất ở "
        'các câu như "điều kiện xét học bổng là gì?", "học phí ngành này bao nhiêu?" '
        'hay "thủ tục bảo lưu thế nào?". Bạn thử hỏi mình một câu như vậy nhé!'
    ),
    (
        "Mình xin phép không trả lời nội dung này vì nó ngoài phạm vi học vụ. "
        "Còn nếu bạn đang băn khoăn về đăng ký học phần, lịch thi, học bổng hay giấy tờ "
        "ở trường, mình rất sẵn lòng giúp."
    ),
    (
        "Rất tiếc, mình được thiết kế để hỗ trợ học vụ nên chưa thể giúp câu này. "
        "Mình có thể giúp bạn tra cứu quy chế đào tạo, chuẩn đầu ra, học bổng, học phí "
        "hoặc hướng dẫn các thủ tục như chuyển ngành, bảo lưu, xin bảng điểm."
    ),
]


def off_topic_reply() -> str:
    return OFF_TOPIC_TEMPLATES[random.randrange(len(OFF_TOPIC_TEMPLATES))]
