from typing import ClassVar


class SuggestionService:
    """Generate deterministic follow-up prompts for the initial chat experience."""

    DEFAULT_SUGGESTIONS: ClassVar[list[str]] = [
        "Điều kiện để đăng ký học cải thiện điểm là gì?",
        "Quy trình xin bảo lưu kết quả học tập gồm những bước nào?",
        "Thời hạn đóng học phí học kỳ này là khi nào?",
    ]

    def generate_suggestions(
        self,
        query: str,
        intent: str | None = None,
        limit: int = 3,
    ) -> list[str]:
        """Return focused follow-up prompts without calling an LLM."""

        del intent
        query_lower = query.lower()
        if "học phí" in query_lower or "tiền" in query_lower:
            suggestions = [
                "Thời gian đóng học phí đợt 2 là khi nào?",
                "Quy định gia hạn đóng học phí cho sinh viên khó khăn?",
                "Mức thu học phí tín chỉ ngành hiện tại là bao nhiêu?",
            ]
        elif "đăng ký" in query_lower or "môn học" in query_lower:
            suggestions = [
                "Số tín chỉ tối đa được đăng ký trong một học kỳ là bao nhiêu?",
                "Quy trình đăng ký học vượt khối lượng như thế nào?",
                "Hạn chót rút bớt môn học không ghi điểm F là khi nào?",
            ]
        else:
            suggestions = self.DEFAULT_SUGGESTIONS
        return suggestions[:limit]
