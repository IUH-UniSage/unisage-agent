"""Calculation node (07) - placeholder, deterministic, no LLM.

The real node (parameter extraction via `agents/calculation_extractor.yaml`
+ a calculator tool over GPA/credits/tuition) is not built yet, and where
its input data comes from (the student's grades, per-program tuition) is
still undecided. Until then the node only tells the student the calculation
part isn't available, so a calculation question is answered honestly
instead of being pushed through retrieval as if it were a regulation
question.
"""

CALCULATION_PLACEHOLDER_TEMPLATE = (
    "Phần tính toán (GPA, tín chỉ, học phí) hiện đang được phát triển nên mình "
    "chưa tính giúp bạn được. Bạn có thể tự tính theo công thức trong quy chế "
    "đào tạo, hoặc liên hệ Phòng Đào tạo để được hỗ trợ nhé."
)
