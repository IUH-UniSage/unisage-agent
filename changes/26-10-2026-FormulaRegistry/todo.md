# Todo - registry công thức, bỏ LLM tự tính (26/10/2026)

## Phase 1: registry + công thức mới
- [ ] `Formula` thêm `version`, `source`, `outputs`; `CalculationResult.values`, `.version`; các công thức cũ điền `values`
- [ ] `admission_score_2026` (xuôi), `transcript_score_3hk`, `weighted_score` + triggers + docs/specs/<formula>.md
- [ ] Tests formulas (d0307 = 25,4; học bạ Kiểm toán; weighted d0312 = 6,8)

## Phase 2: solver tính ngược
- [ ] `app/calculation/solver.py`: `solve(formula_id, params, unknown, target)`, điểm chữ làm mục tiêu, kiểm tra giá trị liền kề, không có nghiệm
- [ ] Tests: course_score CK để A+ ; admission ĐHB ≥ 25,5 (d0308); transcript HK1 Lý = 5,5 (d0309); tuition tín chỉ tối đa dưới ngân sách; weighted cuối kỳ (d0313 = 7,2; d0311 = 9,5)

## Phase 3: node + graph + prompt
- [ ] Extractor schema mới (`solve_for`, `target`, `none`), prompt viết lại, `parse_extraction`
- [ ] Bỏ nhánh LLM: `LlmAnswered`, `_ask_llm`, `llm_outcome`, handoff, review-redo, template, settings, `.env.example`, `docs/guide/cau-hinh.md`
- [ ] `none` → chuyển task sang advisory đầu lượt; resume plan cũ `llm` → câu cố định
- [ ] `CalculationPlan` thêm `solve_for`/`target`; Literal formula_id mới; `LastCalculation`
- [ ] Prompt phân loại: registry qua `{calculation_formulas}`, bỏ rule 5b; prompt advisory: luật "hiển thị công thức, không tự tính"
- [ ] Tests graph: viết lại các test nhánh LLM, thêm test none→advisory, inverse qua node, resume inverse
- [ ] ruff, mypy, pytest toàn bộ

## Phase 4: docs + eval + PR
- [ ] SPEC-calculation-node.md, SPEC-calc-engine.md, known-gaps, rag-pipeline, DECISIONS, PRODUCT, dev-onboard
- [ ] Chạy lại 38 câu calculation (`none`) → report.md trong thư mục này
- [ ] Commit theo nhóm (calculation / graph / prompt / docs), push, PR xếp chồng trên PR #28
