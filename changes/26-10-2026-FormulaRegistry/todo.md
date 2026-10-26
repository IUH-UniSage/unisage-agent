# Todo - registry công thức, bỏ LLM tự tính (26/10/2026)

## Phase 1: registry + công thức mới
- [x] `Formula` thêm `version`, `source`, `outputs`; `CalculationResult.values`, `.version`; các công thức cũ điền `values` (commit 65eef87)
- [x] `admission_score_2026` (xuôi), `transcript_score_3hk`, `weighted_score` + triggers (docs: SPEC-calc-engine)
- [x] Tests formulas (d0307 = 25,4; học bạ Kiểm toán; weighted d0312 = 6,8)

## Phase 2: solver tính ngược
- [x] `app/calculation/solver.py` (commit 9e84e89): lưới theo bước làm tròn (`Formula.rounding_of`), vùng hợp lệ nhị phân, điểm chữ, kiểm tra liền kề, không có nghiệm
- [x] Tests: CK = 10 để A+; ĐHB ≥ 25,5 (d0308); HK1 Lý = 5,5 (d0309); 8 tín chỉ dưới 8 triệu; weighted d0313 = 7,1 (6,95 → 7,0), d0311 = 9,5

## Phase 3: node + graph + prompt
- [x] Extractor schema mới (`solve_for`, `target`, `none`), prompt viết lại, `parse_extraction` (model thắng router) (commit ce3ab25)
- [x] Bỏ nhánh LLM: `LlmAnswered`, `_ask_llm`, `llm_outcome`, handoff, review-redo, template, settings, `.env.example`, `docs/guide/cau-hinh.md`
- [x] `none` → `Deferred` → advisory_tasks đầu lượt (`_plan_calculations`); resume plan cũ `llm` → `legacy_llm`
- [x] `CalculationPlan` thêm `solve_for`/`target` (`CalculationTarget`); Literal formula_id mới
- [x] Prompt phân loại: registry qua `{calculation_formulas}`, bỏ rule 5b; prompt advisory: luật "hiển thị công thức, không tự tính"
- [x] Tests graph: 15 test nhánh LLM bỏ, 7 test mới (none→advisory, giải ngược qua graph, resume inverse, lookup lọc như advisory)
- [x] ruff sạch (trừ lỗi cũ không liên quan), mypy sạch trên file đã sửa, 2723 test pass

## Phase 4: docs + eval + PR
- [x] SPEC-calculation-node.md, SPEC-calc-engine.md, known-gaps, rag-pipeline, DECISIONS, PRODUCT, dev-onboard
- [x] Chạy lại 38 câu calculation (`none`) → report.md (7/8 câu công thức đúng bằng Python; 30 câu còn lại đi advisory, luật "không tự tính" chưa đủ mạnh - chờ chốt)
- [x] Commit theo nhóm (calculation / solver / node+graph+prompt / docs), push, PR xếp chồng trên PR #28
