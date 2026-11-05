# Plan — Benchmark cross-encoder chỉ retrieval trên bộ official (bước 1)

Tiền thân: `changes/22-10-2026-CrossEncoderBenchmark/` (26/10, bộ demo: RRF chạm trần, không đo
được giá trị của CE). Bộ official đã nạp (`unisage-official-chunks`, 1 074 tài liệu, ~24 900
chunk, nhiều thông báo/quyết định gần trùng theo năm) nên đo lại ở đây. Bước 2 (eval e2e trên
mẫu, với cấu hình thắng) là việc sau.

## Quyết định

- **Chỉ retrieval, không LLM**: không classifier, không HyDE, không tách câu hỏi, không sinh câu
  trả lời, không judge. Câu hỏi đi thẳng vào hybrid search của graph (`RetrievalService._search`:
  3 dense + BM25, RRF k = 1, filter quyền). OpenAI chỉ dùng embedding `text-embedding-3-small`.
- **Pool 64 một lần, ghi ra file**: mọi phương án đọc cùng `pools.jsonl` / `chunks.jsonl`; chấm
  CE ghi `ce.jsonl`; chỉ số tính lại offline được (`per_query.jsonl`, `summary.json`).
- **Giống graph ở bước 09**: sàn cosine `CHAT_RETRIEVAL_DENSE_FLOOR` 0,45 (BM25-only qua theo cờ)
  áp trên N chunk đầu trước khi chấm; passage = `passage_text` của node CE (tiêu đề + heading + nội dung).
- **K = 8** (`CHAT_CROSS_ENCODER_KEEP` = `CHAT_CONTEXT_MAX_CHUNKS` = 8). `none` = 8 chunk đầu sau sàn
  của 40 chunk đầu (`CHAT_RETRIEVAL_MAX_CHUNKS=40` của `.env.eval`).
- **Service CE đang chạy (8403) để nguyên**, không spawn, không tắt; 1 request mỗi (truy vấn, cỡ pool),
  tuần tự.

## Việc

- [x] `evals/benchmark/retrieval_pool.py` (`collect` / `ce` / `score`) + `task benchmark:pool` + test
      `tests/evals/test_benchmark_retrieval_pool.py`.
- [x] Chuẩn bị bộ câu: 352 câu (normal 207, calculation 59, access thấy 86) × 2 biến thể = 704 truy vấn;
      + 102 câu access không được thấy × 2 = 204 truy vấn kiểm tra rò quyền.
- [x] Lấy pool 64 (+ pool 40 ở độ sâu graph để đối chiếu).
- [x] Chấm CE ở 16 / 40 / 64.
- [x] Tính chỉ số (`assets/analyze.py`), vẽ hình (`assets/plot_*.py`).
- [x] `report.md`, `report-full.md`, `report.xlsx`; thêm sheet " 03-11" vào workbook 26/10.

## Lệnh

```
task benchmark:pool env=eval -- collect --out benchmark_results/03-11-2026-ce-official
task benchmark:pool env=eval -- ce      --out benchmark_results/03-11-2026-ce-official
task benchmark:pool env=eval -- score   --out benchmark_results/03-11-2026-ce-official
python changes/03-11-2026-CrossEncoderBenchmarkOfficial/assets/analyze.py benchmark_results/03-11-2026-ce-official
python changes/03-11-2026-CrossEncoderBenchmarkOfficial/assets/plot_all.py benchmark_results/03-11-2026-ce-official changes/03-11-2026-CrossEncoderBenchmarkOfficial/assets
```
