# Plan — Cross-encoder rerank: service, nối sau cờ, benchmark so với LLM rerank (UNISAGE-106)

Spec: `docs/specs/SPEC-cross-encoder-rerank.md`. Nhánh: `feature/huydh-unisage-106-benchmark-cross-encoder`
(agent). Tiền thân: "Phase sau: Cross-encoder" trong `changes/20-10-2026-HybridRetrieval/plan.md`.

## Overview

Bước rerank hiện tốn 1,5–5 s mỗi lượt và dao động. Dựng service `bge-reranker-v2-m3` riêng,
nối vào graph như một chế độ tắt mặc định, rồi đo trên cùng bộ ứng viên xem cross-encoder
có nhanh hơn mà giữ (hoặc tăng) độ chính xác không. Kết quả là báo cáo số liệu + khuyến nghị;
bật mặc định và đăng ký registry là ticket sau.

## Architecture Decisions

- **Service riêng, project uv riêng** (`reranker/`): agent/worker không ôm torch; Windows
  deploy chạy native nếu torch CUDA cài được, không thì WSL2. Baseline transformers trực tiếp
  (cùng điểm với FlagEmbedding, bỏ được datasets/peft/ir-datasets) + FastAPI; chưa dùng
  Infinity/TEI (chưa xác minh hỗ trợ v2-m3).
- **Chế độ rerank là một setting** (`CHAT_RERANK_MODE`), không phải cờ bật/tắt rời rạc: ba
  giá trị `llm | cross_encoder | cross_encoder_then_llm`; mọi đường đều rơi về `llm` khi
  service lỗi. Lý do: benchmark và eval cần đổi chế độ bằng một biến môi trường.
- **Giữ top-K, không gate theo điểm CE**: điểm BGE là logit, ngưỡng phải hiệu chuẩn trên
  phân phối thật; benchmark này sinh ra phân phối đó.
- **Passage = heading_path + content**, `max_length=1024`, ghi tỷ lệ cắt; `summary` là biến
  thể dự phòng nếu cắt nhiều.
- **Benchmark tách khỏi graph**: `evals/benchmark` gọi thẳng hàm rerank của từng biến thể
  trên bộ ứng viên đã ghi (`runs/<run>`), không chạy classifier/HyDE/Qdrant, nên loại được
  nhiễu retrieval và rẻ. Tầng câu trả lời đo bằng `evals.run` sẵn có với chế độ bật.
- **VRAM**: harness không giữ model; tự spawn service khi cần và luôn tắt ở `finally`, xác
  nhận bằng nvidia-smi trước khi kết thúc.
- **Khung báo cáo theo vya-ocr** (`-a`/`-p`/`-c`, md + json theo timestamp) để đọc quen.

## Dependency Graph

```
Task 1 service ──┬── Task 2 agent client + node + chế độ ──┬── Task 5 eval e2e
                 └── Task 3 harness (-a) ── Task 4 (-p, -c, VRAM) ── Task 6 chạy đủ, viết kết quả
Task 7 docs/task sau Task 2–4
```

## Task List

### Phase 1: Service

- [x] Task 1: `reranker/` (pyproject uv, `app.py` FastAPI: lifespan load bge-reranker-v2-m3 FP16 qua transformers,
      `/rerank` batch + `ranking`, `/health`, giải phóng khi tắt), `taskfiles/reranker.yml`
      (`reranker:serve`, `reranker:test`), test với scorer giả, chạy thật trên 4080 một lần.

### Checkpoint: Service

- [ ] `curl /health` → loaded=true, cuda; 16 passage → 16 điểm; Ctrl+C → VRAM về 0.

### Phase 2: Nối vào graph

- [x] Task 2: settings `CHAT_RERANK_MODE`, `CROSS_ENCODER_URL`, `CROSS_ENCODER_TIMEOUT_SECONDS`,
      `CHAT_CROSS_ENCODER_KEEP`; `cross_encoder_client.py`; node `cross_encoder_rerank.py`
      (top-K, `metadata.ce_score`, chế độ hai tầng, fail-over LLM); rẽ nhánh trong
      `streaming_graph` + trace `09a_CrossEncoderRerankNode`; fingerprint cache; tests.

### Checkpoint: Graph

- [ ] `CHAT_RERANK_MODE=llm` test cũ xanh; `cross_encoder` chạy tay trên eval: log node CE, tắt
      service → warning + LLM rerank.

### Phase 3: Benchmark

- [x] Task 3: `evals/benchmark/`: `candidates.py` (đọc `runs/<run>`, fetch chunk theo id),
      `variants.py` (3 biến thể gọi cùng giao diện `rerank(question, chunks) -> kept`),
      `metrics.py` (recall@k, NDCG@16, MRR, empty rate, Jaccard ổn định), `report.py`,
      `cli.py` với `-a`; `task benchmark:run`.
- [x] Task 4: `-p` (warm-up, iterations, sampler CPU/RAM/VRAM), `-c` (mức đồng thời,
      semaphore, `--include-llm`), spawn/tắt service, kiểm tra VRAM trước/sau, ghi vào báo cáo.

### Checkpoint: Benchmark

- [ ] `task benchmark:run -- hybrid-on --limit 20` ra md + json đủ 3 bảng; sau khi xong
      `nvidia-smi` sạch.

### Phase 4: Đo

- [x] Task 5: eval tầng câu trả lời: `runs/ce-on`, `runs/ce-then-llm` so với `runs/hybrid-on`
      (llm) cùng điều kiện; bảng recall@5 / pass / fallback / p95.
- [x] Task 6: chạy đủ benchmark trên 4080 (`-a --repeats 3`, `-p`, `-c`), ghi bảng vào plan,
      khuyến nghị; ghi việc cần đo lại trên 3060.

### Phase 5: Tài liệu

- [x] Task 7: `docs/guide/benchmark-rerank.md`, `docs/guide/cau-hinh.md` (4 biến), `.env.example`,
      `.env.eval.example`, `dev-onboard.txt` mục 15 (ngoài repo), spec khớp code.

### Checkpoint: Hoàn tất

- [x] `task test` 2671 passed (bỏ tests/e2e lỗi import có sẵn trên main), ruff/mypy sạch trên file đổi; PR lên `main` (26/10).

### Phase 6 (26/10 chiều, theo review): chống bịa khi không còn LLM rerank

Bối cảnh: `none`/CE không có gate "không tài liệu nào trả lời"; 5 câu demo không có tài liệu cho
thấy nhánh normal từ chối đúng, nhánh calculation còn tự tính từ giả định. Thứ tự do review chốt.

- [x] Task 8: **Calculation gate + kiểm tra trích dẫn/số liệu sau khi sinh.** `app/graph/grounding.py`:
      trích số trong câu trả lời, tập "có nguồn" = câu hỏi + giá trị đã biết + chunk được trích dẫn
      (normal: chunk trích dẫn, không có trích dẫn thì toàn ngữ cảnh), chấp nhận số suy ra bằng một
      phép tính từ các số có nguồn; chỉ số `[n]` ngoài phạm vi bị loại. Nhánh calculation (LLM):
      số không có nguồn → ghi chú giới hạn vào câu trả lời + cảnh báo `CALC_UNGROUNDED_INPUT`;
      prompt `chat_calculation_llm.yaml` thêm luật không tự điền giả định, phân biệt "không đủ dữ
      liệu" với 0. Nhánh normal: số không có nguồn → chú thích cuối câu trả lời (stream đã đi,
      không sinh lại) + `ANSWER_UNGROUNDED_NUMBERS`; không trích dẫn mà có số → `ANSWER_NO_CITATION`.
      Eval: `checks.ungrounded`, `checks.no_citation` từ warning_codes, cột trong summary.
- [x] Task 9: **Bộ câu unanswerable phân nhóm** (dataset gateway, `source_set=demo`, id d0337–d0356):
      5 nhóm × 4 câu: `not_in_corpus`, `partial`, `false_premise`, `near_miss` (đúng chủ đề, sai
      năm/chương trình/đối tượng, kèm `forbidden_source_patterns`), `calc_missing_input`. Trường mới
      `unanswerable_kind`. Metrics: pass theo nhóm (partial = trả lời phần có + nói rõ giới hạn),
      thêm `fabricated` (không từ chối và có số không nguồn hoặc trích tài liệu cấm), summary theo
      nhóm. Chạy eval **chỉ** các câu mới với `none`, `cross_encoder`, `llm`.
- [x] Task 10: **Auto rerank có log tín hiệu, ngưỡng từ dữ liệu.** Fusion ghi vào metadata chunk:
      `rrf_score`, `dense_rank`, `bm25_rank`. `app/graph/evidence.py`: tín hiệu mỗi sub-query (cosine
      top-1, RRF top-1, khoảng cách top-1/top-2, đồng thuận dense/BM25 trong top-3, điểm CE top-1 nếu
      có) → `ok | weak | none`, log mỗi lượt. Nghiên cứu ngưỡng: chạy retrieval 139 câu, so tín hiệu
      với "tài liệu đúng trong top-8", chọn ngưỡng, ghi vào report. `CHAT_RERANK_MODE=auto`: đường
      `none`, gọi LLM rerank khi có sub-query `weak`; không ép tỷ lệ.
- [x] Task 11: **Tách evidence sufficiency khỏi chính sách web.** `app/graph/web_policy.py`: quyết định
      tìm web từ (sufficiency, topic speculative, cờ bật) thay vì chỉ "sub-query trống"; `none`/`weak`
      → fallback khi bật; log quyết định. Web đang tắt trong eval: test đơn vị + log, đánh giá web
      fallback là ticket riêng.

### Checkpoint: Phase 6

- [x] Eval 20 câu unanswerable ở 3 chế độ: none 20/20, CE 20/20, llm 19/20 (1 bịa, d0350); report.md 8.2; 2693 test pass; push 26/10.

## Risks and Mitigations

- **torch CUDA trên Windows 3060**: nếu wheel không khớp driver → WSL2; service tách nên
  agent không ảnh hưởng.
- **Chunk dài bị cắt ở 1024 token**: đo tỷ lệ cắt trong `-a`; dự phòng rerank trên `summary`.
- **LLM rerank tốn phí khi lặp 3 lần**: giới hạn `--limit`, `-c` không chạy LLM nếu không có
  `--include-llm`.
- **Harness quên tắt service → VRAM kẹt**: `finally` + kiểm tra nvidia-smi; `task benchmark:run`
  in cảnh báo đỏ nếu VRAM sau cao hơn trước quá 200 MB.
- **Điểm CE không có ngưỡng**: không gate; ghi histogram điểm để ticket sau hiệu chuẩn.

## Kết quả đo (26/10/2026, RTX 4080 SUPER)

Báo cáo so sánh 4 phương án: `report.md`; bản đầy đủ từng bước và từng câu: `report-full.md` (cùng thư mục). Tóm tắt:

| Trục | `none` (RRF, không rerank) | `llm` (hiện tại) | `cross_encoder` | `cross_encoder_then_llm` |
|---|---|---|---|---|
| Độ trễ bước rerank p50 / p95 | 0 ms | 1,7 s / 3,3 s | 0,11 s / 0,15 s | 1,7 s / 2,6 s |
| Độ trễ cả lượt p50 / p95 (eval, 154 câu) | 4,9 s / 8,2 s | 5,3 s / 9,5 s | **4,6 s / 7,5 s** | 6,6 s / 17,9 s |
| recall@8 bước rerank (139 câu, cùng ứng viên) | **0,928** (= trần) | 0,880 | 0,921 | 0,876 |
| normal recall@5 e2e | **111/116** | 104/116 | 110/116 | 108/116 |
| calculation pass e2e | 19/37 | 18/38 | 20/38 | **21/38** |
| calculation recall@5 e2e (theo trích dẫn) | 19/38 | **23/38** | 20/38 | 20/38 |
| Từ chối trả lời | 31 | 41 | 32 | **29** |
| Tái lập qua 3 lượt | 100 % | 49,6 % | 100 % | 49,6 % |
| USD / lượt | **0,00483** | 0,00557 | 0,00503 | 0,00540 |
| Chịu tải service (1 câu, 30 req/mức) | — | — | mức 1/2/4/8: p50 146/269/538/1076 ms, ~7,3 req/s | — |
| VRAM | — | — | 1 839 MiB rảnh, 2 987 MiB đỉnh, về 494 MiB sau khi tắt | — |

Mốc không rerank (`none`: 8 chunk đầu theo RRF sau sàn dense) recall@8 0,928 = trần của bộ ứng
viên: trên corpus này bước rerank không còn việc để làm; `none` và `cross_encoder` hoà trong mức
nhiễu ở tầng câu trả lời, cả hai hơn `llm` rõ (+6–7 câu normal, −9–10 từ chối, −10–13 % chi phí,
nhanh hơn 0,5–0,7 s). Tiêu chí thành công của spec (cross-encoder so với LLM): 3/5 đạt trọn, 2/5
đạt một phần (MRR 0,815 so với 0,890; calculation recall@5 20 so với 23 do mất trích dẫn).
Chi tiết report.md (bản so sánh 4 phương án); từng câu lệch/trượt: report-full.md 5.3, 7.6.

**Khuyến nghị**: đặt `CHAT_RERANK_MODE=none` mặc định (rẻ nhất, không cần GPU, bằng cross-encoder
về chất lượng); giữ `cross_encoder` làm lựa chọn cho corpus mà retrieval xếp kém; tắt `llm`; không
triển khai `cross_encoder_then_llm`. Điều kiện: gate web search khác khi bật lại web search; kiểm
tra trích dẫn nhánh calculation; đo lại trên bộ official trước khi khái quát; sửa rate limit
credential RERANK (70/154 lượt lỗi trong run gốc).

**Cần đo lại trên Windows 3060**: `-p`, `-c` (kỳ vọng p50 ~250–300 ms, p95 ~350 ms, ~3 req/s);
`-a` không cần (tất định).

## Open Questions

- Passage có nên kèm `summary` không: **đã đo** — service báo `truncated=0` ở mọi request (không
  chunk nào vượt 1024 token với heading_path + content), nên chưa cần biến thể `summary`.
- Gate web search khi không còn LLM rerank (report-full.md 8.2 mục 3): ticket sau.
- Trích dẫn nhánh calculation ở chế độ `cross_encoder` (report-full.md 7.4): ticket sau.
