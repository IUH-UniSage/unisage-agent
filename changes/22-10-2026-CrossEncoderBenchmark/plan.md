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

- [ ] Task 5: eval tầng câu trả lời: `runs/ce-on`, `runs/ce-then-llm` so với `runs/hybrid-on`
      (llm) cùng điều kiện; bảng recall@5 / pass / fallback / p95.
- [ ] Task 6: chạy đủ benchmark trên 4080 (`-a --repeats 3`, `-p`, `-c`), ghi bảng vào plan,
      khuyến nghị; ghi việc cần đo lại trên 3060.

### Phase 5: Tài liệu

- [ ] Task 7: `docs/guide/benchmark-rerank.md`, `docs/guide/cau-hinh.md` (4 biến), `.env.example`,
      `.env.eval.example`, `dev-onboard.txt` mục 15 (ngoài repo), spec khớp code.

### Checkpoint: Hoàn tất

- [ ] `task test`, ruff, mypy sạch trên file đổi; PR lên `main`.

## Risks and Mitigations

- **torch CUDA trên Windows 3060**: nếu wheel không khớp driver → WSL2; service tách nên
  agent không ảnh hưởng.
- **Chunk dài bị cắt ở 1024 token**: đo tỷ lệ cắt trong `-a`; dự phòng rerank trên `summary`.
- **LLM rerank tốn phí khi lặp 3 lần**: giới hạn `--limit`, `-c` không chạy LLM nếu không có
  `--include-llm`.
- **Harness quên tắt service → VRAM kẹt**: `finally` + kiểm tra nvidia-smi; `task benchmark:run`
  in cảnh báo đỏ nếu VRAM sau cao hơn trước quá 200 MB.
- **Điểm CE không có ngưỡng**: không gate; ghi histogram điểm để ticket sau hiệu chuẩn.

## Kết quả đo (điền khi chạy)

(chưa có)

## Open Questions

- Passage có nên kèm `summary` không: đo rồi quyết (xem spec).
