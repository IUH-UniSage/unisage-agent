# Todo — Cross-encoder rerank (UNISAGE-106)

Plan: `changes/22-10-2026-CrossEncoderBenchmark/plan.md`. Spec: `docs/specs/SPEC-cross-encoder-rerank.md`.

## Task 1: Service reranker

**Description:** `reranker/pyproject.toml` (fastapi, uvicorn, transformers + torch CUDA 13 — không dùng FlagEmbedding, xem ghi chú),
`reranker/app.py`: lifespan load `AutoModelForSequenceClassification` bge-reranker-v2-m3 FP16 (cùng cách tính điểm với `FlagReranker`),
`POST /rerank {query, passages, top_k?} -> {scores, ranking, model, took_ms}` (batch
`RERANKER_BATCH_SIZE`, `max_length=1024`), `GET /health`; tắt → del model + `empty_cache`.
`taskfiles/reranker.yml`: `reranker:serve` (`uv run --project reranker uvicorn ... --port 8403`),
`reranker:test`. Scorer được inject để test không tải model.

**Acceptance criteria:**

- [x] `/rerank` 16 passage → 16 điểm cùng thứ tự đầu vào; `ranking` giảm dần theo điểm; `top_k` cắt `ranking`.
- [x] Rỗng passages → 400; passage > max_length bị cắt, không lỗi.
- [x] `/health` báo loaded/device/vram; sau SIGTERM `nvidia-smi` không còn tiến trình.

**Verification:**

- [x] `task reranker:test`; chạy thật: `task reranker:serve` + `curl` 16 passage, ghi took_ms lần đầu/lần sau.

**Dependencies:** none · **Files:** `reranker/*`, `taskfiles/reranker.yml`, `Taskfile.yml` include · **Scope:** M

**Kết quả (22/10, RTX 4080 SUPER):** `reranker:test` 6/6; chạy thật: model nạp ~50 s lần đầu
(tải 2.3 GB), `/health` `vram_used_mb=1083`; `/rerank` 16 passage: lần đầu 334 ms (warm-up),
5 lần sau 10–11 ms e2e; chunk học phí CNTT xếp hạng 1. VRAM `nvidia-smi` 7546 → 8987 MiB khi
serve → 7546 MiB sau Ctrl+C (log "reranker unloaded, VRAM released"). Ghi chú: thay FlagEmbedding
bằng transformers trực tiếp vì FlagEmbedding 1.4.2 kéo theo datasets/peft/ir-datasets; điểm là
logit[0] y hệt `FlagReranker.compute_score`. Response có thêm `truncated` (số passage vượt
`max_length`) để trả lời câu hỏi mở về chunk dài.

## Task 2: Agent: chế độ rerank + node CE + fail-over

**Description:** `config.py` 4 setting; `app/integrations/cross_encoder_client.py` (httpx async,
timeout, lỗi → `CrossEncoderUnavailable`); `app/graph/nodes/cross_encoder_rerank.py`:
`cross_encoder_rerank(questions, rerank_result, client) -> TurnRerankResult` giữ top-K mỗi
sub-query theo điểm CE, `metadata["ce_score"]`, `score` giữ nguyên; `streaming_graph` rẽ nhánh:
`llm` như cũ; `cross_encoder` → node CE; `cross_encoder_then_llm` → CE rồi LLM rerank trên K;
CE lỗi → log + LLM rerank. Trace `09a_CrossEncoderRerankNode`. Fingerprint cache thêm chế độ + K.

**Acceptance criteria:**

- [ ] Client giả trả điểm → giữ đúng top-K, thứ tự theo điểm, `ce_score` có mặt, `score` không đổi.
- [ ] Client ném lỗi/timeout → kết quả bằng LLM rerank, log warning một dòng.
- [ ] Chế độ hai tầng: LLM nhận đúng K ứng viên.
- [ ] Đổi `CHAT_RERANK_MODE` → khoá cache khác (test fingerprint).

**Verification:**

- [ ] `uv run pytest tests/graph -q`; chạy tay trên eval với service bật rồi tắt.

**Dependencies:** Task 1 · **Files:** như trên · **Scope:** M

## Task 3: Harness `-a`

**Description:** `evals/benchmark/candidates.py` (đọc `runs/<run>/results.jsonl`: id, category,
câu độc lập của sub-query đầu, 16 chunk_id, expected_document_ids; fetch chunk qua
`RetrievalService.fetch` SA; bỏ câu thiếu), `variants.py` (3 biến thể, cùng chữ ký, biến thể
LLM dùng `llm_rerank` thật qua registry), `metrics.py`, `report.py` (md + json), `cli.py`
(argparse theo vya-ocr: run id, `-a|-p|-c|--all`, `--variant`, `--repeats`, `--limit`, `--out`,
`--serve/--no-serve`), `taskfiles/benchmark.yml`.

**Acceptance criteria:**

- [ ] Cùng bộ ứng viên cho mọi biến thể; số câu in ở đầu báo cáo.
- [ ] Bảng `-a`: recall@8, recall@5, NDCG@16, MRR, empty rate, theo category; biến thể LLM có cột ổn định (Jaccard, lệch recall) khi `--repeats > 1`.
- [ ] Tỷ lệ passage bị cắt ở 1024 token được ghi (service trả `truncated` hoặc ước lượng).

**Verification:**

- [ ] `uv run pytest tests/evals/test_benchmark_metrics.py -q`; `task benchmark:run -- hybrid-on -a --limit 10 --variant cross_encoder`.

**Dependencies:** Task 2 · **Scope:** M

## Task 4: Harness `-p`, `-c`, VRAM

**Description:** `resource_sampler.py` (thread 0,2 s: CPU % qua `/proc/stat` hoặc psutil, RAM,
VRAM qua `nvidia-smi --query-gpu=memory.used`), `-p` (warm-up 3, `--iterations`, tuần tự),
`-c` (1 câu cố định, `--concurrency-levels 1,2,4,8`, `--conc-requests 30`, semaphore,
`--include-llm`), spawn service nếu `/health` không trả lời và tắt ở `finally`, VRAM
trước/sau, cảnh báo nếu không trả.

**Acceptance criteria:**

- [ ] Bảng `-p`: P50/P95/avg ms, request/s, avg CPU, peak RAM, peak VRAM, error rate cho từng biến thể.
- [ ] Bảng `-c`: mỗi mức: P50/P95, throughput, lỗi, peak VRAM.
- [ ] Kết thúc (kể cả Ctrl+C) service do harness spawn bị tắt; báo cáo ghi VRAM trước/sau.

**Verification:**

- [ ] `task benchmark:run -- hybrid-on -p --limit 10`; `-c --conc-requests 10`; `nvidia-smi` sau đó.

**Dependencies:** Task 3 · **Scope:** M

## Task 5: Eval tầng câu trả lời

**Description:** `CHAT_RERANK_MODE=cross_encoder` và `cross_encoder_then_llm` chạy `evals.run`
(demo normal + calculation, exact, web off, cache tắt) ra `runs/ce-on`, `runs/ce-then-llm`;
so với `runs/hybrid-on`.

**Acceptance criteria:**

- [ ] Bảng recall@5, pass calculation, ticket fallback, latency p50/p95 cho 3 chế độ trong plan.

**Dependencies:** Task 2 · **Scope:** S (thời gian chạy)

## Task 6: Chạy đủ benchmark trên 4080, khuyến nghị

- [ ] `-a --repeats 3` (3 biến thể), `-p --iterations 5`, `-c` (CE; LLM nếu đồng ý tốn phí).
- [ ] Bảng + histogram điểm CE + khuyến nghị bật/không; danh sách việc đo lại trên 3060.

**Dependencies:** Task 4, Task 5 · **Scope:** S

## Task 7: Tài liệu

- [ ] `docs/guide/benchmark-rerank.md` (theo BENCHMARK.md của vya-ocr), `cau-hinh.md` 4 biến, `.env.example`, `.env.eval.example`, `dev-onboard.txt` mục 15, spec khớp code.

**Dependencies:** Task 2–4 · **Scope:** S

## Checkpoint: Hoàn tất

- [ ] `task test`, `uv run ruff check app evals tests`, `uv run mypy app evals` sạch trên file đổi.
- [ ] PR lên `main`.
