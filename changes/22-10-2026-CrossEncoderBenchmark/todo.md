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

- [x] Client giả trả điểm → giữ đúng top-K, thứ tự theo điểm, `ce_score` có mặt, `score` không đổi.
- [x] Client ném lỗi/timeout → kết quả bằng LLM rerank, log warning một dòng.
- [x] Chế độ hai tầng: LLM nhận đúng K ứng viên.
- [x] Đổi `CHAT_RERANK_MODE` → khoá cache khác (test fingerprint).

**Verification:**

- [x] `uv run pytest tests/graph -q`; chạy tay trên eval với service bật rồi tắt.

**Dependencies:** Task 1 · **Files:** như trên · **Scope:** M

**Kết quả (22/10):** `app/integrations/cross_encoder_client.py` (httpx, một deadline
`CROSS_ENCODER_TIMEOUT_SECONDS`, mọi lỗi → `CrossEncoderUnavailableError` tiếng Việt),
`app/graph/nodes/cross_encoder_rerank.py` (gather một request/sub-query, passage = heading_path +
content, top-K theo điểm CE, `metadata["ce_score"]`, score giữ nguyên, không bao giờ làm rỗng
sub-query). Graph chính và node Calculation cùng rẽ nhánh theo `CHAT_RERANK_MODE`; CE lỗi →
warning `CROSS_ENCODER_FAILED` + LLM rerank + không ghi cache. Fingerprint cache thêm
`rerank_mode`, `ce_keep`. Tests: node 6, client 8, wiring 3, fingerprint 1; toàn bộ suite 2665
passed (bỏ `tests/e2e` vì lỗi import `ClarificationStateRepository` có sẵn trên main). Chạy tay
trên eval để ở Task 5.

## Task 3: Harness `-a`

**Description:** `evals/benchmark/candidates.py` (đọc `runs/<run>/results.jsonl`: id, category,
câu độc lập của sub-query đầu, 16 chunk_id, expected_document_ids; fetch chunk qua
`RetrievalService.fetch` SA; bỏ câu thiếu), `variants.py` (3 biến thể, cùng chữ ký, biến thể
LLM dùng `llm_rerank` thật qua registry), `metrics.py`, `report.py` (md + json), `cli.py`
(argparse theo vya-ocr: run id, `-a|-p|-c|--all`, `--variant`, `--repeats`, `--limit`, `--out`,
`--serve/--no-serve`), `taskfiles/benchmark.yml`.

**Acceptance criteria:**

- [x] Cùng bộ ứng viên cho mọi biến thể; số câu in ở đầu báo cáo.
- [x] Bảng `-a`: recall@8, recall@5, NDCG@16, MRR, empty rate, theo category; biến thể LLM có cột ổn định (Jaccard, lệch recall) khi `--repeats > 1`.
- [x] Tỷ lệ passage bị cắt ở 1024 token được ghi (service trả `truncated` hoặc ước lượng).

**Verification:**

- [x] `uv run pytest tests/evals/test_benchmark_metrics.py -q`; `task benchmark:run -- hybrid-on -a --limit 10 --variant cross_encoder`.

**Dependencies:** Task 2 · **Scope:** M

## Task 4: Harness `-p`, `-c`, VRAM

**Description:** `resource_sampler.py` (thread 0,2 s: CPU % qua `/proc/stat` hoặc psutil, RAM,
VRAM qua `nvidia-smi --query-gpu=memory.used`), `-p` (warm-up 3, `--iterations`, tuần tự),
`-c` (1 câu cố định, `--concurrency-levels 1,2,4,8`, `--conc-requests 30`, semaphore,
`--include-llm`), spawn service nếu `/health` không trả lời và tắt ở `finally`, VRAM
trước/sau, cảnh báo nếu không trả.

**Acceptance criteria:**

- [x] Bảng `-p`: P50/P95/avg ms, request/s, avg CPU, peak RAM, peak VRAM, error rate cho từng biến thể.
- [x] Bảng `-c`: mỗi mức: P50/P95, throughput, lỗi, peak VRAM.
- [x] Kết thúc (kể cả Ctrl+C) service do harness spawn bị tắt; báo cáo ghi VRAM trước/sau.

**Verification:**

- [x] `task benchmark:run -- hybrid-on -p --limit 10`; `-c --conc-requests 10`; `nvidia-smi` sau đó.

**Dependencies:** Task 3 · **Scope:** M

**Kết quả Task 3 + 4 (26/10):** `evals/benchmark/` (candidates, variants, metrics, runner,
resource_sampler, service, report, cli, `__main__`), `taskfiles/benchmark.yml`
(`task benchmark:run env=eval -- <run> [-a|-p|-c|--all] ...`). Smoke trên eval, 3 câu,
`--variant cross_encoder --all`: service tự spawn (VRAM 494 → 1839 MiB), -a recall@8 1.0,
2 lượt cùng thứ tự 100%; -p 16 passage thật p50 87 ms (service took_ms ~131 ms với chunk dài,
khác 10 ms của passage ngắn ở Task 1); -c mức 1: p50 144 ms, mức 2: p50 267 ms (GPU xử lý tuần
tự); tắt service → VRAM về 494 MiB, "đã nhả VRAM: có". Tests `tests/evals/test_benchmark_rerank.py`
6/6 (biến thể giả, không Qdrant/service). Biến thể LLM chưa chạy thật (cần backend eval + key),
để Task 6.

## Task 5: Eval tầng câu trả lời

**Description:** `CHAT_RERANK_MODE=cross_encoder` và `cross_encoder_then_llm` chạy `evals.run`
(demo normal + calculation, exact, web off, cache tắt) ra `runs/ce-on`, `runs/ce-then-llm`;
so với `runs/hybrid-on`.

**Acceptance criteria:**

- [x] Bảng recall@5, pass calculation, ticket fallback, latency p50/p95 cho 3 chế độ trong plan.

**Dependencies:** Task 2 · **Scope:** S (thời gian chạy)

**Kết quả Task 5 (26/10):** `runs/ce-on`, `runs/ce-then-llm` so với `runs/hybrid-on`: normal
recall@5 110 / 108 / 104 trên 116; calculation pass 20 / 21 / 18 trên 38, recall@5 20 / 20 / 23;
từ chối 32 / 29 / 41; p50 toàn lượt 4,6 / 6,6 / 5,3 s; p95 7,5 / 17,9 / 9,5 s. Chi tiết
`report.md` mục 7. Bổ sung 26/10 chiều: chế độ `none` (`runs/rrf-only`): normal recall@5
111/116, calculation pass 19/37, từ chối 31, p50 4,9 s, USD/lượt 0,00483 — hoà `cross_encoder`,
khuyến nghị cuối chuyển sang `none` mặc định (`report.md` 1.3, 7.7).

## Task 6: Chạy đủ benchmark trên 4080, khuyến nghị

- [x] `-a --repeats 3` (3 biến thể), `-p --iterations 5`, `-c` (CE; LLM nếu đồng ý tốn phí).
- [x] Bảng + histogram điểm CE + khuyến nghị bật/không; danh sách việc đo lại trên 3060.

**Dependencies:** Task 4, Task 5 · **Scope:** S

**Kết quả Task 6 (26/10):** bước A–D chạy đủ trên 4080 (`benchmark_results/step-{a,b,c,d}-*`);
bảng và khuyến nghị trong `plan.md` "Kết quả đo" và `report.md`. Sự cố: lần đầu -a LLM hỏng vì
rate limit 15 lượt/phút → thêm `--llm-rpm`/retry; biến thể LLM cần `lifespan` app.

## Task 7: Tài liệu

- [x] `docs/guide/benchmark-rerank.md` (theo BENCHMARK.md của vya-ocr), `cau-hinh.md` 4 biến, `.env.example`, `.env.eval.example`, `dev-onboard.txt` mục 15, spec khớp code.

**Dependencies:** Task 2–4 · **Scope:** S

**Kết quả Task 7 (26/10):** `docs/guide/benchmark-rerank.md`, `cau-hinh.md` 4 biến,
`.env.example`, `.env.eval.example`, `dev-onboard.txt` mục 15, spec đồng bộ (transformers thay
FlagEmbedding, trường `truncated`), `report.md` 538 dòng.

## Checkpoint: Hoàn tất

- [x] `task test`, `uv run ruff check app evals tests`, `uv run mypy app evals` sạch trên file đổi.
- [x] PR lên `main`.
