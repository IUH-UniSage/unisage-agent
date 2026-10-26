# Benchmark rerank: LLM rerank so với cross-encoder

Thiết kế: `docs/specs/SPEC-cross-encoder-rerank.md`. Kết quả đo 26/10/2026 và khuyến nghị:
`changes/22-10-2026-CrossEncoderBenchmark/report.md` (đầy đủ, từng số liệu kèm đọc số) và
`plan.md` cùng thư mục (bảng tóm tắt).

Bước 09a của graph hiện gọi model LLM (purpose `RERANK`, không có thì Extraction) một lần mỗi
lượt để chọn chunk nào trả lời câu hỏi con: 3–6 s cho 16 chunk. Cross-encoder
`BAAI/bge-reranker-v2-m3` chấm từng cặp (câu hỏi, chunk) trong vài chục đến vài trăm ms trên
GPU. Bộ công cụ này đo xem đổi sang cross-encoder (hoặc cross-encoder rồi mới LLM) có nhanh hơn
mà giữ được độ chính xác không.

## Ba phần

| Phần | Ở đâu | Chạy |
|---|---|---|
| Service cross-encoder | `reranker/` (project uv riêng, có torch; agent không có torch) | `task reranker:sync` một lần, rồi `task reranker:serve` |
| Chế độ rerank của agent | `CHAT_RERANK_MODE` + node `09a_CrossEncoderRerankNode` | đặt biến trong `.env` / `.env.eval` |
| Harness benchmark | `evals/benchmark/` | `task benchmark:run env=eval -- <run> ...` |

## Service

```
task reranker:sync          # cài torch CUDA + transformers vào reranker/.venv (~2,5 GB, một lần)
task reranker:serve         # http://127.0.0.1:8403 ; lần đầu tải model ~2,3 GB vào ~/.cache/huggingface
task reranker:health        # {"status":"ok","model":...,"device":"cuda","loaded":true,"vram_used_mb":1083}
task reranker:test          # test HTTP với scorer giả, không tải model
```

- `POST /rerank` `{"query", "passages": [...], "top_k"?}` → `{"scores", "ranking", "truncated",
  "model", "took_ms", "batched"}`. `scores` theo thứ tự passage, `ranking` là chỉ số tốt nhất trước,
  `truncated` là số passage vượt `RERANKER_MAX_LENGTH` (1024 token) bị cắt.
- Biến môi trường của service: `RERANKER_MODEL`, `RERANKER_MAX_LENGTH` (**512**, đo 26/10: bằng
  hoặc hơn 1024 về độ chính xác, nhanh 1,6 lần; 384 nhanh hơn nữa nhưng cắt 44 % passage),
  `RERANKER_BATCH_SIZE` (16), `RERANKER_DEVICE` (`cuda`/`cpu`), `RERANKER_FP16` (`true`),
  `RERANKER_MAX_PASSAGES` (64), `RERANKER_BATCH_WINDOW_MS` (0: gom các request đến cùng lúc thành
  một forward, không chờ thêm; GPU bão hoà compute nên cửa sổ > 0 không tăng throughput),
  `RERANKER_MAX_BATCH_PAIRS` (64); cổng/host qua `task reranker:serve RERANKER_PORT=8403
  RERANKER_HOST=127.0.0.1`. Response có `batched` (số request chung một forward), `/health` có
  `batches`/`requests`.
- Ctrl+C / SIGTERM gỡ model và `torch.cuda.empty_cache()`: `nvidia-smi` về mức trước khi chạy.
  Chỉ bind `127.0.0.1`, không có auth: production đặt sau mạng nội bộ.
- Điểm là logit (có thể âm), không phải xác suất; node không dùng ngưỡng, chỉ giữ top-K.
- VRAM: ~1,1 GB theo torch, ~1,4 GB theo `nvidia-smi` (FP16, RTX 4080 SUPER). CPU chạy được
  nhưng chậm, chỉ để test.

## Chế độ rerank của agent

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `CHAT_RERANK_MODE` | `llm` | `llm`: như cũ, không gọi service. `none`: không rerank gì, sàn cosine rồi lấy thứ tự retrieval (RRF) cắt 8, mốc "không rerank" đo được e2e. `cross_encoder`: node CE giữ top-K mỗi câu hỏi con, **bỏ** LLM rerank. `cross_encoder_then_llm`: CE cắt 16 → K, LLM rerank chấm K chunk đó |
| `CROSS_ENCODER_URL` | `http://127.0.0.1:8403` | Service |
| `CROSS_ENCODER_TIMEOUT_SECONDS` | `2.0` | Một deadline cho cả request |
| `CHAT_CROSS_ENCODER_KEEP` | `8` | K chunk giữ mỗi câu hỏi con |

- Service lỗi/timeout ở hai chế độ CE → log warning, cảnh báo AI admin `CROSS_ENCODER_FAILED`,
  chạy LLM rerank như chế độ `llm` (fail-over), và lượt đó không ghi retrieval cache.
- `score` của chunk giữ nguyên (cosine hoặc 0,70 của BM25-only); điểm CE nằm ở
  `metadata["ce_score"]`. Node không bao giờ làm rỗng một câu hỏi con có ứng viên, nên ở chế độ
  `cross_encoder` quyết định "không chunk nào trả lời → web search" không còn; chế độ hai tầng
  giữ quyết định đó cho LLM.
- Retrieval cache: fingerprint có `rerank_mode` và `ce_keep`, đổi chế độ là khoá khác.
- Trace: `09a_CrossEncoderRerankNode` (có `elapsed_ms` như mọi node); node Calculation dùng cùng
  nhánh.

## Harness benchmark

Nguồn ứng viên là một run eval đã ghi (`runs/<run>/results.jsonl`, trường `retrievals`): mỗi
câu lấy câu hỏi độc lập + 16 `chunk_id` của câu hỏi con đầu, fetch lại nội dung từ Qdrant theo
id với đúng filter quyền của lượt đó. Mọi biến thể chấm **cùng** danh sách này; ground truth là
`expected_document_ids` của câu.

```
task benchmark:run env=eval -- hybrid-on                         # cả 3 bảng, cả 3 biến thể
task benchmark:run env=eval -- hybrid-on -a --repeats 3          # Bảng 1
task benchmark:run env=eval -- hybrid-on -p --iterations 5       # Bảng 2
task benchmark:run env=eval -- hybrid-on -c --concurrency-levels 1,2,4,8 --conc-requests 30
task benchmark:run env=eval -- hybrid-on -a --variant cross_encoder --limit 30 --category calculation
task benchmark:help
```

| Cờ | Ý nghĩa |
|---|---|
| `-a` / `-p` / `-c` / `--all` | Bảng 1 accuracy, Bảng 2 performance, Bảng 3 concurrency; không cờ = `--all` |
| `--variant V` (lặp được) | `none`, `llm_rerank`, `cross_encoder`, `cross_encoder_then_llm`; mặc định cả 4 |
| `--repeats N` | Số lượt `-a` của biến thể LLM (3); cross-encoder luôn 2 lượt để kiểm tra tất định |
| `--limit N`, `--category C`, `--id ID` | Thu hẹp bộ câu |
| `--keep K` | K của cross-encoder (mặc định `CHAT_CROSS_ENCODER_KEEP`) |
| `--warmup 3`, `--iterations 5` | `-p`: bỏ 3 request đầu, rồi 5 vòng qua toàn bộ câu ở concurrency 1 |
| `--concurrency-levels 1,2,4,8`, `--conc-requests 30` | `-c`: 1 câu cố định (câu nhiều ứng viên nhất), `asyncio.Semaphore` mỗi mức |
| `--include-llm` | `-c` cũng bắn biến thể LLM (tốn phí API); mặc định `-c` chỉ chạy cross-encoder |
| `--serve` / `--no-serve` | Tự chạy service nếu chưa có (mặc định) / báo lỗi nếu chưa có |
| `--out DIR` | Mặc định `benchmark_results/<timestamp>/` (gitignore) |

- Biến thể LLM gọi `llm_rerank` thật với model RERANK của môi trường (`get_graph_models`):
  cần backend eval đang chạy và key có sẵn. Chỉ `--variant cross_encoder` thì không cần.
- Harness **không** nạp model trong tiến trình: nếu `CROSS_ENCODER_URL` chưa trả lời, nó spawn
  `uv run --project reranker uvicorn ...`, chờ `/health` báo `loaded`, và **luôn** tắt ở
  `finally`, chờ `nvidia-smi` về mức trước rồi mới ghi báo cáo (mục "VRAM trước / sau"). Service
  có sẵn thì để nguyên và báo cáo ghi chú.
- Báo cáo `rerank_benchmark.md` + `.json`: đầu ghi GPU/driver, model, run nguồn, số câu dùng
  được và lý do bỏ; Bảng 1 có recall@8, recall@5, NDCG@16, MRR, tỷ lệ loại hết, số chunk giữ
  trung bình, lỗi, p50/p95 của riêng bước rerank, phân rã theo category, và bảng ổn định giữa
  các lượt (Jaccard tập giữ, tỷ lệ cùng thứ tự, lệch recall@8); Bảng 2 có p50/p95/TB/max,
  req/s, CPU %, RAM, VRAM đỉnh, GPU %; Bảng 3 theo mức đồng thời. JSON có từng lượt từng câu
  (`kept_ids`, `ce_scores`) để soi câu lệch.

Định nghĩa: recall@k = phần tài liệu kỳ vọng xuất hiện trong k chunk giữ đầu; NDCG@16 nhị phân
theo `document_id`, lý tưởng là mọi chunk đúng trong bộ ứng viên xếp trước; MRR = 1/hạng của
chunk đúng đầu tiên; lỗi = lượt biến thể không trả kết quả (service/model lỗi), không tính
fail-open.

## Đo tầng câu trả lời (e2e)

Benchmark chỉ đo bước rerank. Để biết câu trả lời có đổi không, chạy eval cùng điều kiện với
`runs/hybrid-on` (cache tắt: `RETRIEVAL_CACHE_REDIS_URL=` trống), đổi `CHAT_RERANK_MODE`:

```
task reranker:serve                                   # terminal riêng
CHAT_RERANK_MODE=cross_encoder task eval:run env=eval -- --set demo --category normal --category calculation --variant exact --web off --out runs/ce-on
CHAT_RERANK_MODE=cross_encoder_then_llm task eval:run env=eval -- --set demo --category normal --category calculation --variant exact --web off --out runs/ce-then-llm
```

So `summary.json`: recall@5, pass calculation, ticket fallback, p50/p95 toàn lượt.

## Windows (RTX 3060 12 GB)

- `task reranker:sync` cần torch CUDA wheel cho Windows; nếu `uv sync` không lấy được wheel CUDA,
  chạy service trong WSL2 và trỏ `CROSS_ENCODER_URL` sang địa chỉ WSL.
- Số cần đo lại trên 3060: Bảng 2 (p50/p95, VRAM đỉnh) và Bảng 3; Bảng 1 là tất định với cùng
  model nên không đổi.
