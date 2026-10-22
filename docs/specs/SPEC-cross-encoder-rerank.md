# SPEC — Cross-encoder rerank: service, nối vào graph sau cờ, benchmark so với LLM rerank (UNISAGE-106)

> Tiếp nối "Phase sau: Cross-encoder" trong `changes/20-10-2026-HybridRetrieval/plan.md`
> (quyết định 20/10/2026). Spec này **chỉ thêm**: mặc định `CHAT_RERANK_MODE=llm` thì
> pipeline y hệt hôm nay. Đăng ký reranker vào model registry / web admin là ticket sau.

## Ý định đã chốt (phỏng vấn 22/10/2026)

- **Mục tiêu**: giảm độ trễ bước rerank (LLM rerank hiện 1,5–5 s mỗi lượt, dao động giữa
  các lần chạy) mà **không giảm** độ chính xác câu trả lời; tăng được càng tốt.
- **Chạy ở đâu**: benchmark chất lượng và chọn cấu hình trên máy Linux dev (RTX 4080 SUPER
  16 GB) trước; số tải (P95, peak VRAM) đo lại trên máy deploy Windows RTX 3060 12 GB sau,
  vì 3060 chậm hơn ~2 lần và ít VRAM hơn.
- **Phạm vi ticket**: (1) service cross-encoder riêng; (2) nối vào graph như một chế độ của
  bước rerank, **mặc định tắt**, service sập thì rơi về LLM rerank; (3) bộ benchmark ba chế
  độ Accuracy / Performance / Concurrency theo khung `vya-ocr` (`task benchmark:run`), báo
  cáo md + json, **tiến trình benchmark trả VRAM khi xong**. Không bật mặc định, không
  đăng ký registry, không sửa backend/web.
- **Độ chính xác đo hai tầng**: tầng rerank trên cùng một bộ ứng viên (recall@8, NDCG@16,
  MRR, tỷ lệ loại hết, độ ổn định qua 3 lần chạy) và tầng câu trả lời (`evals.run` bật/tắt
  chế độ trên cùng bộ câu hỏi). Không chạy RAGAS trong ticket này.

## Assumptions (sửa nếu sai)

- Model `BAAI/bge-reranker-v2-m3` (568M, đa ngữ, có tiếng Việt), FP16, `max_length=1024`
  cho cặp query + passage. Tải từ Hugging Face lần đầu (~1,1 GB FP16), cache tại
  `~/.cache/huggingface`.
- Service dùng FlagEmbedding (`FlagReranker`) + FastAPI + uvicorn, 1 worker, load model một
  lần, chạy native trên Linux; Windows chạy native nếu torch CUDA cài được, không thì WSL2.
- Bước rerank nhận tối đa 16 ứng viên/sub-query (`CHAT_RETRIEVAL_MAX_CHUNKS`), chunk dài tới
  ~800 token; một request rerank = 16 cặp.
- Điểm BGE là logit, không phải xác suất: không dùng ngưỡng chung; ticket này **không** dùng
  điểm làm gate (giữ top-K), hiệu chuẩn ngưỡng là việc sau khi có phân phối điểm từ benchmark.
- Bộ ứng viên chuẩn cho benchmark lấy từ một run eval đã ghi (`runs/<run>/results.jsonl`,
  trường `retrievals` có `chunk_id` của 16 ứng viên mỗi sub-query); nội dung chunk fetch lại
  từ Qdrant theo id với quyền SUPER_ADMIN. Ground truth = `expected_document_ids`.

## Objective

### Hiện trạng

Node `09a_LLMRerankNode` gửi 16 ứng viên (đã qua sàn cosine 0,45 và giữ thứ tự RRF) cho một
model chat (purpose RERANK, hiện gpt-5-mini) kèm câu hỏi, nhận JSON "giữ chunk nào, vì sao",
giữ theo thứ tự ứng viên rồi cắt `CHAT_CONTEXT_MAX_CHUNKS=8`. Chi phí: một lượt LLM
1,5–5 s mỗi lượt chat, kết quả thay đổi giữa các lần chạy (21/10: cùng câu, lần giữ 4 chunk
đúng, lần giữ cả 16 kể cả chunk nó tự nhận là không liên quan). Khi model lỗi → giữ nguyên
thứ tự điểm (fail-open).

### Thiết kế

1. **Service reranker** (`reranker/` trong repo agent, project uv riêng, không đụng
   dependency của agent):
   - `POST /rerank` body `{"query": str, "passages": [str], "top_k"?: int}` → `{"scores":
     [float], "ranking": [int], "model": str, "took_ms": int}`; điểm theo thứ tự passages,
     `ranking` là chỉ số sắp giảm dần. Batch nội bộ `RERANKER_BATCH_SIZE` (mặc định 16),
     `max_length=1024`, FP16 khi có CUDA, CPU khi không (chậm, chỉ để test).
   - `GET /health` → `{"status","model","device","loaded":bool,"vram_used_mb"}`.
   - Load model lúc khởi động (lifespan); tắt bằng SIGTERM thì giải phóng model và
     `torch.cuda.empty_cache()`.
   - Chạy: `task reranker:serve` (uvicorn, cổng `RERANKER_PORT` mặc định 8403).
   - Không auth: chỉ bind `127.0.0.1` mặc định; production đặt sau mạng nội bộ.
2. **Agent nối vào graph** (`app/integrations/cross_encoder_client.py`,
   `app/graph/nodes/cross_encoder_rerank.py`):
   - Cài đặt: `CHAT_RERANK_MODE` ∈ {`llm` (mặc định), `cross_encoder`,
     `cross_encoder_then_llm`}, `CROSS_ENCODER_URL` (`http://127.0.0.1:8403`),
     `CROSS_ENCODER_TIMEOUT_SECONDS` (2.0), `CHAT_CROSS_ENCODER_KEEP` (8: số chunk giữ mỗi
     sub-query; chế độ hai tầng dùng số này làm đầu vào cho LLM rerank).
   - `cross_encoder`: mỗi sub-query gửi (câu hỏi độc lập, 16 passage = heading_path +
     content) tới service, nhận điểm, giữ top-K theo điểm giảm dần; `score` của chunk giữ
     nguyên (cosine/0,70), điểm CE ghi vào `metadata["ce_score"]` để trace và benchmark.
     Không "loại hết" theo ngưỡng; sub-query luôn có ≥1 chunk nếu có ứng viên.
   - `cross_encoder_then_llm`: CE cắt 16 → K rồi LLM rerank như hiện nay trên K chunk.
   - Lỗi/timeout service → log warning, chạy LLM rerank như chế độ `llm` (fail-over, không
     fail-open về điểm). Node trace `09a_CrossEncoderRerankNode` với `elapsed_ms`.
   - Retrieval cache: `config_fingerprint` thêm `CHAT_RERANK_MODE` và `CHAT_CROSS_ENCODER_KEEP`
     (kết quả sau rerank khác nhau theo chế độ).
3. **Benchmark** (`evals/benchmark/`, lệnh `task benchmark:run -- <run> [-a|-p|-c|--all]
   [--variant V]... [--repeats N] [--limit N] [--out DIR]`), biến thể `llm_rerank`,
   `cross_encoder`, `cross_encoder_then_llm`; mặc định không cờ = cả 3 chế độ, cả 3 biến thể:
   - Nguồn: `runs/<run>` (vd `runs/hybrid-on`): mỗi câu lấy câu độc lập + 16 `chunk_id`
     của sub-query đầu, fetch nội dung từ Qdrant; bỏ câu không có `expected_document_ids`
     hoặc không có ứng viên. Cùng bộ này cho mọi biến thể.
   - `-a` Accuracy: mỗi biến thể chọn 8/16 cho từng câu. recall@8, recall@5 (tài liệu kỳ
     vọng có trong top), NDCG@16 và MRR (relevance nhị phân theo document_id), tỷ lệ loại hết.
     Biến thể có LLM chạy `--repeats 3`: độ ổn định = Jaccard trung bình của tập giữ giữa
     các lần và độ lệch recall; CE chạy 1 lần (tất định) nhưng kiểm tra 2 lần cho ra cùng
     thứ tự. Phân rã theo category (normal / calculation).
   - `-p` Performance: warm-up 3 request, rồi `--iterations` (mặc định 5) vòng tuần tự qua
     toàn bộ câu, concurrency 1; đo ms của riêng bước rerank; thread nền lấy mẫu CPU %,
     RAM, VRAM (nvidia-smi) mỗi 0,2 s. Ra P50/P95/avg, throughput (request/s), peak VRAM.
     LLM rerank đo cùng cách (độ trễ API) để so.
   - `-c` Concurrency: 1 câu cố định, mức 1 → 2 → 4 → 8 (`--concurrency-levels`), mỗi mức
     `--conc-requests` (30) request qua `asyncio.Semaphore`; P50/P95, throughput, lỗi,
     peak VRAM. Biến thể LLM chỉ chạy khi `--include-llm` (tốn phí).
   - Báo cáo `benchmark_results/<timestamp>/rerank_benchmark.md` + `.json` (gitignore); đầu
     báo cáo ghi GPU, driver, model, run nguồn, số câu; cuối ghi **VRAM trước / sau** khi
     benchmark kết thúc.
   - VRAM: harness không load model trong tiến trình; nếu service chưa chạy, harness tự
     spawn (`--serve`, mặc định bật) và **luôn** tắt ở `finally`, chờ VRAM về mức trước
     (đọc nvidia-smi) rồi mới in báo cáo; service có sẵn thì để nguyên và ghi chú.
4. **Đo tầng câu trả lời**: `task eval:run env=eval -- --set demo --category normal
   --category calculation --variant exact --web off` với `CHAT_RERANK_MODE=cross_encoder`
   (và `cross_encoder_then_llm`) so với run `llm` cùng điều kiện, cache tắt; so recall@5,
   pass calculation, ticket fallback, latency p50/p95 toàn lượt. Ghi vào plan.

### Acceptance

- `task reranker:serve` lên, `GET /health` báo `loaded=true`, device cuda; `POST /rerank`
  với 16 passage trả 16 điểm, `ranking` đúng thứ tự giảm dần.
- `CHAT_RERANK_MODE=llm` (mặc định): mọi test cũ xanh, không gọi service.
- `CHAT_RERANK_MODE=cross_encoder`: node CE giữ top-K, trace có `elapsed_ms`; tắt service
  → warning + LLM rerank chạy, câu trả lời vẫn có.
- `task benchmark:run -- hybrid-on -a --variant cross_encoder` ra báo cáo với đủ cột; chạy
  xong `nvidia-smi` không còn tiến trình của benchmark/service do nó spawn.
- Benchmark đầy đủ trên 4080 ghi vào plan với khuyến nghị bật/không bật.

## Tech Stack

Service: Python 3.12, FastAPI, uvicorn, FlagEmbedding, torch (CUDA 12.x wheel), project uv
riêng `reranker/pyproject.toml`. Agent: httpx (đã có). Benchmark: asyncio, nvidia-smi qua
subprocess (không thêm pynvml), psutil nếu đã có trong lock, không thì đọc `/proc`.

## Commands

```
task reranker:serve                      # service 8403 (Ctrl+C giải phóng VRAM)
task reranker:test
task benchmark:run -- hybrid-on          # cả 3 chế độ, cả 3 biến thể
task benchmark:run -- hybrid-on -a --repeats 3
task benchmark:run -- hybrid-on -p --iterations 5
task benchmark:run -- hybrid-on -c --concurrency-levels 1,2,4,8 --conc-requests 30
task benchmark:run -- hybrid-on -a --variant cross_encoder --limit 30
CHAT_RERANK_MODE=cross_encoder task eval:run env=eval -- --set demo --category normal --category calculation --variant exact --web off --out runs/ce-on
```

## Project Structure

```
reranker/pyproject.toml, reranker/app.py, reranker/tests/        → service riêng
app/integrations/cross_encoder_client.py                           → httpx client
app/graph/nodes/cross_encoder_rerank.py                            → node + chế độ hai tầng
app/graph/streaming_graph.py                                       → rẽ nhánh theo CHAT_RERANK_MODE
app/core/config.py                                                 → 4 setting mới
app/integrations/retrieval_cache.py                                → fingerprint thêm chế độ
evals/benchmark/{cli,runner,candidates,variants,metrics,report,resource_sampler}.py
taskfiles/benchmark.yml, taskfiles/reranker.yml
docs/guide/benchmark-rerank.md, docs/guide/cau-hinh.md, .env.example
```

## Code Style

Như repo: docstring nói vì sao, log một dòng mỗi lượt, không nuốt lỗi; service tách biệt
không import `app.*`.

## Testing Strategy

- Service: test FastAPI với scorer giả (không tải model) cho batching, thứ tự `ranking`,
  `/health`; test tích hợp thật đánh dấu `gpu` chạy tay.
- Agent: test node CE với client giả (giữ top-K, giữ nguyên `score`, fallback khi lỗi),
  test streaming_graph rẽ nhánh theo chế độ, test fingerprint đổi theo chế độ.
- Benchmark: test metrics (recall/NDCG/MRR/Jaccard) bằng số liệu tay; test report dựng từ
  JSON cố định; CLI dry-run không cần GPU.

## Boundaries

- **Always**: cùng bộ ứng viên cho mọi biến thể; fail-over về LLM rerank; VRAM trả lại sau
  benchmark; mặc định `llm`.
- **Ask first**: đổi model, bật mặc định, chạy biến thể LLM trong `-c` (tốn phí).
- **Never**: dùng điểm CE làm gate trong ticket này; đưa torch vào dependency của agent;
  ghi báo cáo vào git.

## Success Criteria

1. P95 bước rerank của `cross_encoder` < 500 ms trên 3060 (kỳ vọng < 200 ms trên 4080) với 16
   ứng viên, concurrency 4 không OOM, queue không tăng.
2. recall@8 của `cross_encoder` ≥ `llm_rerank` − 1 điểm trên bộ demo normal + calculation;
   NDCG@16 và MRR không thấp hơn.
3. Độ ổn định: CE cho cùng thứ tự qua các lần chạy; ghi được mức dao động của LLM rerank.
4. Tầng câu trả lời: recall@5 và pass calculation của `cross_encoder` ≥ `llm` − 1 điểm; tỷ lệ
   ticket fallback không tăng; p95 toàn lượt giảm.
5. `nvidia-smi` sau benchmark: không còn tiến trình do benchmark tạo; VRAM về mức trước.

## Quyết định đã chốt (22/10/2026)

1. Benchmark trên Linux 4080 trước; số tải đo lại trên Windows 3060.
2. Phạm vi: service + nối sau cờ tắt + benchmark; không registry/web.
3. Độ chính xác đo hai tầng như trên; RAGAS không chạy.
4. Giữ top-K, không gate theo điểm CE.

## Open Questions

- Passage đưa cho CE: `heading_path + content` (như BM25) hay thêm `summary`? Bắt đầu với
  heading + content, đo độ dài token thật sau tokenize; nếu > 5 % ứng viên bị cắt ở 1024
  thì thử biến thể rerank trên `summary` và so bằng `-a`.
