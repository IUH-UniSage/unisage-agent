# Plan: thêm giám khảo Qwen cho `evals/judge` (28/10/2026)

Ticket: UNISAGE-106. Lý do: chấm RAGAS 444 lượt demo bằng Gemini (`runs/20261028-demo-exact-weboff-none`)
quá chậm - flash-lite preview quá tải, flash-lite 3.5 chỉ 6-29 điểm/phút dù 21 key không bị giới hạn
(chậm do độ trễ mỗi lời gọi). Đã dừng ở 637/1 776 điểm (checkpoint `judge.jsonl` giữ nguyên).

## Mục tiêu

`task eval:judge` chọn được nhà cung cấp giám khảo: `gemini` (như hiện tại) hoặc `openai_compat` (mọi
endpoint tương thích OpenAI Chat Completions; Qwen qua DashScope là trường hợp đầu tiên). Không đổi
cách tính điểm (cùng metric RAGAS, cùng prompt tiếng Việt đã cache, cùng ngưỡng 0,7), chỉ đổi model
gọi.

## Người dùng cung cấp (không dán vào chat; ghi vào `unisage-agent/.env.eval`)

```
JUDGE_OPENAI_BASE_URL=<base url, VD https://dashscope-intl.aliyuncs.com/compatible-mode/v1>
JUDGE_OPENAI_API_KEYS=<key1>,<key2>        # một hoặc nhiều key, phân cách dấu phẩy
```
Cần chốt thêm: tên model chấm (VD `qwen-plus`, `qwen-flash`...), model embedding (VD
`text-embedding-v4`) hoặc giữ embedding Gemini, giới hạn RPM mỗi key, giá USD/1M token vào/ra.

## Thiết kế

1. **`evals/judge/openai_compat.py`** (mới)
   - `OpenAICompatCaller`: gọi `POST {base}/chat/completions` và `/embeddings` bằng `httpx.AsyncClient`
     (đã có trong môi trường judge), dùng lại `KeyPool` của `gemini_pool.py` để xoay vòng key và giới hạn
     RPM; phân loại lỗi: 429 → cool down (đọc `Retry-After`), 401/403 → loại key, 5xx/503 → backoff, quá
     thời gian → backoff, còn lại → `CallFailed`. Không bao giờ in key (dùng `mask`).
   - `OpenAICompatJudgeLLM(InstructorBaseRagasLLM)`: `agenerate(prompt, response_model)` gửi
     `response_format={"type": "json_schema", ...}` (nếu endpoint không hỗ trợ thì `json_object` + schema
     chèn vào system prompt), `temperature=0`, tắt thinking nếu model có; parse bằng
     `response_model.model_validate_json`; thử model dự phòng khi model chính không sẵn sàng.
   - `OpenAICompatEmbeddings(BaseRagasEmbedding)`: `/embeddings`; nếu người dùng không có model
     embedding thì dùng tiếp `GeminiEmbeddings` (answer_relevancy) - ghi rõ trong record.
   - Ghi usage (`prompt_tokens`, `completion_tokens`) vào `Ledger` như `_book` của Gemini.
2. **`ragas_metrics.RagasScorerFactory`**: thêm trường `provider`; `gemini` giữ nguyên, `openai_compat`
   dựng caller/LLM/embedding mới. Metric, prompt đã localize, `RagasScorer` không đổi.
3. **`cli.py`**
   - Cờ mới: `--judge-provider gemini|openai_compat` (mặc định `gemini`), `--base-url-env`,
     `--keys-env` (mặc định `JUDGE_OPENAI_BASE_URL` / `JUDGE_OPENAI_API_KEYS`).
   - `PRICES` thêm model Qwen người dùng chốt; model lạ → cảnh báo và tính theo giá cao nhất đã biết
     (không âm thầm dùng giá Gemini như bây giờ).
   - `read_keys` theo provider; thông báo lỗi nêu đúng tên biến.
4. **Không trộn giám khảo trong một bảng điểm**: mỗi record đã có `model`. Thêm cờ
   `--rejudge-other-models`: bỏ qua record do model khác chấm (chấm lại bằng giám khảo mới). Khuyến nghị
   dùng thư mục riêng: tạo `runs/20261028-demo-exact-weboff-none-judge-qwen` chỉ chứa `results.jsonl`
   và `config.json` (không copy `judge.jsonl`), để giữ bảng điểm Gemini dở dang làm đối chứng.
5. **Hiệu chuẩn trước khi chấm hết** (rẻ, ~5 phút): chấm 40 lượt đầu bằng Qwen, so với 40 lượt đó đã
   chấm bằng Gemini: độ lệch trung bình từng metric, tỷ lệ đồng ý đạt/không đạt ở ngưỡng 0,7. Lệch quá
   lớn ở faithfulness hoặc answer_coverage (đồng ý < 80 %) thì dừng, báo lại trước khi chấm hết.
6. **Báo cáo Excel**: `build_report.py` ghi giám khảo đã dùng (model chấm, model embedding) ở mục 1;
   nếu record trộn model thì nêu tỷ lệ.

## Kiểm thử

- `tests/evals/test_judge_openai_compat.py`: transport giả (`httpx.MockTransport`): JSON schema đúng →
  parse; 429 có `Retry-After` → cool down rồi gọi key khác; 401 → loại key; 503 nhiều lần → model dự
  phòng; quá thời gian → backoff; usage ghi vào ledger; key không xuất hiện trong log/exception.
- `test_judge.py`: chọn provider theo cờ; thiếu biến môi trường → thoát 2 với thông báo đúng tên biến;
  `--rejudge-other-models` chọn đúng record cần chấm lại; ước tính chi phí theo giá Qwen.
- Chạy thật: `--dry-run` (chi phí), rồi hiệu chuẩn 40 lượt (mục 5).

## Thứ tự

1. Code + test provider (khoảng 1-1,5 giờ), không cần key.
2. Người dùng điền `.env.eval`, chốt model/RPM/giá.
3. Dry-run → hiệu chuẩn 40 lượt → báo kết quả.
4. Chấm toàn bộ vào thư mục riêng → dựng 2 sheet Excel (Report / Chi tiết 28-10-2026).
5. Commit (`[UNISAGE-106]`), cập nhật PR #29, `docs/guide/benchmark-rerank.md`/hướng dẫn eval.

## Rủi ro

- Endpoint không hỗ trợ `json_schema` → dùng `json_object`; RAGAS cần output đúng schema, sai nhiều
  thì điểm NaN tăng: theo dõi NaN trong hiệu chuẩn.
- Qwen chấm khác Gemini → điểm không so trực tiếp với báo cáo 13/10; báo cáo phải ghi rõ giám khảo.
- Dữ liệu (câu hỏi, chunk tài liệu, câu trả lời) gửi tới nhà cung cấp Qwen: bộ eval chỉ gồm tài liệu
  công khai đã thu thập và tài liệu private demo; cần người dùng xác nhận được phép gửi ra ngoài.
