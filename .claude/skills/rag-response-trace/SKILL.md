---
name: rag-response-trace
description: Trace vì sao chatbot RAG UniSage (unisage-agent) trả lời như vậy và vì sao lệch đáp án kỳ vọng, cho một câu hỏi trong bộ "Câu Hỏi Demo RAG.xlsx" (sheet "Câu hỏi không trùng", ID dạng d0001). Đầu vào là mã câu hỏi + log server + ảnh chụp câu trả lời. Skill đối chiếu evidence trong PDF với chunk thật sự được retrieve, rerank và đưa vào prompt, rồi chỉ ra tầng gây lỗi. Dùng skill này mỗi khi user gửi mã dXXXX kèm log hoặc ảnh response, hoặc hỏi "sao AI trả lời sai câu này", "trace câu dXXXX", "kiểm tra hệ thống có lấy đúng chunk không", "tại sao bot từ chối/bịa", kể cả khi không nhắc chữ "trace".
---

# Trace câu trả lời RAG theo mã câu hỏi

## Bối cảnh

User test chatbot trên **server deploy** bằng **tài khoản SA (SUPER_ADMIN)**. SA có claim wildcard nên thấy mọi tài liệu. Vì vậy phân quyền gần như không phải nguyên nhân, và câu hỏi chính cần trả lời là:

> Hệ thống có lấy đúng chunk chứa đáp án ra và đưa vào prompt không? Nếu có thì vì sao LLM vẫn trả lời sai?

User gửi 3 thứ:
1. **Mã câu hỏi** (vd `d0042`): một dòng trong sheet "Câu hỏi không trùng" của `/home/huy/Main/questions/Câu Hỏi Demo RAG.xlsx`.
2. **Log server**: stderr của unisage-agent, dán vào chat hoặc gửi đường dẫn file.
3. **Ảnh response của AI**: đọc ảnh để lấy câu trả lời thật và các trích dẫn `[n]`.

Thiếu thứ nào thì vẫn làm phần làm được, và nói rõ thiếu gì ảnh hưởng tới kết luận nào. Ví dụ thiếu log thì chỉ so được đáp án với tài liệu, không biết được chunk nào đã được lấy ra.

Kiến trúc chi tiết (node, retrieval, ingest, những gì log có và không có, prompt, cấu hình) nằm trong `references/pipeline.md`. Đọc file đó khi cần giải thích cơ chế hoặc trích đường dẫn code.

## Quy trình

Chạy script bằng venv của unisage-agent, với `-I` để không nạp nhầm module ở thư mục hiện tại:

```bash
PY=/home/huy/Main/unisage-agent/.venv/bin/python
S=/home/huy/Main/unisage-agent/.claude/skills/rag-response-trace/scripts
```

### Bước 1: Xác định "đúng" trông như thế nào

```bash
$PY -I $S/lookup_case.py <ID>
```

Script in ra:
- dòng Excel: câu hỏi, đáp án kỳ vọng, evidence, Doc IDs, intent, routing mode, cần web search...;
- thông tin manifest của từng tài liệu kỳ vọng: `file_name`, `quality`, số trang, đường dẫn PDF tuyệt đối;
- trang chứa từng số liệu trong đáp án kỳ vọng, kèm cảnh báo khi cùng tài liệu có nhiều giá trị khác nhau (vd "15 phút" và "30 phút");
- **trang và đoạn văn chứa evidence trong PDF**.

Cột Evidence thường chỉ phủ **một** ý của đáp án. Với các ý còn lại (tên phòng ban, điều kiện...), dò thêm bằng `--find "<cụm>"`, có thể lặp lại cờ này. Có thể đọc thẳng PDF bằng pymupdf quanh trang tìm được.

Đoạn văn này là "chunk lẽ ra phải được lấy". Ghi lại tên file, `file_id` và một cụm từ đặc trưng của đoạn đó để dò trong log ở bước 3.

Chú ý ngay:
- `quality` khác `ok`, hoặc PDF không có text-layer: ingest không OCR, nên chunk của tài liệu này có thể rỗng hoặc rác. Đây thường là nguyên nhân gốc.
- Evidence không tìm thấy trong PDF: có thể evidence do LLM sinh bộ câu hỏi diễn đạt lại, hoặc nằm trong bảng/ảnh. Đọc PDF quanh trang nghi ngờ trước khi kết luận.
- Case không cần tài liệu (`social`, `off_topic`, `unanswerable`, `web_search`): kỳ vọng là đi đúng nhánh hoặc từ chối đúng, không phải lấy đúng chunk. Xem bước 4.

### Bước 2: Đọc câu trả lời thực tế từ ảnh

Chép lại nguyên văn các ý chính và các `[n]` được trích. So với "Đáp án kỳ vọng" theo **từng ý**, vì câu hỏi demo thường có 2 ý. Phân loại từng ý: đúng / sai số liệu / sai đối tượng / thiếu / từ chối / bịa. Nếu ảnh có ô nguồn trích dẫn, ghi lại tên tài liệu và trang hiện trong đó.

### Bước 3: Tóm tắt log của lượt chat

Lưu log user dán vào file trong scratchpad (hoặc dùng đường dẫn user đưa), rồi chạy:

```bash
$PY -I $S/parse_log.py <log> --expect <file_id> --expect "<file_name>"
# log có nhiều lượt: thêm --message-id <id>; script liệt kê các id có trong log
```

Script in ra:
- chuỗi node và intent suy ra;
- `user_id`;
- text HyDE / sub-query dùng để embed;
- dòng LLM rerank (kept/dropped kèm score), tài liệu kỳ vọng được đánh dấu ✅;
- các chunk `[n]` trong context gửi LLM;
- kết luận tài liệu kỳ vọng có nằm trong context hay không.

Kiểm tra đầu tiên: `user_id` không phải `guest` và academic_metadata cho thấy vai trò SUPER_ADMIN. Nếu là guest thì lượt chat không chạy dưới SA, và nguyên nhân có thể là phân quyền. Báo user điều này trước.

Cột Persona / Quyền người hỏi / Kỳ vọng thấy tài liệu mô tả người hỏi giả định. Khi test bằng SA thì bỏ qua các cột này, vì SA luôn thấy tài liệu. Riêng case `access_private` có "Kỳ vọng thấy tài liệu = Không": test bằng SA không kiểm được phân quyền, chỉ kiểm được retrieval. Ghi một dòng nhắc điều này trong báo cáo, không coi là lỗi.

Có source trong context hoặc rerank không khớp tài liệu kỳ vọng thì tra xem đó là tài liệu nào:

```bash
$PY -I $S/lookup_case.py --doc "<object_key hoặc một phần tên file>"
```

Không khớp manifest nghĩa là file được upload ngoài bộ dataset. Ghi rõ điều đó trong báo cáo.

### Bước 4: Khoanh vùng tầng lỗi

Đi theo pipeline từ trên xuống. **Nguyên nhân gốc là tầng sớm nhất mà sự lệch của nó làm hỏng đầu ra của tầng kế tiếp.** Một tầng có thể lệch mà vẫn vô hại. Ví dụ HyDE bịa số liệu nhưng retrieval vẫn xếp chunk đúng cao nhất: khi đó HyDE chỉ là yếu tố góp phần, và gốc nằm ở tầng sau, nơi chunk đúng bị mất. Ghi tầng gốc trong phần Kết luận, các tầng góp phần ghi trong phần Đường đi.

| # | Tầng | Câu hỏi kiểm tra | Bằng chứng trong log |
|---|---|---|---|
| A | Định tuyến | Intent suy ra có khớp cột "Intent" / "Case" không? Câu học vụ có bị đẩy vào social, off-topic hay greeting không? | chuỗi `node=` |
| B | Biến đổi truy vấn | HyDE có giữ đúng đối tượng (giảng viên hay sinh viên, năm, khoa, hệ đào tạo) không? Câu "tài liệu giả định" có bịa số liệu hoặc đơn vị khác với tài liệu thật không? Câu nhiều ý có bị tách thiếu ý không? | khối HyDE / SQ |
| C | Retrieval + ngưỡng | Tài liệu kỳ vọng có nằm trong danh sách ứng viên của rerank (kept hoặc dropped) không? | dòng `LLM rerank SQ…` |
| D | LLM rerank | Tài liệu kỳ vọng có bị **dropped** không? Lý do giữ các chunk khác có hợp lý không? Đáp án có nằm sau 800 ký tự đầu của chunk không? | kept/dropped |
| E | Context | Chunk chứa **đúng đoạn evidence** (không chỉ đúng file) có nằm trong `<academic_context>` không? | khối context |
| F | Sinh câu trả lời | Context đã có evidence mà câu trả lời vẫn sai, thiếu hoặc từ chối? Model có trích nhầm `[n]` không, có lấy chunk của tài liệu khác hay năm khác không, có áp `current_date` để loại văn bản không? | so ảnh với context |
| G | Fallback / web | Có `11_TicketFallbackNode`, hoặc web search lấp chỗ trống bằng nguồn ngoài không? | node 09b / 11 |

Lưu ý khi so khớp:
- Đúng file nhưng sai chunk vẫn là lỗi retrieval. So **nội dung** chunk với đoạn evidence ở bước 1, không chỉ so tên file.
- Score trong dòng rerank là cosine tốt nhất qua 3 vector. Nếu chunk đúng có score thấp hơn chunk sai thì đó là bằng chứng mạnh cho lỗi ở tầng B hoặc C.
- Với case `unanswerable` / `off_topic` / `social`, "đúng" nghĩa là từ chối hoặc đi đúng nhánh. Bot trả lời được bằng chunk khác là lỗi ngưỡng/rerank quá lỏng (C/D) hoặc lỗi prompt (F).
- Case `calculation`: kiểm tra context có đủ **mọi** con số cần cho phép tính không, rồi tự tính lại để biết model sai số liệu hay sai phép tính.

### Bước 5: Khi log không đủ

Tài liệu kỳ vọng không xuất hiện ở bất kỳ đâu trong log nghĩa là log không phân biệt được ba khả năng: chưa được ingest / ingest ra chunk rỗng, nằm ngoài top-k, hoặc dưới ngưỡng. Không được đoán. Kết luận ở mức "chưa xác định giữa X/Y/Z" và đề nghị user kiểm tra trên server, theo thứ tự rẻ tới đắt:

1. Trang quản lý tài liệu trên web: tài liệu đã ingest chưa, trạng thái gì, có bao nhiêu chunk.
2. `GET /api/v1/ai/documents/{document_id}/chunks/indexed`: tìm chunk chứa cụm từ evidence. Không có thì là lỗi ingest hoặc chunking (evidence bị cắt đôi, chunk rỗng do PDF scan).
3. Chunk có tồn tại thì là lỗi xếp hạng: so nội dung chunk với text HyDE, và hỏi ngưỡng `CHAT_RERANK_SCORE_THRESHOLD` đang đặt trên server.

Tương tự, khi log thiếu khối HyDE và khối context (APP_DEBUG tắt trên server), chỉ còn dòng rerank và chuỗi node. Nói rõ điều này và đề nghị bật `APP_DEBUG` để trace lại.

### Bước 6: Viết báo cáo

Trả lời trong chat **và** lưu file `/home/huy/Main/questions/traces/<ID>.md`. Nếu file đã có thì thêm mục mới ở cuối, có ghi ngày giờ: mỗi lần test lại là một lần chạy, giữ lịch sử để so sánh trước và sau khi sửa.

Dùng đúng khung sau và viết bằng tiếng Việt:

```markdown
# <ID> — <tóm tắt câu hỏi ≤ 12 từ>
_Trace lúc <YYYY-MM-DD HH:mm> · case <Case> · intent kỳ vọng <Intent>_

## Kết luận
<1–3 câu: tầng gây lỗi (A–G) + nguyên nhân gốc + mức chắc chắn (chắc chắn / nhiều khả năng / chưa xác định)>

## So sánh câu trả lời
| Ý | Kỳ vọng | Thực tế | Đánh giá |
|---|---|---|---|

## Tài liệu kỳ vọng
- <file_name> (`<file_id>`, quality=<…>) — evidence ở trang <p>: "<trích ngắn>"

## Đường đi trong pipeline
- Định tuyến: <chuỗi node rút gọn> → <đúng/sai>
- Truy vấn embed: <tóm tắt HyDE, chỉ ra chỗ lệch nếu có>
- Ứng viên rerank: <tài liệu kỳ vọng: kept/dropped/không có, score; chunk thắng thế và score>
- Context gửi LLM: <[n] nào chứa evidence, hoặc không có>
- Sinh câu trả lời: <model dùng [n] nào, sai ở đâu>

## Bằng chứng
<trích dòng log / đoạn chunk then chốt, ngắn gọn>

## Đề xuất
- <hành động cụ thể: re-ingest có OCR, đổi chunking, chỉnh prompt HyDE, chỉnh ngưỡng, sửa prompt sinh, sửa đáp án kỳ vọng nếu đáp án sai… kèm file/tham số liên quan>

## Cần kiểm tra thêm (nếu có)
- <việc user cần làm trên server để chốt kết luận>
```

Nguyên tắc viết:
- Mỗi kết luận phải trỏ về một bằng chứng cụ thể: dòng log, đoạn chunk, đoạn PDF hoặc ảnh.
- Phân biệt rõ "thấy trong log" và "suy luận". Log có điểm mù (xem `references/pipeline.md` mục 4), nên đừng lấp điểm mù bằng phỏng đoán.
- Có khi lỗi nằm ở **đáp án kỳ vọng** (câu hỏi do LLM sinh, `reviewed=false`): evidence không khớp PDF, hoặc tài liệu có văn bản mới hơn. Khi đó hãy nói thẳng, kèm trích PDF làm chứng.
- Không chép nguyên log dài vào báo cáo. Chỉ trích những dòng quyết định kết luận.
