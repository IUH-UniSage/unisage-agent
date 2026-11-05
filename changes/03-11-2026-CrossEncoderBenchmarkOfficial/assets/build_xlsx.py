"""Bảng tính cho benchmark CE trên bộ official (03/11/2026).

1. report.xlsx mới cạnh report.md (cấu trúc như workbook 26/10: Tổng quan + mục lục, các mục đánh số,
   Theo câu, Hình).
2. Thêm 4 sheet " 03-11" vào changes/22-10-2026-CrossEncoderBenchmark/report.xlsx (sao lưu trước thành
   report.truoc-03-11-2026.xlsx, không đụng sheet cũ).

Chạy: uv run --isolated --no-project --with openpyxl python assets/build_xlsx.py <thư mục kết quả>
"""
import json
import shutil
import sys
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

RES = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
NEW_DIR = HERE.parent
OLD_XLSX = NEW_DIR.parent / "22-10-2026-CrossEncoderBenchmark" / "report.xlsx"
S = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
NF = json.loads((RES / "nofloor" / "summary_nofloor.json").read_text(encoding="utf-8"))
G = S["groups"]
V = ["none", "ce16", "ce40", "ce64"]

NAVY, SUB, ZEBRA, BORDER = "1F3864", "DCE6F1", "F7F7F7", "BFBFBF"
FONT = "Times New Roman"
thin = Side(style="thin", color=BORDER)
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)


def vn(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


class Sheet:
    """Viết tuần tự theo hàng, cùng kiểu với workbook 26/10."""

    def __init__(self, ws, widths):
        self.ws, self.row = ws, 1
        for col, w in widths.items():
            ws.column_dimensions[col].width = w
        self.ncols = len(widths)
        ws.freeze_panes = "A2"

    def title(self, text, size=14):
        c = self.ws.cell(self.row, 1, text)
        c.font = Font(name=FONT, bold=True, size=size, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=NAVY)
        c.alignment = Alignment(vertical="center", wrap_text=True)
        self.ws.merge_cells(start_row=self.row, start_column=1, end_row=self.row, end_column=self.ncols)
        self.ws.row_dimensions[self.row].height = 28
        self.row += 2

    def sub(self, text):
        c = self.ws.cell(self.row, 1, text)
        c.font = Font(name=FONT, bold=True, size=12, color=NAVY)
        c.fill = PatternFill("solid", fgColor=SUB)
        c.alignment = Alignment(vertical="center", wrap_text=True)
        self.ws.merge_cells(start_row=self.row, start_column=1, end_row=self.row, end_column=self.ncols)
        self.row += 2

    def para(self, text, bold=False, height=None):
        c = self.ws.cell(self.row, 1, text)
        c.font = Font(name=FONT, size=11, bold=bold)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        self.ws.merge_cells(start_row=self.row, start_column=1, end_row=self.row, end_column=self.ncols)
        if height:
            self.ws.row_dimensions[self.row].height = height
        else:
            width = sum(self.ws.column_dimensions[chr(65 + i)].width for i in range(self.ncols))
            self.ws.row_dimensions[self.row].height = max(18, 15 * (len(text) // max(1, int(width * 1.1)) + 1))
        self.row += 1

    def table(self, header, rows, bold_cols=None, bold_best=None, number_formats=None, height=20):
        """bold_best: {row_index: 'max'|'min'} in đậm giá trị tốt nhất của hàng (cột 2 trở đi, theo
        số đầu tiên của chuỗi), như workbook 26/10."""
        for j, h in enumerate(header, 1):
            c = self.ws.cell(self.row, j, h)
            c.font = Font(name=FONT, bold=True, size=11, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor=NAVY)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = BOX
            c.number_format = "@"
        self.ws.row_dimensions[self.row].height = height
        self.row += 1
        best = {}
        for ri, how in (bold_best or {}).items():
            if ri >= len(rows):
                continue
            vals = [_num(v) for v in rows[ri][1:]]
            vals = [v for v in vals if v is not None]
            if vals:
                best[ri] = max(vals) if how == "max" else min(vals)
        for i, r in enumerate(rows):
            for j, v in enumerate(r, 1):
                c = self.ws.cell(self.row, j, v)
                is_text = isinstance(v, str)
                c.font = Font(name=FONT, size=11)
                c.alignment = Alignment(
                    horizontal="left" if (j == 1 or (is_text and not _numlike(v))) else "right",
                    vertical="top",
                    wrap_text=True,
                )
                c.border = BOX
                if is_text:
                    c.number_format = "@"
                elif number_formats and j - 1 in number_formats:
                    c.number_format = number_formats[j - 1]
                if i % 2 == 1:
                    c.fill = PatternFill("solid", fgColor=ZEBRA)
                if i in best and j > 1 and _num(v) is not None and abs(_num(v) - best[i]) < 1e-9:
                    c.font = Font(name=FONT, size=11, bold=True)
            self.ws.row_dimensions[self.row].height = height
            self.row += 1
        self.row += 1

    def image(self, path, width_px=None):
        img = Image(str(path))
        if width_px:
            ratio = width_px / img.width
            img.width, img.height = width_px, int(img.height * ratio)
        self.ws.add_image(img, f"A{self.row}")
        self.row += int(img.height / 20) + 2


def _num(v):
    try:
        return float(str(v).replace(",", ".").split()[0])
    except (ValueError, IndexError):
        return None


def _numlike(v):
    s = str(v).replace(",", "").replace(" ", "").replace("/", "").replace("ms", "").replace("%", "")
    s = s.replace("+", "").replace("−", "").replace("-", "").replace("—", "")
    return s.replace(".", "").isdigit() or v in ("—",)


# ---------------------------------------------------------------------------------------------
# Nội dung dùng chung

VARIANT_ROWS = [
    ["none", "sàn cosine 0,45, lấy 8 chunk đầu theo thứ tự RRF trong 40 chunk đầu", "CHAT_RERANK_MODE=none"],
    ["ce16", "sàn trên 16 chunk đầu RRF, CE chấm, giữ top-8 theo điểm CE", "cross_encoder, CHAT_RETRIEVAL_MAX_CHUNKS=16"],
    ["ce40", "như trên với 40 chunk đầu", ".env.eval hiện tại (cross_encoder, MAX_CHUNKS=40)"],
    ["ce64", "như trên với cả pool 64", "mở rộng thêm"],
]
GLOSSARY = [
    ["truy vấn", "một câu hỏi ở một biến thể: question (đúng chính tả, có dấu) hoặc question_user (kiểu người dùng, nhiều câu không dấu)"],
    ["pool N", "N chunk đầu theo thứ tự RRF của hybrid search (dense + BM25) với filter quyền của persona"],
    ["sàn", "CHAT_RETRIEVAL_DENSE_FLOOR 0,45: bỏ chunk cosine dense < 0,45 trước rerank; chunk chỉ BM25 tìm thấy được giữ theo cờ"],
    ["recall@k", "phần tài liệu kỳ vọng có trong k chunk giữ đầu (cấp tài liệu)"],
    ["hit@5", "1 nếu ít nhất một tài liệu kỳ vọng nằm trong 5 chunk đầu"],
    ["MRR", "trung bình 1/hạng của chunk đúng đầu tiên trong 8 chunk giữ"],
    ["NDCG@8", "NDCG nhị phân cấp tài liệu trên 8 chunk giữ (mỗi tài liệu đúng tính một lần)"],
    ["trần pool N", "recall@N của chính pool; 'sau sàn' = chỉ chunk qua sàn (cái CE thấy)"],
    ["cứu được / làm rơi", "số truy vấn mà recall@k của CE cao hơn / thấp hơn none"],
    ["p50 / p95", "50 % / 95 % request nhanh hơn giá trị này (client đo, concurrency 1)"],
]
CONCLUSIONS = [
    "1. Trên bộ official, CE có việc để làm: none recall@8 0,890, trần pool 40 sau sàn 0,942; ce40 0,917 (+2,7 điểm), recall@5 +3,7, recall@1 +8,2, MRR +0,063; cứu 30, làm rơi 9.",
    "2. Lợi ích gần như toàn bộ ở câu kiểu người dùng: question_user recall@8 0,803 → 0,861 (+5,8), recall@5 +7,9. Câu đúng chính tả: RRF đã 0,977 (trần 0,997), CE không thêm ở recall@8.",
    "3. Pool 16 → 40 đáng (+1,0 điểm recall@8, +120 ms); 40 → 64 không (trần +0,005, recall@8 −0,003, +115 ms).",
    "4. CE làm rơi ở các nhóm tài liệu gần trùng (bản đăng ký PGS của từng người, đề cương ôn thi gần tên, portfolio từng sinh viên).",
    "5. 61 truy vấn ce40 chưa đạt: 25 tài liệu không có trong pool 64, 12 bị sàn 0,45 cắt, 2 ở hạng 41–64, chỉ 22 do CE xếp sai.",
    "6. Bỏ sàn khi bật CE: ce40 recall@8 0,917 → 0,933 (+11 truy vấn, 0 kém đi).",
    "7. Câu không dấu (77 truy vấn) là điểm yếu lớn nhất của retrieval: trần pool 64 0,714, recall@8 0,44–0,53 mọi phương án.",
    "8. q0054 không trượt ở retrieval-only (hạng 1); d0315 trượt thật: tài liệu tiếng Anh, câu tiếng Việt, không vào pool 64.",
    "9. 0 rò quyền trên 204 truy vấn không được thấy.",
]
RECS = [
    "• Bước 2 dùng ce40: CHAT_RERANK_MODE=cross_encoder, CHAT_RETRIEVAL_MAX_CHUNKS=40, CHAT_CROSS_ENCODER_KEEP=8 (đúng .env.eval hiện tại).",
    "• Thêm nhánh ce40 + sàn 0 (CHAT_RETRIEVAL_DENSE_FLOOR=0) vào bước 2; đo cả từ chối và bịa trên 20 câu unanswerable trước khi đổi.",
    "• Giữ none làm mốc ở bước 2.",
    "• Ticket riêng: gập dấu cho BM25 hoặc khôi phục dấu; truy vấn song ngữ cho tài liệu tiếng Anh; rà nhãn nhóm tài liệu gần trùng.",
]


def metric_rows(group, metrics=("recall@1", "recall@3", "recall@5", "recall@8", "hit@5", "mrr", "ndcg@8")):
    names = {"mrr": "MRR", "ndcg@8": "NDCG@8"}
    return [[names.get(m, m)] + [vn(G[group][v][m]) for v in V] for m in metrics]


def best_rows(n):
    return {i: "max" for i in range(n)}


def lat_rows():
    L = S["latency"]
    return [
        [s, vn(L[s]["cand_mean"], 1), f"{L[s]['p50']:.0f} ms", f"{L[s]['p95']:.0f} ms", f"{L[s]['p99']:.0f} ms",
         f"{L[s]['max']:.0f} ms", f"{L[s]['service_p50']} / {L[s]['service_p95']} ms", f"{L[s]['truncated_share'] * 100:.0f} %"]
        for s in ("16", "40", "64")
    ]


def per_query_rows():
    rows = []
    for r in S["all_rows"]:
        rows.append([
            r["key"], r["category"], r["persona"], r["text"], "; ".join(r["expected_titles"]),
            r["none_r8"], r["ce16_r8"], r["ce40_r8"], r["ce64_r8"],
            r["first_rank_rrf"] if r["first_rank_rrf"] is not None else "—",
            r["none_rank"] if r["none_rank"] is not None else "—",
            r["ce40_rank"] if r["ce40_rank"] is not None else "—",
            r["top1_cos"],
        ])
    return rows


PQ_HEADER = ["Truy vấn", "Category", "Persona", "Câu hỏi", "Tài liệu kỳ vọng", "recall@8 none", "recall@8 ce16",
             "recall@8 ce40", "recall@8 ce64", "Hạng RRF (pool 64)", "Hạng none (sau sàn)", "Hạng ce40", "Cosine dense top-1"]
PQ_WIDTHS = {"A": 16, "B": 12, "C": 24, "D": 60, "E": 48, "F": 11, "G": 11, "H": 11, "I": 11, "J": 11, "K": 11, "L": 11, "M": 11}
PQ_FORMATS = {5: "0.00", 6: "0.00", 7: "0.00", 8: "0.00", 12: "0.000"}
FIGS = [
    ("fig1-pipeline.png", "Hình 1. Luồng benchmark"),
    ("fig2-recall-at-k.png", "Hình 2. recall@k theo phương án"),
    ("fig3-category.png", "Hình 3. recall@5 / @8 theo category"),
    ("fig4-exact-vs-user.png", "Hình 4. question so với question_user"),
    ("fig5-ceiling.png", "Hình 5. Trần so với đạt được theo cỡ pool"),
    ("fig6-rescued-dropped.png", "Hình 6. Cứu được / làm rơi so với none"),
    ("fig7-ce-latency.png", "Hình 7. Độ trễ CE theo cỡ pool"),
    ("fig8-rank-distribution.png", "Hình 8. Hạng của tài liệu đúng đầu tiên"),
]


def write_overview(sh, title, toc, date_line):
    sh.title(title, size=16)
    sh.ws.cell(sh.row - 1, 1, date_line).font = Font(name=FONT, bold=True, size=11)
    sh.para("Ngày đo 03/11/2026 · RTX 4080 SUPER 16 GB · Qdrant unisage-official-chunks (env eval, 1 074 tài liệu, ~24 900 chunk) · 352 câu × 2 biến thể = 704 truy vấn · chỉ retrieval, không LLM · chi tiết: changes/03-11-2026-CrossEncoderBenchmarkOfficial/report.md, report-full.md")
    sh.para("Bốn phương án, cùng pool 64 ứng viên mỗi truy vấn, giữ K = 8 chunk:")
    sh.row += 1
    sh.table(["Phương án", "Làm gì", "Ứng với cấu hình"], VARIANT_ROWS)
    if toc:
        sh.sub("Mục lục (bấm để chuyển sheet)")
        for name, label in toc:
            c = sh.ws.cell(sh.row, 1, f"{name}  —  {label}")
            c.hyperlink = f"#'{name}'!A1"
            c.font = Font(name=FONT, size=11, color="0563C1", underline="single")
            sh.row += 1
        sh.row += 1
    sh.sub("Bảng tổng (704 truy vấn)")
    rows = metric_rows("all")
    rows.append(["trần pool (sau sàn)", "0,942 (pool 40)", vn(G["all"]["ceiling_floored"]["16"]), vn(G["all"]["ceiling_floored"]["40"]), vn(G["all"]["ceiling_floored"]["64"])])
    rows.append(["cứu / rơi @8 so với none", "—"] + [f"{G['all'][v]['rescued@8']} / {G['all'][v]['dropped@8']}" for v in V[1:]])
    rows.append(["độ trễ CE p50 / p95", "0"] + [f"{S['latency'][s]['p50']:.0f} / {S['latency'][s]['p95']:.0f} ms" for s in ("16", "40", "64")])
    rows.append(["ce40 khi bỏ sàn: recall@8", "—", "—", vn(NF["all"]["ce40"]["recall@8"]), "—"])
    sh.table(["Chỉ số", "none", "ce16", "ce40", "ce64"], rows, bold_best={**best_rows(7), 9: "min"})
    sh.sub("Kết luận")
    for t in CONCLUSIONS:
        sh.para(t)
    sh.row += 1
    sh.sub("Khuyến nghị cho bước 2")
    for t in RECS:
        sh.para(t)


def write_accuracy(sh, heading):
    sh.title(heading)
    sh.sub("Tổng, 704 truy vấn")
    sh.table(["Chỉ số", "none", "ce16", "ce40", "ce64"], metric_rows("all"), bold_best=best_rows(7))
    sh.sub("Theo category: recall@5 / recall@8 / MRR")
    rows = []
    for cat, lab in (("normal", "normal"), ("calculation", "calculation"), ("access", "access (thấy)")):
        for m in ("recall@5", "recall@8", "mrr"):
            rows.append([f"{lab} (n = {G[cat]['n']}) · {m if m != 'mrr' else 'MRR'}"] + [vn(G[cat][v][m]) for v in V])
    sh.table(["Category · chỉ số", "none", "ce16", "ce40", "ce64"], rows, bold_best=best_rows(len(rows)))
    sh.sub("Theo biến thể câu hỏi: recall@5 / recall@8")
    rows = []
    for g, lab in (("exact", "question"), ("user", "question_user")):
        for cat in ("normal", "calculation", "access"):
            key = f"{cat}/{g}"
            rows.append([f"{lab} · {cat} (n = {G[key]['n']})"] + [f"{vn(G[key][v]['recall@5'])} / {vn(G[key][v]['recall@8'])}" for v in V])
        rows.append([f"{lab} · tất cả (n = {G[g]['n']})"] + [f"{vn(G[g][v]['recall@5'])} / {vn(G[g][v]['recall@8'])}" for v in V])
    rows.append(["question_user có dấu (n = 275)", "0,865 / 0,904", "0,918 / 0,936", "0,938 / 0,955", "0,938 / 0,949"])
    rows.append(["question_user không dấu (n = 77)", "0,409 / 0,442", "0,461 / 0,481", "0,513 / 0,526", "0,519 / 0,526"])
    sh.table(["Nhóm", "none", "ce16", "ce40", "ce64"], rows)
    sh.sub("Trần (recall trong pool)")
    rows = []
    for g, lab in (("all", "cả hai biến thể"), ("exact", "question"), ("user", "question_user")):
        rows.append([f"trước sàn, {lab}"] + [vn(G[g]["ceiling"][n]) for n in ("8", "16", "40", "64")])
        rows.append([f"sau sàn, {lab}"] + [vn(G[g]["ceiling_floored"][n]) for n in ("8", "16", "40", "64")])
    sh.table(["Trần", "pool 8", "pool 16", "pool 40", "pool 64"], rows)
    sh.sub("Cứu được / làm rơi so với none (số truy vấn, trên 704)")
    rows = [[f"recall@{k}"] + [f"{G['all'][v][f'rescued@{k}']} / {G['all'][v][f'dropped@{k}']}" for v in V[1:]] for k in (1, 5, 8)]
    sh.table(["Chỉ số", "ce16", "ce40", "ce64"], rows)
    sh.sub("Đo phụ: ce40 khi bỏ sàn cosine (CHAT_RETRIEVAL_DENSE_FLOOR = 0)")
    rows = []
    for g, lab in (("all", "tất cả"), ("exact", "question"), ("user", "question_user"), ("normal", "normal"), ("calculation", "calculation"), ("access", "access")):
        rows.append([f"{lab} (n = {NF[g]['n']})", vn(G[g]["ce40"]["recall@8"]), vn(NF[g]["ce40"]["recall@8"]), vn(NF[g]["none"]["recall@8"]),
                     f"{NF[g]['vs_ce40_floor']['better']} / {NF[g]['vs_ce40_floor']['worse']}"])
    sh.table(["Nhóm", "ce40 recall@8 (sàn 0,45)", "ce40 recall@8 (không sàn)", "none recall@8 (không sàn)", "tốt hơn / kém hơn"], rows)


def write_latency(sh, heading):
    sh.title(heading)
    sh.table(["Cỡ pool", "Passage TB sau sàn", "p50", "p95", "p99", "Max", "Service p50 / p95", "Bị cắt ở 512"], lat_rows())
    sh.para("Request đầu tiên 1 022 ms (khởi động); service dùng chung với hệ thống live; ~5,3 ms mỗi passage, gần tuyến tính.")


def write_per_query(sh, heading):
    sh.title(heading)
    sh.table(PQ_HEADER, per_query_rows(), number_formats=PQ_FORMATS, height=30)
    sh.ws.auto_filter.ref = f"A3:M{sh.row - 2}"
    sh.ws.freeze_panes = "B4"


def write_misses(sh, heading, rows_src):
    sh.title(heading)
    rows = [[r["key"], r["category"], r["text"], "; ".join(r["expected_titles"]), r["none_r8"], r["ce40_r8"],
             r["first_rank_rrf"] if r["first_rank_rrf"] is not None else "—",
             r["ce40_rank"] if r["ce40_rank"] is not None else "—", r["reason"]] for r in rows_src]
    sh.table(["Truy vấn", "Category", "Câu hỏi", "Tài liệu kỳ vọng", "recall@8 none", "recall@8 ce40", "Hạng RRF", "Hạng ce40", "Nhóm lỗi"],
             rows, number_formats={4: "0.00", 5: "0.00"}, height=30)
    sh.para("Nhóm lỗi: A = tài liệu không có trong pool 64; B = chỉ ở hạng 41–64; C = trong 40 chunk đầu nhưng mọi chunk của nó dưới sàn 0,45; D = CE thấy nhưng xếp ngoài top-8 (hoặc chỉ 1/2 tài liệu).")


def write_figs(sh, heading):
    sh.title(heading)
    for name, cap in FIGS:
        sh.para(cap, bold=True)
        sh.image(HERE / name, width_px=1100)


# phân nhóm lỗi ce40 (A/B/C/D) từ file thô
pools = {}
for line in open(RES / "pools.jsonl", encoding="utf-8"):
    r = json.loads(line)
    pools[r["key"]] = r
ce = {}
for line in open(RES / "ce.jsonl", encoding="utf-8"):
    r = json.loads(line)
    ce[r["key"]] = r


def reason(key):
    p = pools[key]
    exp = set(p["expected_document_ids"])
    if not exp & {e["document_id"] for e in p["pool"]}:
        return "A"
    in40 = [e for e in p["pool"][:40] if e["document_id"] in exp]
    if not in40:
        return "B"
    ids40 = set(ce[key]["sizes"]["40"]["ids"])
    if not any(e["chunk_id"] in ids40 for e in in40):
        return "C"
    return "D"


misses = sorted(({**r, "reason": reason(r["key"])} for r in S["ce40_misses"]), key=lambda r: (r["reason"], r["key"]))
changed = [{**r, "reason": "cứu được" if r["ce40_r8"] > r["none_r8"] else "làm rơi"} for r in S["changed_ce40"]]

# ---------------------------------------------------------------------------------------------
# 1. workbook mới
wb = Workbook()
wb.remove(wb.active)
toc = [
    ("0 Thuật ngữ", "0. Thuật ngữ"), ("1 Kết luận", "1. Kết luận"), ("2 Cách đo", "2. Cách đo"),
    ("3 Accuracy", "3. Accuracy chỉ retrieval"), ("4 Độ trễ CE", "4. Độ trễ CE theo cỡ pool"),
    ("5 Lỗi ce40", "5. 61 truy vấn ce40 chưa đạt recall@8 = 1"), ("6 Cứu-rơi ce40", "6. 39 truy vấn ce40 khác none"),
    ("Theo câu", "Từng truy vấn (704)"), ("Hình", "Hình 1–8"), ("9 Phụ lục", "9. Phụ lục: lệnh, file"),
]
sh = Sheet(wb.create_sheet("Tổng quan"), {"A": 34, "B": 57, "C": 29, "D": 30, "E": 30})
write_overview(sh, "Benchmark cross-encoder trên bộ official: chỉ retrieval (UNISAGE-106, bước 1)", toc, "Ngày 03/11/2026")
sh = Sheet(wb.create_sheet("0 Thuật ngữ"), {"A": 45, "B": 80})
sh.title("0. Thuật ngữ")
sh.table(["Từ khoá", "Nghĩa"], GLOSSARY)
sh = Sheet(wb.create_sheet("1 Kết luận"), {"A": 42, "B": 30, "C": 30, "D": 30, "E": 30})
sh.title("1. Kết luận")
sh.sub("1.1. Bảng tổng")
sh.table(["Chỉ số", "none", "ce16", "ce40", "ce64"], metric_rows("all"), bold_best=best_rows(7))
sh.sub("1.2. Kết luận")
for t in CONCLUSIONS:
    sh.para(t)
sh.row += 1
sh.sub("1.3. Khuyến nghị")
for t in RECS:
    sh.para(t)
sh.row += 1
sh.sub("1.4. Giới hạn")
for t in [
    "• Không qua classifier, HyDE, tách câu hỏi: câu hỏi đi thẳng vào search; trong graph, HyDE và câu hỏi độc lập (thường có dấu) có thể thu hẹp khoảng cách ở câu kiểu người dùng.",
    "• Mọi phương án đọc tiền tố của pool 64; pool lấy riêng ở độ sâu 40 cho cùng recall@8 cấp tài liệu (0,901) với 8 chunk đầu chưa sàn, thứ tự 40 chunk trùng hoàn toàn 74/704.",
    "• Recall cấp tài liệu, không kiểm tra evidence; nhãn do model sinh, reviewed=false.",
    "• Service CE dùng chung với hệ thống live; độ trễ đo tuần tự.",
]:
    sh.para(t)
sh = Sheet(wb.create_sheet("2 Cách đo"), {"A": 125})
sh.title("2. Cách đo")
for t in [
    "• Bộ câu: dataset/official/questions.jsonl, dòng có expected_doc_ids và expect_visible: normal 207, calculation 59, access 86 = 352 câu × 2 biến thể = 704 truy vấn; + 102 câu access không được thấy × 2 = 204 truy vấn kiểm tra rò quyền.",
    "• expected_doc_ids → document_id của Qdrant bằng evals.run.runner.document_ids; quyền: access_of + security_of như evals.run.",
    "• Retrieval: embed câu hỏi (text-embedding-3-small), RetrievalService._search (3 dense + BM25, RRF k = 1, filter quyền) ở độ sâu 64 và 40; dưới tầng retrieval cache. Ghi pools.jsonl, chunks.jsonl.",
    "• CE: service 127.0.0.1:8403 đang chạy (bge-reranker-v2-m3, max_length 512); sàn 0,45 trên N chunk đầu, một request /rerank mỗi (truy vấn, cỡ pool), tuần tự; 0 lỗi.",
    "• K = 8 (CHAT_CROSS_ENCODER_KEEP = CHAT_CONTEXT_MAX_CHUNKS = 8). Không gọi LLM nào.",
]:
    sh.para(t)
sh = Sheet(wb.create_sheet("3 Accuracy"), {"A": 40, "B": 22, "C": 22, "D": 22, "E": 22})
write_accuracy(sh, "3. Accuracy chỉ retrieval (704 truy vấn)")
sh = Sheet(wb.create_sheet("4 Độ trễ CE"), {"A": 14, "B": 18, "C": 12, "D": 12, "E": 12, "F": 18, "G": 18, "H": 14})
write_latency(sh, "4. Độ trễ CE theo cỡ pool (concurrency 1, RTX 4080 SUPER)")
sh = Sheet(wb.create_sheet("5 Lỗi ce40"), {"A": 16, "B": 12, "C": 60, "D": 48, "E": 11, "F": 11, "G": 10, "H": 10, "I": 10})
write_misses(sh, "5. 61 truy vấn ce40 chưa đạt recall@8 = 1", misses)
sh = Sheet(wb.create_sheet("6 Cứu-rơi ce40"), {"A": 16, "B": 12, "C": 60, "D": 48, "E": 11, "F": 11, "G": 10, "H": 10, "I": 12})
write_misses(sh, "6. 39 truy vấn ce40 khác none (30 cứu được, 9 làm rơi)", changed)
sh = Sheet(wb.create_sheet("Theo câu"), PQ_WIDTHS)
write_per_query(sh, "Từng truy vấn (704): recall@8 theo phương án, hạng tài liệu đúng")
sh = Sheet(wb.create_sheet("Hình"), {"A": 160})
write_figs(sh, "Hình 1–8")
sh = Sheet(wb.create_sheet("9 Phụ lục"), {"A": 14, "B": 90, "C": 40})
sh.title("9. Phụ lục: lệnh, file")
sh.table(["Bước", "Lệnh", "Ra"], [
    ["Lấy pool", "task benchmark:pool env=eval -- collect --out benchmark_results/03-11-2026-ce-official", "pools.jsonl, chunks.jsonl"],
    ["Chấm CE", "task benchmark:pool env=eval -- ce --out benchmark_results/03-11-2026-ce-official", "ce.jsonl"],
    ["Chỉ số", "task benchmark:pool env=eval -- score --out ...; python assets/analyze.py <out>", "per_query.jsonl, summary.json"],
    ["Hình", "python assets/plot_all.py <out> assets; python assets/plot_pipeline.py assets/fig1-pipeline.png", "assets/fig*.png"],
    ["Excel", "uv run --isolated --no-project --with openpyxl python assets/build_xlsx.py <out>", "report.xlsx + 4 sheet 03-11 ở workbook 26/10"],
])
wb.properties.creator = "openpyxl"
out = NEW_DIR / "report.xlsx"
wb.save(out)
print("saved", out)

# ---------------------------------------------------------------------------------------------
# 2. thêm sheet vào workbook 26/10
backup = OLD_XLSX.with_name("report.truoc-03-11-2026.xlsx")
if not backup.exists():
    shutil.copy2(OLD_XLSX, backup)
    print("backup", backup)
old = load_workbook(backup)  # luôn dựng lại từ bản sao lưu: chạy lại không nhân đôi sheet
before = list(old.sheetnames)
sh = Sheet(old.create_sheet("Tổng quan 03-11"), {"A": 34, "B": 57, "C": 29, "D": 30, "E": 30})
write_overview(sh, "Benchmark CE trên bộ official, chỉ retrieval (03/11/2026)", [
    ("Accuracy 03-11", "Accuracy chỉ retrieval"), ("Theo câu 03-11", "Từng truy vấn (704)"), ("Hình 03-11", "Hình 1–8"),
], "Ngày 03/11/2026")
sh = Sheet(old.create_sheet("Accuracy 03-11"), {"A": 40, "B": 22, "C": 22, "D": 22, "E": 22})
write_accuracy(sh, "Accuracy chỉ retrieval trên bộ official (704 truy vấn, 03/11/2026)")
sh.sub("Độ trễ CE theo cỡ pool")
sh.table(["Cỡ pool", "Passage TB", "p50", "p95", "p99"], [r[:5] for r in lat_rows()])
sh = Sheet(old.create_sheet("Theo câu 03-11"), PQ_WIDTHS)
write_per_query(sh, "Từng truy vấn (704), 03/11/2026")
sh = Sheet(old.create_sheet("Hình 03-11"), {"A": 160})
write_figs(sh, "Hình benchmark 03/11/2026")
assert old.sheetnames[: len(before)] == before
old.save(OLD_XLSX)
print("updated", OLD_XLSX, "sheets:", old.sheetnames[len(before):])
