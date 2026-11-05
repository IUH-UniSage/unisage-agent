"""Hình 1: luồng benchmark retrieval-only (03/11/2026). Chạy: python plot_pipeline.py <out.png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out = sys.argv[1]
INK, INK2, SURF = "#1a1a19", "#5f5e57", "#fcfcfb"
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#9a998f", "#8a5cd1"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(15, 8.6), dpi=160)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_xlim(0, 150); ax.set_ylim(0, 86); ax.axis("off")


def box(x, y, w, h, title, body, color, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                linewidth=1.6, edgecolor=color, facecolor=fill or "#ffffff"))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10.5, color=INK, weight="bold")
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=8.6, color=INK2, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=INK2, text=None, tx=0, ty=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, fontsize=8, color=INK2, ha="center")


ax.text(1, 84.5, "Luồng benchmark retrieval-only trên bộ official (03/11/2026)", fontsize=13, color=INK, weight="bold", va="top")
ax.text(1, 80.6, "Không classifier, không HyDE, không tách câu hỏi, không sinh câu trả lời, không LLM. OpenAI chỉ dùng cho embedding text-embedding-3-small.",
        fontsize=9, color=INK2, va="top")

# hàng 1: bộ câu → embed → hybrid
box(1, 52, 26, 23, "1. Bộ câu official", "questions.jsonl: 592 dòng\ncó expected_doc_ids + expect_visible\nnormal 207 · calculation 59\naccess (thấy) 86 = 352 câu\n× 2 biến thể: question / question_user\n= 704 truy vấn", BLUE)
box(1, 26, 26, 21, "Quyền theo persona", "asker.department_access\n→ UUID phòng ban (departments.json)\n→ AcademicSecurityContext\n(đúng như evals.run)\n+ 102 câu access không được thấy\n× 2 = 204 truy vấn kiểm tra rò", PURPLE)
box(33, 52, 24, 23, "2. Embed câu hỏi", "text-embedding-3-small\n(credential EMBEDDING của\nregistry env eval)\nmột lần mỗi truy vấn,\nbatch 16", GREY)
box(63, 52, 33, 23, "3. Hybrid search", "RetrievalService._search\nQdrant unisage-official-chunks\n(1 074 tài liệu, ~24 900 chunk)\n3 vector dense + BM25 (câu hỏi gốc),\nmỗi danh sách lấy limit × 2,\ngộp RRF (k = 1), cùng filter quyền\nTiêu đề tài liệu đã nằm trong text index", BLUE)
box(102, 52, 46, 23, "4. Pool 64 theo thứ tự RRF → pools.jsonl", "mỗi chunk: chunk_id, document_id, hạng RRF,\ncosine dense tốt nhất, cờ BM25-only, hạng dense/BM25\n+ pool 40 lấy riêng ở độ sâu của graph (đối chiếu)\nchunks.jsonl: passage = \"Tài liệu: <tiêu đề>\"\n+ heading_path + nội dung (passage_text của node CE)\nRetrieval cache ở tầng graph: không bị đọc", BLUE)
arrow(27, 63.5, 33, 63.5)
arrow(57, 63.5, 63, 63.5)
arrow(96, 63.5, 102, 63.5)
arrow(14, 47.3, 14, 52, PURPLE)
ax.annotate("", xy=(79, 52), xytext=(27, 36), arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4, connectionstyle="arc3,rad=0.15"))
ax.text(29, 31, "filter quyền áp cho cả dense lẫn BM25", fontsize=8, color=PURPLE)

# hàng 2: sàn + 4 phương án
box(102, 31, 46, 16, "5. Sàn cosine 0,45 (bước 09 của graph)", "bỏ chunk dense < CHAT_RETRIEVAL_DENSE_FLOOR,\ngiữ chunk BM25-only theo cờ\náp trên N chunk đầu trước khi chấm", GREY)
arrow(125, 52, 125, 47.3)

xs = [33, 61, 89, 117]
specs = [
    ("none", "8 chunk đầu theo RRF\n(trong 40 chunk đầu)\n= CHAT_RERANK_MODE=none", GREY),
    ("ce16", "CE chấm 16 chunk đầu RRF\ngiữ top-8 theo điểm CE", ORANGE),
    ("ce40", "CE chấm 40 chunk đầu\ngiữ top-8 = cấu hình eval\nhiện tại (MAX_CHUNKS=40)", ORANGE),
    ("ce64", "CE chấm cả pool 64\ngiữ top-8", ORANGE),
]
for x, (name, body, col) in zip(xs, specs):
    box(x, 6, 25, 17, name, body, col, fill="#fff7f2" if col == ORANGE else "#f6f6f4")
ax.annotate("", xy=(45.5, 23.4), xytext=(102, 39), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, connectionstyle="arc3,rad=0.1"))
for x in xs[1:]:
    ax.annotate("", xy=(x + 12.5, 23.4), xytext=(125, 31), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
ax.text(61, 1.2, "CE = bge-reranker-v2-m3 · service 127.0.0.1:8403 (GPU, max_length 512) · 1 request /rerank mỗi (truy vấn, cỡ pool), tuần tự → ce.jsonl",
        fontsize=8.4, color=ORANGE, ha="left")

box(1, 3, 26, 19, "6. Chỉ số (cấp tài liệu)", "recall@1/3/5/8, hit@5, MRR, NDCG@8\ntrần: tài liệu đúng trong 16/40/64\ncứu được / làm rơi so với none\nđộ trễ CE p50/p95 theo cỡ pool\ntheo category × exact/user", GREEN)
ax.annotate("", xy=(27.4, 12.5), xytext=(33, 12.5), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4))
fig.tight_layout()
fig.savefig(out, facecolor=SURF)
print("saved", out)
