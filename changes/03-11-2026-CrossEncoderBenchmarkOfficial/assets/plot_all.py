"""Hình 2–8 của benchmark CE trên bộ official (03/11/2026), từ summary.json của analyze.py.
Chạy: python plot_all.py <thư mục kết quả có summary.json> <thư mục assets>"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

S = json.loads((Path(sys.argv[1]) / "summary.json").read_text(encoding="utf-8"))
OUT = Path(sys.argv[2])
INK, INK2, SURF, GRID = "#1a1a19", "#5f5e57", "#fcfcfb", "#e6e5df"
V = ["none", "ce16", "ce40", "ce64"]
C = {"none": "#2a78d6", "ce16": "#f5b79a", "ce40": "#eb6834", "ce64": "#a83e12"}
NAME = {"none": "none (RRF, không rerank)", "ce16": "ce16", "ce40": "ce40 (cấu hình eval)", "ce64": "ce64"}
CAT = {"normal": "normal", "calculation": "calculation", "access": "access (thấy)"}
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
G = S["groups"]


def vn(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def style(ax, title=None):
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)


def fig(w, h):
    f = plt.figure(figsize=(w, h), dpi=160); f.patch.set_facecolor(SURF); return f


def save(f, name, bottom=0.0):
    f.tight_layout(rect=(0, bottom, 1, 1)); f.savefig(OUT / name, facecolor=SURF); plt.close(f); print("saved", name)


# Hình 2: recall@k
f = fig(15, 5.4)
for i, grp in enumerate(["all", "exact", "user"]):
    ax = f.add_subplot(1, 3, i + 1)
    curves = S["curves"][grp]
    lo = min(min(c) for c in curves.values())
    ceil40 = G[grp]["ceiling_floored"]["40"]
    ax.axhline(ceil40, color="#9a998f", linewidth=1, linestyle=(0, (4, 4)))
    ax.text(8, ceil40 - 0.004, f"trần pool 40 sau sàn {vn(ceil40)}", fontsize=8.5, color=INK2, va="top", ha="right")
    for v in V:
        ax.plot(range(1, 9), curves[v], color=C[v], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURF, label=f"{NAME[v]}  @8 {vn(curves[v][-1])}")
    ax.set_xticks(range(1, 9)); ax.set_xlabel("k (số chunk giữ đầu)")
    hi = max(ceil40, max(max(c) for c in curves.values()))
    ax.set_ylim(max(0, lo - 0.05), min(1.005, hi + 0.02))
    title = {"all": "Cả hai biến thể", "exact": "question (đúng chính tả)", "user": "question_user (kiểu người dùng)"}[grp]
    style(ax, f"{title} · n = {G[grp]['n']}")
    if i == 0:
        ax.set_ylabel("recall@k (trung bình truy vấn)")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
f.suptitle("Hình 2. recall@k, k = 1..8, theo phương án (chỉ retrieval, bộ official 352 câu)", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig2-recall-at-k.png")

# Hình 3: recall@5 và @8 theo category × phương án
f = fig(14, 5.2)
for i, m in enumerate(["recall@5", "recall@8"]):
    ax = f.add_subplot(1, 2, i + 1)
    cats = ["normal", "calculation", "access"]
    w = 0.2
    for j, v in enumerate(V):
        xs = [k + (j - 1.5) * w for k in range(len(cats))]
        vals = [G[c][v][m] for c in cats]
        ax.bar(xs, vals, width=w - 0.02, color=C[v], label=NAME[v])
        for x, y in zip(xs, vals):
            ax.text(x, y - 0.008, vn(y), ha="center", fontsize=8, color="white" if v != "ce16" else INK, rotation=90, va="top")
    for k, c in enumerate(cats):
        ce = G[c]["ceiling_floored"]["40"]
        ax.plot([k - 0.42, k + 0.42], [ce, ce], color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels([f"{CAT[c]}\n(n = {G[c]['n']})" for c in cats])
    ax.set_ylim(0.5, 1.02)
    style(ax, f"{m} theo category (vạch đứt: trần pool 40 sau sàn; trục từ 0,5)")
    if i == 0:
        ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.13))
f.suptitle("Hình 3. recall@5 / recall@8 theo category × phương án (gộp question + question_user)", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig3-category.png")

# Hình 4: exact vs user, theo category
f = fig(14, 5.2)
for i, var in enumerate(["exact", "user"]):
    ax = f.add_subplot(1, 2, i + 1)
    cats = ["normal", "calculation", "access"]
    w = 0.2
    for j, v in enumerate(V):
        xs = [k + (j - 1.5) * w for k in range(len(cats) + 1)]
        vals = [G[f"{c}/{var}"][v]["recall@8"] for c in cats] + [G[var][v]["recall@8"]]
        ax.bar(xs, vals, width=w - 0.02, color=C[v], label=NAME[v])
        for x, y in zip(xs, vals):
            ax.text(x, y - 0.008, vn(y), ha="center", fontsize=8, color="white" if v != "ce16" else INK, rotation=90, va="top")
    ax.set_xticks(range(4)); ax.set_xticklabels([CAT[c] for c in cats] + ["tất cả"])
    ax.set_ylim(0.5, 1.02)
    style(ax, {"exact": "question (đúng chính tả, có dấu)", "user": "question_user (kiểu người dùng, thường không dấu)"}[var] + " · recall@8 (trục từ 0,5)")
    if i == 0:
        ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.08))
f.suptitle("Hình 4. Câu hỏi chuẩn so với câu kiểu người dùng: recall@8 theo phương án", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig4-exact-vs-user.png")

# Hình 5: trần so với đạt được theo cỡ pool
f = fig(13, 5.2)
for i, grp in enumerate(["exact", "user"]):
    ax = f.add_subplot(1, 2, i + 1)
    sizes = ["16", "40", "64"]
    raw = [G[grp]["ceiling"][s] for s in sizes]
    flo = [G[grp]["ceiling_floored"][s] for s in sizes]
    ach = [G[grp][f"ce{s}"]["recall@8"] for s in sizes]
    x = range(3)
    ax.bar([k - 0.27 for k in x], raw, width=0.26, color="#c9c8bf", label="trần pool (trước sàn)")
    ax.bar(list(x), flo, width=0.26, color="#9a998f", label="trần sau sàn 0,45 (CE thấy)")
    ax.bar([k + 0.27 for k in x], ach, width=0.26, color=[C[f"ce{s}"] for s in sizes], label="CE đạt recall@8")
    for xs, vals in (([k - 0.27 for k in x], raw), (list(x), flo), ([k + 0.27 for k in x], ach)):
        for xx, y in zip(xs, vals):
            ax.text(xx, y + 0.008, vn(y), ha="center", fontsize=8.5, color=INK)
    none8 = G[grp]["none"]["recall@8"]
    ax.axhline(none8, color=C["none"], linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xticks(list(x)); ax.set_xticklabels([f"pool {s}" for s in sizes]); ax.set_ylim(0.5, 1.06)
    style(ax, {"exact": "question", "user": "question_user"}[grp] + f" · n = {G[grp]['n']} · vạch xanh: none @8 {vn(none8)} (trục từ 0,5)")
    if i == 0:
        f.legend(handles=[Patch(color="#c9c8bf", label="trần pool (trước sàn)"), Patch(color="#9a998f", label="trần sau sàn 0,45 (cái CE thấy)"),
                          Patch(color=C["ce40"], label="CE đạt recall@8 (màu theo ce16/ce40/ce64)")], loc="lower center", ncol=3, frameon=False, fontsize=9)
f.suptitle("Hình 5. Trần (tài liệu đúng có trong pool) so với recall@8 CE đạt được, theo cỡ pool", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig5-ceiling.png", bottom=0.07)

# Hình 6: cứu được / làm rơi so với none
f = fig(14, 4.8)
for i, (m, lab) in enumerate((("1", "recall@1"), ("5", "recall@5"), ("8", "recall@8"))):
    ax = f.add_subplot(1, 3, i + 1)
    ce = ["ce16", "ce40", "ce64"]
    resc = [G["all"][v][f"rescued@{m}"] for v in ce]
    drop = [G["all"][v][f"dropped@{m}"] for v in ce]
    x = range(3)
    ax.bar([k - 0.2 for k in x], resc, width=0.38, color="#1baf7a", label="cứu được (none kém hơn → CE tốt hơn)")
    ax.bar([k + 0.2 for k in x], drop, width=0.38, color="#d64545", label="làm rơi (none tốt hơn → CE kém hơn)")
    for k in x:
        ax.text(k - 0.2, resc[k] + 0.5, str(resc[k]), ha="center", fontsize=9, color=INK)
        ax.text(k + 0.2, drop[k] + 0.5, str(drop[k]), ha="center", fontsize=9, color=INK)
        ax.text(k, -0.09 * max(resc + drop + [1]), f"ròng {resc[k] - drop[k]:+d}", ha="center", fontsize=9, color=INK2)
    ax.set_xticks(list(x)); ax.set_xticklabels(ce)
    ax.set_ylim(-0.14 * max(resc + drop + [1]), max(resc + drop + [1]) * 1.18)
    style(ax, f"{lab} · số truy vấn (trên 704)")
    if i == 0:
        f.legend(handles=[Patch(color="#1baf7a", label="cứu được: CE có recall cao hơn none"), Patch(color="#d64545", label="làm rơi: CE có recall thấp hơn none")],
                 loc="lower center", ncol=2, frameon=False, fontsize=9)
f.suptitle("Hình 6. Truy vấn CE cứu được và làm rơi so với none, theo cỡ pool", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig6-rescued-dropped.png", bottom=0.07)

# Hình 7: độ trễ CE theo cỡ pool
f = fig(14, 5.0)
L = S["latency"]
ax = f.add_subplot(1, 2, 1)
sizes = sorted(L, key=int)
data = [[ms for _, ms in S["latency_points"][s]] for s in sizes]
bp = ax.boxplot(data, positions=range(len(sizes)), widths=0.5, patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, markerfacecolor=INK2, markeredgecolor="none", alpha=0.5),
                medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
for patch, s in zip(bp["boxes"], sizes):
    patch.set_facecolor(C[f"ce{s}"]); patch.set_alpha(0.85); patch.set_edgecolor(SURF)
for k, s in enumerate(sizes):
    ax.text(k + 0.3, L[s]["p50"], f"p50 {L[s]['p50']:.0f} ms\np95 {L[s]['p95']:.0f} ms", fontsize=8.8, color=INK, va="center")
ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"ce{s}\nTB {vn(L[s]['cand_mean'], 1)} passage" for s in sizes])
ax.set_ylabel("ms / request (client đo, concurrency 1)"); ax.set_xlim(-0.5, len(sizes) - 0.1)
style(ax, "Phân phối độ trễ một request /rerank")
ax = f.add_subplot(1, 2, 2)
for s in sizes:
    pts = S["latency_points"][s]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=8, alpha=0.35, color=C[f"ce{s}"], label=f"ce{s}", edgecolors="none")
ax.set_xlabel("số passage gửi CE (sau sàn 0,45)"); ax.set_ylabel("ms")
style(ax, "Độ trễ theo số passage: gần tuyến tính")
leg = ax.legend(frameon=False, loc="upper left", markerscale=2.5)
for h in leg.legend_handles:
    h.set_alpha(1)
f.suptitle("Hình 7. Độ trễ CE (bge-reranker-v2-m3, max_length 512, RTX 4080 SUPER) theo cỡ pool", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig7-ce-latency.png")

# Hình 8: phân phối hạng tài liệu đúng đầu tiên
f = fig(13, 5.0)
ax = f.add_subplot(1, 1, 1)
buckets = ["1", "2", "3", "4", "5", "6", "7", "8", "9–16", "17–40", "41–64", "không có"]
D = S["rank_dist"]
series = [("none", "none (thứ tự RRF sau sàn)", C["none"]), ("ce40", "ce40 (thứ tự CE trên 40)", C["ce40"]), ("ce64", "ce64 (thứ tự CE trên 64)", C["ce64"])]
w = 0.27
for j, (key, lab, col) in enumerate(series):
    vals = [D[key].get(b, 0) for b in buckets]
    xs = [k + (j - 1) * w for k in range(len(buckets))]
    ax.bar(xs, vals, width=w - 0.02, color=col, label=lab)
    for x, y in zip(xs, vals):
        if y:
            ax.text(x, y + 3, str(y), ha="center", fontsize=7.5, color=INK, rotation=90, va="bottom")
ax.axvline(7.5, color=INK2, linestyle=(0, (3, 3)), linewidth=1)
ax.text(7.45, ax.get_ylim()[1] * 0.92, "← trong 8 chunk prompt đọc", ha="right", fontsize=9, color=INK2)
ax.set_xticks(range(len(buckets))); ax.set_xticklabels(buckets)
ax.set_xlabel("hạng của chunk đầu tiên thuộc tài liệu kỳ vọng ('không có': không nằm trong ứng viên sau sàn: none và ce64 xét cả pool 64, ce40 xét 40 chunk đầu)")
ax.set_ylabel("số truy vấn (trên 704)")
style(ax, None); ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
f.suptitle("Hình 8. Hạng của tài liệu đúng đầu tiên: none so với ce40 / ce64", x=0.01, ha="left", fontsize=11.5, color=INK)
save(f, "fig8-rank-distribution.png")
