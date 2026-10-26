import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]
ml = [256, 384, 512, 768, 1024]
recall8 = [0.928, 0.928, 0.928, 0.921, 0.921]; recall5 = [0.885, 0.910, 0.888, 0.888, 0.888]; mrr = [0.810, 0.826, 0.828, 0.815, 0.815]
p50 = [62, 78, 97, 138, 139]; rps = [17.08, 13.10, 10.40, 7.77, 7.33]; trunc = [4065/6720, 2975/6720, 1939/6720, 572/6720, 12/6720]
INK, INK2, GRID, SURF = "#1a1a19", "#5f5e57", "#e6e5df", "#fcfcfb"; C1, C2, C3 = "#eb6834", "#2a78d6", "#9a998f"
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
f = plt.figure(figsize=(15, 4.4), dpi=160); f.patch.set_facecolor(SURF)
def style(ax, t):
    ax.set_facecolor(SURF); [ax.spines[s].set_visible(False) for s in ("top", "right")]; ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0); ax.set_title(t, loc="left", fontsize=10.5, color=INK)
ax = f.add_subplot(141); style(ax, "Độ chính xác theo max_length\n(139 câu, cùng 16 ứng viên, tất định)")
for ys, c, lab, m in ((recall8, C1, "recall@8", "o"), (recall5, C2, "recall@5", "s"), (mrr, C3, "MRR", "^")):
    ax.plot(ml, ys, color=c, linewidth=2, marker=m, markersize=7, markeredgecolor=SURF, markeredgewidth=1.5, label=lab)
    ax.text(ml[-1] + 15, ys[-1], f"{lab} {ys[-1]:.3f}".replace(".", ","), fontsize=9, color=INK, va="center")
ax.axhline(0.928, color="#9a998f", linewidth=1, linestyle=(0, (4, 4))); ax.text(260, 0.930, "trần 0,928", fontsize=8.5, color=INK2)
ax.set_xticks(ml); ax.set_xlim(230, 1250); ax.set_ylim(0.78, 0.95); ax.set_xlabel("max_length (token mỗi cặp câu hỏi + chunk)")
ax = f.add_subplot(142); style(ax, "Độ trễ p50 (ms)\nconcurrency 1, 16 chunk mỗi request")
ax.bar(ml, p50, width=56, color=C1)
for x, v in zip(ml, p50): ax.text(x, v + 3, f"{v}", ha="center", fontsize=9, color=INK)
ax.set_xticks(ml); ax.set_xlabel("max_length"); ax.set_ylim(0, 170); ax.set_ylabel("ms")
ax = f.add_subplot(143); style(ax, "Throughput (req/s)\nconcurrency 1")
ax.plot(ml, rps, color=C2, linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.5)
for x, v in zip(ml, rps): ax.text(x, v + 0.7, f"{v:.1f}".replace(".", ","), ha="center", fontsize=9, color=INK)
ax.set_xticks(ml); ax.set_ylim(0, 20); ax.set_ylabel("request / giây"); ax.set_xlabel("max_length")
ax = f.add_subplot(144); style(ax, "Tỷ lệ passage bị cắt ở max_length\n(6 720 passage của 139 câu × 3 lượt)")
ax.bar(ml, [t * 100 for t in trunc], width=56, color=C3)
for x, t in zip(ml, trunc): ax.text(x, t * 100 + 1.5, f"{t*100:.0f}%", ha="center", fontsize=9, color=INK)
ax.set_xticks(ml); ax.set_ylim(0, 75); ax.set_ylabel("% passage bị cắt"); ax.set_xlabel("max_length")
f.tight_layout(); f.savefig(out, facecolor=SURF); print("saved", out)
