"""fig9: 38 câu calculation qua 4 lần chạy chế độ none (pass / qua node calculation / từ chối)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
runs = ["rrf-only\n(trước Phase 6)", "calc-none-v2\n(grounding + prompt 1)", "calc-none-v3\n(route + prompt 2)", "calc-none-v4\n(+ số liệu câu hỏi)"]
series = [("pass (đúng số, qua node calculation)", [19, 23, 24, 26], "#2a78d6"),
          ("qua node calculation", [28, 30, 33, 32], "#eb6834"),
          ("từ chối / nêu giới hạn", [7, 9, 7, 5], "#9a998f")]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
w = 0.26
for i, (name, ys, color) in enumerate(series):
    xs = [x + (i - 1) * (w + 0.02) for x in range(len(runs))]
    bars = ax.bar(xs, ys, width=w, color=color, label=name, zorder=3)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.5, str(y), ha="center", va="bottom", fontsize=10, color="#1a1a19")
ax.axhline(38, color="#9a998f", linewidth=1, linestyle=(0, (4, 4)), zorder=1)
ax.text(-0.45, 38.6, "38 câu", color="#5f5e57", fontsize=9.5)
ax.set_xticks(range(len(runs))); ax.set_xticklabels(runs, fontsize=10)
ax.set_ylim(0, 42); ax.set_ylabel("số câu")
ax.grid(axis="y", color="#e6e5df", linewidth=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left", fontsize=9.5, ncol=3, bbox_to_anchor=(0, 1.08))
ax.set_title("Calculation 38 câu, chế độ none: pass / routing / từ chối qua 4 lần chạy", loc="left", fontsize=12, pad=28)
fig.tight_layout(); fig.savefig(out)
