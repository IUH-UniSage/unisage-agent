import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
k = list(range(1, 9))
series = [
    ("Thứ tự retrieval (RRF, không rerank)", [0.745, 0.831, 0.860, 0.896, 0.917, 0.924, 0.928, 0.928], "#2a78d6", "o"),
    ("cross_encoder", [0.712, 0.766, 0.831, 0.871, 0.888, 0.906, 0.914, 0.921], "#eb6834", "s"),
    ("llm_rerank", [0.829, 0.868, 0.879, 0.880, 0.880, 0.880, 0.880, 0.880], "#1baf7a", "^"),
    ("cross_encoder_then_llm", [0.811, 0.860, 0.872, 0.874, 0.876, 0.876, 0.876, 0.876], "#eda100", "D"),
]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.axhline(0.928, color="#9a998f", linewidth=1, linestyle=(0, (4, 4)), zorder=1)
ax.text(1.0, 0.932, "trần của bộ ứng viên 0,928 (tài liệu kỳ vọng có trong các ứng viên fetch lại được)", color="#5f5e57", fontsize=9, va="bottom")
label_offsets = {"Thứ tự retrieval (RRF, không rerank)": 0.0, "cross_encoder": -0.006, "llm_rerank": 0.0, "cross_encoder_then_llm": -0.012}
for name, ys, color, marker in series:
    ax.plot(k, ys, color=color, linewidth=2, marker=marker, markersize=7, markerfacecolor=color,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name, zorder=3)
    ax.annotate(f"{name}  {ys[-1]:.3f}".replace(".", ","), xy=(8, ys[-1]), xytext=(8.15, ys[-1] + label_offsets[name]),
                color="#1a1a19", fontsize=9.5, va="center")
ax.set_xlim(0.7, 11.3); ax.set_ylim(0.68, 0.96)
ax.set_xticks(k); ax.set_xlabel("k (số chunk giữ đầu)")
ax.set_ylabel("recall@k (trung bình 139 câu)")
ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.928])
ax.set_yticklabels([f"{v:.3f}".replace(".", ",").rstrip("0").rstrip(",") if v != 0.928 else "0,928" for v in [0.70, 0.75, 0.80, 0.85, 0.90, 0.928]])
ax.grid(axis="y", color="#e6e5df", linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right"): ax.spines[side].set_visible(False)
for side in ("left", "bottom"): ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors="#5f5e57", length=0)
ax.set_title("Đường cong recall@k trên cùng 16 ứng viên của 139 câu (runs/hybrid-on)\nLLM lấy trung bình 3 lượt; cross-encoder tất định; k = 8 là số chunk prompt sinh câu trả lời đọc",
             loc="left", fontsize=11, color="#1a1a19")
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(out, facecolor=fig.get_facecolor())
print("saved", out)
