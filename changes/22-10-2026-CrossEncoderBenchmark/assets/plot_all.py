import json, pathlib, statistics as st, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(exist_ok=True)
C = {"retrieval": "#2a78d6", "none": "#2a78d6", "cross_encoder": "#eb6834", "llm_rerank": "#1baf7a", "cross_encoder_then_llm": "#eda100"}
NAME = {"retrieval": "Thứ tự retrieval (RRF)", "cross_encoder": "cross_encoder", "llm_rerank": "llm_rerank", "cross_encoder_then_llm": "cross_encoder_then_llm"}
INK, INK2, GRID, SURF = "#1a1a19", "#5f5e57", "#e6e5df", "#fcfcfb"
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})

def style(ax, title=None):
    ax.set_facecolor(SURF)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    if title: ax.set_title(title, loc="left", fontsize=10.5, color=INK)

def fig(w, h): 
    f = plt.figure(figsize=(w, h), dpi=160); f.patch.set_facecolor(SURF); return f

def save(f, name):
    f.tight_layout(); f.savefig(OUT / name, facecolor=SURF); plt.close(f); print("saved", name)

def vn(x, d=3): return f"{x:.{d}f}".replace(".", ",")
def pct(v, q):
    v = sorted(v); return v[min(len(v)-1, int(q*len(v)))]

def load(run): return {r["id"]: r for r in (json.loads(l) for l in pathlib.Path(run, "results.jsonl").read_text().splitlines() if l.strip()) if not r.get("quota")}
def doc(c): return c.split(":")[0]
rows = load("runs/hybrid-on")
A = json.loads(pathlib.Path("benchmark_results/step-a-ce-accuracy/rerank_benchmark.json").read_text())
B = json.loads(pathlib.Path("benchmark_results/step-b-ce-perf/rerank_benchmark.json").read_text())
Cj = json.loads(pathlib.Path("benchmark_results/step-c-llm-accuracy/rerank_benchmark.json").read_text())
D = json.loads(pathlib.Path("benchmark_results/step-d-llm-perf/rerank_benchmark.json").read_text())

# ---- Fig 2: CE top-1 score distribution, relevant vs not
ce = A["accuracy"][0]
rel, irr = [], []
for r in ce["runs"]:
    if r["repeat"] != 1 or not r["kept_ids"]: continue
    exp = set(rows[r["question_id"]]["expected_document_ids"])
    (rel if doc(r["kept_ids"][0]) in exp else irr).append(r["detail"]["ce_scores"][0])
f = fig(10, 4.6); ax = f.add_subplot(111); style(ax, "Điểm cross-encoder (logit) của chunk xếp hạng 1, theo chunk đó có thuộc tài liệu kỳ vọng hay không (139 câu, bước A)")
bins = [x/2 for x in range(-14, 22)]
ax.hist(rel, bins=bins, color=C["cross_encoder"], alpha=0.85, label=f"hạng 1 đúng tài liệu (n={len(rel)}, median {vn(st.median(rel),2)})", edgecolor=SURF, linewidth=0.8)
ax.hist(irr, bins=bins, color=C["retrieval"], alpha=0.75, label=f"hạng 1 sai tài liệu (n={len(irr)}, median {vn(st.median(irr),2)})", edgecolor=SURF, linewidth=0.8)
ax.axvline(st.median(rel), color=C["cross_encoder"], linestyle=(0,(4,3)), linewidth=1.2); ax.axvline(st.median(irr), color=C["retrieval"], linestyle=(0,(4,3)), linewidth=1.2)
ax.set_xlabel("điểm cross-encoder của chunk hạng 1"); ax.set_ylabel("số câu"); ax.legend(frameon=False)
ax.text(0.01, 0.97, "Hai phân phối chồng nhau rộng (sai có tới 8,3; đúng xuống −0,8): không có ngưỡng tách được đúng/sai → không dùng điểm làm gate.", transform=ax.transAxes, fontsize=9, color=INK2, va="top")
save(f, "fig2-ce-score-distribution.png")

# ---- Fig 3: CE latency histogram + concurrency
perf = B["performance"][0]; lat = perf["latencies_ms"]; conc = B["concurrency"]
f = fig(12, 4.4)
ax = f.add_subplot(131); style(ax, f"Độ trễ bước rerank cross-encoder\n{len(lat)} phép đo, concurrency 1")
ax.hist(lat, bins=range(50, 190, 8), color=C["cross_encoder"], edgecolor=SURF, linewidth=0.8)
for q, lab, ha, dx in ((0.5, "p50", "right", -1.5), (0.95, "p95", "left", 1.5)):
    v = pct(lat, q); ax.axvline(v, color=INK2, linestyle=(0,(4,3)), linewidth=1); ax.text(v+dx, ax.get_ylim()[1]*0.9, f"{lab} {v:.0f} ms", fontsize=9, color=INK2, ha=ha)
ax.set_xlabel("ms"); ax.set_ylabel("số phép đo")
levels = [c["concurrency"] for c in conc]
ax = f.add_subplot(132); style(ax, "Độ trễ theo mức đồng thời\n1 câu cố định, 30 request mỗi mức")
x = range(len(levels))
ax.bar([i-0.2 for i in x], [c["p50_ms"] for c in conc], width=0.38, color=C["cross_encoder"], label="p50")
ax.bar([i+0.2 for i in x], [c["p95_ms"] for c in conc], width=0.38, color="#f5b79a", label="p95")
for i, c in enumerate(conc): ax.text(i-0.2, c["p50_ms"]+15, f"{c['p50_ms']:.0f}", ha="center", fontsize=9, color=INK)
ax.set_xticks(list(x)); ax.set_xticklabels([str(l) for l in levels]); ax.set_xlabel("số request đồng thời"); ax.set_ylabel("ms"); ax.legend(frameon=False, loc="upper left")
ax = f.add_subplot(133); style(ax, "Throughput theo mức đồng thời\nphẳng ~7,3 req/s: service xử lý tuần tự")
ax.plot(levels, [c["throughput_rps"] for c in conc], color=C["cross_encoder"], linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.5)
for l, c in zip(levels, conc): ax.text(l, c["throughput_rps"]+0.25, vn(c["throughput_rps"],2), ha="center", fontsize=9, color=INK)
ax.set_xticks(levels); ax.set_ylim(0, 10); ax.set_xlabel("số request đồng thời"); ax.set_ylabel("request / giây")
save(f, "fig3-ce-latency-concurrency.png")

# ---- Fig 4: metrics grouped bars (step A + C + baseline)
base = {"recall@8": 0.928, "recall@5": 0.917, "NDCG@16": 0.733, "MRR": 0.846}
aggs = {"cross_encoder": ce["aggregate"]}
for v in Cj["accuracy"]: aggs[v["variant"]] = v["aggregate"]
metrics = ["recall@8", "recall@5", "NDCG@16", "MRR"]
key = {"recall@8": "recall_at_8", "recall@5": "recall_at_5", "NDCG@16": "ndcg_at_16", "MRR": "mrr"}
order = ["retrieval", "cross_encoder", "llm_rerank", "cross_encoder_then_llm"]
f = fig(11, 4.8); ax = f.add_subplot(111); style(ax, "Bốn chỉ số xếp hạng trên cùng 139 câu × 16 ứng viên (bước A và C; LLM trung bình 3 lượt)")
w = 0.2
for j, name in enumerate(order):
    vals = [base[m] if name == "retrieval" else aggs[name][key[m]] for m in metrics]
    xs = [i + (j-1.5)*w for i in range(len(metrics))]
    ax.bar(xs, [v or 0 for v in vals], width=w-0.02, color=C[name], label=NAME[name])
    for xx, v in zip(xs, vals):
        if v is None:
            ax.text(xx, 0.02, "không có:\nkhông chọn gì", ha="center", fontsize=7.5, color=INK2, rotation=90, va="bottom")
        else:
            ax.text(xx, v+0.008, vn(v), ha="center", fontsize=8.5, color=INK, rotation=90, va="bottom")
ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics); ax.set_ylim(0, 1.12); ax.set_ylabel("giá trị (0–1)")
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(f, "fig4-rank-metrics.png")

# ---- Fig 5: kept-count distribution + stability
f = fig(12, 4.4)
ax = f.add_subplot(121); style(ax, "Số chunk mỗi biến thể giữ lại trên 16 ứng viên\n(417 lượt mỗi biến thể LLM, 278 lượt cross-encoder)")
ks = list(range(0, 9))
dist = {}
for name in ("cross_encoder", "llm_rerank", "cross_encoder_then_llm"):
    runs = ce["runs"] if name == "cross_encoder" else next(v for v in Cj["accuracy"] if v["variant"] == name)["runs"]
    cnt = collections.Counter(r["scores"]["kept"] for r in runs if r["scores"])
    total = sum(cnt.values()); dist[name] = [cnt.get(k, 0)/total for k in ks]
for j, name in enumerate(dist):
    ax.bar([k + (j-1)*0.27 for k in ks], dist[name], width=0.25, color=C[name], label=NAME[name])
ax.set_xticks(ks); ax.set_xlabel("số chunk giữ"); ax.set_ylabel("tỷ lệ lượt"); ax.legend(frameon=False)
ax.text(0.02, 0.95, "LLM: median 2 chunk, 11 lượt loại hết\ncross-encoder: luôn 8", transform=ax.transAxes, fontsize=9, color=INK2, va="top")
ax = f.add_subplot(122); style(ax, "Tái lập qua các lượt lặp\nJaccard tập giữ (đậm) và tỷ lệ câu cùng thứ tự mọi lượt (gạch chéo)")
names = ["cross_encoder", "llm_rerank", "cross_encoder_then_llm"]
jac = [1.0] + [next(v for v in Cj["accuracy"] if v["variant"] == n)["stability"]["mean_jaccard"] for n in names[1:]]
ident = [1.0] + [next(v for v in Cj["accuracy"] if v["variant"] == n)["stability"]["identical_rate"] for n in names[1:]]
x = range(3)
ax.bar([i-0.2 for i in x], jac, width=0.38, color=[C[n] for n in names])
ax.bar([i+0.2 for i in x], ident, width=0.38, color=[C[n] for n in names], alpha=0.45, hatch="//", edgecolor=SURF)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#9a998f", label="Jaccard TB tập giữ"), Patch(facecolor="#9a998f", alpha=0.45, hatch="//", edgecolor=SURF, label="câu cùng thứ tự mọi lượt")], frameon=False, loc="upper right")
for i in x:
    ax.text(i-0.2, jac[i]+0.02, vn(jac[i]), ha="center", fontsize=9, color=INK); ax.text(i+0.2, ident[i]+0.02, f"{ident[i]*100:.1f}%".replace(".", ","), ha="center", fontsize=9, color=INK)
ax.set_xticks(list(x)); ax.set_xticklabels(["cross_encoder\n(2 lượt)", "llm_rerank\n(3 lượt)", "cross_encoder_then_llm\n(3 lượt)"]); ax.set_ylim(0, 1.25)
save(f, "fig5-kept-stability.png")

# ---- Fig 6: rerank-step latency per variant (box + percentile labels)
lat_by = {"cross_encoder": lat}
for p_ in D["performance"]: lat_by[p_["variant"]] = p_["latencies_ms"]
f = fig(11, 4.6); ax = f.add_subplot(111); style(ax, "Độ trễ riêng bước rerank, concurrency 1 (cross-encoder 695 phép đo; LLM 139 phép đo, không tính thời gian chờ rate limit)")
names = ["cross_encoder", "llm_rerank", "cross_encoder_then_llm"]
bp = ax.boxplot([lat_by[n] for n in names], positions=range(3), widths=0.5, showfliers=True, patch_artist=True,
                flierprops=dict(marker="o", markersize=3, markerfacecolor=INK2, markeredgecolor="none", alpha=0.5),
                medianprops=dict(color=INK, linewidth=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
for patch, n in zip(bp["boxes"], names): patch.set_facecolor(C[n]); patch.set_alpha(0.8); patch.set_edgecolor(SURF)
for i, n in enumerate(names):
    v = lat_by[n]; ax.text(i+0.3, pct(v,0.5), f"p50 {pct(v,0.5):.0f} ms\np95 {pct(v,0.95):.0f} ms\nmax {max(v):.0f} ms", fontsize=9, color=INK, va="center")
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_ylabel("ms"); ax.set_xlim(-0.5, 3.1)
save(f, "fig6-rerank-latency.png")

# ---- Fig 7: e2e comparison
E = {"none": load("runs/rrf-only"), "llm_rerank": load("runs/hybrid-on"), "cross_encoder": load("runs/ce-on"), "cross_encoder_then_llm": load("runs/ce-then-llm")}
def e2e(rs):
    ch = lambda r: r.get("checks") or {}
    normal = [r for r in rs.values() if r["category"] == "normal"]; calc = [r for r in rs.values() if r["category"] == "calculation"]
    return dict(
        normal_recall=sum(1 for r in normal if ch(r).get("recall_at_5") is True)/len(normal)*100,
        calc_pass=sum(1 for r in calc if ch(r).get("pass") is True)/len(calc)*100,
        calc_recall=sum(1 for r in calc if ch(r).get("recall_at_5") is True)/len(calc)*100,
        refusal=sum(1 for r in rs.values() if ch(r).get("refusal_hint")),
        p50=pct([r["total_ms"] for r in rs.values() if r.get("total_ms")], .5), p95=pct([r["total_ms"] for r in rs.values() if r.get("total_ms")], .95),
        usd=st.mean(r["usage"]["usd"] for r in rs.values() if r.get("usage"))*1000, calls=st.mean(r["usage"]["calls"] for r in rs.values() if r.get("usage")))
S = {n: e2e(rs) for n, rs in E.items()}
names = ["none", "llm_rerank", "cross_encoder", "cross_encoder_then_llm"]
SHORT = ["none\n(RRF)", "llm", "cross_\nencoder", "ce_then_\nllm"]
f = fig(14, 4.8)
ax = f.add_subplot(141); style(ax, "Độ chính xác e2e (%)\n154 câu, cache tắt")
ms = [("normal_recall", "normal\nrecall@5"), ("calc_pass", "calc\npass"), ("calc_recall", "calc\nrecall@5")]
for j, n in enumerate(names):
    xs = [i + (j-1.5)*0.21 for i in range(3)]; vals = [S[n][m] for m, _ in ms]
    ax.bar(xs, vals, width=0.2, color=C[n], label=n)
    for xx, v in zip(xs, vals): ax.text(xx, v+1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in ms]); ax.set_ylim(0, 118); ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
ax = f.add_subplot(142); style(ax, "Số câu có dấu hiệu từ chối\n(refusal_hint) trên 154")
vals = [S[n]["refusal"] for n in names]; ax.bar(range(4), vals, color=[C[n] for n in names], width=0.6)
for i, v in enumerate(vals): ax.text(i, v+1, str(v), ha="center", fontsize=9.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT); ax.set_ylim(0, 50)
ax = f.add_subplot(143); style(ax, "Độ trễ cả lượt (ms)\np50 đậm, p95 nhạt")
for i, n in enumerate(names):
    ax.bar(i-0.2, S[n]["p50"], width=0.38, color=C[n]); ax.bar(i+0.2, S[n]["p95"], width=0.38, color=C[n], alpha=0.45)
    ax.text(i-0.2, S[n]["p50"]+300, f"{S[n]['p50']/1000:.1f}s".replace(".", ","), ha="center", fontsize=8.5, color=INK); ax.text(i+0.2, S[n]["p95"]+300, f"{S[n]['p95']/1000:.1f}s".replace(".", ","), ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT); ax.set_ylim(0, 21000)
ax = f.add_subplot(144); style(ax, "Chi phí LLM mỗi lượt\n(USD × 1000, số lượt gọi)")
vals = [S[n]["usd"] for n in names]; ax.bar(range(4), vals, color=[C[n] for n in names], width=0.6)
for i, n in enumerate(names): ax.text(i, vals[i]+0.1, f"{vals[i]:.2f}\n({S[n]['calls']:.2f} gọi)".replace(".", ","), ha="center", fontsize=8, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels(SHORT); ax.set_ylim(0, 7.5)
save(f, "fig7-e2e-comparison.png")

# ---- Fig 8: failure reasons per run
def normal_reason(r):
    exp = set(r.get("expected_document_ids") or []); nodes = r.get("nodes") or []; rets = r.get("retrievals") or []
    if any(n.startswith("05_") for n in nodes) or not rets: return "A. off-topic /\nkhông retrieval"
    if r.get("used_ticket_fallback"): return "E. ticket"
    ranks = [i for i, ch in enumerate(rets[0]["chunks"]) if doc(ch["chunk_id"]) in exp]
    if not ranks: return "B. retrieval\nkhông lấy được"
    hit = [i for i, c in enumerate(r.get("context") or []) if isinstance(c, dict) and c.get("document_id") in exp]
    return "C. rerank loại\nkhỏi ngữ cảnh" if not hit else "D. trong ngữ cảnh,\nngoài top-5"
def calc_reason(r):
    ch = r.get("checks") or {}; nodes = r.get("nodes") or []
    if any(n.startswith("05_") for n in nodes): return "R1. off-topic"
    if "calculation" not in (ch.get("outcome") or []): return "R2. không qua\nCalculationNode"
    if ch.get("asked_back"): return "R3. hỏi lại"
    return "R4. con số sai /\ntừ chối tính"
f = fig(12, 4.6)
for k_, (cat, fn, title, key_, order_) in enumerate((("normal", normal_reason, "Vì sao câu normal trượt recall@5", "recall_at_5", ["A. off-topic /\nkhông retrieval", "B. retrieval\nkhông lấy được", "C. rerank loại\nkhỏi ngữ cảnh", "D. trong ngữ cảnh,\nngoài top-5", "E. ticket"]),
                                                      ("calculation", calc_reason, "Vì sao câu calculation trượt pass", "pass", ["R1. off-topic", "R2. không qua\nCalculationNode", "R3. hỏi lại", "R4. con số sai /\ntừ chối tính"]))):
    ax = f.add_subplot(1, 2, k_+1); style(ax, title)
    cnts = {}
    for n in ["llm_rerank", "cross_encoder", "none"]:
        c_ = collections.Counter(fn(r) for r in E[n].values() if r["category"] == cat and (r.get("checks") or {}).get(key_) is False)
        cnts[n] = [c_.get(o, 0) for o in order_]
    x = range(len(order_))
    for j, n in enumerate(cnts):
        xs = [i + (j-1)*0.28 for i in x]; ax.bar(xs, cnts[n], width=0.26, color=C[n], label=f"{n} (tổng {sum(cnts[n])})")
        for xx, v in zip(xs, cnts[n]):
            if v: ax.text(xx, v+0.15, str(v), ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(order_, fontsize=8.5); ax.set_ylabel("số câu"); ax.legend(frameon=False)
save(f, "fig8-failure-reasons.png")
