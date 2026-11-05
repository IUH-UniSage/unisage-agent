"""Tổng hợp số liệu benchmark CE trên bộ official (03/11/2026).

Đọc pools.jsonl, ce.jsonl, per_query.jsonl (do `evals.benchmark.retrieval_pool` ghi) trong
thư mục kết quả, ghi summary.json bên cạnh. Chạy:
    python assets/analyze.py benchmark_results/03-11-2026-ce-official
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

RES = Path(sys.argv[1])
FLOOR = 0.45
KEEP = 8
VARIANTS = ["none", "ce16", "ce40", "ce64"]
METRICS = ["recall@1", "recall@3", "recall@5", "recall@8", "hit@5", "mrr", "ndcg@8"]


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pct(vals, q):
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))] if v else None


pools = {r["key"]: r for r in rd(RES / "pools.jsonl")}
ce = {r["key"]: r for r in rd(RES / "ce.jsonl")}
pq = rd(RES / "per_query.jsonl")
vis = [r for r in pq if r["expect_visible"]]
hidden = [r for r in pq if not r["expect_visible"]]


def floored_ceiling(key, n):
    """Phần tài liệu kỳ vọng có trong N ứng viên sau sàn cosine (cái CE thực sự thấy)."""
    p = pools[key]
    exp = set(p["expected_document_ids"])
    docs = {e["document_id"] for e in p["pool"][:n] if e["score"] >= FLOOR or e["lexical_only"]}
    return len(docs & exp) / len(exp)


def agg(rows):
    out = {"n": len(rows)}
    for v in VARIANTS:
        out[v] = {m: sum(r["variants"][v][m] for r in rows) / len(rows) for m in METRICS}
    out["ceiling"] = {str(n): sum(r["in_pool"][str(n)] for r in rows) / len(rows) for n in (8, 16, 40, 64)}
    out["ceiling_floored"] = {str(n): sum(floored_ceiling(r["key"], n) for r in rows) / len(rows) for n in (8, 16, 40, 64)}
    for v in VARIANTS[1:]:
        base = [r["variants"]["none"]["recall@8"] for r in rows]
        new = [r["variants"][v]["recall@8"] for r in rows]
        out[v]["rescued@8"] = sum(1 for a, b in zip(base, new) if b > a)
        out[v]["dropped@8"] = sum(1 for a, b in zip(base, new) if b < a)
        b5 = [r["variants"]["none"]["recall@5"] for r in rows]
        n5 = [r["variants"][v]["recall@5"] for r in rows]
        out[v]["rescued@5"] = sum(1 for a, b in zip(b5, n5) if b > a)
        out[v]["dropped@5"] = sum(1 for a, b in zip(b5, n5) if b < a)
        b1 = [r["variants"]["none"]["recall@1"] for r in rows]
        n1 = [r["variants"][v]["recall@1"] for r in rows]
        out[v]["rescued@1"] = sum(1 for a, b in zip(b1, n1) if b > a)
        out[v]["dropped@1"] = sum(1 for a, b in zip(b1, n1) if b < a)
    return out


groups = {"all": vis}
for cat in ("normal", "calculation", "access"):
    groups[cat] = [r for r in vis if r["category"] == cat]
for var in ("exact", "user"):
    groups[var] = [r for r in vis if r["variant"] == var]
    for cat in ("normal", "calculation", "access"):
        groups[f"{cat}/{var}"] = [r for r in vis if r["category"] == cat and r["variant"] == var]
summary = {"groups": {g: agg(rows) for g, rows in groups.items()}}

# recall@k curves k = 1..8 (all, exact, user)
def curve(rows, v):
    res = []
    for k in range(1, 9):
        s = 0.0
        for r in rows:
            exp = set(r["expected_document_ids"])
            doc_of = {e["chunk_id"]: e["document_id"] for e in pools[r["key"]]["pool"]}
            docs = {doc_of[c] for c in r["variants"][v]["kept_ids"][:k]}
            s += len(docs & exp) / len(exp)
        res.append(s / len(rows))
    return res


summary["curves"] = {g: {v: curve(groups[g], v) for v in VARIANTS} for g in ("all", "exact", "user")}

# latency + candidates after floor
lat = defaultdict(list); svc = defaultdict(list); cands = defaultdict(list); trunc = defaultdict(int); npass = defaultdict(int)
for r in vis:
    for size, ms in r["ce_ms"].items():
        lat[size].append(ms)
        cands[size].append(r["ce_candidates"][size])
        trunc[size] += r["ce_truncated"][size]
        npass[size] += r["ce_candidates"][size]
for key, c in ce.items():
    for size, e in c["sizes"].items():
        svc[size].append(e["service_ms"])
summary["latency"] = {
    s: {
        "n": len(lat[s]), "p50": pct(lat[s], .5), "p95": pct(lat[s], .95), "p99": pct(lat[s], .99),
        "max": max(lat[s]), "mean": sum(lat[s]) / len(lat[s]),
        "service_p50": pct(svc[s], .5), "service_p95": pct(svc[s], .95),
        "cand_mean": sum(cands[s]) / len(cands[s]), "cand_p50": pct(cands[s], .5), "cand_min": min(cands[s]),
        "truncated_share": trunc[s] / npass[s] if npass[s] else 0,
    }
    for s in sorted(lat, key=int)
}
summary["latency_points"] = {s: list(zip(cands[s], lat[s])) for s in lat}

# rank distribution: none vs ce40 (first correct doc in the variant's ordering over the full pool)
def bucket(rank):
    if rank is None:
        return "không có"
    if rank <= 8:
        return str(rank)
    if rank <= 16:
        return "9–16"
    if rank <= 40:
        return "17–40"
    return "41–64"


dist = {}
for v in ("none", "ce40", "ce64"):
    c = Counter()
    for r in vis:
        rank = r["first_rank_rrf_floored"] if v == "none" else r["variants"][v].get("first_rank_full")
        c[bucket(rank)] += 1
    dist[v] = dict(c)
dist["rrf_unfloored"] = dict(Counter(bucket(r["first_rank_rrf"]) for r in vis))
summary["rank_dist"] = dist

# floor impact: expected doc in pool64 but all its chunks under floor
summary["floor_lost"] = {
    "pool64_has_doc": sum(1 for r in vis if r["in_pool"]["64"] > 0),
    "floored_has_doc": sum(1 for r in vis if r["first_rank_rrf_floored"] is not None),
}
# graph-depth pool agreement
summary["graph_pool"] = {
    "same_order": sum(1 for r in vis if r["graph_pool_same_order"]),
    "overlap_mean": sum(r["graph_pool_overlap"] for r in vis) / len(vis),
    "n": len(vis),
}
summary["leaks"] = {"hidden_queries": len(hidden), "leaks": sum(1 for r in hidden if r["leak"]),
                    "leak_keys": [r["key"] for r in hidden if r["leak"]]}

# per-question misses for ce40 (recall@8 < 1)
def row_info(r):
    p = pools[r["key"]]
    return {
        "key": r["key"], "id": r["id"], "variant": r["variant"], "category": r["category"],
        "persona": r["persona"], "text": r["text"], "expected_titles": r["expected_titles"],
        "first_rank_rrf": r["first_rank_rrf"], "first_rank_rrf_floored": r["first_rank_rrf_floored"],
        **{f"{v}_r8": r["variants"][v]["recall@8"] for v in VARIANTS},
        **{f"{v}_rank": r["variants"][v].get("first_rank_full", r["variants"][v]["first_rank"]) for v in VARIANTS},
        "none_rank": r["first_rank_rrf_floored"],
        "top_titles_rrf": [],
        "top1_cos": max((e["score"] for e in p["pool"] if not e["lexical_only"]), default=None),
    }


summary["ce40_misses"] = [row_info(r) for r in vis if r["variants"]["ce40"]["recall@8"] < 1]
summary["none_misses"] = [row_info(r) for r in vis if r["variants"]["none"]["recall@8"] < 1]
summary["changed_ce40"] = [row_info(r) for r in vis if r["variants"]["ce40"]["recall@8"] != r["variants"]["none"]["recall@8"]]
summary["focus"] = [row_info(r) for r in vis if r["id"] in ("q0054", "d0315")]
summary["all_rows"] = [row_info(r) for r in vis]
(RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
g = summary["groups"]["all"]
for v in VARIANTS:
    print(v, {m: round(x, 3) for m, x in g[v].items()})
print("ceiling", g["ceiling"], g["ceiling_floored"])
print("latency", {s: (l["p50"], l["p95"], round(l["cand_mean"], 1)) for s, l in summary["latency"].items()})
print("leaks", summary["leaks"]["leaks"], "/", summary["leaks"]["hidden_queries"])
