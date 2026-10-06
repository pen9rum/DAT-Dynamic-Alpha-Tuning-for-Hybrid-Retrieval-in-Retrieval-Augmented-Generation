"""Top-5 Slate Frontier-DAT (validation only; test untouched). Judge: openai/gpt-4.1-mini.

For each query: the exact top-5 Actionable Ranking Frontier (from m3c_topk_geometry); every unique document
shown once under a neutral random ID (D1, D2, ...); every executable slate shown only as an ordered list of
IDs under a random slate ID (S1, S2, ...). No alpha, interval, source, endpoint or global identity is shown.
The judge picks ONE slate (listwise, whole-slate judgment). Two independent passes (re-randomized document IDs
and slate order; within-slate order never changed). Consensus: same slate in both passes -> execute it at the
feasible alpha closest to alpha_g; otherwise abstain to alpha_g. One-pass result kept for comparison.

  python -m experiments.m3c_topk_slate --dataset nq
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import hashlib
import json
import pickle
import re

import numpy as np
import pandas as pd

from sedat.config import get_dotted, load_config, resolve, setup_logging, write_json
from sedat.m3.cache import build_m3_cache
from sedat.m3b.bench import load_bench
from sedat.m3b.evaluate import paired_diff
from sedat.m3b.llm import SmallLLM
from sedat.m3c.exact_frontier import WIDTH_EPS, order_at
from sedat.m3c.sensor import short
from sedat.retrieval.normalization import normalize

VERSION = "slate-v1"
SYSTEM = ("You are a retrieval evaluator. You will see a query, a set of passages with IDs, and several candidate "
          "retrieval results (slates). Each slate is an ordered list of 5 passage IDs, rank 1 first.\n"
          "Which retrieval slate is most likely to provide the evidence needed to answer the query correctly? "
          "Consider the slate as a whole: evidence coverage, complementary information, redundancy, and ranking position.\n"
          "Return JSON only, no explanation: {\"best\": \"<slate ID>\"}")


def build_prompt(qid, query, slates, texts, pass_no):
    rng = np.random.default_rng(int(hashlib.sha1(f"{VERSION}|{qid}|pass{pass_no}".encode()).hexdigest()[:8], 16))
    docs = sorted({d for sl in slates for d in sl})
    doc_ids = {d: f"D{i + 1}" for i, d in enumerate([docs[j] for j in rng.permutation(len(docs))])}
    order = rng.permutation(len(slates))
    lab = {f"S{i + 1}": slates[j] for i, j in enumerate(order)}
    body = f"QUERY:\n{query}\n\nPASSAGES:\n" + "\n\n".join(
        f"{doc_ids[d]}: {texts[d]}" for d in sorted(docs, key=lambda x: int(doc_ids[x][1:])))
    body += "\n\nSLATES (ordered, rank 1 first):\n" + "\n".join(
        f"{L}: [{', '.join(doc_ids[d] for d in sl)}]" for L, sl in lab.items())
    body += "\n\nWhich slate is most likely to provide the evidence needed to answer the query correctly?\nJSON:"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}], lab


def parse(text, lab):
    m = re.search(r"\{.*?\}", re.sub(r"```(?:json)?", "", text or "", flags=re.I), re.S)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    key = str(d.get("best", "")).strip().upper() if isinstance(d, dict) else ""
    return lab.get(key)


def feasible(intervals, target):
    best, dist = None, np.inf
    for a, b in intervals:
        x = float(np.clip(target, a + min(1e-6, (b - a) / 2), b - min(1e-6, (b - a) / 2))) if b - a > WIDTH_EPS else float(a)
        if abs(x - target) < dist - 1e-15:
            best, dist = x, abs(x - target)
    return best


def evaluate(top, gold):
    t5 = list(top[:5])
    hits = [x in gold for x in t5]
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), 5)))
    return {"recall@5": sum(hits) / len(gold), "ndcg@5": dcg / idcg, "hit@1": float(hits[0]),
            "allsupport@5": float(gold <= set(t5)), "allrel@10": float(gold <= set(top[:10]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2))   # 1: one-pass only (no consensus spend)
    ap.add_argument("--geometry", default=None)                        # default: latest topk_geometry_* dir
    args = ap.parse_args()
    setup_logging()
    ds = args.dataset
    out = resolve(f"runs/milestone3c/{ds}/topk_slate_{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True)
    geo = args.geometry or sorted(glob.glob(str(resolve("runs/milestone3c/topk_geometry_*"))))[-1]
    G = pickle.load(open(f"{geo}/{ds}_k5_frontiers.pkl", "rb"))
    ag = G["alpha_global"]
    FR = G["frontiers"]
    cfg = load_config(f"configs/m3/{ds}.yaml")
    cache = build_m3_cache(cfg)
    B = load_bench(ds)
    va = [int(r) for r in B.rows("val")]
    pq = sorted(glob.glob(str(resolve(f"runs/milestone3c/{ds}/exact_frontier_*/per_query.parquet"))))[-1]
    prev = pd.read_parquet(pq).set_index("qid")
    corp = cache.corpus.set_index("doc_id")
    need = sorted({d for f in FR.values() for sl in f["slates"] for d in sl})
    missing = [x for x in need if x not in corp.index or not corp.at[x, "text"]]
    texts = {}
    if missing:
        from sedat.retrieval.pyserini_backend import LuceneBM25, setup_java
        setup_java(get_dotted(cfg, "paths.java_home"))
        lb = LuceneBM25(cfg["retrieval"]["sparse_index"])
        for x in missing:
            y = lb.doc(x)
            texts[x] = short(y["title"], y["text"])
    for x in need:
        texts.setdefault(x, short(corp.at[x, "title"], corp.at[x, "text"]))
    llm = SmallLLM(args.model, args.budget, max_tokens=20)

    def judge(r):
        qid = B.qids[r]
        f = FR[qid]
        if len(f["slates"]) < 2:
            return r, (None, None, 0, 0.0)
        picks, tok, cost = [], 0, 0.0
        for p in range(1, args.passes + 1):
            msgs, lab = build_prompt(qid, B.query[r], f["slates"], texts, p)
            o = llm.call(llm.key(f"{VERSION}|pass{p}", qid, "none", "static", msgs), msgs)
            picks.append(parse(o["text"], lab))
            tok += o["prompt_tokens"] + o["completion_tokens"]
            cost += o["cost"]
        return r, (picks[0], picks[1] if len(picks) > 1 else None, tok, cost)

    with cf.ThreadPoolExecutor(args.workers) as ex:
        J = dict(ex.map(judge, va))
    print(f"judging done: spent ${llm.spent_new:.4f} {llm.stats}", flush=True)

    rows = []
    for r in va:
        qid = B.qids[r]
        p = cache.pools[qid]
        s = normalize(p.sparse_score, p.sparse_rank, "minmax")
        d = normalize(p.dense_score, p.dense_rank, "minmax")
        ids = p.doc_id
        gold = FR[qid]["gold"]
        top = lambda a: [ids[i] for i in order_at(s, d, a)[:10]]
        p1, p2, tok, cost = J[r]
        a1 = feasible(FR[qid]["intervals"][p1], ag) if p1 is not None else ag
        a2 = feasible(FR[qid]["intervals"][p1], ag) if (p1 is not None and p1 == p2) else ag
        if p1 is not None:
            assert tuple(top(a1)[:5]) == tuple(p1), "selected slate is not executable at its alpha"
        alphas = {"sparse": 0.0, "dense": 1.0, "global": ag, "dat_endpoints": float(prev.at[qid, "alpha_endpoints"]),
                  "top1_frontier": float(prev.at[qid, "alpha_exact_frontier"]), "slate_onepass": a1, "slate_consensus": a2}
        rec = {"qid": qid, "n_slates": len(FR[qid]["slates"]), "agree": (p1 == p2) if p1 is not None else None,
               "p1_valid": p1 is not None or len(FR[qid]["slates"]) < 2, "p2_valid": p2 is not None or len(FR[qid]["slates"]) < 2,
               "tokens": tok, "cost": cost}
        for name, a in alphas.items():
            for m, v in evaluate(top(a), gold).items():
                rec[f"{name}|{m}"] = v
            rec[f"{name}|alpha"] = a
            rec[f"{name}|slate"] = tuple(top(a)[:5])
        orc = [evaluate(list(sl), gold) for sl in FR[qid]["slates"]]
        for m in ("recall@5", "ndcg@5", "hit@1", "allsupport@5"):
            rec[f"oracle|{m}"] = max(o[m] for o in orc)
        e0, e1 = tuple(top(0.0)[:5]), tuple(top(1.0)[:5])
        ng = lambda sl: len(gold & set(sl))
        sel = rec["slate_consensus|slate"]
        rec["complementary_win"] = bool(abs(a2 - ag) > 1e-12 and ng(sel) > max(ng(e0), ng(e1)))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_query.parquet", index=False)

    primary = "allsupport@5" if ds in ("hotpotqa", "musique") else "recall@5"
    pols = ["sparse", "dense", "global", "dat_endpoints", "top1_frontier", "slate_onepass", "slate_consensus"]
    mets = ["recall@5", "ndcg@5", "hit@1", "allsupport@5"] + (["allrel@10"] if ds in ("hotpotqa", "musique") else [])
    table = {p_: {m: float(df[f"{p_}|{m}"].mean()) for m in mets} for p_ in pols}
    table["oracle"] = {m: float(df[f"oracle|{m}"].mean()) for m in mets if f"oracle|{m}" in df}
    rec_ = lambda p_, m: (table[p_][m] - table["global"][m]) / (table["oracle"][m] - table["global"][m]) \
        if table["oracle"][m] - table["global"][m] > 1e-12 else None
    paired = {}
    for m in (primary, "ndcg@5", "hit@1") + (("recall@5",) if ds in ("hotpotqa", "musique") else ()):
        paired[m] = {f"slate_consensus - {b}": paired_diff(df[f"slate_consensus|{m}"], df[f"{b}|{m}"])
                     for b in ("global", "dat_endpoints", "top1_frontier", "slate_onepass")}
        paired[m]["slate_onepass - top1_frontier"] = paired_diff(df[f"slate_onepass|{m}"], df[f"top1_frontier|{m}"])
        paired[m]["oracle - slate_consensus"] = paired_diff(df[f"oracle|{m}"], df[f"slate_consensus|{m}"])
    multi = df[df["n_slates"] >= 2]
    diff_top1 = df[df["slate_consensus|slate"] != df["top1_frontier|slate"]]
    dd = diff_top1[f"slate_consensus|{primary}"] - diff_top1[f"top1_frontier|{primary}"]
    res = {"dataset": ds, "model": args.model, "alpha_global": ag, "n_val": len(df), "primary_metric": primary,
           "table": table, "recovery_vs_exact_oracle": {p_: {m: rec_(p_, m) for m in table["oracle"]} for p_ in pols},
           "paired": paired,
           "stability": {"queries_judged": int(len(multi)), "two_pass_agreement": float(multi["agree"].mean()),
                         "abstention_rate (disagree)": float(1 - multi["agree"].mean()),
                         "parse_valid_pass1": float(multi["p1_valid"].mean()), "parse_valid_pass2": float(multi["p2_valid"].mean()),
                         "alpha_change_rate_onepass": float((df["slate_onepass|alpha"] - ag).abs().gt(1e-12).mean()),
                         "alpha_change_rate_consensus": float((df["slate_consensus|alpha"] - ag).abs().gt(1e-12).mean())},
           "vs_top1_frontier": {"frac_different_top5": float(len(diff_top1) / len(df)),
                                "benefit": float((dd > 1e-12).mean()) if len(dd) else None,
                                "neutral": float((dd.abs() <= 1e-12).mean()) if len(dd) else None,
                                "harm": float((dd < -1e-12).mean()) if len(dd) else None},
           "complementary_winning_slates": float(df["complementary_win"].mean()),
           "frontier_size": {"slates_mean": float(df["n_slates"].mean()), "slates_max": int(df["n_slates"].max())},
           "cost": {"new_spend_usd": llm.spent_new, "stats": llm.stats, "tokens_per_judged_query": float(multi["tokens"].mean()),
                    "usd_per_judged_query": float(multi["cost"].mean()), "overflow_queries": 0}}
    ex_ = df[df["complementary_win"]].head(3)
    res["examples"] = []
    for _, x in ex_.iterrows():
        res["examples"].append({"qid": x["qid"], "query": B.query[B.qids.tolist().index(x["qid"])],
                                "gold_in_selected": int(len(FR[x["qid"]]["gold"] & set(x["slate_consensus|slate"]))),
                                "gold_in_alpha0": int(len(FR[x["qid"]]["gold"] & set(x["sparse|slate"]))),
                                "gold_in_alpha1": int(len(FR[x["qid"]]["gold"] & set(x["dense|slate"]))),
                                "selected_alpha": float(x["slate_consensus|alpha"]),
                                "selected_titles": [corp.at[d, "title"] if d in corp.index else d for d in x["slate_consensus|slate"]]})
    write_json(out / "topk_slate_results.json", res)
    print(json.dumps({p_: {m: round(v, 4) for m, v in t.items()} for p_, t in table.items()}, indent=0))
    print("paired", {k: (round(v["mean"], 4), [round(c, 4) for c in v["ci95"]]) for k, v in paired[primary].items()})
    print("stability", res["stability"], "vs_top1", res["vs_top1_frontier"])
    print(f"out: {out}")


if __name__ == "__main__":
    main()
