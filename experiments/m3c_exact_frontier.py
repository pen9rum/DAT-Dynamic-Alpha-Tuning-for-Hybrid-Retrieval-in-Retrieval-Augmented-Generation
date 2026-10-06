"""Exact Alpha Frontier validation (NQ / FiQA / HotpotQA validation only; test untouched).

Replaces the 11-point frontier with the exact upper envelope (sedat.m3c.exact_frontier); everything
else is frozen: judge (default openai/gpt-4.1-mini), prompt, blinding, two-order averaging, decision
rule, delta grid, 2-fold cross-fitting. Exact Frontier-DAT executes alpha_hat = clip(alpha_g, a, b)
on the chosen document's feasible interval (nudged inside when clip lands on a tie point).

Conditions: global, DAT endpoints, 11-grid Frontier-DAT, exact Frontier-DAT, exposure-matched top-m
union with m = |exact frontier| per query, grid oracle, exact continuous-alpha oracle.

  python -m experiments.m3c_exact_frontier --dataset nq
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from sedat.config import get_dotted, load_config, resolve, setup_logging, write_json
from sedat.m3.cache import build_m3_cache
from sedat.m3b.bench import load_bench
from sedat.m3b.evaluate import paired_diff
from sedat.m3b.llm import SmallLLM
from sedat.m3c.exact_frontier import exact_frontier, order_at
from sedat.m3c.frontier import FRONTIER_VERSION, build_call, parse
from sedat.m3c.sensor import short
from sedat.retrieval.normalization import normalize

DELTAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, np.inf]
EPS = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    setup_logging()
    out = resolve(f"runs/milestone3c/{args.dataset}/exact_frontier_{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True)
    cfg = load_config(f"configs/m3/{args.dataset}.yaml")
    cache = build_m3_cache(cfg)
    B = load_bench(args.dataset)
    g, R, alphas = B.g, B.R["hit@1"], B.alphas
    A, ag = len(alphas), float(alphas[B.g])
    va = [int(r) for r in B.rows("val")]
    labels = cache.labels
    t = pd.read_parquet(cache.dir / "reward_table_minmax.parquet", columns=["qid", "alpha_idx", "top_doc_ids"])
    tops = t[t["qid"].isin(set(B.qids[va]))].pivot(index="qid", columns="alpha_idx", values="top_doc_ids")
    pools = cache.pools

    Q = {}
    grid_mismatch = 0
    for r in va:
        qid = B.qids[r]
        p = pools[qid]
        s = normalize(p.sparse_score, p.sparse_rank, "minmax")
        d = normalize(p.dense_score, p.dense_rank, "minmax")
        ids = p.doc_id
        tl = [list(tops.at[qid, a]) for a in range(A)]
        for a in range(A):                                    # exact ranking == cached reward-table ranking
            grid_mismatch += int(ids[order_at(s, d, alphas[a])[0]] != tl[a][0])
        F = exact_frontier(s, d)
        exact = [ids[i] for i in F.docs]
        grid = list(dict.fromkeys(x[0] for x in tl))
        gdoc = tl[g][0]
        s_ord = list(ids[np.argsort(p.sparse_rank, kind="stable")])
        d_ord = list(ids[np.argsort(p.dense_rank, kind="stable")])
        endpoints = list(dict.fromkeys([gdoc, s_ord[0], d_ord[0]]))
        inter = [x for pair in zip(s_ord, d_ord) for x in pair]
        union = list(dict.fromkeys([gdoc] + inter))[:max(len(exact), len(endpoints))]
        gold = set(labels.at[qid, "gold_doc_ids"])
        ep_docs = {ids[order_at(s, d, 0.0)[0]], ids[order_at(s, d, 1.0)[0]]}
        Q[r] = {"s": s, "d": d, "ids": ids, "F": F, "idx": {x: i for i, x in enumerate(ids)}, "tl": tl,
                "rank": [{x: k + 1 for k, x in enumerate(v)} for v in tl], "gold": gold, "gdoc": gdoc,
                "sets": {"endpoints": endpoints, "grid_frontier": grid, "exact_frontier": exact, "union": union},
                "interior_only": [x for x in exact if x not in ep_docs]}
    assert ids[order_at(s, d, ag)[0]] == gdoc

    need = sorted({x for q in Q.values() for c in q["sets"].values() for x in c})
    corp = cache.corpus.set_index("doc_id")
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

    llm = SmallLLM(args.model, args.budget, max_tokens=60)
    TAG = {"endpoints": f"{FRONTIER_VERSION}|endpoints", "union": f"{FRONTIER_VERSION}|union",
           "grid_frontier": FRONTIER_VERSION, "exact_frontier": FRONTIER_VERSION}   # identical sets -> cache

    def judge(item):
        r, cond = item
        docs = Q[r]["sets"][cond]
        if len(docs) < 2:
            return item, None
        ok = []
        for rev in (False, True):
            msgs, lab2doc = build_call(B.qids[r], B.query[r], docs, texts, reverse=rev)
            o = llm.call(llm.key(TAG[cond] + ("|rev" if rev else ""), B.qids[r], "none", "static", msgs), msgs)
            pp = parse(o["text"], lab2doc)
            if pp is not None:
                ok.append(pp)
        return item, ({x: float(np.mean([y[x] for y in ok])) for x in docs} if ok else None)

    conds = ["endpoints", "grid_frontier", "exact_frontier", "union"]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        J = dict(ex.map(judge, [(r, c) for r in va for c in conds]))
    print(f"judging done: new spend ${llm.spent_new:.4f} {llm.stats}", flush=True)

    # ---------------- decisions: returns (executed alpha value, rank-1 doc at that alpha, chosen doc or None)
    def top_doc(r, alpha):
        q = Q[r]
        return q["ids"][order_at(q["s"], q["d"], alpha)[0]]

    def decide(r, cond, delta):
        q = Q[r]
        sc = J[(r, cond)]
        if sc is None or max(sc.values()) - sc[q["gdoc"]] < max(delta, EPS):
            return ag, q["gdoc"], None
        best = max(sc.values())
        tied = [x for x, v in sc.items() if v >= best - EPS and x != q["gdoc"]]
        if cond == "exact_frontier":                          # continuous alpha: clip(alpha_g, interval)
            opts = [(abs(q["F"].feasible_alpha(q["idx"][x], ag) - ag), q["F"].feasible_alpha(q["idx"][x], ag), x)
                    for x in tied]
            _, a_hat, x = min(opts)
            return a_hat, top_doc(r, a_hat), x
        b = min((q["rank"][a].get(x, 10 ** 6), abs(a - g), a, x) for x in tied for a in range(A))
        if b[0] >= 10 ** 6:
            return ag, q["gdoc"], None
        return float(alphas[b[2]]), q["tl"][b[2]][0], b[3]

    def reward(r, doc):
        return float(doc in Q[r]["gold"])

    def allrel10(r, alpha):
        q = Q[r]
        top = set(q["ids"][order_at(q["s"], q["d"], alpha)[:10]])
        return float(q["gold"] <= top)

    perm = np.random.default_rng(0).permutation(va)
    halves = [np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])]
    rg = np.array([reward(r, Q[r]["gdoc"]) for r in va])
    ro_grid = R[va].max(1)
    ro_exact = np.array([float(bool(Q[r]["gold"] & set(Q[r]["sets"]["exact_frontier"]))) for r in va])

    def rec(rp, ro):
        den = ro.mean() - rg.mean()
        return float((rp.mean() - rg.mean()) / den) if den > EPS else None

    def boot(rp, ro, n=2000):
        rng = np.random.default_rng(0)
        v = [x for x in ((lambda s_: (rp[s_].mean() - rg[s_].mean()) / (ro[s_].mean() - rg[s_].mean())
                          if ro[s_].mean() - rg[s_].mean() > EPS else None)(rng.integers(0, len(va), len(va)))
                         for _ in range(n)) if x is not None]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    res = {"dataset": args.dataset, "model": args.model, "alpha_global": ag, "n_val": len(va),
           "grid_vs_exact_ranking_mismatches": grid_mismatch, "conditions": {}}
    dec = {}
    for cond in conds:
        chosen, picked = {}, []
        for i in (0, 1):
            sel, ev = halves[i], halves[1 - i]
            best = max(((np.mean([reward(int(r), decide(int(r), cond, dl)[1]) for r in sel]),
                         -dl if np.isfinite(dl) else -99, dl) for dl in DELTAS))
            picked.append(str(best[2]))
            for r in ev:
                chosen[int(r)] = decide(int(r), cond, best[2])
        dec[cond] = chosen
        rp = np.array([reward(r, chosen[r][1]) for r in va])
        ad = np.array([abs(chosen[r][0] - ag) > 1e-12 for r in va])
        gold_in = np.mean([bool(Q[r]["gold"] & set(Q[r]["sets"][cond])) for r in va])
        entry = {"hit@1": float(rp.mean()), "recovery_vs_grid_oracle": rec(rp, ro_grid),
                 "recovery_vs_grid_oracle_ci": boot(rp, ro_grid),
                 "recovery_vs_exact_oracle": rec(rp, ro_exact), "coverage": float(ad.mean()),
                 "precision": float((rp[ad] > rg[ad]).mean()) if ad.any() else None,
                 "harm": float((rp[ad] < rg[ad]).mean()) if ad.any() else None,
                 "gold_in_candidates": float(gold_in),
                 "mean_candidates": float(np.mean([len(Q[r]["sets"][cond]) for r in va])), "picked_delta": picked}
        if args.dataset == "hotpotqa":
            entry["allrel@10"] = float(np.mean([allrel10(r, chosen[r][0]) for r in va]))
        res["conditions"][cond] = entry
        print(cond, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()
                     if k not in ("recovery_vs_grid_oracle_ci",)}, flush=True)
    hits = {c: np.array([reward(r, dec[c][r][1]) for r in va]) for c in conds}
    res["paired_hit@1"] = {f"exact - {c}": paired_diff(hits["exact_frontier"], hits[c])
                           for c in ("grid_frontier", "endpoints", "union")}
    res["global"] = {"hit@1": float(rg.mean())}
    if args.dataset == "hotpotqa":
        res["global"]["allrel@10"] = float(np.mean([allrel10(r, ag) for r in va]))
    res["oracle"] = {"grid_11": float(ro_grid.mean()), "exact_continuous": float(ro_exact.mean())}

    sizes = np.array([len(Q[r]["sets"]["exact_frontier"]) for r in va])
    gsizes = np.array([len(Q[r]["sets"]["grid_frontier"]) for r in va])
    missed = np.array([bool(set(Q[r]["sets"]["exact_frontier"]) - set(Q[r]["sets"]["grid_frontier"])) for r in va])
    changed = np.array([dec["exact_frontier"][r][2] != dec["grid_frontier"][r][2] for r in va])
    interior = [r for r in va if Q[r]["interior_only"]]
    int_gold = [r for r in interior if set(Q[r]["interior_only"]) & Q[r]["gold"]]
    useful = [r for r in int_gold if Q[r]["gdoc"] not in Q[r]["gold"]]
    sel_int = [r for r in va if dec["exact_frontier"][r][2] in Q[r]["interior_only"]]
    res["frontier_stats"] = {
        "exact_size_mean": float(sizes.mean()), "exact_size_median": float(np.median(sizes)),
        "exact_size_hist": pd.Series(sizes).value_counts().sort_index().to_dict(),
        "grid_size_mean": float(gsizes.mean()),
        "frac_exact_has_docs_missed_by_grid": float(missed.mean()),
        "frac_grid_has_docs_not_in_exact": float(np.mean([bool(set(Q[r]["sets"]["grid_frontier"]) -
                                                              set(Q[r]["sets"]["exact_frontier"])) for r in va])),
        "frac_selected_doc_changed_exact_vs_grid": float(changed.mean())}
    res["geometry"] = {
        "frac_only_endpoint_docs_on_frontier": float(1 - len(interior) / len(va)),
        "frac_has_interior_only_doc": float(len(interior) / len(va)),
        "frac_interior_only_doc_is_gold": float(len(int_gold) / len(va)),
        "P(useful interior candidate) = interior gold and global wrong": float(len(useful) / len(va)),
        "judge_selects_interior_only_doc": int(len(sel_int)),
        "interior_selection_improves_vs_global": float(np.mean([reward(r, dec["exact_frontier"][r][1]) > rg[va.index(r)]
                                                                for r in sel_int])) if sel_int else None,
        "interior_selection_hurts_vs_global": float(np.mean([reward(r, dec["exact_frontier"][r][1]) < rg[va.index(r)]
                                                             for r in sel_int])) if sel_int else None}
    res["cost"] = {"new_spend_usd": llm.spent_new, "stats": llm.stats}
    write_json(out / "exact_frontier_results.json", res)
    pd.DataFrame([{"qid": B.qids[r], "exact_size": len(Q[r]["sets"]["exact_frontier"]),
                   "grid_size": len(Q[r]["sets"]["grid_frontier"]),
                   "exact_intervals": str({Q[r]["ids"][i]: Q[r]["F"].intervals[i] for i in Q[r]["F"].docs}),
                   **{f"alpha_{c}": dec[c][r][0] for c in conds}} for r in va]).to_parquet(
        out / "per_query.parquet", index=False)
    print(f"out: {out}")


if __name__ == "__main__":
    main()
