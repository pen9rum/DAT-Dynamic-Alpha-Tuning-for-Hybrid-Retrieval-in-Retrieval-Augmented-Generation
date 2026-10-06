"""Exact top-k Actionable Ranking Frontier: geometry and oracle diagnostics (validation only; no LLM calls).

For k in {1, 3, 5}: number of executable ordered slates per query, unique documents on the frontier,
interior slates (different from both endpoint slates), interior slates carrying relevant evidence missing from
both endpoint slates, and oracle ladders  global -> endpoint-only oracle (best of alpha = 0, 1, alpha_g)
-> exact k-frontier oracle  for cut-k metrics (k = 1: Hit@1; k = 3, 5: Recall@k, nDCG@k, all-support@k).
Also estimates the k = 5 judge prompt size (to set an overflow rule before any LLM call).
Saves per-query k = 5 frontiers for the slate-judge experiment.

  python -m experiments.m3c_topk_geometry
"""
import argparse
import datetime as dt
import pickle

import numpy as np
import pandas as pd

from sedat.config import load_config, resolve, setup_logging, write_json
from sedat.m3.cache import build_m3_cache
from sedat.m3b.bench import load_bench
from sedat.m3c.topk_frontier import topk_frontier
from sedat.retrieval.normalization import normalize

DATASETS = ["nq", "fiqa", "hotpotqa"]
KS = [1, 3, 5]


def metrics(slate_ids, gold, k):
    top = list(slate_ids)[:k]
    hits = [x in gold for x in top]
    dcg = sum(1 / np.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(gold), k)))
    return {"hit@1": float(hits[0]) if hits else 0.0, f"recall@{k}": sum(hits) / len(gold),
            f"ndcg@{k}": dcg / idcg if idcg else 0.0, f"allsupport@{k}": float(gold <= set(top))}


def stats(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()
    setup_logging()
    out = resolve(f"runs/milestone3c/topk_geometry_{dt.datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True)
    res = {}
    for ds in args.datasets:
        cfg = load_config(f"configs/m3/{ds}.yaml")
        cache = build_m3_cache(cfg)
        B = load_bench(ds)
        ag = float(B.alphas[B.g])
        va = [int(r) for r in B.rows("val")]
        labels = cache.labels
        corp = cache.corpus.set_index("doc_id")
        tlen = corp["text"].str.len()
        per_k = {k: [] for k in KS}
        saved = {}
        for r in va:
            qid = B.qids[r]
            p = cache.pools[qid]
            s = normalize(p.sparse_score, p.sparse_rank, "minmax")
            d = normalize(p.dense_score, p.dense_rank, "minmax")
            ids = p.doc_id
            gold = set(labels.at[qid, "gold_doc_ids"])
            for k in KS:
                F = topk_frontier(s, d, k)
                to_ids = lambda sl: tuple(ids[i] for i in sl)
                e0, e1, eg = F.slate_at(0.0), F.slate_at(1.0), F.slate_at(ag)
                ep_docs = set(to_ids(e0)) | set(to_ids(e1))
                interior = [sl for sl in F.slates if sl not in (e0, e1)]
                m_all = {sl: metrics(to_ids(sl), gold, k) for sl in F.slates}
                keys = list(m_all[F.slates[0]])
                rec = {"qid": qid, "n_slates": len(F.slates),
                       "n_unique_docs": len({i for sl in F.slates for i in sl}),
                       "has_interior_slate": bool(interior),
                       "interior_new_gold": bool(any((set(to_ids(sl)) - ep_docs) & gold for sl in interior)),
                       **{f"global_{m}": m_all[eg][m] for m in keys},
                       **{f"endpoint_oracle_{m}": max(m_all[e0][m], m_all[e1][m], m_all[eg][m]) for m in keys},
                       **{f"exact_oracle_{m}": max(v[m] for v in m_all.values()) for m in keys}}
                if k == 5:
                    udocs = sorted({ids[i] for sl in F.slates for i in sl})
                    rec["prompt_chars_est"] = int(sum(min(int(tlen.get(x, 450)), 450) + 20 for x in udocs) + 30 * len(F.slates))
                    saved[qid] = {"slates": [to_ids(sl) for sl in F.slates],
                                  "intervals": {to_ids(sl): F.intervals[sl] for sl in F.slates}, "gold": gold}
                per_k[k].append(rec)
        with open(out / f"{ds}_k5_frontiers.pkl", "wb") as f:
            pickle.dump({"alpha_global": ag, "frontiers": saved}, f)
        res[ds] = {"alpha_global": ag, "n_val": len(va)}
        for k in KS:
            df = pd.DataFrame(per_k[k])
            mets = [c[len("global_"):] for c in df.columns if c.startswith("global_")]
            lad = {}
            for m in mets:
                g, e, o = df[f"global_{m}"].mean(), df[f"endpoint_oracle_{m}"].mean(), df[f"exact_oracle_{m}"].mean()
                lad[m] = {"global": float(g), "endpoint_oracle": float(e), "exact_oracle": float(o),
                          "interior_share_of_gap": float((o - e) / (o - g)) if o - g > 1e-12 else None}
            res[ds][f"k={k}"] = {"slates_per_query": stats(df["n_slates"]),
                                 "unique_docs_on_frontier": stats(df["n_unique_docs"]),
                                 "frac_interior_slate_differs_from_both_endpoints": float(df["has_interior_slate"].mean()),
                                 "frac_interior_slate_with_relevant_doc_missing_from_both_endpoints": float(df["interior_new_gold"].mean()),
                                 "oracle_ladder": lad}
            if k == 5:
                res[ds]["k=5"]["prompt_chars_est"] = stats(df["prompt_chars_est"])
                big = df.sort_values("n_slates", ascending=False).head(5)
                res[ds]["k=5"]["largest_queries"] = big[["qid", "n_slates", "n_unique_docs", "prompt_chars_est"]].to_dict("records")
                for thr in (10, 15, 20, 26):
                    res[ds]["k=5"][f"frac_queries_with_more_than_{thr}_slates"] = float((df["n_slates"] > thr).mean())
            print(ds, f"k={k}", res[ds][f"k={k}"]["slates_per_query"], {m: (round(v['global'], 3), round(v['endpoint_oracle'], 3),
                  round(v['exact_oracle'], 3)) for m, v in res[ds][f"k={k}"]["oracle_ladder"].items()}, flush=True)
    write_json(out / "topk_geometry.json", res)
    print(f"out: {out}")


if __name__ == "__main__":
    main()
