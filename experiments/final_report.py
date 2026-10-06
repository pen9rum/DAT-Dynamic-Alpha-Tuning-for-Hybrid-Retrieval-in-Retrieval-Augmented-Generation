"""Aggregate runs/final/<split>/<dataset>/per_query.parquet into every paper number (tables A-E).

  python -m experiments.final_report --split test
"""
import argparse
import json

import numpy as np
import pandas as pd

from sedat.config import load_config, resolve, write_json
from sedat.m3.cache import build_m3_cache
from sedat.m3b.evaluate import paired_diff

DATASETS = ["nq", "fiqa", "hotpotqa", "musique"]
TOP1 = ["sparse", "dense", "global", "dat", "endpoint_judge", "union", "top1_frontier"]
TOP5 = ["sparse", "dense", "global", "dat", "endpoint_judge", "top1_frontier", "slate"]
PRIMARY5 = {"nq": "recall@5", "fiqa": "recall@5", "hotpotqa": "allsupport@5", "musique": "allsupport@5"}
METRICS5 = {"nq": ["recall@5", "ndcg@5", "hit@1"], "fiqa": ["recall@5", "ndcg@5", "hit@1"],
            "hotpotqa": ["allsupport@5", "recall@5", "ndcg@5", "allrel@10", "hit@1"],
            "musique": ["allsupport@5", "recall@5", "ndcg@5", "allrel@10", "hit@1"]}
EPS = 1e-12


def rec(m, g, o):
    return float((m - g) / (o - g)) if o - g > EPS else None


def dist(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90))}


def summarize(ds, df):
    n = len(df)
    S = {"n": n, "alpha_global": float(df["global|alpha"].iloc[0])}
    # ---- Table A: top-1
    o1 = df["oracle1|hit@1"].mean()
    g1 = df["global|hit@1"].mean()
    S["top1"] = {m: {"hit@1": float(df[f"{m}|hit@1"].mean()), "recall@5": float(df[f"{m}|recall@5"].mean())}
                 for m in TOP1}
    S["top1"]["oracle"] = {"hit@1": float(o1)}
    S["top1_recovery"] = {m: rec(df[f"{m}|hit@1"].mean(), g1, o1) for m in ["dat", "endpoint_judge", "union", "top1_frontier"]}
    S["top1_paired_hit@1"] = {f"top1_frontier - {b}": paired_diff(df["top1_frontier|hit@1"], df[f"{b}|hit@1"])
                              for b in ["dat", "endpoint_judge", "global", "union"]}
    S["top1_paired_hit@1"]["dat - global"] = paired_diff(df["dat|hit@1"], df["global|hit@1"])
    S["top1_paired_hit@1"]["oracle - top1_frontier"] = paired_diff(df["oracle1|hit@1"], df["top1_frontier|hit@1"])
    ad = (df["top1_frontier|alpha"] - df["global|alpha"]).abs() > EPS
    dd = df["top1_frontier|hit@1"] - df["global|hit@1"]
    S["top1_frontier_actions"] = {"coverage": float(ad.mean()), "benefit": float((dd[ad] > 0).mean()) if ad.any() else None,
                                  "harm": float((dd[ad] < 0).mean()) if ad.any() else None}
    # ---- Table B: top-5
    mets, prim = METRICS5[ds], PRIMARY5[ds]
    S["top5_primary"] = prim
    S["top5"] = {m: {k: float(df[f"{m}|{k}"].mean()) for k in mets} for m in TOP5}
    S["top5"]["oracle"] = {k: float(df[f"oracle5|{k}"].mean()) if f"oracle5|{k}" in df else
                           (float(o1) if k == "hit@1" else None) for k in mets}
    S["top5_recovery"] = {m: {k: rec(S["top5"][m][k], S["top5"]["global"][k], S["top5"]["oracle"][k])
                              for k in mets if S["top5"]["oracle"][k] is not None}
                          for m in ["dat", "endpoint_judge", "top1_frontier", "slate"]}
    S["top5_paired"] = {k: {f"slate - {b}": paired_diff(df[f"slate|{k}"], df[f"{b}|{k}"])
                            for b in ["top1_frontier", "global", "dat", "endpoint_judge"]} for k in mets}
    ch = df["slate|slate"].apply(tuple) != df["global|slate"].apply(tuple)
    d5 = df[f"slate|{prim}"] - df[f"global|{prim}"]
    h5 = df["slate|hit@1"] - df["global|hit@1"]
    S["slate_judge"] = {"changed_vs_global": float(ch.mean()),
                        "benefit": float((d5[ch] > EPS).mean()), "neutral": float((d5[ch].abs() <= EPS).mean()),
                        "harm": float((d5[ch] < -EPS).mean()),
                        "hit@1_benefit": float((h5[ch] > EPS).mean()), "hit@1_harm": float((h5[ch] < -EPS).mean()),
                        "complementary_win": float(df["slate|complementary_win"].mean()),
                        "interior_selected": float(df["slate|interior_selected"].mean())}
    # ---- Table E: structure
    e1 = df["d1|endpoint_oracle"].mean()
    S["structure_top1"] = {"frac_interior_only_rank1_winner": float(df["d1|has_interior_only"].mean()),
                           "frac_interior_only_gold_and_global_wrong": float(df["d1|interior_gold_global_wrong"].mean()),
                           "global": float(g1), "endpoint_oracle": float(e1), "exact_oracle": float(o1),
                           "interior_share_of_oracle_gap": float((o1 - e1) / (o1 - g1)) if o1 - g1 > EPS else None}
    S["structure_top5"] = {"frac_interior_slate_distinct_from_both_endpoints": float(df["d5|has_interior_slate"].mean()),
                           "frac_interior_slate_adds_gold_missing_from_both_endpoints": float(df["d5|interior_adds_gold"].mean()),
                           "frac_interior_slate_more_gold_than_either_endpoint": float(df["d5|interior_more_gold_than_either"].mean())}
    for k in ("recall@5", "ndcg@5", "allsupport@5"):
        g_, e_, o_ = df[f"global|{k}"].mean(), df[f"d5|endpoint_oracle|{k}"].mean(), df[f"oracle5|{k}"].mean()
        S["structure_top5"][k] = {"global": float(g_), "endpoint_oracle": float(e_), "exact_oracle": float(o_),
                                  "interior_share_of_oracle_gap": float((o_ - e_) / (o_ - g_)) if o_ - g_ > EPS else None}
    # ---- Table D: efficiency
    eff = {}
    for name, c, docs in [("dat", "dat", None), ("endpoint_judge", "endpoint_judge", "eff|n_endpoint_docs"),
                          ("union", "union", "eff|n_union_docs"), ("top1_frontier", "top1_frontier", "eff|n_exact_docs"),
                          ("slate", "slate", "eff|n_slate_docs")]:
        e = {"docs_shown": float(2.0 if docs is None else df[docs].mean()),
             "calls_per_query": float(df[f"eff|{c}|calls"].mean()),
             "input_tokens": float(df[f"eff|{c}|in"].mean()), "output_tokens": float(df[f"eff|{c}|out"].mean()),
             "usd_per_query": float(df[f"eff|{c}|cost"].mean()), "usd_total": float(df[f"eff|{c}|cost"].sum()),
             "judge_latency_s_sequential": dist(df[f"eff|{c}|lat"])}
        eff[name] = e
    eff["top1_frontier"]["frontier_outcomes"] = dist(df["eff|n_exact_docs"])
    eff["top1_frontier"]["construction_ms"] = dist(df["eff|t_top1_frontier"] * 1000)
    eff["slate"]["frontier_outcomes"] = dist(df["eff|n_slates"])
    eff["slate"]["construction_ms"] = dist(df["eff|t_top5_frontier"] * 1000)
    eff["slate"]["unique_docs_shown"] = float(df["eff|n_slate_docs"].mean())
    S["efficiency"] = eff
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    out = {}
    for ds in DATASETS:
        p = resolve(f"runs/final/{args.split}/{ds}/per_query.parquet")
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        out[ds] = summarize(ds, df)
        out[ds]["run_meta"] = json.load(open(resolve(f"runs/final/{args.split}/{ds}/run_meta.json"), encoding="utf-8"))
        if ds == "musique":
            L = build_m3_cache(load_config("configs/m3/musique.yaml")).labels
            hops = df["qid"].map(lambda q: int(L.at[q, "n_hops"]))
            out[ds]["by_hops"] = {int(h): {"n": int((hops == h).sum()),
                                           **{f"{m}|{k}": float(df.loc[hops == h, f"{m}|{k}"].mean())
                                              for m in ["global", "dat", "top1_frontier", "slate"]
                                              for k in ["allsupport@5", "recall@5", "hit@1"]},
                                           "oracle|allsupport@5": float(df.loc[hops == h, "oracle5|allsupport@5"].mean())}
                                  for h in sorted(hops.unique())}
    write_json(resolve(f"runs/final/{args.split}_summary.json"), out)
    for ds, S in out.items():
        print("==", ds, "n", S["n"])
        print("  top1", {m: round(v["hit@1"], 4) for m, v in S["top1"].items()})
        print("  rec1", {m: (round(v, 3) if v is not None else None) for m, v in S["top1_recovery"].items()})
        for k, v in S["top1_paired_hit@1"].items():
            print(f"    {k:32s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p_signflip']:.4f}")
        p5 = S["top5_primary"]
        print("  top5", p5, {m: round(v[p5], 4) if v[p5] is not None else None for m, v in S["top5"].items()})
        for k, v in S["top5_paired"][p5].items():
            print(f"    {k:32s} {v['mean']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p_signflip']:.4f}")


if __name__ == "__main__":
    main()
