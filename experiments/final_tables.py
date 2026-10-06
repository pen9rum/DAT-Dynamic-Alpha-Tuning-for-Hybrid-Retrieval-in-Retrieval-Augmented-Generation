"""Render paper tables A-E (markdown) from runs/final/<split>_summary.json -- no hand-copied numbers.

  python -m experiments.final_tables --split test > runs/final/tables_test.md
"""
import argparse
import json

from sedat.config import resolve

DS = ["nq", "fiqa", "hotpotqa", "musique"]
NAME = {"sparse": "Sparse (α=0)", "dense": "Dense (α=1)", "global": "Global α", "dat": "DAT (faithful)",
        "endpoint_judge": "Endpoint judge (DAT-style)", "union": "Exposure-matched top-m union",
        "top1_frontier": "**Exact Top-1 Frontier-DAT**", "slate": "**Exact Top-5 Slate Frontier-DAT**", "oracle": "Exact continuous-α oracle"}


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:+.0f}%"


def ci(p, d=3):
    return f"{p['mean']:+.{d}f} [{p['ci95'][0]:+.{d}f}, {p['ci95'][1]:+.{d}f}] (p={p['p_signflip']:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    S = json.load(open(resolve(f"runs/final/{a.split}_summary.json"), encoding="utf-8"))
    ds = [d for d in DS if d in S]
    L = []
    # ---- A
    L += ["### Table A — Top-1 retrieval (Hit@1; recovery of the continuous-α oracle gap in parentheses)", "",
          "| Method | " + " | ".join(f"{d} (n={S[d]['n']})" for d in ds) + " |", "|---" * (len(ds) + 1) + "|"]
    for m in ["sparse", "dense", "global", "dat", "endpoint_judge", "union", "top1_frontier", "oracle"]:
        cells = []
        for d in ds:
            v = S[d]["top1"][m]["hit@1"]
            r = S[d]["top1_recovery"].get(m)
            cells.append(f(v) + (f" ({pct(r)})" if r is not None else ""))
        L.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    L += ["", "Secondary (Recall@5 at the executed α): " + "; ".join(
        f"{d}: " + ", ".join(f"{k} {f(S[d]['top1'][k]['recall@5'])}" for k in ["global", "dat", "endpoint_judge", "union", "top1_frontier"])
        for d in ds), ""]
    # ---- B
    L += ["### Table B — Top-5 / multi-evidence retrieval", ""]
    for d in ds:
        mets = list(S[d]["top5"]["global"].keys())
        L += [f"**{d}** (n={S[d]['n']}; primary = {S[d]['top5_primary']})", "",
              "| Method | " + " | ".join(mets) + " |", "|---" * (len(mets) + 1) + "|"]
        for m in ["sparse", "dense", "global", "dat", "endpoint_judge", "top1_frontier", "slate", "oracle"]:
            L.append(f"| {NAME[m]} | " + " | ".join(f(S[d]["top5"][m].get(k)) for k in mets) + " |")
        r = S[d]["top5_recovery"]["slate"]
        L += ["", "Slate oracle-gap recovery: " + ", ".join(f"{k} {pct(v)}" for k, v in r.items()) +
              " · Top-1 Frontier: " + ", ".join(f"{k} {pct(v)}" for k, v in S[d]["top5_recovery"]["top1_frontier"].items()), ""]
        if "by_hops" in S[d]:
            L += ["| hops | n | Global | DAT | Top-1 Frontier | Top-5 Slate | Oracle (all-support@5) |", "|---|---|---|---|---|---|---|"]
            for h, v in S[d]["by_hops"].items():
                L.append(f"| {h} | {v['n']} | {f(v['global|allsupport@5'])} | {f(v['dat|allsupport@5'])} | "
                         f"{f(v['top1_frontier|allsupport@5'])} | {f(v['slate|allsupport@5'])} | {f(v['oracle|allsupport@5'])} |")
            L.append("")
    # ---- C
    L += ["### Table C — Paired comparisons (mean difference, paired bootstrap 95% CI, sign-flip p as secondary)", "",
          "| Comparison | Metric | " + " | ".join(ds) + " |", "|---|---|" + "---|" * len(ds)]
    for k in ["top1_frontier - dat", "top1_frontier - endpoint_judge", "top1_frontier - global", "top1_frontier - union"]:
        L.append(f"| {k} | Hit@1 | " + " | ".join(ci(S[d]["top1_paired_hit@1"][k]) for d in ds) + " |")
    for k in ["slate - top1_frontier", "slate - global", "slate - dat", "slate - endpoint_judge"]:
        L.append(f"| {k} | primary top-5 | " + " | ".join(ci(S[d]["top5_paired"][S[d]["top5_primary"]][k]) for d in ds) + " |")
    for k in ["slate - top1_frontier"]:
        L.append(f"| {k} | Hit@1 | " + " | ".join(ci(S[d]["top5_paired"]["hit@1"][k]) for d in ds) + " |")
        L.append(f"| {k} | nDCG@5 | " + " | ".join(ci(S[d]["top5_paired"]["ndcg@5"][k]) for d in ds) + " |")
    L.append("")
    L.append("Primary top-5 metric: " + ", ".join(f"{d} = {S[d]['top5_primary']}" for d in ds) + ".")
    L.append("")
    # ---- D
    L += ["### Table D — Efficiency / cost (per query; judge latency = sum of sequential call latencies)", ""]
    L += ["| Dataset | Method | docs shown | executable outcomes mean / median / p90 | frontier build (ms, mean) | calls | input tok | output tok | judge latency s (mean / p90) | USD / query | USD total |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in ds:
        E = S[d]["efficiency"]
        for m in ["dat", "endpoint_judge", "union", "top1_frontier", "slate"]:
            e = E[m]
            fo = e.get("frontier_outcomes")
            outc = f"{fo['mean']:.1f} / {fo['median']:.0f} / {fo['p90']:.0f}" if fo else ("2 (fixed)" if m == "dat" else "–")
            cm = f"{e['construction_ms']['mean']:.2f}" if "construction_ms" in e else "–"
            L.append(f"| {d} | {NAME[m]} | {e['docs_shown']:.2f} | {outc} | {cm} | {e['calls_per_query']:.2f} | {e['input_tokens']:.0f} | "
                     f"{e['output_tokens']:.1f} | {e['judge_latency_s_sequential']['mean']:.2f} / {e['judge_latency_s_sequential']['p90']:.2f} | "
                     f"{e['usd_per_query']:.5f} | {e['usd_total']:.3f} |")
    L.append("")
    # ---- E
    L += ["### Table E — Structural diagnostics (no judge involved)", "",
          "| | " + " | ".join(ds) + " |", "|---" * (len(ds) + 1) + "|"]
    t1 = lambda d: S[d]["structure_top1"]
    t5 = lambda d: S[d]["structure_top5"]
    L.append("| Top-1: interior-only rank-1 winner exists | " + " | ".join(f(t1(d)["frac_interior_only_rank1_winner"]) for d in ds) + " |")
    L.append("| Top-1: interior-only doc relevant & global wrong | " + " | ".join(f(t1(d)["frac_interior_only_gold_and_global_wrong"]) for d in ds) + " |")
    L.append("| Top-1: global → endpoint oracle → exact oracle (Hit@1) | " + " | ".join(
        f"{f(t1(d)['global'])} → {f(t1(d)['endpoint_oracle'])} → {f(t1(d)['exact_oracle'])}" for d in ds) + " |")
    L.append("| Top-1: interior share of oracle gap | " + " | ".join(pct(t1(d)["interior_share_of_oracle_gap"]).lstrip("+") for d in ds) + " |")
    L.append("| Top-5: interior slate distinct from both endpoint slates | " + " | ".join(f(t5(d)["frac_interior_slate_distinct_from_both_endpoints"]) for d in ds) + " |")
    L.append("| Top-5: interior slate adds relevant doc missing from both endpoints | " + " | ".join(f(t5(d)["frac_interior_slate_adds_gold_missing_from_both_endpoints"]) for d in ds) + " |")
    L.append("| Top-5: interior slate holds more relevant docs than either endpoint | " + " | ".join(f(t5(d)["frac_interior_slate_more_gold_than_either_endpoint"]) for d in ds) + " |")
    for k in ["recall@5", "allsupport@5"]:
        L.append(f"| Top-5 {k}: global → endpoint oracle → exact oracle | " + " | ".join(
            f"{f(t5(d)[k]['global'])} → {f(t5(d)[k]['endpoint_oracle'])} → {f(t5(d)[k]['exact_oracle'])}" for d in ds) + " |")
        L.append(f"| Top-5 {k}: interior share of oracle gap | " + " | ".join(pct(t5(d)[k]["interior_share_of_oracle_gap"]).lstrip("+") for d in ds) + " |")
    L += ["", "### Judge behaviour (Top-5 Slate)", "", "| | " + " | ".join(ds) + " |", "|---" * (len(ds) + 1) + "|"]
    for k, lab in [("changed_vs_global", "top-5 differs from global"), ("benefit", "primary benefit (among changed)"),
                   ("neutral", "primary neutral"), ("harm", "primary harm"), ("hit@1_benefit", "Hit@1 benefit (among changed)"),
                   ("hit@1_harm", "Hit@1 harm"), ("complementary_win", "complementary win (more relevant than either endpoint)"),
                   ("interior_selected", "interior slate selected")]:
        L.append(f"| {lab} | " + " | ".join(f(S[d]["slate_judge"][k]) for d in ds) + " |")
    ta = S[ds[0]]["top1_frontier_actions"]
    L += ["", "Top-1 Frontier actions (coverage / benefit / harm among changed): " + "; ".join(
        f"{d} {f(S[d]['top1_frontier_actions']['coverage'])} / {f(S[d]['top1_frontier_actions']['benefit'])} / {f(S[d]['top1_frontier_actions']['harm'])}" for d in ds)]
    print("\n".join(L))


if __name__ == "__main__":
    main()
