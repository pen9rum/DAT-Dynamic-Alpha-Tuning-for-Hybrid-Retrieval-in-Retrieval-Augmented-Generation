"""Final frozen evaluation (WSDM short paper): every method in one pass over one split.

Methods (executed alpha -> full hybrid ranking via order_at, the retriever's exact rule):
  sparse (0), dense (1), global (alpha_g), dat (faithful DAT: paper prompt, paper alpha rule),
  endpoint_judge (frontier-v1 judge on {global, sparse, dense} top-1s + delta gate = earlier "DAT endpoints"),
  union (exposure-matched top-m union, frontier-v1 judge + delta gate), top1_frontier (Exact Top-1 Frontier-DAT),
  slate (Exact Top-5 Slate Frontier-DAT, one-pass listwise judge), and exact continuous-alpha oracles.
Judges, prompts, blinding, alpha mapping and tie rules are imported unchanged from the validated code.

  val : cache replay; selects the delta gate per condition on the FULL validation set with the frozen
        selection rule (max mean Hit@1, ties -> smaller delta) and writes it for the freeze.
  test: requires the frozen config (code hashes must match), uses its deltas, runs exactly once.

  python -m experiments.final_eval --dataset nq --split val
  python -m experiments.final_eval --dataset nq --split test --frozen runs/final/final_frozen_config.json
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import time

import numpy as np
import pandas as pd

from sedat.config import get_dotted, load_config, resolve, setup_logging, write_json
from sedat.llm.prompts import DAT_JUDGE_PAPER
from sedat.m3.cache import build_m3_cache
from sedat.m3b.bench import load_bench
from sedat.m3b.llm import SmallLLM
from sedat.m3c.exact_frontier import exact_frontier, order_at
from sedat.m3c.frontier import FRONTIER_VERSION, build_call, parse
from sedat.m3c.sensor import short
from sedat.m3c.topk_frontier import topk_frontier
from sedat.policies.dat_style import dat_alpha, parse_scores
from sedat.retrieval.normalization import normalize
from experiments.m3c_topk_slate import VERSION as SLATE_VERSION, build_prompt, evaluate, feasible
from experiments.m3c_topk_slate import parse as parse_slate

MODEL = "openai/gpt-4.1-mini"
DAT_VERSION = "dat-paper-v1"
DELTAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, np.inf]
EPS = 1e-9
K = 5
GATED = ["endpoint_judge", "top1_frontier", "union"]
TAG = {"endpoint_judge": f"{FRONTIER_VERSION}|endpoints", "union": f"{FRONTIER_VERSION}|union",
       "top1_frontier": FRONTIER_VERSION}
SETS = {"endpoint_judge": "endpoints", "union": "union", "top1_frontier": "exact"}
HASHED = ["experiments/final_eval.py", "experiments/m3c_topk_slate.py", "src/sedat/m3c/exact_frontier.py",
          "src/sedat/m3c/topk_frontier.py", "src/sedat/m3c/frontier.py", "src/sedat/m3c/sensor.py",
          "src/sedat/m3b/llm.py", "src/sedat/llm/prompts.py", "src/sedat/policies/dat_style.py",
          "src/sedat/retrieval/normalization.py", "src/sedat/m3b/bench.py", "src/sedat/m3/cache.py",
          "src/sedat/m3b/evaluate.py", "src/sedat/config.py"]


def code_hashes():
    return {f: hashlib.sha256(resolve(f).read_bytes()).hexdigest() for f in HASHED}


def select_delta(rows, decide, reward):
    """Frozen rule (as in the 2-fold cross-fit, applied to the whole validation set)."""
    return max(((np.mean([reward(r, decide(r, dl)[1]) for r in rows]), -dl if np.isfinite(dl) else -99, dl)
                for dl in DELTAS))[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    setup_logging()
    ds, split = args.dataset, args.split
    out = resolve(f"runs/final/{split}/{ds}")
    frozen = None
    if split == "test":
        assert args.frozen, "test requires the frozen config"
        frozen = json.load(open(resolve(args.frozen), encoding="utf-8"))
        now = code_hashes()
        bad = [f for f in HASHED if frozen["code_sha256"].get(f) != now[f]]
        assert not bad, f"code changed since freeze: {bad}"
        assert not (out / "per_query.parquet").exists(), f"test already run for {ds} -- single run only"
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(f"configs/m3/{ds}.yaml")
    cache = build_m3_cache(cfg)
    B = load_bench(ds)
    alphas, g = B.alphas, B.g
    A, ag = len(alphas), float(alphas[g])
    if frozen:
        assert abs(frozen["datasets"][ds]["alpha_global"] - ag) < 1e-12
    rows = [int(r) for r in B.rows(split)]
    labels = cache.labels
    t = pd.read_parquet(cache.dir / "reward_table_minmax.parquet", columns=["qid", "alpha_idx", "top_doc_ids"])
    tops = t[t["qid"].isin(set(B.qids[rows]))].pivot(index="qid", columns="alpha_idx", values="top_doc_ids")
    pools = cache.pools

    Q = {}
    for r in rows:
        qid = B.qids[r]
        p = pools[qid]
        s = normalize(p.sparse_score, p.sparse_rank, "minmax")
        d = normalize(p.dense_score, p.dense_rank, "minmax")
        ids = p.doc_id
        tl = [list(tops.at[qid, a]) for a in range(A)]
        t0 = time.perf_counter()
        F = exact_frontier(s, d)
        t1 = time.perf_counter()
        FK = topk_frontier(s, d, K)
        t2 = time.perf_counter()
        exact = [ids[i] for i in F.docs]
        gdoc = ids[order_at(s, d, ag)[0]]
        assert gdoc == tl[g][0]
        s_ord = list(ids[np.argsort(p.sparse_rank, kind="stable")])
        d_ord = list(ids[np.argsort(p.dense_rank, kind="stable")])
        endpoints = list(dict.fromkeys([gdoc, s_ord[0], d_ord[0]]))
        inter = [x for pair in zip(s_ord, d_ord) for x in pair]
        union = list(dict.fromkeys([gdoc] + inter))[:max(len(exact), len(endpoints))]
        to_ids = lambda sl: tuple(ids[i] for i in sl)
        Q[r] = {"s": s, "d": d, "ids": ids, "F": F, "idx": {x: i for i, x in enumerate(ids)}, "tl": tl,
                "rank": [{x: k + 1 for k, x in enumerate(v)} for v in tl], "gdoc": gdoc,
                "gold": set(labels.at[qid, "gold_doc_ids"]),
                "sets": {"endpoints": endpoints, "exact": exact, "union": union},
                "sparse_top1": s_ord[0], "dense_top1": d_ord[0],
                "slates": [to_ids(sl) for sl in FK.slates],
                "intervals": {to_ids(sl): FK.intervals[sl] for sl in FK.slates},
                "t_top1_frontier": t1 - t0, "t_top5_frontier": t2 - t1}

    corp = cache.corpus.set_index("doc_id")
    need = sorted({x for q in Q.values() for c in q["sets"].values() for x in c} |
                  {x for q in Q.values() for sl in q["slates"] for x in sl} |
                  {x for q in Q.values() for x in (q["sparse_top1"], q["dense_top1"])})
    missing = [x for x in need if x not in corp.index or not corp.at[x, "text"]]
    raw = {}
    if missing:
        from sedat.retrieval.pyserini_backend import LuceneBM25, setup_java
        setup_java(get_dotted(cfg, "paths.java_home"))
        lb = LuceneBM25(cfg["retrieval"]["sparse_index"])
        for x in missing:
            y = lb.doc(x)
            raw[x] = (y["title"], y["text"])
    for x in need:
        raw.setdefault(x, (corp.at[x, "title"], corp.at[x, "text"]))
    texts = {x: short(*v) for x, v in raw.items()}                       # frozen 450-char judge view

    def full(x):                                                          # DAT: full passage, no truncation
        ti, tx = raw[x]
        return ((ti or "").strip() + ". " if ti else "") + " ".join((tx or "").split())

    budget = args.budget
    llm1 = SmallLLM(MODEL, budget, max_tokens=60)        # frontier-v1 judge (top-1 conditions)
    llm5 = SmallLLM(MODEL, budget, max_tokens=20)        # slate judge
    llmd = SmallLLM(MODEL, budget, max_tokens=20)        # DAT judge

    def acc(o):
        return {"in": o["prompt_tokens"], "out": o["completion_tokens"], "cost": o["cost"], "lat": o["latency"] or 0.0}

    def job(item):
        r, cond = item
        q, qid = Q[r], B.qids[r]
        calls = []
        if cond in GATED:
            docs = q["sets"][SETS[cond]]
            if len(docs) < 2:
                return item, (None, calls)
            ok = []
            for rev in (False, True):
                msgs, lab2doc = build_call(qid, B.query[r], docs, texts, reverse=rev)
                o = llm1.call(llm1.key(TAG[cond] + ("|rev" if rev else ""), qid, "none", "static", msgs), msgs)
                calls.append(acc(o))
                pp = parse(o["text"], lab2doc)
                if pp is not None:
                    ok.append(pp)
            return item, ({x: float(np.mean([y[x] for y in ok])) for x in docs} if ok else None, calls)
        if cond == "dat":
            msgs = [{"role": "user", "content": DAT_JUDGE_PAPER.format(
                question=B.query[r], vector_reference=full(q["dense_top1"]), bm25_reference=full(q["sparse_top1"]))}]
            o = llmd.call(llmd.key(DAT_VERSION, qid, "none", "static", msgs), msgs)
            calls.append(acc(o))
            return item, (parse_scores(o["text"] or ""), calls)
        if len(q["slates"]) < 2:                                          # slate
            return item, (None, calls)
        msgs, lab = build_prompt(qid, B.query[r], q["slates"], texts, 1)
        o = llm5.call(llm5.key(f"{SLATE_VERSION}|pass1", qid, "none", "static", msgs), msgs)
        calls.append(acc(o))
        return item, (parse_slate(o["text"], lab), calls)

    _ = cache.pools
    with cf.ThreadPoolExecutor(args.workers) as ex:
        J = dict(ex.map(job, [(r, c) for r in rows for c in GATED + ["dat", "slate"]]))
    spend = llm1.spent_new + llm5.spent_new + llmd.spent_new
    stats = {k: llm1.stats[k] + llm5.stats[k] + llmd.stats[k] for k in llm1.stats}
    print(f"judging done: new spend ${spend:.4f} {stats}", flush=True)

    def top_doc(r, alpha):
        q = Q[r]
        return q["ids"][order_at(q["s"], q["d"], alpha)[0]]

    def decide(r, cond, delta):                       # identical to experiments.m3c_exact_frontier.decide
        q = Q[r]
        sc = J[(r, cond)][0]
        if sc is None or max(sc.values()) - sc[q["gdoc"]] < max(delta, EPS):
            return ag, q["gdoc"], None
        best = max(sc.values())
        tied = [x for x, v in sc.items() if v >= best - EPS and x != q["gdoc"]]
        if cond == "top1_frontier":
            opts = [(abs(q["F"].feasible_alpha(q["idx"][x], ag) - ag), q["F"].feasible_alpha(q["idx"][x], ag), x)
                    for x in tied]
            _, a_hat, x = min(opts)
            return a_hat, top_doc(r, a_hat), x
        b = min((q["rank"][a].get(x, 10 ** 6), abs(a - g), a, x) for x in tied for a in range(A))
        if b[0] >= 10 ** 6:
            return ag, q["gdoc"], None
        return float(alphas[b[2]]), q["tl"][b[2]][0], b[3]

    reward = lambda r, doc: float(doc in Q[r]["gold"])
    if frozen:
        deltas = {c: float(frozen["datasets"][ds]["delta"][c]) for c in GATED}
    else:
        deltas = {c: float(select_delta(rows, lambda r, dl, c=c: decide(r, c, dl), reward)) for c in GATED}
    print("deltas", deltas, flush=True)

    recs = []
    n_dat_fail = n_slate_fail = 0
    for r in rows:
        q, qid = Q[r], B.qids[r]
        al = {"sparse": 0.0, "dense": 1.0, "global": ag}
        sc = J[(r, "dat")][0]
        n_dat_fail += sc is None
        al["dat"] = float(dat_alpha(*(sc if sc is not None else (0, 0))))
        for c in GATED:
            al[c] = float(decide(r, c, deltas[c])[0])
        pick = J[(r, "slate")][0]
        if len(q["slates"]) >= 2 and pick is None:
            n_slate_fail += 1
        al["slate"] = feasible(q["intervals"][pick], ag) if pick is not None else ag
        rec = {"qid": qid, "dat_scores": str(sc)}
        for name, a in al.items():
            top = [q["ids"][i] for i in order_at(q["s"], q["d"], a)[:10]]
            if name == "slate" and pick is not None:
                assert tuple(top[:K]) == tuple(pick)
            for m, v in evaluate(top, q["gold"]).items():
                rec[f"{name}|{m}"] = v
            rec[f"{name}|alpha"] = a
            rec[f"{name}|slate"] = list(top[:K])
        sl_eval = [evaluate(list(sl), q["gold"]) for sl in q["slates"]]
        rec["oracle1|hit@1"] = float(bool(q["gold"] & set(q["sets"]["exact"])))
        for m in ("recall@5", "ndcg@5", "allsupport@5"):
            rec[f"oracle5|{m}"] = max(o[m] for o in sl_eval)
        # structural diagnostics
        r0, r1 = top_doc(r, 0.0), top_doc(r, 1.0)
        interior1 = [x for x in q["sets"]["exact"] if x not in (r0, r1)]
        rec["d1|has_interior_only"] = bool(interior1)
        rec["d1|interior_gold_global_wrong"] = bool(set(interior1) & q["gold"]) and q["gdoc"] not in q["gold"]
        rec["d1|endpoint_oracle"] = float(bool({r0, r1, q["gdoc"]} & q["gold"]))
        e0 = tuple(q["ids"][i] for i in order_at(q["s"], q["d"], 0.0)[:K])
        e1 = tuple(q["ids"][i] for i in order_at(q["s"], q["d"], 1.0)[:K])
        eg = tuple(q["ids"][i] for i in order_at(q["s"], q["d"], ag)[:K])
        int5 = [sl for sl in q["slates"] if sl not in (e0, e1)]
        ng = lambda sl: len(q["gold"] & set(sl))
        rec["d5|has_interior_slate"] = bool(int5)
        rec["d5|interior_adds_gold"] = bool(any((set(sl) - set(e0) - set(e1)) & q["gold"] for sl in int5))
        rec["d5|interior_more_gold_than_either"] = bool(any(ng(sl) > max(ng(e0), ng(e1)) for sl in int5))
        for m in ("recall@5", "ndcg@5", "allsupport@5"):
            rec[f"d5|endpoint_oracle|{m}"] = max(evaluate(list(x), q["gold"])[m] for x in (e0, e1, eg))
        rec["slate|complementary_win"] = bool(pick is not None and ng(pick) > max(ng(e0), ng(e1)))
        rec["slate|interior_selected"] = bool(pick is not None and pick not in (e0, e1))
        # efficiency
        rec["eff|n_exact_docs"] = len(q["sets"]["exact"])
        rec["eff|n_endpoint_docs"] = len(q["sets"]["endpoints"])
        rec["eff|n_union_docs"] = len(q["sets"]["union"])
        rec["eff|n_slates"] = len(q["slates"])
        rec["eff|n_slate_docs"] = len({x for sl in q["slates"] for x in sl})
        rec["eff|t_top1_frontier"] = q["t_top1_frontier"]
        rec["eff|t_top5_frontier"] = q["t_top5_frontier"]
        for c in GATED + ["dat", "slate"]:
            calls = J[(r, c)][1]
            rec[f"eff|{c}|calls"] = len(calls)
            for f in ("in", "out", "cost", "lat"):
                rec[f"eff|{c}|{f}"] = float(sum(x[f] for x in calls))
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_parquet(out / "per_query.parquet", index=False)
    write_json(out / "run_meta.json", {"dataset": ds, "split": split, "n": len(rows), "alpha_global": ag,
                                       "deltas": deltas, "new_spend_usd": spend, "llm_stats": stats,
                                       "dat_parse_failures": n_dat_fail, "slate_parse_failures": n_slate_fail,
                                       "code_sha256": code_hashes(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
    show = ["sparse", "dense", "global", "dat", "endpoint_judge", "union", "top1_frontier", "slate"]
    print({n: round(df[f"{n}|hit@1"].mean(), 4) for n in show}, "oracle1", round(df["oracle1|hit@1"].mean(), 4))
    print({n: round(df[f"{n}|allsupport@5"].mean(), 4) for n in show}, round(df["oracle5|allsupport@5"].mean(), 4))
    print({n: round(df[f"{n}|recall@5"].mean(), 4) for n in show}, round(df["oracle5|recall@5"].mean(), 4))
    print(f"out: {out}  spend ${spend:.4f}  dat_fail {n_dat_fail}  slate_fail {n_slate_fail}")


if __name__ == "__main__":
    main()
