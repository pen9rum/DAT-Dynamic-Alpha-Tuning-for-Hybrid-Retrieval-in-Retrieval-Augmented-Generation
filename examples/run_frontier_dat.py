"""End-to-end example: Exact Top-1 Frontier-DAT on a few validation queries of a prepared dataset.

Per query: frozen BM25 + BGE candidate pool -> min-max normalization -> exact actionable frontier ->
blinded LLM relevance judgment of the frontier documents (two presentation orders, averaged) ->
adapt only if the best document beats alpha_g's rank-1 document by >= delta -> execute the feasible alpha closest
to alpha_g. Requires the retrieval cache (see README "Pipeline") and an OpenRouter key in .env
(OPENROUTER_API_KEY=sk-or-...). Responses are cached in cache/m3b_llm_cache.sqlite.

  python -m examples.run_frontier_dat --dataset fiqa --n 5 --delta 1.5
"""
import argparse

import numpy as np

from sedat.config import get_dotted, load_config
from sedat.m3.cache import build_m3_cache
from sedat.m3b.bench import load_bench
from sedat.m3b.llm import SmallLLM
from sedat.m3c.exact_frontier import exact_frontier, order_at
from sedat.m3c.frontier import FRONTIER_VERSION, build_call, parse
from sedat.m3c.sensor import short
from sedat.retrieval.normalization import normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiqa")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--delta", type=float, default=1.5, help="gate threshold on the judged score gain")
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--budget", type=float, default=0.10, help="hard USD cap for new API calls")
    args = ap.parse_args()

    cfg = load_config(f"configs/m3/{args.dataset}.yaml")
    cache = build_m3_cache(cfg)
    B = load_bench(args.dataset)
    ag = float(B.alphas[B.g])
    corp = cache.corpus.set_index("doc_id")
    llm = SmallLLM(args.model, args.budget, max_tokens=60)
    lb = None

    def text(x):
        nonlocal lb
        if x in corp.index and corp.at[x, "text"]:
            return short(corp.at[x, "title"], corp.at[x, "text"])
        if lb is None:
            from sedat.retrieval.pyserini_backend import LuceneBM25, setup_java
            setup_java(get_dotted(cfg, "paths.java_home"))
            lb = LuceneBM25(cfg["retrieval"]["sparse_index"])
        y = lb.doc(x)
        return short(y["title"], y["text"])

    hits = {"global": [], "frontier": []}
    for r in [int(x) for x in B.rows("val")][: args.n]:
        qid, query = B.qids[r], B.query[r]
        p = cache.pools[qid]
        s = normalize(p.sparse_score, p.sparse_rank, "minmax")
        d = normalize(p.dense_score, p.dense_rank, "minmax")
        ids = p.doc_id
        gold = set(cache.labels.at[qid, "gold_doc_ids"])
        F = exact_frontier(s, d)
        docs = [ids[i] for i in F.docs]
        gdoc = ids[order_at(s, d, ag)[0]]
        alpha = ag
        print(f"\nQ: {query}\n  frontier: {len(docs)} actionable rank-1 documents (alpha_g = {ag})")
        if len(docs) >= 2:
            texts = {x: text(x) for x in docs}
            runs = []
            for rev in (False, True):                          # two presentation orders -> average
                msgs, lab2doc = build_call(qid, query, docs, texts, reverse=rev)
                o = llm.call(llm.key(FRONTIER_VERSION + ("|rev" if rev else ""), qid, "none", "static", msgs), msgs)
                sc = parse(o["text"], lab2doc)
                if sc is not None:
                    runs.append(sc)
            if runs:
                score = {x: float(np.mean([z[x] for z in runs])) for x in docs}
                best = max(score.values())
                for x in docs:
                    iv = [(round(a, 3), round(b, 3)) for a, b in F.intervals[F.docs[docs.index(x)]]]
                    tag = " <- alpha_g top-1" if x == gdoc else ""
                    print(f"    score {score[x]:.1f}  alpha {iv}{' [relevant]' if x in gold else ''}{tag}")
                if best - score[gdoc] >= max(args.delta, 1e-9):
                    tied = [x for x, v in score.items() if v >= best - 1e-9 and x != gdoc]
                    alpha = min((abs(F.feasible_alpha(F.docs[docs.index(x)], ag) - ag),
                                 F.feasible_alpha(F.docs[docs.index(x)], ag)) for x in tied)[1]
        top1 = ids[order_at(s, d, alpha)[0]]
        hits["global"].append(gdoc in gold)
        hits["frontier"].append(top1 in gold)
        print(f"  executed alpha = {alpha:.4f}  ->  Hit@1 global {int(gdoc in gold)} | Frontier-DAT {int(top1 in gold)}")
    print(f"\nHit@1 over {args.n} queries: global {np.mean(hits['global']):.3f} | Frontier-DAT {np.mean(hits['frontier']):.3f}")
    print(f"LLM calls: {llm.stats}, new spend ${llm.spent_new:.4f}")


if __name__ == "__main__":
    main()
