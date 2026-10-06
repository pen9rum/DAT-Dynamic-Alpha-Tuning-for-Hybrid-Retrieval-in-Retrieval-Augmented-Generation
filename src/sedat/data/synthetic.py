"""Synthetic benchmark with analytically known answers.

Bypasses the retrievers and writes a candidate cache directly, so the whole
downstream pipeline (fusion -> reward table -> oracle analysis -> online
learning) can be verified in seconds.

Normalized (min-max) pool scores per query type -- gold = g:

  type A (sparse-favoring)   sparse: g=1.0 x=0.0 f=0.6    dense: g=0.0 x=1.0 f=0.6
                             gold is top-1  iff  1-alpha > 0.6   iff alpha <= 0.3
  type B (dense-favoring)    mirror of A                   iff alpha >= 0.7
  type E (hybrid-only)       sparse: g=0.75 x1=1.0 x2=0.0  dense: g=0.75 x1=0.0 x2=1.0
                             gold is top-1  iff  0.25 < alpha < 0.75  iff alpha in [0.3, 0.7]
  + `n_filler` docs with normalized scores in [0, 0.5] (never top-1).

With fractions (pA, pB, pE) = (0.4, 0.4, 0.2) and reward = hit@1:
  global optimum  alpha in {0.3, 0.7} -> 0.6      per-query oracle -> 1.0
  oracle gap 0.4 = routing part 0.2 (A/B) + hybrid-specific part 0.2 (E)
  categories A=0.4 B=0.4 E=0.2 C=D=0

Raw scores = normalized * type-specific scale + offset, so the *type is visible
at decision time* only through raw-score features (e.g. BM25 top-1 magnitude),
as it would be for a real retriever -- min-max normalization hides it from fusion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..retrieval.candidates import write_cache

LAYOUTS = {
    "A": {"sparse": [1.0, 0.0, 0.6], "dense": [0.0, 1.0, 0.6]},
    "B": {"sparse": [0.0, 1.0, 0.6], "dense": [1.0, 0.0, 0.6]},
    "E": {"sparse": [0.75, 1.0, 0.0], "dense": [0.75, 0.0, 1.0]},
}
# raw-score scale per type: (sparse scale, dense scale)
SCALES = {"A": (20.0, 0.20), "B": (6.0, 0.50), "E": (12.0, 0.35)}


def expected_values(fractions: dict[str, float]) -> dict:
    pA, pB, pE = (fractions.get(k, 0.0) for k in "ABE")
    glob = max(pA + pE, pB + pE, pA, pB, pE)
    return {"global": glob, "oracle": pA + pB + pE, "routing": pA + pB,
            "gap": pA + pB + pE - glob, "categories": {"A": pA, "B": pB, "E": pE}}


def write_synthetic_cache(cfg: dict, cache_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    fr = cfg.get("type_fractions", {"A": 0.4, "B": 0.4, "E": 0.2})
    n_per_split = cfg.get("n_queries", {"train": 2000, "val": 500, "test": 1000})
    n_filler = cfg.get("n_filler", 7)
    noise = cfg.get("feature_noise", 0.15)       # relative noise on raw-score scales

    q_rows, cand_parts, lab_rows, corpus_rows = [], [], [], []
    for split, n in n_per_split.items():
        counts = {t: int(round(fr.get(t, 0) * n)) for t in "ABE"}
        types = np.array([t for t, c in counts.items() for _ in range(c)])
        rng.shuffle(types)
        for j, t in enumerate(types):
            qid = f"{split}-{j:05d}"
            s_norm = np.array(LAYOUTS[t]["sparse"] + list(rng.uniform(0, 0.5, n_filler)))
            d_norm = np.array(LAYOUTS[t]["dense"] + list(rng.uniform(0, 0.5, n_filler)))
            ss, ds = SCALES[t]
            ss *= np.exp(rng.normal(0, noise))
            ds *= np.exp(rng.normal(0, noise))
            s_raw = s_norm * ss + rng.uniform(0, 2)
            d_raw = d_norm * ds + rng.uniform(0.2, 0.4)
            n_docs = len(s_norm)
            doc_ids = [f"{qid}-d{k}" for k in range(n_docs)]
            base = len(corpus_rows)
            corpus_rows.extend({"doc_id": d, "title": qid, "text": f"synthetic passage {d}"} for d in doc_ids)
            s_rank = np.argsort(np.argsort(-s_raw, kind="stable"), kind="stable") + 1
            d_rank = np.argsort(np.argsort(-d_raw, kind="stable"), kind="stable") + 1
            cand_parts.append(pd.DataFrame({
                "qid": qid, "doc_idx": np.arange(base, base + n_docs, dtype=np.int32),
                "doc_id": doc_ids, "sparse_score": s_raw, "dense_score": d_raw,
                "sparse_rank": s_rank.astype(np.int32), "dense_rank": d_rank.astype(np.int32),
                "in_sparse_topk": True, "in_dense_topk": True}))
            q_rows.append({"qid": qid, "query": f"synthetic query {qid}", "split": split,
                           "group": qid, "n_tokens": 3, "n_tokens_in_vocab": 3})
            lab_rows.append({"qid": qid, "gold_doc_ids": [doc_ids[0]],
                             "sparse_gold_rank": int(s_rank[0]), "dense_gold_rank": int(d_rank[0]),
                             "type": t})   # latent type is a LABEL, not decision-time info

    write_cache(cache_dir, pd.DataFrame(corpus_rows), pd.DataFrame(q_rows),
                pd.concat(cand_parts, ignore_index=True), pd.DataFrame(lab_rows))
