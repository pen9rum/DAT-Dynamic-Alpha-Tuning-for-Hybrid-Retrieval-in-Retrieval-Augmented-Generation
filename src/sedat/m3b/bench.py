"""Milestone 3B benchmark view: frozen M3 caches -> everything the controller needs.

Decision-time information per query (never labels):
  query text, frozen query embedding, standardized retrieval-state features,
  sparse / dense top-1 title + snippet, a few raw retrieval statistics for the prompt.
Labels (feedback / evaluation only): the counterfactual reward matrix R[q, alpha].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import alpha_grid, load_config
from ..evaluation.oracle_analysis import tune_global_alpha
from ..m3.cache import build_m3_cache
from ..m3.features import raw_features
from ..state import FEATURE_NAMES

EVAL_METRICS = ["hit@1", "ndcg@10", "allrel@10"]


@dataclass
class Bench:
    name: str
    alphas: np.ndarray
    g: int                                  # val-tuned alpha_global index (hit@1)
    qids: np.ndarray                        # row order used by every array below
    split: np.ndarray
    query: np.ndarray
    emb: np.ndarray                         # L2-normalized frozen query embeddings
    state: np.ndarray                       # train-standardized retrieval-state features
    view: list                              # per-query dict of prompt-facing retrieval info
    R: dict                                 # metric -> (n, n_alpha) rewards  (LABELS)

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def _snip(title: str, text: str, n: int = 110) -> str:
    t = (title or "").strip()
    x = " ".join((text or "").split())
    s = (t + " — " if t else "") + x
    return s[:n] + ("…" if len(s) > n else "")


def load_bench(name: str) -> Bench:
    cfg = load_config(f"configs/m3/{name}.yaml")
    cache = build_m3_cache(cfg)
    alphas = alpha_grid(cfg)
    raw = raw_features(cfg, cache)
    qids = np.array(raw["qid"], dtype=object)
    q = cache.queries.set_index("qid").loc[list(qids)]
    split = q["split"].to_numpy()
    table = pd.read_parquet(cache.dir / "reward_table_minmax.parquet")
    R = {m: table.pivot(index="qid", columns="alpha_idx", values=m).loc[list(qids)].to_numpy(np.float64)
         for m in EVAL_METRICS if m in table.columns}
    g = tune_global_alpha(R["hit@1"][split == "val"], alphas)

    emb = raw["q_emb"].astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    st = raw["state"].astype(np.float64)
    tr = split == "train"
    mu, sd = st[tr].mean(0), st[tr].std(0)
    sd[sd < 1e-9] = 1.0
    state = (st - mu) / sd

    corpus = cache.corpus
    title = corpus["title"].to_numpy()
    text = corpus["text"].to_numpy()
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    view = []
    for i, qid in enumerate(qids):
        p = cache.pools[qid]
        so = np.argsort(p.sparse_rank, kind="stable")
        do = np.argsort(p.dense_rank, kind="stable")
        s1, d1 = p.doc_idx[so[0]], p.doc_idx[do[0]]
        f = raw["state"][i]
        view.append({
            "bm25_top1": float(p.sparse_score[so[0]]),
            "bm25_margin": float(f[fi["sparse_margin12"]]),
            "dense_top1": float(p.dense_score[do[0]]),
            "dense_margin": float(f[fi["dense_margin12"]]),
            "overlap10": float(f[fi["jaccard10"]]),
            "same_top1": bool(so[0] == do[0]),
            "q_tokens": int(q["n_tokens"].iat[i]),
            "bm25_doc": _snip(title[s1], text[s1]),
            "dense_doc": _snip(title[d1], text[d1]),
            "bm25_long": _snip(title[s1], text[s1], 600),
            "dense_long": _snip(title[d1], text[d1], 600),
        })
    return Bench(name, alphas, g, qids, split, q["query"].to_numpy(), emb, state, view, R)
