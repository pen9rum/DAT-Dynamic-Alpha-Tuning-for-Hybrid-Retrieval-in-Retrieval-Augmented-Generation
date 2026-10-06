"""Candidate pools from large-corpus retrievers (Pyserini backend).

For a query text: sparse top-`depth` and dense top-`depth`; pool = union of the
top-K of each; every pool doc gets BOTH exact raw scores. Ranks come from the
depth lists (rank > depth is censored to depth+1 -- only RRF normalization and
diagnostics use ranks; min-max/z-score use exact scores).

Used for (a) single-query candidate caches (Experiment A) and (b) every hop query
of the multi-hop engine (Experiment B), so both experiments share one code path.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


class PoolBuilder:
    def __init__(self, sparse, dense, sparse_top_k: int = 100, dense_top_k: int = 100, depth: int = 1000):
        self.sparse, self.dense = sparse, dense
        self.ks, self.kd, self.depth = sparse_top_k, dense_top_k, depth
        self.doc_cache: dict[str, dict] = {}

    def build(self, keys: list[str], texts: list[str], batch_size: int = 128,
              desc: str = "pools") -> tuple[pd.DataFrame, np.ndarray]:
        """Returns (long candidate frame keyed by `key`, query embeddings aligned with keys)."""
        parts, embs = [], []
        for start in tqdm(range(0, len(keys), batch_size), desc=desc):
            ks, ts = keys[start:start + batch_size], texts[start:start + batch_size]
            S = self.sparse.search(ks, ts, self.depth)
            Q = self.dense.encode(ts)
            D = self.dense.search(ks, Q, self.depth)
            embs.append(Q)
            for i, (k, t) in enumerate(zip(ks, ts)):
                s_ids, s_sc = S.get(k, ([], np.zeros(0)))
                d_ids, d_sc = D[k]
                s_rank = {d: r + 1 for r, d in enumerate(s_ids)}
                d_rank = {d: r + 1 for r, d in enumerate(d_ids)}
                s_map, d_map = dict(zip(s_ids, s_sc)), dict(zip(d_ids, d_sc))
                pool = sorted(set(s_ids[:self.ks]) | set(d_ids[:self.kd]))
                miss_d = [d for d in pool if d not in d_map]
                if miss_d:
                    d_map.update(zip(miss_d, self.dense.score(Q[i], miss_d)))
                miss_s = [d for d in pool if d not in s_map]
                if miss_s:
                    s_map.update(zip(miss_s, self.sparse.score_many(t, miss_s)))
                sp = np.array([s_map[d] for d in pool])
                parts.append(pd.DataFrame({
                    "qid": k, "doc_id": pool, "sparse_score": sp,
                    "dense_score": np.array([d_map[d] for d in pool], dtype=np.float64),
                    "sparse_rank": np.array([s_rank.get(d, self.depth + 1) for d in pool], dtype=np.int32),
                    "dense_rank": np.array([d_rank.get(d, self.depth + 1) for d in pool], dtype=np.int32),
                    "in_sparse_topk": [s_rank.get(d, 10 ** 9) <= self.ks for d in pool],
                    "in_dense_topk": [d_rank.get(d, 10 ** 9) <= self.kd for d in pool]}))
            self._last = (S, D)
        return pd.concat(parts, ignore_index=True), np.concatenate(embs)

    def gold_ranks(self, key: str, gold: list[str]) -> tuple[int, int]:
        """Best gold rank per retriever from the most recent batch (censored at depth+1)."""
        S, D = self._last
        s_ids, d_ids = S.get(key, ([], None))[0], D[key][0]
        sr = min((s_ids.index(g) + 1 for g in gold if g in s_ids), default=self.depth + 1)
        dr = min((d_ids.index(g) + 1 for g in gold if g in d_ids), default=self.depth + 1)
        return sr, dr

    def docs(self, doc_ids) -> pd.DataFrame:
        todo = [d for d in doc_ids if d not in self.doc_cache]
        for d in tqdm(todo, desc="doc texts", disable=len(todo) < 2000):
            self.doc_cache[d] = self.sparse.doc(d)
        return pd.DataFrame([{"doc_id": d, **self.doc_cache[d]} for d in doc_ids],
                            columns=["doc_id", "title", "text"])
