"""Decision-time feature sets for Milestone 3A (no gold information anywhere).

  query  : before retrieval -- query embedding + lightweight lexical features
  dat    : what the original DAT judge sees -- query, the sparse top-1 and dense top-1
           documents (as frozen embeddings of their text) and their scores
  state  : query features + top-k score statistics of both retrievers (margins,
           concentration/entropy, overlap, rank agreement)
  all    : union (supplementary)

Embedding blocks are reduced with PCA fit on TRAIN only; all features are standardized
with TRAIN statistics.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..config import get_dotted, resolve
from ..retrieval.candidates import CandidateCache
from ..state import FEATURE_NAMES, pool_features

log = logging.getLogger(__name__)

WH = ["what", "who", "when", "where", "which", "why", "how"]
_TOK = re.compile(r"\w+", re.UNICODE)


def lexical(query: str, n_tokens: int, n_in_vocab: int) -> list[float]:
    toks = _TOK.findall(query)
    low = [t.lower() for t in toks]
    wh = [float(any(t == w for t in low[:3])) for w in WH]
    return [float(n_tokens), n_in_vocab / n_tokens if n_tokens else 0.0, float(len(query)), float(len(toks)),
            np.mean([t.isdigit() for t in toks]) if toks else 0.0,
            np.mean([t[:1].isupper() for t in toks[1:]]) if len(toks) > 1 else 0.0,
            float('"' in query or "'" in query), float(query.strip().endswith("?")),
            *wh, float(sum(wh) == 0)]


LEX_NAMES = ["n_tokens", "frac_in_vocab", "n_chars", "n_words", "frac_digit", "frac_cap", "has_quote",
             "ends_q"] + [f"wh_{w}" for w in WH] + ["wh_none"]
EXTRA_STATE = ["sparse_std10", "dense_std10", "sparse_rel_margin", "dense_rel_margin"]
DAT_SCALARS = ["sparse_top1_bm25", "dense_top1_cos", "cos_of_sparse_top1", "bm25_of_dense_top1", "same_top1",
               "cos_between_top1s"]


def _doc_embedder(cfg: dict, cache: CandidateCache):
    """doc_ids -> embeddings, using the SAME frozen dense encoder as retrieval."""
    if (cache.dir / "doc_emb.npy").exists():         # SQuAD (Milestone-1 pipeline): rows == corpus rows
        E = np.load(cache.dir / "doc_emb.npy")
        row = {d: i for i, d in enumerate(cache.corpus["doc_id"])}
        return lambda ids: E[[row[d] for d in ids]]
    from ..pipeline import make_suite_retrievers  # noqa: F401  (sets up Java before faiss/pyserini)
    from ..retrieval.pyserini_backend import FaissBGE, setup_java
    setup_java(get_dotted(cfg, "paths.java_home"))
    r = cfg["retrieval"]
    dense = FaissBGE(r["dense_index"], model=r["dense"]["model"], mmap=r["dense"].get("mmap", False))
    return lambda ids: np.stack([dense.index.reconstruct(int(dense.row[d])) for d in ids])


def raw_features(cfg: dict, cache: CandidateCache) -> dict[str, np.ndarray]:
    """Unscaled feature blocks, rows aligned with cache.queries. Cached to disk."""
    path = cache.dir / "m3_raw_features.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    q = cache.queries
    lex, state, dat, s1_ids, d1_ids = [], [], [], [], []
    for r in q.itertuples():
        p = cache.pools[r.qid]
        lex.append(lexical(r.query, int(r.n_tokens), int(r.n_tokens_in_vocab)))
        f = pool_features(p, int(r.n_tokens), int(r.n_tokens_in_vocab))
        so = np.argsort(p.sparse_rank, kind="stable")
        do = np.argsort(p.dense_rank, kind="stable")
        s10, d10 = p.sparse_score[so[:10]], p.dense_score[do[:10]]
        state.append([f[n] for n in FEATURE_NAMES] + [
            float(s10.std()), float(d10.std()),
            float((s10[0] - s10[1]) / abs(s10[0])) if len(s10) > 1 and s10[0] else 0.0,
            float((d10[0] - d10[1]) / abs(d10[0])) if len(d10) > 1 and d10[0] else 0.0])
        i_s, i_d = so[0], do[0]
        dat.append([p.sparse_score[i_s], p.dense_score[i_d], p.dense_score[i_s], p.sparse_score[i_d],
                    float(i_s == i_d)])
        s1_ids.append(p.doc_id[i_s])
        d1_ids.append(p.doc_id[i_d])
    embed = _doc_embedder(cfg, cache)
    log.info("embedding top-1 documents for %d queries", len(q))
    S1 = embed(s1_ids).astype(np.float32)
    D1 = embed(d1_ids).astype(np.float32)
    cos12 = (S1 * D1).sum(1) / (np.linalg.norm(S1, axis=1) * np.linalg.norm(D1, axis=1) + 1e-9)
    out = {"qid": q["qid"].to_numpy(dtype=object), "lex": np.array(lex, dtype=np.float64),
           "state": np.array(state, dtype=np.float64),
           "dat": np.column_stack([np.array(dat, dtype=np.float64), cos12]),
           "q_emb": cache.query_emb.astype(np.float32), "s1_emb": S1, "d1_emb": D1}
    np.savez(path, **out)
    return out


class FeatureBuilder:
    """Fit PCA + standardization on train rows; transform any rows."""

    def __init__(self, raw: dict[str, np.ndarray], train_rows: np.ndarray, pca_dim: int = 32, seed: int = 0):
        self.raw, self.dim = raw, pca_dim
        self.pca = {}
        for k in ("q_emb", "s1_emb", "d1_emb"):
            n = min(pca_dim, len(train_rows) - 1, raw[k].shape[1])
            self.pca[k] = PCA(n_components=n, random_state=seed).fit(raw[k][train_rows])
        self.blocks = {k: self.pca[k].transform(raw[k]) for k in self.pca}
        self.blocks.update({k: raw[k] for k in ("lex", "state", "dat")})
        self._stats = {}
        self.train_rows = train_rows

    def matrix(self, feature_set: str) -> np.ndarray:
        parts = {"query": ["q_emb", "lex"],
                 "dat": ["q_emb", "s1_emb", "d1_emb", "dat"],
                 "state": ["q_emb", "lex", "state"],
                 "all": ["q_emb", "lex", "state", "s1_emb", "d1_emb", "dat"]}[feature_set]
        X = np.column_stack([self.blocks[p] for p in parts]).astype(np.float64)
        mu = X[self.train_rows].mean(0)
        sd = X[self.train_rows].std(0)
        sd[sd < 1e-9] = 1.0
        return (X - mu) / sd
