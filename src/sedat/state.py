"""Decision-time state.

A `State` contains ONLY what is legally available when the policy chooses alpha:
the query, both retrievers' outputs (ids, raw scores, optionally passage texts)
and features computed from them. It is built from `queries.parquet` and
`candidates.parquet` -- never from `labels.parquet`.

Past experience E_{<t} is NOT part of State: a self-improving policy keeps its
own memory (see experience/) and conditions on it internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retrieval.candidates import CandidateCache, Pool

FEATURE_NAMES = [
    "sparse_top1", "sparse_margin12", "sparse_top1_minus_mean10", "sparse_entropy10",
    "dense_top1", "dense_margin12", "dense_top1_minus_mean10", "dense_entropy10",
    "top1_agree", "jaccard5", "jaccard10",
    "log2_dense_rank_of_sparse_top1", "log2_sparse_rank_of_dense_top1",
    "pool_overlap", "n_query_tokens", "frac_query_tokens_in_vocab",
]


@dataclass(frozen=True)
class State:
    qid: str
    query: str
    sparse_top: list[tuple[str, float]]          # (doc_id, raw score), best first
    dense_top: list[tuple[str, float]]
    features: dict[str, float]
    sparse_top_texts: list[str] = field(default_factory=list)
    dense_top_texts: list[str] = field(default_factory=list)
    query_embedding: np.ndarray | None = None

    def feature_vector(self, names: list[str] = FEATURE_NAMES) -> np.ndarray:
        return np.array([self.features[n] for n in names], dtype=np.float64)


def _entropy(x: np.ndarray) -> float:
    z = x - x.max()
    p = np.exp(z) / np.exp(z).sum()
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def _side(scores: np.ndarray, order: np.ndarray, prefix: str) -> dict[str, float]:
    top = scores[order[:10]]
    return {f"{prefix}_top1": float(top[0]),
            f"{prefix}_margin12": float(top[0] - top[1]) if len(top) > 1 else 0.0,
            f"{prefix}_top1_minus_mean10": float(top[0] - top.mean()),
            f"{prefix}_entropy10": _entropy(top)}


def pool_features(pool: Pool, n_tokens: int, n_in_vocab: int) -> dict[str, float]:
    s_ord = np.argsort(pool.sparse_rank, kind="stable")
    d_ord = np.argsort(pool.dense_rank, kind="stable")
    f = {**_side(pool.sparse_score, s_ord, "sparse"), **_side(pool.dense_score, d_ord, "dense")}
    s5, d5 = set(pool.doc_idx[s_ord[:5]]), set(pool.doc_idx[d_ord[:5]])
    s10, d10 = set(pool.doc_idx[s_ord[:10]]), set(pool.doc_idx[d_ord[:10]])
    f.update({
        "top1_agree": float(s_ord[0] == d_ord[0]),
        "jaccard5": len(s5 & d5) / len(s5 | d5),
        "jaccard10": len(s10 & d10) / len(s10 | d10),
        "log2_dense_rank_of_sparse_top1": float(np.log2(pool.dense_rank[s_ord[0]])),
        "log2_sparse_rank_of_dense_top1": float(np.log2(pool.sparse_rank[d_ord[0]])),
        "pool_overlap": float((pool.in_sparse_topk & pool.in_dense_topk).mean()),
        "n_query_tokens": float(n_tokens),
        "frac_query_tokens_in_vocab": n_in_vocab / n_tokens if n_tokens else 0.0,
    })
    return f


def build_states(cache: CandidateCache, qids: list[str] | None = None, top_n: int = 3,
                 with_texts: bool = False) -> dict[str, State]:
    q = cache.queries.set_index("qid")
    row_of = {qid: i for i, qid in enumerate(cache.queries["qid"])}
    qids = qids if qids is not None else q.index.tolist()
    states = {}
    for qid in qids:
        pool = cache.pools[qid]
        s_ord = np.argsort(pool.sparse_rank, kind="stable")[:top_n]
        d_ord = np.argsort(pool.dense_rank, kind="stable")[:top_n]
        states[qid] = State(
            qid=qid, query=q.at[qid, "query"],
            sparse_top=[(pool.doc_id[i], float(pool.sparse_score[i])) for i in s_ord],
            dense_top=[(pool.doc_id[i], float(pool.dense_score[i])) for i in d_ord],
            features=pool_features(pool, int(q.at[qid, "n_tokens"]), int(q.at[qid, "n_tokens_in_vocab"])),
            sparse_top_texts=[cache.doc_text(pool.doc_idx[i]) for i in s_ord] if with_texts else [],
            dense_top_texts=[cache.doc_text(pool.doc_idx[i]) for i in d_ord] if with_texts else [],
            query_embedding=cache.query_emb[row_of[qid]] if cache.query_emb is not None else None,
        )
    return states


class FeatureScaler:
    """Standardizer fit on *unlabeled* decision-time features (e.g. the train split)."""

    def __init__(self, names: list[str] = FEATURE_NAMES):
        self.names = names
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, states: list[State]) -> "FeatureScaler":
        X = np.stack([s.feature_vector(self.names) for s in states])
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def __call__(self, state: State) -> np.ndarray:
        return (state.feature_vector(self.names) - self.mu) / self.sd
