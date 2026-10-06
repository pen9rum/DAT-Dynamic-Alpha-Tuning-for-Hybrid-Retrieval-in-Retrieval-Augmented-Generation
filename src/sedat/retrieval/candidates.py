"""Candidate cache: retrieve once, fuse under any alpha for free.

For each query: take the union of sparse top-K and dense top-K; store the *exact*
raw scores of both retrievers for every pool doc plus each doc's full-corpus rank
under each retriever. Every downstream experiment (alpha sweep, oracle, policy
learning) reads this cache and never re-runs retrieval.

Files (in the retrieval cache dir):
  corpus.parquet       doc_id, title, text
  queries.parquet      qid, query, split, group, n_tokens, n_tokens_in_vocab   (decision-time)
  candidates.parquet   qid, doc_idx, doc_id, sparse_score, dense_score,
                       sparse_rank, dense_rank, in_sparse_topk, in_dense_topk  (decision-time)
  labels.parquet       qid, gold_doc_ids, sparse_gold_rank, dense_gold_rank     (LABELS)
  query_emb.npy        dense query embeddings, rows aligned with queries.parquet
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.base import RetrievalDataset

log = logging.getLogger(__name__)

FILES = ("corpus.parquet", "queries.parquet", "candidates.parquet", "labels.parquet")


def cache_exists(cache_dir: Path) -> bool:
    return all((cache_dir / f).exists() for f in FILES)


def _full_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every doc for each query row (ties -> order by doc index)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return ranks


def build_candidates(ds: RetrievalDataset, sparse, dense, sparse_top_k: int, dense_top_k: int,
                     cache_dir: Path, batch_size: int = 256) -> None:
    texts = ds.corpus["text"].tolist()
    log.info("indexing BM25 over %d docs", len(texts))
    sparse.index(texts)
    dense.index(texts)

    q = ds.queries.copy()
    ntok = [sparse.n_query_terms(t) for t in q["query"]]
    q["n_tokens"] = [a for a, _ in ntok]
    q["n_tokens_in_vocab"] = [b for _, b in ntok]

    log.info("encoding %d queries", len(q))
    q_emb = dense.encode_queries(q["query"].tolist())
    np.save(cache_dir / "query_emb.npy", q_emb)

    cand_parts, label_rows = [], []
    for start in tqdm(range(0, len(q), batch_size), desc="candidates"):
        chunk = q.iloc[start:start + batch_size]
        s_all = sparse.score_all(chunk["query"].tolist())
        d_all = q_emb[start:start + batch_size] @ dense.doc_emb.T
        s_rank, d_rank = _full_ranks(s_all), _full_ranks(d_all)
        for i, qid in enumerate(chunk["qid"]):
            in_s = s_rank[i] <= sparse_top_k
            in_d = d_rank[i] <= dense_top_k
            pool = np.flatnonzero(in_s | in_d)            # sorted by corpus index
            cand_parts.append(pd.DataFrame({
                "qid": qid, "doc_idx": pool.astype(np.int32),
                "sparse_score": s_all[i, pool], "dense_score": d_all[i, pool],
                "sparse_rank": s_rank[i, pool].astype(np.int32),
                "dense_rank": d_rank[i, pool].astype(np.int32),
                "in_sparse_topk": in_s[pool], "in_dense_topk": in_d[pool]}))
            gold_idx = [ds.doc_index[g] for g in ds.qrels[qid]]
            label_rows.append({"qid": qid, "gold_doc_ids": list(ds.qrels[qid]),
                               "sparse_gold_rank": int(min(s_rank[i, gold_idx])),
                               "dense_gold_rank": int(min(d_rank[i, gold_idx]))})

    cands = pd.concat(cand_parts, ignore_index=True)
    cands["doc_id"] = ds.corpus["doc_id"].to_numpy()[cands["doc_idx"].to_numpy()]
    write_cache(cache_dir, ds.corpus, q, cands, pd.DataFrame(label_rows))


def write_cache(cache_dir: Path, corpus: pd.DataFrame, queries: pd.DataFrame,
                candidates: pd.DataFrame, labels: pd.DataFrame) -> None:
    corpus.to_parquet(cache_dir / "corpus.parquet", index=False)
    queries.to_parquet(cache_dir / "queries.parquet", index=False)
    candidates.to_parquet(cache_dir / "candidates.parquet", index=False)
    labels.to_parquet(cache_dir / "labels.parquet", index=False)
    log.info("wrote candidate cache to %s (%d queries, %.1f cands/query)", cache_dir,
             len(queries), len(candidates) / max(len(queries), 1))


@dataclass
class Pool:
    """One query's candidate pool as aligned arrays (decision-time information only)."""
    qid: str
    doc_idx: np.ndarray
    doc_id: np.ndarray
    sparse_score: np.ndarray
    dense_score: np.ndarray
    sparse_rank: np.ndarray
    dense_rank: np.ndarray
    in_sparse_topk: np.ndarray
    in_dense_topk: np.ndarray


class CandidateCache:
    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir)
        if not cache_exists(self.dir):
            raise FileNotFoundError(f"no candidate cache in {self.dir}; run experiments.cache_retrieval")
        self.queries = pd.read_parquet(self.dir / "queries.parquet")
        self.candidates = pd.read_parquet(self.dir / "candidates.parquet")
        self._corpus: pd.DataFrame | None = None
        self._labels: pd.DataFrame | None = None
        self._pools: dict[str, Pool] | None = None
        emb = self.dir / "query_emb.npy"
        self.query_emb = np.load(emb) if emb.exists() else None

    # decision-time ------------------------------------------------------------
    @property
    def pools(self) -> dict[str, Pool]:
        if self._pools is None:
            self._pools = {}
            for qid, g in self.candidates.groupby("qid", sort=False):
                self._pools[qid] = Pool(qid, g["doc_idx"].to_numpy(), g["doc_id"].to_numpy(),
                                        g["sparse_score"].to_numpy(np.float64),
                                        g["dense_score"].to_numpy(np.float64),
                                        g["sparse_rank"].to_numpy(), g["dense_rank"].to_numpy(),
                                        g["in_sparse_topk"].to_numpy(bool), g["in_dense_topk"].to_numpy(bool))
        return self._pools

    @property
    def corpus(self) -> pd.DataFrame:
        if self._corpus is None:
            self._corpus = pd.read_parquet(self.dir / "corpus.parquet")
        return self._corpus

    def doc_text(self, doc_idx: int) -> str:
        return self.corpus["text"].iat[int(doc_idx)]

    def split_qids(self, split: str) -> list[str]:
        return self.queries.loc[self.queries["split"] == split, "qid"].tolist()

    # labels: evaluation / analysis only ----------------------------------------
    @property
    def labels(self) -> pd.DataFrame:
        if self._labels is None:
            self._labels = pd.read_parquet(self.dir / "labels.parquet").set_index("qid")
        return self._labels
