"""Counterfactual reward table: every query x every alpha.

Long format, one row per (qid, alpha):
  qid, split, alpha, alpha_idx, gold_rank, <metric columns>, top_doc_ids, top_scores

Once built, any alpha-selection policy can be evaluated (offline or as a simulated
online stream) by table lookup -- no retrieval, no fusion.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import alpha_grid, get_dotted, reward_table_path
from ..retrieval.candidates import CandidateCache
from ..retrieval.hybrid import fuse, rank_orders
from ..retrieval.normalization import normalize
from .metrics import all_metrics

log = logging.getLogger(__name__)


def build_reward_table(cache: CandidateCache, alphas: np.ndarray, normalization: str,
                       rrf_k: int = 60, ks: list[int] = (1, 3, 5, 10, 20), ndcg_k: int = 10,
                       topk_store: int = 10) -> pd.DataFrame:
    labels = cache.labels
    split_of = dict(zip(cache.queries["qid"], cache.queries["split"]))
    A = len(alphas)
    cols: dict[str, list] = {c: [] for c in ("qid", "split", "alpha", "alpha_idx", "gold_rank",
                                             "top_doc_ids", "top_scores")}
    rank_mats, n_golds = [], []
    for qid, pool in tqdm(cache.pools.items(), desc=f"reward table [{normalization}]"):
        s = normalize(pool.sparse_score, pool.sparse_rank, normalization, rrf_k)
        d = normalize(pool.dense_score, pool.dense_rank, normalization, rrf_k)
        fused = fuse(s, d, alphas)
        orders = rank_orders(fused, s, d)
        ranks = np.empty_like(orders)
        ranks[np.arange(A)[:, None], orders] = np.arange(1, orders.shape[1] + 1)[None, :]

        gold = list(labels.at[qid, "gold_doc_ids"])
        pos = {doc: i for i, doc in enumerate(pool.doc_id)}
        gr = np.full((A, len(gold)), np.inf)
        for j, g in enumerate(gold):
            if g in pos:
                gr[:, j] = ranks[:, pos[g]]
        rank_mats.append(gr)
        n_golds.append(len(gold))

        for a in range(A):
            top = orders[a, :topk_store]
            cols["qid"].append(qid)
            cols["split"].append(split_of[qid])
            cols["alpha"].append(float(alphas[a]))
            cols["alpha_idx"].append(a)
            cols["gold_rank"].append(float(gr[a].min()))
            cols["top_doc_ids"].append(pool.doc_id[top].tolist())
            cols["top_scores"].append(fused[a, top].tolist())

    G = max(n_golds)
    R = np.full((len(rank_mats) * A, G), np.inf)
    for i, gr in enumerate(rank_mats):
        R[i * A:(i + 1) * A, :gr.shape[1]] = gr
    ng = np.repeat(np.asarray(n_golds, dtype=np.float64), A)
    df = pd.DataFrame(cols)
    for name, vals in all_metrics(R, ng, list(ks), ndcg_k).items():
        df[name] = vals
    return df


def load_or_build_reward_table(cfg: dict, cache: CandidateCache, normalization: str | None = None,
                               force: bool = False) -> pd.DataFrame:
    norm = normalization or get_dotted(cfg, "fusion.normalization", "minmax")
    path: Path = reward_table_path(cfg, norm)
    if path.exists() and not force:
        log.info("loading reward table %s", path.name)
        return pd.read_parquet(path)
    df = build_reward_table(cache, alpha_grid(cfg), norm,
                            rrf_k=get_dotted(cfg, "fusion.rrf_k", 60),
                            ks=get_dotted(cfg, "evaluation.metrics_k", [1, 3, 5, 10, 20]),
                            ndcg_k=get_dotted(cfg, "evaluation.ndcg_k", 10),
                            topk_store=get_dotted(cfg, "evaluation.topk_store", 10))
    df.to_parquet(path, index=False)
    log.info("wrote reward table %s (%d rows)", path, len(df))
    return df


def reward_matrix(table: pd.DataFrame, metric: str, qids: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Wide view: (qids, R) with R[i, a] = metric of query i under alpha index a."""
    wide = table.pivot(index="qid", columns="alpha_idx", values=metric)
    if qids is not None:
        wide = wide.loc[qids]
    return wide.index.tolist(), wide.to_numpy(dtype=np.float64)
