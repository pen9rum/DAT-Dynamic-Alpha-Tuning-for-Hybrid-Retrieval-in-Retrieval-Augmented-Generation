"""Retrieval metrics from gold ranks (1-based; np.inf = not in candidate pool).

All metrics are vectorized over a (n_rows, n_gold) rank matrix, padded with inf.
"""
from __future__ import annotations

import numpy as np


def hit_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    return (np.min(ranks, axis=-1) <= k).astype(np.float64)


def recall_at_k(ranks: np.ndarray, k: int, n_gold: np.ndarray) -> np.ndarray:
    return (ranks <= k).sum(axis=-1) / n_gold


def allrel_at_k(ranks: np.ndarray, k: int, n_gold: np.ndarray) -> np.ndarray:
    """All-support success: every relevant doc is in the top-k (multi-hop evidence)."""
    return ((ranks <= k).sum(axis=-1) >= n_gold).astype(np.float64)


def mrr(ranks: np.ndarray) -> np.ndarray:
    return 1.0 / np.min(ranks, axis=-1)


def ndcg_at_k(ranks: np.ndarray, k: int, n_gold: np.ndarray) -> np.ndarray:
    gains = np.where(ranks <= k, 1.0 / np.log2(np.minimum(ranks, 1e12) + 1), 0.0).sum(axis=-1)
    ideal = np.array([np.sum(1.0 / np.log2(np.arange(2, min(int(g), k) + 2))) for g in np.atleast_1d(n_gold)])
    return gains / ideal


def all_metrics(ranks: np.ndarray, n_gold: np.ndarray, ks: list[int], ndcg_k: int) -> dict[str, np.ndarray]:
    out = {f"hit@{k}": hit_at_k(ranks, k) for k in ks}
    out.update({f"recall@{k}": recall_at_k(ranks, k, n_gold) for k in ks})
    out.update({f"allrel@{k}": allrel_at_k(ranks, k, n_gold) for k in ks})
    out["mrr"] = mrr(ranks)
    out[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranks, ndcg_k, n_gold)
    return out
