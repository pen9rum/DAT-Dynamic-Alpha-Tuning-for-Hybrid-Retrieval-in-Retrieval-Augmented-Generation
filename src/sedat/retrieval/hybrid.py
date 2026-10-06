"""Hybrid fusion R(q,d; alpha) = alpha * dense~ + (1 - alpha) * sparse~.

alpha = 0 -> sparse only, alpha = 1 -> dense only (DAT's convention).
"""
from __future__ import annotations

import numpy as np

from .normalization import normalize


def fuse(sparse_norm: np.ndarray, dense_norm: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """(n_alpha, n_docs) fused scores. Rounded so float noise cannot break ties."""
    a = np.asarray(alphas, dtype=np.float64)[:, None]
    return np.round(a * dense_norm[None, :] + (1 - a) * sparse_norm[None, :], 10)


def rank_orders(fused: np.ndarray, sparse_norm: np.ndarray, dense_norm: np.ndarray) -> np.ndarray:
    """Deterministic ranking for each alpha row: by fused score, ties broken by the
    unweighted mean of both normalized scores, then by pool position (which the
    candidate cache orders by corpus index). Returns (n_alpha, n_docs) indices."""
    secondary = np.round((sparse_norm + dense_norm) / 2, 10)
    pos = np.arange(fused.shape[1])
    return np.stack([np.lexsort((pos, -secondary, -row)) for row in fused])


def rank_pool(sparse_raw: np.ndarray, dense_raw: np.ndarray, sparse_full_rank: np.ndarray,
              dense_full_rank: np.ndarray, alphas: np.ndarray, method: str = "minmax",
              rrf_k: int = 60) -> np.ndarray:
    """Convenience: normalize + fuse + rank for one query's candidate pool."""
    s = normalize(sparse_raw, sparse_full_rank, method, rrf_k)
    d = normalize(dense_raw, dense_full_rank, method, rrf_k)
    return rank_orders(fuse(s, d, alphas), s, d)
