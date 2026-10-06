"""Per-query score normalization over the candidate pool.

Alpha is only meaningful relative to a normalization: the same alpha means
different things under min-max and z-score. Every reward table is keyed on the
normalization that produced it.
"""
from __future__ import annotations

import numpy as np

METHODS = ("minmax", "zscore", "rank", "rrf")


def normalize(raw: np.ndarray, full_rank: np.ndarray, method: str, rrf_k: int = 60) -> np.ndarray:
    """raw: raw retriever scores of the pool docs. full_rank: 1-based rank of each
    pool doc in that retriever's full-corpus ranking (used by rrf)."""
    raw = np.asarray(raw, dtype=np.float64)
    if method == "minmax":
        lo, hi = raw.min(), raw.max()
        return np.zeros_like(raw) if hi - lo < 1e-12 else (raw - lo) / (hi - lo)
    if method == "zscore":
        sd = raw.std()
        return np.zeros_like(raw) if sd < 1e-12 else (raw - raw.mean()) / sd
    if method == "rank":
        # linear rank within the pool: best -> 1, worst -> 0 (ties share the better rank)
        n = len(raw)
        if n == 1:
            return np.ones(1)
        pool_rank = (raw[None, :] > raw[:, None]).sum(axis=1)   # 0-based
        return 1.0 - pool_rank / (n - 1)
    if method == "rrf":
        return 1.0 / (rrf_k + np.asarray(full_rank, dtype=np.float64))
    raise ValueError(f"unknown normalization {method!r}; choose from {METHODS}")
