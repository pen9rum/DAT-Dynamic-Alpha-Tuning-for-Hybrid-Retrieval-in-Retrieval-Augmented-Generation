"""Policy registry. Add new policies here so experiments can select them by name."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..state import State
from .bandit import LinUCBAlpha, UCBAlpha
from .base import RetrievalPolicy
from .dat_style import DATAlpha
from .fixed import FixedAlpha, GlobalTunedAlpha
from .oracle import OracleAlpha

__all__ = ["RetrievalPolicy", "FixedAlpha", "GlobalTunedAlpha", "OracleAlpha", "DATAlpha",
           "UCBAlpha", "LinUCBAlpha", "make_policy"]


def make_policy(name: str, alphas: np.ndarray, *, R_val: np.ndarray | None = None,
                featurize: Callable[[State], np.ndarray] | None = None, dim: int | None = None,
                params: dict | None = None, seed: int = 0) -> RetrievalPolicy:
    p = dict(params or {})
    if name == "sparse_only":
        return FixedAlpha(alphas, 0.0)
    if name == "dense_only":
        return FixedAlpha(alphas, 1.0)
    if name.startswith("fixed_"):
        return FixedAlpha(alphas, float(name.split("_", 1)[1]))
    if name == "global_tuned":
        assert R_val is not None, "global_tuned needs validation rewards"
        return GlobalTunedAlpha(alphas, R_val)
    if name == "ucb":
        return UCBAlpha(alphas, seed=seed, **p)
    if name == "linucb":
        assert featurize is not None and dim is not None
        return LinUCBAlpha(alphas, featurize, dim, seed=seed, **p)
    raise ValueError(f"unknown policy {name!r} (oracle / dat_style are constructed explicitly)")
