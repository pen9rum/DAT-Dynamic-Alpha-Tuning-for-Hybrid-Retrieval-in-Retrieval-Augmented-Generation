from __future__ import annotations

import numpy as np

from ..evaluation.oracle_analysis import tune_global_alpha
from ..state import State
from .base import RetrievalPolicy


class FixedAlpha(RetrievalPolicy):
    """alpha = constant (sparse-only = 0, dense-only = 1, DAT's fixed hybrid, ...)."""
    kind = "static"

    def __init__(self, alphas: np.ndarray, alpha: float):
        super().__init__(alphas)
        self.alpha = self.snap(alpha)
        self.name = f"fixed_{self.alpha:.2f}"

    def select_alpha(self, state: State) -> float:
        return self.alpha


class GlobalTunedAlpha(FixedAlpha):
    """One alpha tuned on validation rewards, frozen for test."""

    def __init__(self, alphas: np.ndarray, R_val: np.ndarray):
        g = tune_global_alpha(R_val, np.asarray(alphas))
        super().__init__(alphas, float(alphas[g]))
        self.name = f"global_tuned({self.alpha:.2f})"
