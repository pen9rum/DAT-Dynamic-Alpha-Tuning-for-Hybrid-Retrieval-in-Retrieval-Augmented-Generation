from __future__ import annotations

import numpy as np

from ..evaluation.oracle_analysis import oracle_select
from ..state import State
from .base import RetrievalPolicy


class OracleAlpha(RetrievalPolicy):
    """Per-query oracle: picks the alpha that maximizes this query's reward using its
    labels. An UPPER BOUND for analysis -- never a deployable baseline."""
    name = "per_query_oracle"
    kind = "oracle"
    deployable = False

    def __init__(self, alphas: np.ndarray, qids: list[str], R: np.ndarray,
                 tie_break: str = "closest_to_global", global_idx: int | None = None):
        super().__init__(alphas)
        idx = oracle_select(R, self.alphas, tie_break, global_idx)
        self._alpha = {q: float(self.alphas[i]) for q, i in zip(qids, idx)}

    def select_alpha(self, state: State) -> float:
        return self._alpha[state.qid]
