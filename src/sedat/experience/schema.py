"""Experience records.

Fields are grouped by *when* the information exists:
  decision-time : known before choosing alpha (same content as State)
  feedback      : revealed after retrieval + evaluation (requires labels)
  oracle        : analysis only; the online runner leaves it None unless
                  explicitly asked (expose_oracle=True), so policies cannot
                  accidentally train on it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Experience:
    # decision-time
    step: int
    qid: str
    query: str
    state_features: np.ndarray
    chosen_alpha: float
    # feedback
    reward: float
    gold_rank: float | None = None
    sparse_rank: float | None = None                  # gold rank under alpha=0 (from reward table)
    dense_rank: float | None = None                   # gold rank under alpha=1
    counterfactual_rewards: np.ndarray | None = None  # only in feedback="full" mode
    # oracle (analysis only)
    oracle_alpha: float | None = None
    metadata: dict = field(default_factory=dict)
