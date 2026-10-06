"""Common policy interface.

The retrieval/evaluation machinery is identical no matter where alpha comes from.
Every policy declares which rung of the ladder it sits on:

  static          alpha = constant
  adaptive        alpha = f(current query / current retrieval outcome)        (DAT)
  self_improving  alpha_t = f(current state, accumulated experience E_{<t})   (target)
  oracle          uses labels of the query being decided -- analysis only
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..experience.schema import Experience
from ..state import State

KINDS = ("static", "adaptive", "self_improving", "oracle")


class RetrievalPolicy(ABC):
    name: str = "policy"
    kind: str = "static"
    deployable: bool = True

    def __init__(self, alphas: np.ndarray):
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.training = True

    @abstractmethod
    def select_alpha(self, state: State) -> float:
        """Choose alpha using decision-time information only."""

    def observe(self, experience: Experience) -> None:
        """Receive post-hoc feedback. Static/adaptive policies ignore it."""

    def update(self) -> None:
        """Improve the decision rule from observed experience (no-op by default)."""

    def train(self, mode: bool = True) -> "RetrievalPolicy":
        """train(False) = greedy evaluation mode (no exploration, no learning)."""
        self.training = mode
        return self

    def eval(self) -> "RetrievalPolicy":
        return self.train(False)

    def snap(self, alpha: float) -> float:
        """Project an arbitrary alpha onto the grid (reward tables exist only there)."""
        return float(self.alphas[int(np.argmin(np.abs(self.alphas - alpha)))])

    def alpha_index(self, alpha: float) -> int:
        return int(np.argmin(np.abs(self.alphas - alpha)))
