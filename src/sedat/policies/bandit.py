"""Bandit policies over the discrete alpha grid (each alpha = one arm).

UCBAlpha      context-free: learns the single best alpha online. Isolates how much
              of any improvement is "just finding the global alpha from experience".
LinUCBAlpha   contextual (disjoint LinUCB): alpha_t = f(state_t, E_{<t}).
              The minimal genuinely self-improving, query-dependent policy.

Both support bandit feedback (only the chosen arm's reward) and full feedback
(Experience.counterfactual_rewards, every arm's reward revealed).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..experience.schema import Experience
from ..state import State
from .base import RetrievalPolicy


def _pick(values: np.ndarray, alphas: np.ndarray, prior: float, rng: np.random.Generator | None) -> int:
    best = np.flatnonzero(values >= values.max() - 1e-12)
    if rng is not None:
        return int(rng.choice(best))
    return int(best[np.argmin(np.abs(alphas[best] - prior))])


class UCBAlpha(RetrievalPolicy):
    name = "ucb_context_free"
    kind = "self_improving"

    def __init__(self, alphas: np.ndarray, c: float = 1.0, prior_alpha: float = 0.5, seed: int = 0):
        super().__init__(alphas)
        self.c, self.prior = c, prior_alpha
        self.rng = np.random.default_rng(seed)
        self.n = np.zeros(len(self.alphas))
        self.s = np.zeros(len(self.alphas))

    def select_alpha(self, state: State) -> float:
        mean = np.where(self.n > 0, self.s / np.maximum(self.n, 1), 0.0)
        if not self.training:
            if self.n.sum() == 0:
                return self.snap(self.prior)
            return float(self.alphas[_pick(np.where(self.n > 0, mean, -np.inf), self.alphas, self.prior, None)])
        if (self.n == 0).any():
            return float(self.alphas[self.rng.choice(np.flatnonzero(self.n == 0))])
        t = self.n.sum()
        ucb = mean + self.c * np.sqrt(2 * np.log(t) / self.n)
        return float(self.alphas[_pick(ucb, self.alphas, self.prior, self.rng)])

    def observe(self, exp: Experience) -> None:
        if not self.training:
            return
        if exp.counterfactual_rewards is not None:
            self.n += 1
            self.s += exp.counterfactual_rewards
        else:
            a = self.alpha_index(exp.chosen_alpha)
            self.n[a] += 1
            self.s[a] += exp.reward


class LinUCBAlpha(RetrievalPolicy):
    name = "linucb"
    kind = "self_improving"

    def __init__(self, alphas: np.ndarray, featurize: Callable[[State], np.ndarray], dim: int,
                 c: float = 1.0, ridge: float = 1.0, prior_alpha: float = 0.5, seed: int = 0):
        super().__init__(alphas)
        self.featurize, self.c, self.prior = featurize, c, prior_alpha
        self.rng = np.random.default_rng(seed)
        d = dim + 1                                                  # + bias
        K = len(self.alphas)
        self.Ainv = np.stack([np.eye(d) / ridge for _ in range(K)])  # (K, d, d)
        self.b = np.zeros((K, d))
        self.count = 0

    def _x(self, state: State) -> np.ndarray:
        return np.append(self.featurize(state), 1.0)

    def select_alpha(self, state: State) -> float:
        x = self._x(state)
        theta = np.einsum("kij,kj->ki", self.Ainv, self.b)
        mean = theta @ x
        if not self.training:
            return float(self.alphas[_pick(mean, self.alphas, self.prior, None)])
        width = np.sqrt(np.einsum("i,kij,j->k", x, self.Ainv, x))
        return float(self.alphas[_pick(mean + self.c * width, self.alphas, self.prior, self.rng)])

    def _update_arm(self, k: int, x: np.ndarray, r: float) -> None:
        Ax = self.Ainv[k] @ x                                        # Sherman-Morrison
        self.Ainv[k] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[k] += r * x

    def observe(self, exp: Experience) -> None:
        if not self.training:
            return
        x = np.append(exp.state_features, 1.0)
        if exp.counterfactual_rewards is not None:
            for k, r in enumerate(exp.counterfactual_rewards):
                self._update_arm(k, x, float(r))
        else:
            self._update_arm(self.alpha_index(exp.chosen_alpha), x, exp.reward)
        self.count += 1
