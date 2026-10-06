from __future__ import annotations

from collections import deque
from typing import Iterator

import numpy as np

from .schema import Experience


class ReplayBuffer:
    """FIFO experience store with optional capacity and uniform sampling."""

    def __init__(self, capacity: int | None = None, seed: int = 0):
        self.buf: deque[Experience] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, exp: Experience) -> None:
        self.buf.append(exp)

    def __len__(self) -> int:
        return len(self.buf)

    def __iter__(self) -> Iterator[Experience]:
        return iter(self.buf)

    def sample(self, n: int) -> list[Experience]:
        idx = self.rng.choice(len(self.buf), size=min(n, len(self.buf)), replace=False)
        return [self.buf[i] for i in idx]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(features X, chosen alphas, rewards) for all stored experiences."""
        if not self.buf:
            return np.empty((0, 0)), np.empty(0), np.empty(0)
        X = np.stack([e.state_features for e in self.buf])
        return X, np.array([e.chosen_alpha for e in self.buf]), np.array([e.reward for e in self.buf])

    def nearest(self, x: np.ndarray, k: int) -> list[Experience]:
        """k most similar past experiences by Euclidean distance in feature space."""
        if not self.buf:
            return []
        X = np.stack([e.state_features for e in self.buf])
        idx = np.argsort(((X - x) ** 2).sum(axis=1))[:k]
        return [self.buf[i] for i in idx]
