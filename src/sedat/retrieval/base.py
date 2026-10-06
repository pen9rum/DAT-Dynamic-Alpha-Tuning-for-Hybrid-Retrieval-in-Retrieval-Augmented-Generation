from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Retriever(ABC):
    """A retriever that can produce exact scores for every corpus document.

    Exhaustive scoring is affordable at DAT-scale corpora (thousands to tens of
    thousands of passages) and gives exact scores for both retrievers on every
    candidate in the union pool -- no missing-score imputation in fusion.
    """
    name: str

    @abstractmethod
    def index(self, texts: list[str]) -> None: ...

    @abstractmethod
    def score_all(self, queries: list[str]) -> np.ndarray:
        """Return a (n_queries, n_docs) float32 score matrix."""
