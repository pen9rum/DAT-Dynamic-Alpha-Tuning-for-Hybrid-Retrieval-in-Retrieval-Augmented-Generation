"""Okapi BM25 as a precomputed sparse doc-term weight matrix.

score(q, d) = sum_{t in q} idf(t) * tf(t,d)(k1+1) / (tf(t,d) + k1(1 - b + b |d|/avgdl))

With W[d, t] precomputed, scoring a batch of queries is one sparse mat-mul and
gives exact scores for *every* document, which the candidate cache needs.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Retriever
from .tokenize import get_tokenizer


class BM25(Retriever):
    name = "sparse"

    def __init__(self, k1: float = 0.9, b: float = 0.4, tokenizer: str = "en"):
        self.k1, self.b = k1, b
        self.tok = get_tokenizer(tokenizer)
        self.vocab: dict[str, int] = {}
        self.W: sp.csr_matrix | None = None

    def index(self, texts: list[str]) -> None:
        rows, cols, vals = [], [], []
        for i, text in enumerate(texts):
            counts: dict[int, int] = {}
            for t in self.tok(text):
                j = self.vocab.setdefault(t, len(self.vocab))
                counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        tf = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                           shape=(len(texts), len(self.vocab)))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if len(dl) else 1.0
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        n = len(texts)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)   # Lucene idf (non-negative)
        tf = tf.tocoo()
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (self.k1 + 1) / denom
        self.W = sp.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)

    def _query_matrix(self, queries: list[str]) -> sp.csr_matrix:
        rows, cols, vals = [], [], []
        for i, q in enumerate(queries):
            for t in self.tok(q):
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i); cols.append(j); vals.append(1.0)
        return sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                             shape=(len(queries), len(self.vocab)))

    def score_all(self, queries: list[str]) -> np.ndarray:
        assert self.W is not None, "call index() first"
        return np.asarray((self._query_matrix(queries) @ self.W.T).todense(), dtype=np.float32)

    def n_query_terms(self, query: str) -> tuple[int, int]:
        """(#tokens, #tokens found in the vocabulary) -- a decision-time feature."""
        toks = self.tok(query)
        return len(toks), sum(t in self.vocab for t in toks)
