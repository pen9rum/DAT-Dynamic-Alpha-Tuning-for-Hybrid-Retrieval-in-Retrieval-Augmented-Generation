"""Dense bi-encoder retrieval (SentenceTransformers), cosine similarity."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import Retriever

log = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, model: str, query_prefix: str = "", passage_prefix: str = "",
                 batch_size: int = 64, max_seq_length: int | None = 512, device: str = "auto",
                 cache_path: Path | None = None):
        from sentence_transformers import SentenceTransformer
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model, device=device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.doc_emb: np.ndarray | None = None

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        emb = self.model.encode([prefix + t for t in texts], batch_size=self.batch_size,
                                normalize_embeddings=True, show_progress_bar=len(texts) > 256,
                                convert_to_numpy=True)
        return emb.astype(np.float32)

    def index(self, texts: list[str]) -> None:
        if self.cache_path is not None and self.cache_path.exists():
            emb = np.load(self.cache_path)
            if emb.shape[0] == len(texts):
                log.info("loaded cached doc embeddings %s", self.cache_path)
                self.doc_emb = emb
                return
        log.info("encoding %d passages on %s", len(texts), self.model.device)
        self.doc_emb = self.encode(texts, self.passage_prefix)
        if self.cache_path is not None:
            np.save(self.cache_path, self.doc_emb)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.encode(queries, self.query_prefix)

    def score_all(self, queries: list[str], query_emb: np.ndarray | None = None) -> np.ndarray:
        assert self.doc_emb is not None, "call index() first"
        q = self.encode_queries(queries) if query_emb is None else query_emb
        return q @ self.doc_emb.T
