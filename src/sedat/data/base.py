"""Generic retrieval dataset.

Labels (`qrels`) are deliberately kept apart from `queries` so that code building
decision-time state can be handed `queries` without any access to gold documents.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass
class RetrievalDataset:
    name: str
    language: str
    corpus: pd.DataFrame            # columns: doc_id, title, text   (row order = doc_idx)
    queries: pd.DataFrame           # columns: qid, query, split, group
    qrels: dict[str, list[str]]     # qid -> relevant doc_ids   (LABELS: evaluation only)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.corpus = self.corpus.reset_index(drop=True)
        self.queries = self.queries.reset_index(drop=True)
        self.doc_index = {d: i for i, d in enumerate(self.corpus["doc_id"])}

    def split(self, name: str) -> pd.DataFrame:
        return self.queries[self.queries["split"] == name]

    def summary(self) -> dict:
        return {"name": self.name, "n_docs": len(self.corpus),
                "n_queries": {s: int((self.queries["split"] == s).sum()) for s in SPLITS},
                "n_groups": {s: int(self.split(s)["group"].nunique()) for s in SPLITS}, **self.meta}


def group_split(groups: list[str], fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Assign whole groups (e.g. Wikipedia articles) to splits so that no article's
    questions leak across train/val/test. Returns group -> split."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    names = [s for s in SPLITS if fractions.get(s, 0) > 0]
    total = sum(fractions[s] for s in names)
    bounds = np.cumsum([fractions[s] / total for s in names]) * len(uniq)
    out, start = {}, 0
    for s, b in zip(names, bounds):
        end = int(round(b))
        for g in uniq[start:end]:
            out[g] = s
        start = end
    return out


def subsample_splits(queries: pd.DataFrame, max_queries: dict | None, seed: int) -> pd.DataFrame:
    if not max_queries:
        return queries
    parts = []
    for s, df in queries.groupby("split", sort=False):
        cap = max_queries.get(s)
        if cap is not None and len(df) > cap:
            df = df.sample(n=cap, random_state=seed)
        parts.append(df)
    return pd.concat(parts).sort_values(["split", "qid"]).reset_index(drop=True)
