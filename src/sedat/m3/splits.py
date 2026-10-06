"""Milestone 3 split discipline: train -> fit, val -> tune, test -> frozen evaluation.

Sources (per dataset, set in configs/m3/<name>.yaml under `m3.split`):
  inherit        keep the Milestone-2 cache's train/val/test (SQuAD: article-level split)
  deterministic  re-split the Milestone-2 queries once (seeded, stratified by group)
  official       BEIR official queries/qrels: test = the Milestone-2 (official test) queries,
                 train/val from the official train/dev files (sampled, seeded)

The resulting query-id -> split map is written once to runs/milestone3/splits/<name>.csv and
re-read verbatim afterwards (frozen before any policy is fit).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import resolve

log = logging.getLogger(__name__)

HF = "https://huggingface.co/datasets/BeIR"


def _get(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def beir_official(name: str, split: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Official BEIR queries + qrels for a split ('train' / 'dev' / 'test')."""
    root = resolve(f"data/raw/beir/{name}")
    qp = _get(f"{HF}/{name}/resolve/main/queries/queries-00000-of-00001.parquet", root / "queries.parquet")
    rp = _get(f"{HF}/{name}-qrels/resolve/main/{split}.tsv", root / f"qrels_{split}.tsv")
    qrels_df = pd.read_csv(rp, sep="\t", dtype=str)
    qrels_df.columns = ["qid", "doc_id", "score"]
    qrels_df = qrels_df[qrels_df["score"].astype(float) > 0]
    qrels = qrels_df.groupby("qid")["doc_id"].apply(list).to_dict()
    q = pd.read_parquet(qp)
    idc = "_id" if "_id" in q.columns else "id"
    q = q.rename(columns={idc: "qid", "text": "query"})[["qid", "query"]]
    q["qid"] = q["qid"].astype(str)
    q = q[q["qid"].isin(set(qrels))].reset_index(drop=True)
    return q, {k: qrels[k] for k in q["qid"]}


def deterministic_split(df: pd.DataFrame, fractions: dict, seed: int, by: str = "group") -> pd.Series:
    rng = np.random.default_rng(seed)
    names = [s for s in ("train", "val", "test") if fractions.get(s, 0) > 0]
    tot = sum(fractions[s] for s in names)
    out = pd.Series("", index=df.index, dtype=object)
    for _, idx in df.groupby(by, sort=True).groups.items():
        idx = rng.permutation(np.asarray(sorted(idx)))
        cuts = np.round(np.cumsum([fractions[s] / tot for s in names]) * len(idx)).astype(int)
        start = 0
        for s, c in zip(names, cuts):
            out.loc[idx[start:c]] = s
            start = c
    return out


def frozen_split_path(name: str) -> Path:
    return resolve(f"runs/milestone3/splits/{name}.csv")
