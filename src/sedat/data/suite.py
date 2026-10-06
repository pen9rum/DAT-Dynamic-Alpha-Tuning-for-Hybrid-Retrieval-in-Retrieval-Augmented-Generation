"""Query/qrels loading for the cross-dataset suite (Pyserini backend).

BEIR (nq, fiqa, scifact, hotpotqa): test topics + qrels shipped with Pyserini.
Local multi-hop (musique, 2wiki): questions.parquet from data/multihop_local.py.

Split protocol (identical for all datasets): the evaluation queries are split once
into val / test with a seeded random permutation (stratified by `group`).
Global alpha is tuned on val; everything is reported on test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import resolve


def _split(df: pd.DataFrame, fractions: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = [s for s in ("train", "val", "test") if fractions.get(s, 0) > 0]
    tot = sum(fractions[s] for s in names)
    out = df.copy()
    out["split"] = ""
    for _, idx in out.groupby("group").groups.items():
        idx = rng.permutation(np.asarray(idx))
        cuts = np.round(np.cumsum([fractions[s] / tot for s in names]) * len(idx)).astype(int)
        start = 0
        for s, c in zip(names, cuts):
            out.loc[idx[start:c], "split"] = s
            start = c
    return out


def load_suite_queries(ds_cfg: dict, seed: int) -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame]:
    """Returns (queries[qid, query, group], qrels qid->gold ids, extra labels[qid, n_hops, qtype])."""
    loader = ds_cfg["loader"]
    if loader == "beir_pyserini":
        from pyserini.search import get_qrels, get_topics
        name = f"beir-v1.0.0-{ds_cfg['beir_name']}-test"
        topics, qrels_raw = get_topics(name), get_qrels(name)
        qrels = {str(q): [str(d) for d, r in docs.items() if int(r) > 0] for q, docs in qrels_raw.items()}
        rows = [{"qid": str(q), "query": t["title"], "group": "all"} for q, t in topics.items()
                if qrels.get(str(q))]
        q = pd.DataFrame(rows)
        extra = pd.DataFrame({"qid": q["qid"], "n_hops": ds_cfg.get("n_hops", 1), "qtype": "beir"})
    elif loader == "local_pyserini":
        qs = pd.read_parquet(resolve(ds_cfg["processed_dir"]) / "questions.parquet")
        qrels = {r.qid: list(r.gold_doc_ids) for r in qs.itertuples()}
        q = pd.DataFrame({"qid": qs["qid"], "query": qs["query"], "group": qs["qtype"].astype(str)})
        extra = qs[["qid", "n_hops", "qtype"]].astype({"qtype": str})
    else:
        raise ValueError(loader)

    q = q.sort_values("qid").reset_index(drop=True)
    cap = ds_cfg.get("max_queries")
    if cap and len(q) > cap:
        q = q.sample(n=cap, random_state=seed).sort_values("qid").reset_index(drop=True)
    q = _split(q, ds_cfg.get("split", {}).get("fractions", {"val": 0.3, "test": 0.7}), seed)
    extra = extra[extra["qid"].isin(set(q["qid"]))]
    return q, {k: qrels[k] for k in q["qid"]}, extra
