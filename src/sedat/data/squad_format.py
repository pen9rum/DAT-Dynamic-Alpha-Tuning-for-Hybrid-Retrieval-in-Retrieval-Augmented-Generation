"""SQuAD-format JSON loader (SQuAD v1.1, DRCD, and anything with the same schema).

Retrieval formulation (as in DAT): corpus = all unique paragraphs of the loaded
files; query = question; relevant document = the paragraph the question was
written on.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from ..config import resolve
from .base import RetrievalDataset, group_split, subsample_splits

log = logging.getLogger(__name__)


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s", url, path)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)


def _doc_id(text: str) -> str:
    return "d" + hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def load_squad_format(cfg: dict, seed: int) -> RetrievalDataset:
    docs: dict[str, dict] = {}
    rows = []
    for spec in cfg["files"]:
        path = resolve(spec["path"])
        if not path.exists():
            if not spec.get("url"):
                raise FileNotFoundError(path)
            _download(spec["url"], path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)["data"]
        for art in data:
            title = art.get("title", "")
            for para in art["paragraphs"]:
                ctx = para["context"].strip()
                did = _doc_id(ctx)
                docs.setdefault(did, {"doc_id": did, "title": title, "text": ctx})
                if not spec.get("queries", True):   # corpus-only file
                    continue
                for qa in para["qas"]:
                    if qa.get("is_impossible"):
                        continue
                    rows.append({"qid": str(qa["id"]), "query": qa["question"].strip(),
                                 "group": title, "gold": did})

    queries = pd.DataFrame(rows).drop_duplicates("qid")
    # a question text repeated on several paragraphs would make the label ambiguous
    queries = queries.drop_duplicates("query", keep=False)

    split_cfg = cfg.get("split", {})
    assign = group_split(queries["group"].tolist(),
                         split_cfg.get("fractions", {"train": 0.5, "val": 0.2, "test": 0.3}), seed)
    queries["split"] = queries["group"].map(assign)
    queries = subsample_splits(queries, cfg.get("max_queries"), seed)

    qrels = dict(zip(queries["qid"], queries["gold"].map(lambda d: [d])))
    corpus = pd.DataFrame(list(docs.values()))
    return RetrievalDataset(name=cfg.get("name", "squad_format"), language=cfg.get("language", "en"),
                            corpus=corpus, queries=queries[["qid", "query", "split", "group"]],
                            qrels=qrels)
