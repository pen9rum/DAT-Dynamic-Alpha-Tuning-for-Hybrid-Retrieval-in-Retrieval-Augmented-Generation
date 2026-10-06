"""Local multi-hop corpora (no prebuilt index exists): MuSiQue and 2WikiMultihopQA.

IRCoT-style corpus: the union of all paragraphs attached to the sampled questions
(each question's gold paragraphs + its dataset-provided distractors), deduplicated
by (title, text). Retrieval for every question runs against the WHOLE pooled corpus,
not only its own paragraphs.

Question sampling exists only because dense encoding on this machine's CPU is
~2 passages/s; raise `n_questions` on better hardware.

Outputs in data/processed/<name>/:
  corpus.jsonl      {"id", "contents", "title", "text"}   (Pyserini JsonCollection)
  questions.parquet qid, query, gold_doc_ids, n_hops, qtype
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import resolve

log = logging.getLogger(__name__)

SOURCES = {
    "musique": ("https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl",
                "data/raw/musique/musique_ans_v1.0_dev.jsonl"),
    "2wiki": ("https://huggingface.co/datasets/framolfese/2WikiMultihopQA/resolve/main/data/validation-00000-of-00001.parquet",
              "data/raw/2wiki/validation.parquet"),
}


def _fetch(name: str) -> Path:
    url, rel = SOURCES[name]
    path = resolve(rel)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
    return path


def _pid(title: str, text: str) -> str:
    return "p" + hashlib.md5(f"{title}\n{text}".encode("utf-8")).hexdigest()[:12]


def _stratified(df: pd.DataFrame, by: str, quota: dict, seed: int) -> pd.DataFrame:
    parts = []
    for key, n in quota.items():
        sub = df[df[by] == key]
        parts.append(sub.sample(n=min(n, len(sub)), random_state=seed))
    return pd.concat(parts)


def _musique(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    qs, paras = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if not r.get("answerable", True):
            continue
        ps = r["paragraphs"]
        ids = [_pid(p["title"], p["paragraph_text"]) for p in ps]
        # gold paragraphs in reasoning order (question_decomposition order)
        gold = [ids[step["paragraph_support_idx"]] for step in r["question_decomposition"]]
        qs.append({"qid": r["id"], "query": r["question"], "gold_doc_ids": list(dict.fromkeys(gold)),
                   "n_hops": len(r["question_decomposition"]), "qtype": r["id"].split("__")[0],
                   "_paras": [(i, p["title"], p["paragraph_text"]) for i, p in zip(ids, ps)]})
    return pd.DataFrame(qs), paras


def _2wiki(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_parquet(path)
    qs = []
    for r in df.itertuples():
        titles = list(r.context["title"])
        texts = [" ".join(s) for s in r.context["sentences"]]
        ids = [_pid(t, x) for t, x in zip(titles, texts)]
        by_title = dict(zip(titles, ids))
        sup = list(dict.fromkeys(r.supporting_facts["title"]))       # evidence order
        gold = [by_title[t] for t in sup if t in by_title]
        if len(gold) != len(sup):
            continue
        qs.append({"qid": r.id, "query": r.question, "gold_doc_ids": gold, "n_hops": len(gold),
                   "qtype": r.type, "_paras": list(zip(ids, titles, texts))})
    return pd.DataFrame(qs), []


def build_multihop_local(name: str, n_questions: dict, seed: int, out_dir: Path) -> None:
    """n_questions: stratification quota, e.g. {"2hop": 350, "3hop1": 100, ...} keyed by qtype
    (musique) or by n_hops (2wiki: {2: 800, 4: 200})."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = _fetch(name)
    qs, _ = (_musique if name == "musique" else _2wiki)(path)
    by = "qtype" if name == "musique" else "n_hops"
    quota = {(int(k) if by == "n_hops" else k): v for k, v in n_questions.items()}
    qs = _stratified(qs, by, quota, seed).sort_values("qid").reset_index(drop=True)

    corpus: dict[str, dict] = {}
    for paras in qs["_paras"]:
        for pid, title, text in paras:
            corpus.setdefault(pid, {"id": pid, "contents": f"{title}\n{text}", "title": title, "text": text})
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in corpus.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    qs.drop(columns="_paras").to_parquet(out_dir / "questions.parquet", index=False)
    log.info("%s: %d questions, %d unique paragraphs -> %s", name, len(qs), len(corpus), out_dir)
