"""Milestone 3C: a cheap LLM as a semantic sensor of the CURRENT retrieval outputs.

One call per query. The LLM sees the query and up to three passages -- sparse top-1, dense top-1,
alpha_global top-1 -- deduplicated and shuffled into anonymous labels A/B/C (seeded per query), so
it cannot know which retriever produced which passage. No memory, no labels.
Scores 0-5 are mapped back to their sources. Parse failure -> invalid (no fabricated scores).
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

SENSOR_VERSION = "sensor-v1"

SYSTEM = (
    "You are a retrieval relevance evaluator.\n"
    "Given a query and several candidate passages, independently judge how useful each passage is for "
    "answering the query.\n"
    "Score each passage from 0 to 5:\n"
    "0 = completely irrelevant\n"
    "1 = weakly related but not useful\n"
    "2 = somewhat related, unlikely to contain the needed information\n"
    "3 = partially useful or plausible evidence\n"
    "4 = highly relevant and likely contains the needed information\n"
    "5 = directly answers the query or contains the key evidence needed to answer it\n"
    "Judge each passage independently.\n"
    "Do not compare passage labels or assume any passage is better because of its position.\n"
    "Return JSON only.\n"
    "Do not explain your reasoning."
)

SOURCES = ("sparse", "dense", "global")
FEATURES = ["llm_sparse_score", "llm_dense_score", "llm_global_score", "llm_dense_minus_sparse",
            "llm_sparse_minus_global", "llm_dense_minus_global", "llm_max_score", "llm_min_score",
            "llm_score_range", "llm_global_is_max", "llm_sparse_is_max", "llm_dense_is_max", "llm_n_unique"]


def short(title: str, text: str, n: int = 450) -> str:
    s = ((title or "").strip() + ". " if title else "") + " ".join((text or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def build_call(qid: str, query: str, docs: dict[str, str], texts: dict[str, str], reverse: bool = False):
    """docs: source -> doc_id. Returns (messages, label -> list of sources).
    reverse=True presents the same passages in the reversed order (position de-biasing)."""
    uniq = list(dict.fromkeys(docs[s] for s in SOURCES))            # dedupe, stable
    rng = np.random.default_rng(int(hashlib.sha1(f"{SENSOR_VERSION}|{qid}".encode()).hexdigest()[:8], 16))
    order = [uniq[i] for i in rng.permutation(len(uniq))]
    if reverse:
        order = order[::-1]
    labels = "ABC"[:len(order)]
    lab_of_doc = dict(zip(order, labels))
    mapping = {lab: [s for s in SOURCES if docs[s] == d] for d, lab in lab_of_doc.items()}
    body = f"QUERY:\n{query}\n\n" + "\n\n".join(f"PASSAGE {lab}:\n{texts[d]}" for d, lab in lab_of_doc.items())
    body += f"\n\nReturn relevance scores for {', '.join(labels)} from 0 to 5."
    example = "{" + ", ".join(f'"{lab}": 0' for lab in labels) + "}"
    body += f"\nJSON format: {example}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}], mapping


_FENCE = re.compile(r"```(?:json)?", re.I)
_JSON = re.compile(r"\{.*?\}", re.S)


def parse(text: str, mapping: dict) -> dict | None:
    """source -> score, or None if the output is unusable (missing label / out of range)."""
    m = _JSON.search(_FENCE.sub("", text or ""))
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for lab, srcs in mapping.items():
        v = d.get(lab, d.get(lab.lower()))
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        if not (0 <= v <= 5) or not np.isfinite(v):
            return None
        for s in srcs:
            out[s] = v
    return out


def features(scores: dict | None, n_unique: int) -> np.ndarray:
    if scores is None:
        return np.full(len(FEATURES), np.nan)
    s, d, g = scores["sparse"], scores["dense"], scores["global"]
    mx, mn = max(s, d, g), min(s, d, g)
    return np.array([s, d, g, d - s, s - g, d - g, mx, mn, mx - mn, float(g == mx), float(s == mx), float(d == mx),
                     float(n_unique)])
