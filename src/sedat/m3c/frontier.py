"""Alpha Frontier Judge: show the small LLM the documents that the alpha grid can actually put at
rank 1 (the "alpha-induced candidate frontier") and ask only for per-document relevance.

The LLM never sees alpha values, retriever names, memory or labels. Candidates are shuffled into
anonymous labels (seeded per query) and judged in two presentation orders (random + reversed);
per-document scores are averaged, which removes the first-slot position bias measured in 3C.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from .sensor import SYSTEM

FRONTIER_VERSION = "frontier-v1"
LABELS = "ABCDEFGHIJK"


def build_call(qid: str, query: str, doc_ids: list[str], texts: dict[str, str], reverse: bool = False):
    rng = np.random.default_rng(int(hashlib.sha1(f"{FRONTIER_VERSION}|{qid}".encode()).hexdigest()[:8], 16))
    order = [doc_ids[i] for i in rng.permutation(len(doc_ids))]
    if reverse:
        order = order[::-1]
    labs = LABELS[:len(order)]
    body = f"QUERY:\n{query}\n\n" + "\n\n".join(f"PASSAGE {l}:\n{texts[d]}" for l, d in zip(labs, order))
    body += f"\n\nReturn relevance scores for {', '.join(labs)} from 0 to 5."
    body += "\nJSON format: {" + ", ".join(f'"{l}": 0' for l in labs) + "}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}], dict(zip(labs, order))


_FENCE = re.compile(r"```(?:json)?", re.I)
_JSON = re.compile(r"\{.*?\}", re.S)


def parse(text: str, lab2doc: dict) -> dict | None:
    m = _JSON.search(_FENCE.sub("", text or ""))
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for lab, doc in lab2doc.items():
        try:
            v = float(d.get(lab, d.get(lab.lower())))
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(v) and 0 <= v <= 5):
            return None
        out[doc] = v
    return out


def choose_alpha(top1: list[str], scores: dict[str, float], g: int, delta: float) -> tuple[int, float]:
    """top1[a] = rank-1 doc under alpha index a. Returns (alpha index, score gain over global's doc).
    Adapt only if the best-judged document beats alpha_global's top-1 by >= delta; then use the alpha
    (closest to global) that puts that document at rank 1. Ties with global's doc -> stay."""
    gdoc = top1[g]
    best = max(scores.values())
    gain = best - scores[gdoc]
    if gain < delta or gain <= 0:
        return g, gain
    best_docs = [d for d, s in scores.items() if s >= best - 1e-9]
    cands = [a for a, d in enumerate(top1) if d in best_docs]
    return min(cands, key=lambda a: (abs(a - g), a)), gain
