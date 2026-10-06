"""DAT-style adaptive policy: an LLM scores both retrievers' top-1 (0-5), a
hand-written rule maps the scores to alpha. Stateless -- `observe` is a no-op.

Judging is the expensive part, so it is separated out: `LLMJudge` calls the LLM
(via OpenRouter, cached), and `experiments.dat_judge` precomputes scores for a
split into a parquet that `PrecomputedJudge` reads back.
"""
from __future__ import annotations

import re
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from ..llm.prompts import DAT_JUDGE
from ..state import State
from .base import RetrievalPolicy


def dat_alpha(s_dense: int, s_bm25: int) -> float:
    """DAT eq.: alpha weights the DENSE score. Rounded to one decimal."""
    if s_dense == 0 and s_bm25 == 0:
        a = 0.5
    elif s_dense == 5 and s_bm25 != 5:
        a = 1.0
    elif s_bm25 == 5 and s_dense != 5:
        a = 0.0
    else:
        a = s_dense / (s_dense + s_bm25)
    return round(a, 1)


def parse_scores(text: str) -> tuple[int, int] | None:
    nums = [int(x) for x in re.findall(r"\b[0-5]\b", text)]
    return (nums[0], nums[1]) if len(nums) >= 2 else None


class Judge(Protocol):
    def __call__(self, state: State) -> tuple[int, int]: ...


class LLMJudge:
    """Returns (S_dense, S_bm25). Needs State built with with_texts=True."""

    def __init__(self, client, max_chars: int = 2000, fallback: tuple[int, int] = (0, 0)):
        self.client, self.max_chars, self.fallback = client, max_chars, fallback
        self.n_parse_failures = 0

    def __call__(self, state: State) -> tuple[int, int]:
        if not state.dense_top_texts or not state.sparse_top_texts:
            raise ValueError("LLMJudge needs states built with with_texts=True")
        prompt = DAT_JUDGE.format(question=state.query,
                                  dense_top1=state.dense_top_texts[0][:self.max_chars],
                                  bm25_top1=state.sparse_top_texts[0][:self.max_chars])
        out = parse_scores(self.client.chat([{"role": "user", "content": prompt}]))
        if out is None:
            self.n_parse_failures += 1
            return self.fallback
        return out


class PrecomputedJudge:
    def __init__(self, path):
        df = pd.read_parquet(path)
        self.scores = {q: (int(v), int(b)) for q, v, b in zip(df["qid"], df["s_dense"], df["s_bm25"])}

    def __call__(self, state: State) -> tuple[int, int]:
        return self.scores[state.qid]


class DATAlpha(RetrievalPolicy):
    name = "dat_style"
    kind = "adaptive"

    def __init__(self, alphas: np.ndarray, judge: Callable[[State], tuple[int, int]]):
        super().__init__(alphas)
        self.judge = judge

    def select_alpha(self, state: State) -> float:
        return self.snap(dat_alpha(*self.judge(state)))
