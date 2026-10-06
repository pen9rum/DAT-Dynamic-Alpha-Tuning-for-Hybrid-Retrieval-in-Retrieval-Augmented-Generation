"""Offline and online evaluation of any RetrievalPolicy against a reward table.

Rewards come from the counterfactual table, so evaluating a policy is a lookup:
alpha_t -> R[q_t, alpha_t]. Labels enter only through `R` (post-hoc feedback).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..experience.schema import Experience
from ..policies.base import RetrievalPolicy
from ..state import State

log = logging.getLogger(__name__)


@dataclass
class RewardLookup:
    """R[i, a] for a set of queries, plus per-alpha gold ranks for feedback."""
    qids: list[str]
    R: np.ndarray
    gold_rank: np.ndarray | None = None

    def __post_init__(self):
        self.row = {q: i for i, q in enumerate(self.qids)}

    @classmethod
    def from_table(cls, table: pd.DataFrame, metric: str, qids: list[str]) -> "RewardLookup":
        R = table.pivot(index="qid", columns="alpha_idx", values=metric).loc[qids].to_numpy(np.float64)
        G = table.pivot(index="qid", columns="alpha_idx", values="gold_rank").loc[qids].to_numpy(np.float64)
        return cls(list(qids), R, G)


def evaluate_policy(policy: RetrievalPolicy, states: dict[str, State], lookup: RewardLookup) -> pd.DataFrame:
    """Frozen (eval-mode) evaluation. Returns per-query chosen alpha and reward."""
    was = policy.training
    policy.eval()
    rows = []
    for q in lookup.qids:
        a = policy.select_alpha(states[q])
        k = policy.alpha_index(a)
        i = lookup.row[q]
        rows.append({"qid": q, "alpha": a, "reward": lookup.R[i, k], "oracle_reward": lookup.R[i].max()})
    policy.train(was)
    return pd.DataFrame(rows)


def run_online(policy: RetrievalPolicy, stream: list[str], states: dict[str, State],
               lookup: RewardLookup, featurize: Callable[[State], np.ndarray], *,
               eval_states: dict[str, State] | None = None, eval_lookup: RewardLookup | None = None,
               checkpoints: list[int] = (), feedback: str = "bandit", expose_oracle: bool = False,
               update_every: int = 1) -> dict:
    """Sequential interaction loop:
        state_t -> alpha_t = pi(state_t; E_<t) -> reward_t -> E_t -> update.
    Regret is measured against the per-query oracle on the stream itself.
    At each checkpoint n (number of experiences observed) the policy is frozen and
    evaluated greedily on the held-out eval set -> learning curve.
    """
    assert feedback in ("bandit", "full")
    ckpts = sorted(set(c for c in checkpoints if c <= len(stream)))
    curve, steps = [], []

    def _checkpoint(n: int):
        if eval_states is not None and eval_lookup is not None:
            ev = evaluate_policy(policy, eval_states, eval_lookup)
            curve.append({"n_experiences": n, "eval_reward": float(ev["reward"].mean()),
                          "eval_mean_alpha": float(ev["alpha"].mean())})

    policy.train(True)
    if 0 in ckpts:
        _checkpoint(0)
    for t, q in enumerate(stream):
        st = states[q]
        a = policy.select_alpha(st)
        k = policy.alpha_index(a)
        i = lookup.row[q]
        r = float(lookup.R[i, k])
        best = float(lookup.R[i].max())
        exp = Experience(
            step=t, qid=q, query=st.query, state_features=featurize(st), chosen_alpha=a, reward=r,
            gold_rank=float(lookup.gold_rank[i, k]) if lookup.gold_rank is not None else None,
            sparse_rank=float(lookup.gold_rank[i, 0]) if lookup.gold_rank is not None else None,
            dense_rank=float(lookup.gold_rank[i, -1]) if lookup.gold_rank is not None else None,
            counterfactual_rewards=lookup.R[i].copy() if feedback == "full" else None,
            oracle_alpha=float(policy.alphas[int(np.argmax(lookup.R[i]))]) if expose_oracle else None)
        policy.observe(exp)
        if (t + 1) % update_every == 0:
            policy.update()
        steps.append({"step": t, "qid": q, "alpha": a, "reward": r, "oracle_reward": best,
                      "regret": best - r})
        if t + 1 in ckpts:
            _checkpoint(t + 1)
    df = pd.DataFrame(steps)
    df["cum_regret"] = df["regret"].cumsum()
    return {"steps": df, "curve": pd.DataFrame(curve)}
