"""Milestone 3B metrics: final summaries, stream blocks, paired comparisons."""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def summary(B, rows, exec_idx) -> dict:
    rows, a = np.asarray(rows), np.asarray(exec_idx)
    R = B.R["hit@1"]
    r, rg, ro = R[rows, a], R[rows, B.g], R[rows].max(1)
    ad = a != B.g
    den = ro.mean() - rg.mean()
    out = {"n": int(len(rows)), "hit@1": float(r.mean()), "global_hit@1": float(rg.mean()), "oracle_hit@1": float(ro.mean()),
           "recovery": float((r.mean() - rg.mean()) / den) if den > EPS else None,
           "adapt_rate": float(ad.mean()),
           "precision": float((r[ad] > rg[ad] + EPS).mean()) if ad.any() else None,
           "harm": float((r[ad] < rg[ad] - EPS).mean()) if ad.any() else None,
           "neutral": float((np.abs(r[ad] - rg[ad]) <= EPS).mean()) if ad.any() else None,
           "regret": float((ro - r).mean())}
    for m in ("ndcg@10", "allrel@10"):
        if m in B.R:
            out[m] = float(B.R[m][rows, a].mean())
            out[f"global_{m}"] = float(B.R[m][rows, B.g].mean())
    return out


def blocks(B, order, exec_by_row: dict, checkpoints, window: int) -> list:
    """Cumulative (first n steps) and local (steps n-window+1..n) metrics along a stream."""
    order = np.asarray(order)
    a = np.array([exec_by_row[int(r)] for r in order])
    out = []
    for n in checkpoints:
        if n <= 0 or n > len(order):
            continue
        lo = max(0, n - window)
        out.append({"n": int(n), "cumulative": summary(B, order[:n], a[:n]), "window": summary(B, order[lo:n], a[lo:n])})
    return out


def paired_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Mean of per-query differences x - y with a bootstrap 95% CI and a sign-flip p-value."""
    d = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    flips = (rng.choice([-1, 1], (n_boot, len(d))) * d).mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_signflip": float((np.abs(flips) >= abs(d.mean()) - 1e-12).mean())}
