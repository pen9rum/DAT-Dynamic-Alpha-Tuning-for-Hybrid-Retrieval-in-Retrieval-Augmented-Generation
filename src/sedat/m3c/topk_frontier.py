"""Exact top-k Actionable Ranking Frontier: every ordered top-k slate S_k(alpha), alpha in [0, 1].

Scores v_i(alpha) = (1 - alpha) s_i + alpha d_i (the retriever's min-max normalized scores). The ordered
top-k can only change where two lines cross, so the exact frontier is obtained by
  1. candidate pruning (exact, not approximate): if at least k documents are strictly above document i at
     BOTH alpha = 0 and alpha = 1, then by linearity they are above it on all of [0, 1], so i is never in
     the top k; only the remaining candidates C can appear in any slate;
  2. breakpoints = all pairwise intersections of candidate lines inside (0, 1), plus 0 and 1;
  3. on every open segment the ordered top-k is constant -> evaluated at the segment midpoint; every
     breakpoint is evaluated at the point itself. Both use the retriever's full ranking rule
     (order_at: rounded fused score -> mean score -> pool position), exactly like the top-1 exact frontier.
Identical ordered slates are deduplicated; each slate keeps the alpha interval(s) where it is the ranking.
k = 1 reproduces exact_frontier().
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exact_frontier import WIDTH_EPS, order_at


@dataclass
class SlateFrontier:
    k: int
    slates: list                                     # tuples of pool indices, ordered by first alpha
    intervals: dict = field(default_factory=dict)    # slate -> sorted list of (start, end); point if start == end

    def feasible_alpha(self, slate: tuple, target: float) -> float:
        """alpha closest to target at which `slate` is the ordered top-k (inside an interval when possible)."""
        best, dist = None, np.inf
        for a, b in self.intervals[slate]:
            if b - a > WIDTH_EPS:
                eta = min(1e-6, (b - a) / 2)
                x = float(np.clip(target, a + eta, b - eta))
            else:
                x = float(a)
            if abs(x - target) < dist - 1e-15:
                best, dist = x, abs(x - target)
        return best

    def slate_at(self, alpha: float):
        for sl, iv in self.intervals.items():
            for a, b in iv:
                if a - 1e-12 <= alpha <= b + 1e-12:
                    return sl
        return None


def candidates(s: np.ndarray, d: np.ndarray, k: int) -> np.ndarray:
    above = (s[None, :] > s[:, None]) & (d[None, :] > d[:, None])      # above[i, j]: j strictly above i at both ends
    return np.flatnonzero(above.sum(axis=1) < k)


def breakpoints(s, d, idx) -> np.ndarray:
    si, di = s[idx], d[idx]
    m = di - si
    dm = m[None, :] - m[:, None]
    db = si[:, None] - si[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = db / dm
    t = t[np.isfinite(t) & (t > 0) & (t < 1)]
    return np.unique(np.r_[0.0, t, 1.0])


def topk_frontier(s, d, k: int) -> SlateFrontier:
    s = np.asarray(s, np.float64)
    d = np.asarray(d, np.float64)
    k = min(k, len(s))
    pts = breakpoints(s, d, candidates(s, d, k))
    raw = []
    for a0, a1 in zip(pts[:-1], pts[1:]):
        if a1 - a0 > WIDTH_EPS:
            raw.append((tuple(order_at(s, d, (a0 + a1) / 2)[:k].tolist()), a0, a1))
    for p in pts:
        raw.append((tuple(order_at(s, d, float(p))[:k].tolist()), float(p), float(p)))
    iv: dict = {}
    for sl, a0, a1 in raw:
        iv.setdefault(sl, []).append((a0, a1))
    for sl, lst in iv.items():
        pos = sorted(x for x in lst if x[1] - x[0] > WIDTH_EPS)
        merged = []
        for a0, a1 in pos:
            if merged and a0 - merged[-1][1] <= WIDTH_EPS:
                merged[-1] = (merged[-1][0], max(merged[-1][1], a1))
            else:
                merged.append((a0, a1))
        pts_i = sorted({x[0] for x in lst if x[1] - x[0] <= WIDTH_EPS
                        and not any(a - WIDTH_EPS <= x[0] <= e + WIDTH_EPS for a, e in merged)})
        iv[sl] = sorted(merged + [(p, p) for p in pts_i])
    slates = sorted(iv, key=lambda sl: (min(a for a, _ in iv[sl]), sl))
    return SlateFrontier(k, slates, iv)
