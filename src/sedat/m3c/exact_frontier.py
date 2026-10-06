"""Exact Alpha Frontier: the documents that are rank 1 for SOME alpha in [0, 1].

Hybrid score of document i (min-max normalized sparse s_i, dense d_i -- exactly as the retriever):
    v_i(alpha) = (1 - alpha) s_i + alpha d_i = b_i + m_i alpha,   b_i = s_i,  m_i = d_i - s_i
Rank 1 at alpha follows the retriever's full ranking rule (sedat.retrieval.hybrid.rank_orders):
    fused score rounded to 1e-10 (desc) -> mean of the two normalized scores (desc) -> pool position (asc).

Construction (no alpha sampling):
  1. walk the upper envelope of the lines from alpha = 0 to 1 with exact (unrounded) arithmetic and
     solve for every breakpoint where a steeper line overtakes the current winner;
  2. the frontier is { rank1_at(alpha) : alpha in {0, 1} U breakpoints U midpoints of the segments },
     evaluated with the retriever's exact ranking rule. On each open segment the rank-1 document is
     constant, so the midpoint represents the whole segment; at breakpoints ties are resolved exactly
     as the retriever resolves them, so a document that genuinely wins only at a tie point is kept
     (zero-width interval) and floating-point noise never creates a candidate that the retriever would
     not actually put first.
Tolerances: VALUE_EPS for value ties while walking, SLOPE_EPS for equal slopes, WIDTH_EPS below which
a segment is treated as a point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROUND = 10
VALUE_EPS = 1e-12
SLOPE_EPS = 1e-12
WIDTH_EPS = 1e-9


def order_at(s: np.ndarray, d: np.ndarray, alpha: float) -> np.ndarray:
    """Full ranking at alpha under the retriever's rule (identical to hybrid.rank_orders)."""
    fused = np.round(alpha * d + (1 - alpha) * s, ROUND)
    sec = np.round((s + d) / 2, ROUND)
    return np.lexsort((np.arange(len(s)), -sec, -fused))


def rank1_at(s: np.ndarray, d: np.ndarray, alpha: float) -> int:
    return int(order_at(s, d, alpha)[0])


@dataclass
class Frontier:
    docs: list                                      # pool indices, ordered by first alpha at which they win
    intervals: dict = field(default_factory=dict)   # pool index -> sorted list of (start, end); start == end: point

    def feasible_alpha(self, i: int, target: float) -> float:
        """alpha closest to `target` at which document i is rank 1; strictly inside a positive-width
        interval (never on a tie point) unless the document only wins at a point."""
        best, dist = None, np.inf
        for a, b in self.intervals[i]:
            if b - a > WIDTH_EPS:
                eta = min(1e-6, (b - a) / 2)
                x = float(np.clip(target, a + eta, b - eta))
            else:
                x = float(a)
            if abs(x - target) < dist - 1e-15:
                best, dist = x, abs(x - target)
        return best


def _right_winner(s, d, b, m, alpha):
    v = b + m * alpha
    tied = np.flatnonzero(v >= v.max() - VALUE_EPS)
    mx = m[tied].max()
    cand = tied[m[tied] >= mx - SLOPE_EPS]
    if len(cand) == 1:
        return int(cand[0])
    sec = np.round((s[cand] + d[cand]) / 2, ROUND)
    return int(cand[np.lexsort((cand, -sec))[0]])


def breakpoints(s: np.ndarray, d: np.ndarray) -> list[float]:
    s = np.asarray(s, np.float64)
    d = np.asarray(d, np.float64)
    b, m = s, d - s
    pts, alpha, guard = [0.0], 0.0, 0
    while alpha < 1.0 and guard < 4 * len(s) + 10:
        guard += 1
        r = _right_winner(s, d, b, m, alpha)
        steeper = np.flatnonzero(m > m[r] + SLOPE_EPS)
        nxt = 1.0
        if len(steeper):
            t = (b[r] - b[steeper]) / (m[steeper] - m[r])
            t = t[t > alpha + 1e-15]
            if len(t):
                nxt = min(float(t.min()), 1.0)
        pts.append(nxt)
        alpha = nxt
    if pts[-1] < 1.0:
        pts.append(1.0)
    return pts


def exact_frontier(s: np.ndarray, d: np.ndarray) -> Frontier:
    s = np.asarray(s, np.float64)
    d = np.asarray(d, np.float64)
    pts = breakpoints(s, d)
    intervals: dict = {}
    first: dict = {}

    def add(i, a0, a1):
        intervals.setdefault(i, []).append((a0, a1))
        first.setdefault(i, a0)

    for a0, a1 in zip(pts[:-1], pts[1:]):
        if a1 - a0 > WIDTH_EPS:
            add(rank1_at(s, d, (a0 + a1) / 2), a0, a1)
    for p in pts:
        add(rank1_at(s, d, p), p, p)
    for i, iv in intervals.items():                          # merge contiguous, drop covered points
        pos = sorted(x for x in iv if x[1] - x[0] > WIDTH_EPS)
        merged = []
        for a0, a1 in pos:
            if merged and a0 - merged[-1][1] <= WIDTH_EPS:
                merged[-1] = (merged[-1][0], max(merged[-1][1], a1))
            else:
                merged.append((a0, a1))
        pts_i = sorted({x[0] for x in iv if x[1] - x[0] <= WIDTH_EPS
                        and not any(a - WIDTH_EPS <= x[0] <= e + WIDTH_EPS for a, e in merged)})
        intervals[i] = sorted(merged + [(p, p) for p in pts_i])
    docs = sorted(intervals, key=lambda i: (min(a for a, _ in intervals[i]), i))
    return Frontier(docs, intervals)
