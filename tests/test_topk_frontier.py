"""Exact top-k Actionable Ranking Frontier vs a dense brute-force alpha sweep."""
import numpy as np

from sedat.m3c.exact_frontier import WIDTH_EPS, exact_frontier, order_at
from sedat.m3c.topk_frontier import candidates, topk_frontier

SWEEP = np.linspace(0.0, 1.0, 50_001)


def brute(s, d, k):
    return {tuple(order_at(s, d, a)[:k].tolist()) for a in SWEEP}


def check(s, d, k):
    s, d = np.asarray(s, float), np.asarray(d, float)
    F = topk_frontier(s, d, k)
    b = brute(s, d, k)
    wide = {sl for sl, iv in F.intervals.items() if any(e - a > WIDTH_EPS for a, e in iv)}
    assert b <= set(F.slates), (b - set(F.slates))
    assert {sl for sl in wide if max(e - a for a, e in F.intervals[sl]) > 1e-4} <= b
    for sl in F.slates:                                    # every slate is executable at its chosen alpha
        for target in (0.0, 0.41, 1.0):
            assert tuple(order_at(s, d, F.feasible_alpha(sl, target))[:k].tolist()) == sl
    return F


def test_k1_equals_exact_top1_frontier():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = rng.integers(3, 30)
        s, d = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        F1 = check(s, d, 1)
        assert {sl[0] for sl in F1.slates} == set(exact_frontier(s, d).docs)


def test_candidate_pruning_is_safe():
    rng = np.random.default_rng(1)
    for _ in range(30):
        n = rng.integers(6, 60)
        s, d = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        for k in (3, 5):
            C = set(candidates(s, d, k).tolist())
            seen = {i for sl in brute(s, d, k) for i in sl}
            assert seen <= C


def test_topk_random_pools():
    rng = np.random.default_rng(2)
    for _ in range(25):
        n = rng.integers(6, 40)
        s, d = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        for k in (3, 5):
            check(s, d, k)


def test_interior_slate_and_ties():
    # documents on a concave curve: every interior alpha produces a different ordered top-3
    th = np.linspace(0, np.pi / 2, 6)
    F = check(np.cos(th), np.sin(th), 3)
    assert len(F.slates) >= 4
    # duplicate lines + three lines meeting at a point
    check([1.0, 1.0, 0.0, 0.5, 0.2], [0.0, 0.0, 1.0, 0.5, 0.2], 3)
