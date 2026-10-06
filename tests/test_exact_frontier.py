"""Exact Alpha Frontier vs a very dense brute-force alpha sweep."""
import numpy as np

from sedat.m3c.exact_frontier import WIDTH_EPS, exact_frontier, rank1_at
from sedat.retrieval.hybrid import fuse, rank_orders

SWEEP = np.linspace(0.0, 1.0, 200_001)


def brute(s, d):
    """rank-1 document at every sweep point + transition points."""
    w = np.array([rank1_at(s, d, a) for a in SWEEP])
    change = np.flatnonzero(w[1:] != w[:-1])
    return set(w.tolist()), [(SWEEP[i] + SWEEP[i + 1]) / 2 for i in change]


def check(s, d):
    s, d = np.asarray(s, float), np.asarray(d, float)
    F = exact_frontier(s, d)
    wide = {i for i, iv in F.intervals.items() if any(b - a > WIDTH_EPS for a, b in iv)}
    bset, trans = brute(s, d)
    # every document the dense sweep sees at rank 1 is on the exact frontier with a positive interval
    assert bset <= set(F.docs), (bset, F.docs)
    assert wide <= bset | {i for i in wide if max(b - a for a, b in F.intervals[i]) < 2e-5}, (wide, bset)
    # interior breakpoints agree with the sweep's transitions (to sweep resolution)
    ends = sorted({round(x, 12) for iv in F.intervals.values() for a, b in iv for x in (a, b) if 0 < x < 1})
    for t in trans:
        assert min(abs(t - e) for e in ends) < 1e-5, (t, ends)
    # every frontier document really is rank 1 at the alpha the method would execute
    for i in F.docs:
        for target in (0.0, 0.37, 1.0):
            assert rank1_at(s, d, F.feasible_alpha(i, target)) == i
    return F


def test_endpoint_only():
    F = check([1.0, 0.0, 0.2], [0.0, 1.0, 0.2])
    assert set(F.docs) == {0, 1}
    (a0, b0), = [iv for iv in F.intervals[0] if iv[1] > iv[0]]
    assert abs(b0 - 0.5) < 1e-12 and a0 == 0.0


def test_one_interior_document():
    F = check([1.0, 0.0, 0.7], [0.0, 1.0, 0.7])
    assert set(F.docs) == {0, 1, 2}
    (a, b), = F.intervals[2]
    assert abs(a - 0.3) < 1e-12 and abs(b - 0.7) < 1e-12
    assert abs(F.feasible_alpha(2, 1.0) - (0.7 - 1e-6)) < 1e-12      # clip(alpha_g, a, b), nudged inside
    assert abs(F.feasible_alpha(2, 0.5) - 0.5) < 1e-12                # global inside -> do not move


def test_multiple_interior_documents():
    th = np.linspace(0, np.pi / 2, 7)
    F = check(np.cos(th), np.sin(th))
    assert set(F.docs) == set(range(7))
    assert F.docs == list(range(7))                                     # ordered along alpha


def test_dominated_documents_never_enter():
    rng = np.random.default_rng(0)
    s = np.r_[1.0, 0.0, 0.7, rng.uniform(0, 0.5, 20)]
    d = np.r_[0.0, 1.0, 0.7, rng.uniform(0, 0.5, 20)]
    F = check(s, d)
    assert set(F.docs) == {0, 1, 2}


def test_breakpoint_ties_follow_ranking_rule():
    # three lines meet at alpha = 0.5; the flat line wins only there, and only if the tie-break favours it
    F = check([1.0, 0.0, 0.5], [0.0, 1.0, 0.5])
    assert set(F.docs) == {0, 1}                                        # position tie-break -> doc 0 at 0.5
    F = check([0.5, 1.0, 0.0], [0.5, 0.0, 1.0])
    assert set(F.docs) == {0, 1, 2} and F.intervals[0] == [(0.5, 0.5)]  # genuine point winner is kept


def test_duplicate_score_lines():
    F = check([1.0, 1.0, 0.0, 0.6], [0.0, 0.0, 1.0, 0.6])
    assert 1 not in F.docs and {0, 2}.issubset(F.docs)


def test_random_pools_and_agreement_with_retriever_ranking():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = rng.integers(3, 40)
        s, d = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
        s[rng.integers(n)], d[rng.integers(n)] = 1.0, 1.0
        F = check(s, d)
        # grid points: exact frontier rank-1 == the retriever's own fusion/ranking code
        grid = np.round(np.linspace(0, 1, 11), 6)
        orders = rank_orders(fuse(s, d, grid), s, d)
        assert set(orders[:, 0].tolist()) <= set(F.docs)
