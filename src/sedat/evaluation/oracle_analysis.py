"""Headroom analysis: is query-dependent alpha worth learning?

Notation: R[i, a] = reward of query i under alpha index a (from the reward table).

  global      max_a  mean_i R[i, a]   (a tuned on VAL, reported on TEST)
  oracle      mean_i max_a R[i, a]    (per query, uses TEST labels -- analysis only)
  routing     mean_i max(R[i, 0], R[i, 1.0])   (per-query pick of sparse OR dense)

  oracle_gap      = oracle  - global
                  = (routing - global)  +  (oracle - routing)
                     "retriever routing"    "hybrid-specific" headroom
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# ----------------------------------------------------------------------------- selection

def endpoint_indices(alphas: np.ndarray) -> tuple[int, int]:
    lo, hi = np.flatnonzero(np.isclose(alphas, 0.0)), np.flatnonzero(np.isclose(alphas, 1.0))
    if not len(lo) or not len(hi):
        raise ValueError("alpha grid must contain 0.0 (sparse only) and 1.0 (dense only)")
    return int(lo[0]), int(hi[0])


def perf_vs_alpha(R: np.ndarray) -> np.ndarray:
    return R.mean(axis=0)


def tune_global_alpha(R_val: np.ndarray, alphas: np.ndarray) -> int:
    """argmax of mean validation reward; ties -> the alpha closest to 0.5."""
    m = perf_vs_alpha(R_val)
    best = np.flatnonzero(m >= m.max() - EPS)
    return int(best[np.argmin(np.abs(alphas[best] - 0.5))])


def argmax_sets(R: np.ndarray) -> np.ndarray:
    """Boolean (n_q, n_alpha): alpha indices achieving each query's best reward."""
    return R >= R.max(axis=1, keepdims=True) - EPS


def oracle_select(R: np.ndarray, alphas: np.ndarray, tie_break: str = "closest_to_global",
                  global_idx: int | None = None) -> np.ndarray:
    """One oracle alpha index per query. Ties are the norm for binary rewards, so the
    tie-break matters for the *distribution* of oracle alphas (not for oracle reward).
      closest_to_global : least deviation from the deployable default (conservative)
      smallest / largest: extreme choices
      center            : middle of the argmax set
    """
    M = argmax_sets(R)
    out = np.empty(len(R), dtype=int)
    for i, row in enumerate(M):
        idx = np.flatnonzero(row)
        if tie_break == "closest_to_global":
            ref = alphas[global_idx] if global_idx is not None else 0.5
            out[i] = idx[np.argmin(np.abs(alphas[idx] - ref))]
        elif tie_break == "smallest":
            out[i] = idx[0]
        elif tie_break == "largest":
            out[i] = idx[-1]
        elif tie_break == "center":
            out[i] = idx[len(idx) // 2]
        else:
            raise ValueError(tie_break)
    return out


# ----------------------------------------------------------------------------- statistics

def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """CI of mean(a - b) over queries."""
    diff = np.asarray(a) - np.asarray(b)
    if n <= 0 or len(diff) == 0:
        return {"mean": float(diff.mean()) if len(diff) else float("nan")}
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(boots, 2.5)),
                                                 float(np.percentile(boots, 97.5))]}


def categorize(R_bin: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Categories on a binary success matrix:
      A sparse succeeds, dense fails     B dense succeeds, sparse fails
      C both succeed                     D both fail, no alpha succeeds
      E both endpoints fail, but some intermediate alpha succeeds (hybrid-only win)
    """
    s, d = R_bin[:, i0] > 0.5, R_bin[:, i1] > 0.5
    anyw = R_bin.max(axis=1) > 0.5
    cat = np.full(len(R_bin), "D", dtype=object)
    cat[s & ~d] = "A"
    cat[d & ~s] = "B"
    cat[s & d] = "C"
    cat[~s & ~d & anyw] = "E"
    return cat


# ----------------------------------------------------------------------------- main analysis

def headroom_analysis(R_val: np.ndarray, R_test: np.ndarray, alphas: np.ndarray, *,
                      tie_break: str = "closest_to_global", n_bootstrap: int = 1000,
                      seed: int = 0) -> dict:
    """All scalar headroom quantities for one reward metric."""
    i0, i1 = endpoint_indices(alphas)
    i05 = int(np.argmin(np.abs(alphas - 0.5)))
    g = tune_global_alpha(R_val, alphas)
    m_val, m_test = perf_vs_alpha(R_val), perf_vs_alpha(R_test)

    best_q = R_test.max(axis=1)
    routing_q = np.maximum(R_test[:, i0], R_test[:, i1])
    three_q = np.maximum(routing_q, R_test[:, g])
    glob_q = R_test[:, g]
    M = argmax_sets(R_test)
    o_idx = oracle_select(R_test, alphas, tie_break, g)
    interior = np.ones(len(alphas), dtype=bool)
    interior[[i0, i1]] = False
    const = R_test.max(axis=1) - R_test.min(axis=1) < EPS

    ladder = {
        "sparse_only": float(m_test[i0]),
        "dense_only": float(m_test[i1]),
        "fixed_0.5": float(m_test[i05]),
        "global_tuned": float(m_test[g]),
        "test_best_fixed (in-sample, reference)": float(m_test.max()),
        "routing_oracle (per-query sparse|dense)": float(routing_q.mean()),
        "three_arm_oracle (sparse|global|dense)": float(three_q.mean()),
        "per_query_oracle": float(best_q.mean()),
    }

    # sensitivity around the global optimum
    sens = {}
    for delta in (1, 2):
        for sign in (-1, 1):
            j = g + sign * delta
            if 0 <= j < len(alphas):
                sens[f"alpha{alphas[j] - alphas[g]:+.2f}"] = float(m_test[j] - m_test[g])

    changes = np.abs(np.diff(R_test, axis=1)) > EPS
    return {
        "global_alpha": float(alphas[g]),
        "global_alpha_idx": g,
        "test_best_fixed_alpha": float(alphas[int(np.argmax(m_test))]),
        "perf_vs_alpha": {"alpha": alphas.tolist(), "val": m_val.tolist(), "test": m_test.tolist()},
        "ladder_test": ladder,
        "oracle_gap": paired_bootstrap(best_q, glob_q, n_bootstrap, seed),
        "gap_decomposition": {
            "routing_minus_global": paired_bootstrap(routing_q, glob_q, n_bootstrap, seed),
            "oracle_minus_routing (hybrid-specific)": paired_bootstrap(best_q, routing_q, n_bootstrap, seed),
            "oracle_minus_three_arm": paired_bootstrap(best_q, three_q, n_bootstrap, seed),
        },
        "gain_oracle_over_best_single_retriever": float(best_q.mean() - max(m_test[i0], m_test[i1])),
        "oracle_alpha_hist (tie-broken)": np.bincount(o_idx, minlength=len(alphas)).tolist(),
        "alpha_optimal_freq (share of queries for which alpha is in argmax set)": M.mean(axis=0).tolist(),
        "frac_oracle_alpha_differs_from_global (tie-broken)": float((o_idx != g).mean()),
        "frac_global_suboptimal (global not in argmax set)": float((~M[:, g]).mean()),
        "frac_ties (argmax set size > 1)": float((M.sum(axis=1) > 1).mean()),
        "frac_constant_reward (alpha irrelevant)": float(const.mean()),
        "mean_argmax_set_size": float(M.sum(axis=1).mean()),
        "sensitivity_around_global": sens,
        "diagnostics": {
            "frac_endpoints_equal_reward": float((np.abs(R_test[:, i0] - R_test[:, i1]) < EPS).mean()),
            "frac_sparse_wins (R(0) > R(1))": float((R_test[:, i0] > R_test[:, i1] + EPS).mean()),
            "frac_dense_wins (R(1) > R(0))": float((R_test[:, i1] > R_test[:, i0] + EPS).mean()),
            "frac_interior_strictly_beats_both_endpoints": float(
                (R_test[:, interior].max(axis=1) > routing_q + EPS).mean()),
            "frac_oracle_alpha_near_endpoint (<=0.1 or >=0.9, tie-broken)": float(
                ((alphas[o_idx] <= 0.1 + EPS) | (alphas[o_idx] >= 0.9 - EPS)).mean()),
            "frac_some_endpoint_optimal": float((M[:, i0] | M[:, i1]).mean()),
            "frac_oracle_strictly_interior (no endpoint optimal)": float((~M[:, i0] & ~M[:, i1]).mean()),
            "alpha_stability: frac of adjacent-alpha steps that change reward": float(changes.mean()),
            "alpha_stability: mean #reward changes along the grid per query": float(changes.sum(axis=1).mean()),
        },
        "n_queries": {"val": int(len(R_val)), "test": int(len(R_test))},
    }


def category_report(R_bin: np.ndarray, alphas: np.ndarray) -> dict:
    i0, i1 = endpoint_indices(alphas)
    cat = categorize(R_bin, i0, i1)
    interior = np.ones(len(alphas), dtype=bool)
    interior[[i0, i1]] = False
    both = cat == "C"
    return {
        "share": {c: float((cat == c).mean()) for c in "ABCDE"},
        "labels": {"A": "sparse wins, dense fails", "B": "dense wins, sparse fails", "C": "both succeed",
                   "D": "both fail, no alpha succeeds", "E": "both fail, some intermediate alpha succeeds"},
        "frac_C_where_some_interior_alpha_fails (hybrid can hurt)": float(
            (R_bin[both][:, interior].min(axis=1) < 0.5).mean()) if both.any() else 0.0,
    }


def distinct_rankings(table: pd.DataFrame, qids: list[str], k: int) -> dict:
    """How many different top-k lists do the alphas produce per query?"""
    sub = table[table["qid"].isin(set(qids))]
    n_topk = sub.groupby("qid")["top_doc_ids"].apply(lambda s: len({tuple(x[:k]) for x in s}))
    n_top1 = sub.groupby("qid")["top_doc_ids"].apply(lambda s: len({x[0] for x in s}))
    return {f"mean_distinct_top{k}_lists": float(n_topk.mean()),
            f"hist_distinct_top{k}_lists": n_topk.value_counts().sort_index().to_dict(),
            "mean_distinct_top1_docs": float(n_top1.mean()),
            "hist_distinct_top1_docs": n_top1.value_counts().sort_index().to_dict()}


def retriever_agreement(labels: pd.DataFrame, qids: list[str]) -> dict:
    """Diagnostic 1: do BM25 and dense rank the gold doc similarly (full corpus)?"""
    lab = labels.loc[qids]
    s = np.log2(lab["sparse_gold_rank"].to_numpy(np.float64))
    d = np.log2(lab["dense_gold_rank"].to_numpy(np.float64))
    return {"frac_gold_rank_equal": float((s == d).mean()),
            "frac_gold_rank_within_2x": float((np.abs(s - d) <= 1).mean()),
            "median_sparse_gold_rank": float(np.median(lab["sparse_gold_rank"])),
            "median_dense_gold_rank": float(np.median(lab["dense_gold_rank"]))}


def per_query_frame(qids: list[str], R: np.ndarray, alphas: np.ndarray, g: int, tie_break: str,
                    R_bin: np.ndarray | None = None) -> pd.DataFrame:
    i0, i1 = endpoint_indices(alphas)
    M = argmax_sets(R)
    o_idx = oracle_select(R, alphas, tie_break, g)
    df = pd.DataFrame(R, columns=[f"a={a:.2f}" for a in alphas])
    df.insert(0, "qid", qids)
    df["global_reward"] = R[:, g]
    df["oracle_reward"] = R.max(axis=1)
    df["oracle_alpha"] = alphas[o_idx]
    df["optimal_alphas"] = [alphas[row].tolist() for row in M]
    df["n_optimal"] = M.sum(axis=1)
    if R_bin is not None:
        df["category"] = categorize(R_bin, i0, i1)
    return df
