"""Toy example (no data, no API key): the exact actionable frontier of a hybrid retriever.

Hybrid score of document i:  v_i(alpha) = alpha * dense_i + (1 - alpha) * sparse_i   (min-max normalized scores).
  * Top-1 frontier: every document that is rank 1 for SOME alpha in [0, 1], with the alpha interval where it wins.
  * Top-k frontier: every ordered top-k list ("slate") executable by some alpha.
Endpoint-only DAT can only reach alpha = 0 or alpha = 1; interior documents/slates are invisible to it.

  python -m examples.toy_frontier
"""
import numpy as np

from sedat.m3c.exact_frontier import exact_frontier, order_at
from sedat.m3c.topk_frontier import topk_frontier

DOCS = ["d0 (lexical match)", "d1 (semantic match)", "d2 (both, moderately)", "d3", "d4", "d5", "d6"]
SPARSE = np.array([1.00, 0.10, 0.75, 0.40, 0.30, 0.20, 0.00])   # min-max normalized BM25
DENSE = np.array([0.05, 1.00, 0.80, 0.30, 0.45, 0.60, 0.20])    # min-max normalized dense


def main():
    print("Rank-1 document at a few alphas:")
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  alpha={a:.2f}: {DOCS[order_at(SPARSE, DENSE, a)[0]]}")

    F = exact_frontier(SPARSE, DENSE)
    print("\nExact top-1 frontier (document -> alpha interval(s) where it is rank 1):")
    for i in F.docs:
        print(f"  {DOCS[i]:24s} {[(round(float(s), 3), round(float(e), 3)) for s, e in F.intervals[i]]}")
    interior = [DOCS[i] for i in F.docs if i not in (order_at(SPARSE, DENSE, 0.0)[0], order_at(SPARSE, DENSE, 1.0)[0])]
    print(f"  interior-only (unreachable for endpoint-only DAT): {interior}")

    ag = 0.5                                                   # a tuned global alpha
    for i in F.docs:
        print(f"  executing {DOCS[i]} with the alpha closest to alpha_g={ag}: {F.feasible_alpha(i, ag):.4f}")

    S = topk_frontier(SPARSE, DENSE, 3)
    print(f"\nExact top-3 frontier: {len(S.slates)} executable ordered slates")
    for sl in S.slates:
        print(f"  {[DOCS[i].split()[0] for i in sl]}  alpha in {[(round(float(s), 3), round(float(e), 3)) for s, e in S.intervals[sl]]}")


if __name__ == "__main__":
    main()
