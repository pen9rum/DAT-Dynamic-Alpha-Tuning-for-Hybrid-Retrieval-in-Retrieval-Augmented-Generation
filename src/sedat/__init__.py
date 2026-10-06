"""sedat: Self-Evolving DAT.

Infrastructure for asking two questions with one pipeline:
  1. How much headroom is there between a single globally tuned sparse/dense
     fusion weight alpha and a per-query oracle alpha?
  2. How much of that headroom can a policy recover from accumulated experience?

Layers (a lower layer never imports a higher one):
  data -> retrieval -> (candidate cache) -> evaluation.reward_table
  policies consume `State` objects (decision-time information only) and
  `Experience` objects (post-hoc feedback); they never touch retrievers or labels.
"""

__version__ = "0.1.0"
