# Frontier-DAT: Actionable-Frontier Dynamic Alpha Tuning for Hybrid Retrieval

Hybrid retrieval scores every document as

```
R(q, d; α) = α · dense~(q, d) + (1 − α) · sparse~(q, d)        (min-max normalized; α = 0 → BM25, α = 1 → dense)
```

DAT (Dynamic Alpha Tuning) chooses α per query by letting an LLM judge only the two **endpoint** results
(BM25 top-1 and dense top-1). Many rankings that the hybrid retriever *can* produce exist only at **interior** α,
and endpoint-only judging never sees them.

**Frontier-DAT** judges the *actionable frontier* instead: the exact set of outcomes that some α ∈ [0, 1] can execute.

- **Exact Top-1 Frontier-DAT**: every document that is rank 1 for some α, computed exactly from the
  upper envelope of the score lines (no α grid). A blinded LLM judges the frontier documents (two presentation
  orders, averaged), and a δ gate decides whether to leave the tuned global α_g. The chosen document is executed with
  the feasible α closest to α_g.
- **Exact Top-5 Slate Frontier-DAT**: every *ordered top-5 ranking* executable by some α. A listwise LLM judge picks
  one slate, which is executed with the α in its feasible interval closest to α_g.

Every output is a ranking that the original sparse–dense retriever executes with a single α, so it is reproducible
and auditable at retrieval time.

## Install

```bash
python -m pip install -e ".[dev]"
cp .env.example .env          # put OPENROUTER_API_KEY=sk-or-... (only needed for the LLM judge)
```

Retrieval uses Pyserini (Lucene BM25, k1 = 0.9, b = 0.4) and BAAI/bge-base-en-v1.5 with a FAISS flat index.
Pyserini needs **Java 21**; set `paths.java_home` in `configs/base.yaml` or put a JDK under `tools/`.

## Quick start

```bash
# 1) No data, no API key: exact top-1 / top-k frontiers on a toy example
python -m examples.toy_frontier

# 2) End-to-end Frontier-DAT on a few validation queries (needs the prepared dataset below + API key)
python -m examples.run_frontier_dat --dataset fiqa --n 5
```

## Pipeline

```bash
# 0) MuSiQue only: build the local Lucene + FAISS index of the pooled paragraph corpus
python -m experiments.build_local_index --config configs/musique.yaml

# 1) Retrieval cache: BM25 top-100 ∪ dense top-100, both retrievers' exact scores for every candidate
python -m experiments.cache_retrieval --config configs/fiqa.yaml          # nq / fiqa / hotpotqa / musique

# 2) Frozen train / val / test split caches (split files: runs/milestone3/splits/<dataset>.csv)
python -m experiments.m3_build_cache --configs configs/m3/fiqa.yaml

# 3) Exact Top-1 Frontier-DAT (validation; δ chosen by 2-fold cross-fitting) + controls
python -m experiments.m3c_exact_frontier --dataset fiqa

# 4) Exact Top-5 Slate Frontier-DAT
python -m experiments.m3c_topk_geometry --datasets fiqa                   # exact top-k frontiers, oracle diagnostics
python -m experiments.m3c_topk_slate --dataset fiqa --passes 1

# 5) Frozen evaluation of all methods on one split, then tables
python -m experiments.final_eval --dataset fiqa --split val
python -m experiments.final_report --split val
python -m experiments.final_tables --split val
```

All LLM responses are cached in `cache/m3b_llm_cache.sqlite`, and every LLM client has a hard USD budget.
The judge model defaults to `openai/gpt-4.1-mini` via OpenRouter (temperature 0, seed 1).

## Layout

```
src/sedat/
  retrieval/     Pyserini BM25 + FAISS BGE backend, normalization, hybrid fusion, candidate pools
  m3c/           exact_frontier.py (exact top-1 frontier), topk_frontier.py (exact top-k slate frontier),
                 frontier.py (blinded frontier judge prompt), sensor.py (judge system prompt)
  m3/, m3b/      frozen split caches, benchmark view, OpenRouter client with cache + budget, paired bootstrap
  policies/      fixed / global-tuned / oracle α, DAT rule (faithful baseline)
  llm/           prompts (DAT baseline prompt, see below)
experiments/     pipeline entry points (above)
examples/        toy frontier example, end-to-end Frontier-DAT example
configs/         base + per-dataset configs (NQ, FiQA, HotpotQA, MuSiQue)
tests/           exact top-1 / top-k frontier vs dense brute-force α sweeps
```

```bash
python -m pytest tests          # exact frontier tests (~3 min); top-k tests compare against a 50,001-point sweep (~6 min)
```

## Baseline attribution

The faithful DAT baseline (`dat_alpha` in `src/sedat/policies/dat_style.py`, `DAT_JUDGE_PAPER` in
`src/sedat/llm/prompts.py`) reproduces the α rule and the judge prompt (Appendix A) of
Hsu & Tzeng, *DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation*,
arXiv:2503.23013. The prompt text belongs to its authors; it is included only to reproduce the baseline.

Datasets are not redistributed: BEIR NQ / FiQA / HotpotQA indexes are downloaded through Pyserini, and MuSiQue
comes from its official release.
