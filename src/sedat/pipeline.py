"""Glue: make sure the candidate cache exists, then hand back loaded objects."""
from __future__ import annotations

import logging

from .config import get_dotted, retrieval_cache_dir, write_json
from .retrieval.candidates import CandidateCache, build_candidates, cache_exists

log = logging.getLogger(__name__)


def ensure_cache(cfg: dict, force: bool = False) -> CandidateCache:
    cache_dir = retrieval_cache_dir(cfg)
    if cache_exists(cache_dir) and not force:
        log.info("using candidate cache %s", cache_dir)
        return CandidateCache(cache_dir)

    seed = cfg.get("seed", 0)
    loader = cfg["dataset"]["loader"]
    if loader == "synthetic":
        from .data.synthetic import write_synthetic_cache
        write_synthetic_cache(cfg["dataset"], cache_dir, seed)
    elif loader == "squad_format":
        from .data.squad_format import load_squad_format
        from .retrieval.bm25 import BM25
        from .retrieval.dense import DenseRetriever
        ds = load_squad_format({"name": cfg["name"], **cfg["dataset"]}, seed)
        log.info("dataset: %s", ds.summary())
        write_json(cache_dir / "dataset_summary.json", ds.summary())
        rcfg = cfg["retrieval"]
        sparse = BM25(**rcfg["sparse"])
        dcfg = dict(rcfg["dense"])
        model = dcfg.pop("model")
        dense = DenseRetriever(model, cache_path=cache_dir / "doc_emb.npy", **dcfg)
        build_candidates(ds, sparse, dense, get_dotted(cfg, "retrieval.candidates.sparse_top_k", 100),
                         get_dotted(cfg, "retrieval.candidates.dense_top_k", 100), cache_dir)
    elif loader in ("beir_pyserini", "local_pyserini"):
        build_suite_cache(cfg, cache_dir)
    else:
        raise ValueError(f"unknown dataset loader {loader!r}")
    write_json(cache_dir / "cache_config.json", {"dataset": cfg["dataset"], "retrieval": cfg.get("retrieval"),
                                                 "seed": seed})
    return CandidateCache(cache_dir)


# ----------------------------------------------------------------------------- Pyserini suite

def make_suite_retrievers(cfg: dict):
    """Lucene BM25 + FAISS BGE for the cross-dataset suite (one backend for all datasets)."""
    from .retrieval.pyserini_backend import FaissBGE, LuceneBM25, setup_java
    setup_java(get_dotted(cfg, "paths.java_home"))
    r = cfg["retrieval"]
    sparse = LuceneBM25(r["sparse_index"], **{k: v for k, v in r["sparse"].items() if k in ("k1", "b", "threads")})
    d = r["dense"]
    dense = FaissBGE(r["dense_index"], model=d["model"], query_prefix=d.get("query_prefix", ""),
                     batch_size=d.get("batch_size", 64), max_seq_length=d.get("max_seq_length", 512),
                     mmap=d.get("mmap", False))
    return sparse, dense


def make_pool_builder(cfg: dict, sparse, dense):
    from .retrieval.pool_builder import PoolBuilder
    c = cfg["retrieval"]["candidates"]
    return PoolBuilder(sparse, dense, c.get("sparse_top_k", 100), c.get("dense_top_k", 100), c.get("rank_depth", 1000))


def build_suite_cache(cfg: dict, cache_dir) -> None:
    import numpy as np
    import pandas as pd
    from .data.suite import load_suite_queries
    from .retrieval.candidates import write_cache
    from .retrieval.pyserini_backend import setup_java
    setup_java(get_dotted(cfg, "paths.java_home"))
    q, qrels, extra = load_suite_queries(cfg["dataset"], cfg.get("seed", 0))
    log.info("%s: %s", cfg["name"], q.groupby("split").size().to_dict())
    sparse, dense = make_suite_retrievers(cfg)
    pb = make_pool_builder(cfg, sparse, dense)

    parts, embs, lab = [], [], []
    bs = get_dotted(cfg, "retrieval.candidates.query_batch", 128)   # one dense flat scan per batch
    for start in range(0, len(q), bs):
        chunk = q.iloc[start:start + bs]
        cands, emb = pb.build(chunk["qid"].tolist(), chunk["query"].tolist(), batch_size=bs,
                              desc=f"pools {start}/{len(q)}")
        parts.append(cands)
        embs.append(emb)
        for qid in chunk["qid"]:
            sr, dr = pb.gold_ranks(qid, qrels[qid])
            lab.append({"qid": qid, "gold_doc_ids": qrels[qid], "sparse_gold_rank": sr, "dense_gold_rank": dr})
    cands = pd.concat(parts, ignore_index=True)
    doc_ids = sorted(cands["doc_id"].unique())
    # fetch stored text only for docs a policy/judge can plausibly look at (top-20 of either
    # retriever) and gold docs; the rest keep empty text (fetched lazily if ever needed)
    top = cands[(cands["sparse_rank"] <= 20) | (cands["dense_rank"] <= 20)]["doc_id"]
    want = set(top) | {g for gs in qrels.values() for g in gs}
    fetched = pb.docs([d for d in doc_ids if d in want]).set_index("doc_id")
    corpus = pd.DataFrame({"doc_id": doc_ids})
    corpus["title"] = corpus["doc_id"].map(fetched["title"]).fillna("")
    corpus["text"] = corpus["doc_id"].map(fetched["text"]).fillna("")
    corpus["fetched"] = corpus["doc_id"].isin(fetched.index)
    cands["doc_idx"] = cands["doc_id"].map({d: i for i, d in enumerate(doc_ids)}).astype(np.int32)
    cands = cands.sort_values(["qid", "doc_idx"]).reset_index(drop=True)
    nt = [sparse.n_query_terms(t) for t in q["query"]]
    q = q.assign(n_tokens=[a for a, _ in nt], n_tokens_in_vocab=[b for _, b in nt])
    np.save(cache_dir / "query_emb.npy", np.concatenate(embs))
    labels = pd.DataFrame(lab).merge(extra, on="qid", how="left")
    write_cache(cache_dir, corpus, q, cands, labels)
