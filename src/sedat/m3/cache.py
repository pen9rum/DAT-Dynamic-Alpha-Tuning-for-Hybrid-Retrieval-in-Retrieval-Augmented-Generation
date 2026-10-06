"""Milestone-3 candidate caches (namespace cache/milestone3/<name>/).

The Milestone-2 cache is read-only input. Pools are deterministic per query, so M2 pools
are reused verbatim and retrieval runs only for queries that M2 never saw (official
train/dev queries). The merged cache has the same file format as every other cache.
"""
from __future__ import annotations

import logging
import shutil

import numpy as np
import pandas as pd

from ..config import get_dotted, load_config, resolve, retrieval_cache_dir, write_json
from ..retrieval.candidates import CandidateCache, cache_exists, write_cache
from .splits import beir_official, deterministic_split, frozen_split_path

log = logging.getLogger(__name__)


def m3_cache_dir(cfg: dict):
    d = resolve(f"cache/milestone3/{cfg['name']}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def m2_config(cfg: dict) -> dict:
    return load_config(resolve(cfg["m3"]["m2_config"]))


def _new_queries(cfg: dict, m2: CandidateCache, seed: int):
    """Official train/val queries not present in the M2 cache."""
    sp = cfg["m3"]["split"]
    name = cfg["dataset"]["beir_name"]
    have = set(m2.queries["qid"])
    parts, qrels = [], {}
    for split in ("train", "val"):
        spec = sp[split]
        if spec.get("from") == "train_holdout":
            continue
        q, r = beir_official(name, spec["from"])
        q = q[~q["qid"].isin(have)].sort_values("qid")
        if spec.get("max") and len(q) > spec["max"]:
            q = q.sample(n=spec["max"], random_state=seed)
        parts.append(q.assign(split=split))
        qrels.update({k: r[k] for k in q["qid"]})
    q = pd.concat(parts, ignore_index=True).sort_values("qid").reset_index(drop=True)
    if sp["val"].get("from") == "train_holdout":   # carve val out of official train, deterministically
        tr = q.loc[q["split"] == "train", "qid"].to_numpy()
        rng = np.random.default_rng(seed)
        val_ids = set(rng.choice(tr, size=int(round(sp["val"]["frac"] * len(tr))), replace=False))
        q.loc[q["qid"].isin(val_ids), "split"] = "val"
    return q, qrels


def build_m3_cache(cfg: dict) -> CandidateCache:
    out = m3_cache_dir(cfg)
    if cache_exists(out):
        return CandidateCache(out)
    m2cfg = m2_config(cfg)
    m2 = CandidateCache(retrieval_cache_dir(m2cfg))
    seed = cfg.get("seed", 0)
    mode = cfg["m3"]["split"]["mode"]
    q2, lab2 = m2.queries.copy(), pd.read_parquet(m2.dir / "labels.parquet")

    if mode in ("inherit", "deterministic"):
        if mode == "deterministic":
            q2["split"] = deterministic_split(q2, cfg["m3"]["split"]["fractions"], seed).to_numpy()
        for f in m2.dir.iterdir():
            if f.is_file() and f.name != "queries.parquet" and not f.name.startswith("reward_table") \
                    and not f.name.startswith("multihop_paths"):
                shutil.copy2(f, out / f.name)
        q2.to_parquet(out / "queries.parquet", index=False)
    elif mode == "official":
        from ..pipeline import make_pool_builder, make_suite_retrievers
        from ..retrieval.pyserini_backend import setup_java
        setup_java(get_dotted(cfg, "paths.java_home"))
        q2["split"] = "test"                       # every M2 query is an official test query
        newq, qrels = _new_queries(cfg, m2, seed)
        assert not set(newq["qid"]) & set(q2["qid"]), "train/val overlap with test"
        log.info("%s: %d new queries %s", cfg["name"], len(newq), newq.groupby("split").size().to_dict())
        sparse, dense = make_suite_retrievers(m2cfg)
        pb = make_pool_builder(m2cfg, sparse, dense)
        bs = get_dotted(m2cfg, "retrieval.candidates.query_batch", 128)
        parts, embs, lab = [], [], []
        for start in range(0, len(newq), bs):
            ch = newq.iloc[start:start + bs]
            c, e = pb.build(ch["qid"].tolist(), ch["query"].tolist(), batch_size=bs,
                            desc=f"m3 pools {start}/{len(newq)}")
            parts.append(c)
            embs.append(e)
            for qid in ch["qid"]:
                sr, dr = pb.gold_ranks(qid, qrels[qid])
                lab.append({"qid": qid, "gold_doc_ids": qrels[qid], "sparse_gold_rank": sr,
                            "dense_gold_rank": dr, "n_hops": m2cfg["dataset"].get("n_hops", 1), "qtype": "beir"})
        newc = pd.concat(parts, ignore_index=True)
        corp2 = m2.corpus
        ids = sorted(set(corp2["doc_id"]) | set(newc["doc_id"]))
        idx = {d: i for i, d in enumerate(ids)}
        top = newc[(newc["sparse_rank"] <= 20) | (newc["dense_rank"] <= 20)]["doc_id"]
        want = (set(top) | {g for gs in qrels.values() for g in gs}) - set(corp2["doc_id"])
        fetched = pb.docs(sorted(want)).assign(fetched=True)
        corp = pd.concat([corp2, fetched], ignore_index=True).drop_duplicates("doc_id").set_index("doc_id")
        corpus = corp.reindex(ids).rename_axis("doc_id").reset_index()
        corpus[["title", "text"]] = corpus[["title", "text"]].fillna("")
        corpus["fetched"] = corpus["fetched"].fillna(False).astype(bool)
        cands = pd.concat([m2.candidates.drop(columns="doc_idx"), newc], ignore_index=True)
        cands["doc_idx"] = cands["doc_id"].map(idx).astype(np.int32)
        nt = [sparse.n_query_terms(t) for t in newq["query"]]
        newq = newq.assign(group="all", n_tokens=[a for a, _ in nt], n_tokens_in_vocab=[b for _, b in nt])
        queries = pd.concat([q2, newq[q2.columns]], ignore_index=True)   # row order == query_emb order
        np.save(out / "query_emb.npy", np.concatenate([m2.query_emb] + embs))
        labels = pd.concat([lab2, pd.DataFrame(lab)], ignore_index=True)
        write_cache(out, corpus, queries, cands.sort_values(["qid", "doc_idx"]).reset_index(drop=True), labels)
    else:
        raise ValueError(mode)

    # freeze the split manifest (re-applied verbatim if it already exists)
    sp_path = frozen_split_path(cfg["name"])
    q = pd.read_parquet(out / "queries.parquet")
    if sp_path.exists():
        frozen = pd.read_csv(sp_path, dtype=str).set_index("qid")["split"]
        assert set(frozen.index) == set(q["qid"]), "frozen split does not match cache queries"
        q["split"] = q["qid"].map(frozen)
        q.to_parquet(out / "queries.parquet", index=False)
    else:
        sp_path.parent.mkdir(parents=True, exist_ok=True)
        q[["qid", "split"]].to_csv(sp_path, index=False)
    write_json(out / "m3_build.json", {"m2_cache": str(m2.dir), "mode": mode,
                                       "n": q.groupby("split").size().to_dict()})
    return CandidateCache(out)
