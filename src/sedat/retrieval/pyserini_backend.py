"""Pyserini-backed retrievers: Lucene BM25 + FAISS flat BGE-base-en-v1.5.

Same backend for every dataset in the cross-dataset suite:
  - BEIR (NQ, FiQA, SciFact, HotpotQA): Pyserini prebuilt indexes
      sparse  beir-v1.0.0-<name>.flat               (Lucene, BM25 k1=0.9 b=0.4)
      dense   beir-v1.0.0-<name>.bge-base-en-v1.5   (FAISS flat, inner product)
  - MuSiQue / 2Wiki: local indexes built with the SAME tools and model
    (`build_local_indexes`), laid out exactly like the prebuilt ones.

Exact scores: when a doc is in one retriever's list but not the other's, its
missing score is computed exactly (Lucene `compute_query_document_score`, FAISS
`reconstruct`) -- never imputed.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..config import resolve

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def setup_java(java_home: str | None = None, pyserini_cache: str | None = "cache/pyserini") -> None:
    """Must run before pyserini/jnius is imported. Uses the portable JDK in tools/."""
    if java_home is None:
        cands = sorted(resolve("tools").glob("jdk-21*"))
        java_home = str(cands[-1]) if cands else os.environ.get("JAVA_HOME")
    if java_home:
        java_home = str(resolve(java_home))
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = str(Path(java_home) / "bin") + os.pathsep + os.environ.get("PATH", "")
    if pyserini_cache:
        os.environ.setdefault("PYSERINI_CACHE", str(resolve(pyserini_cache)))


def _is_local(index: str) -> bool:
    return Path(resolve(index)).is_dir()


class LuceneBM25:
    name = "sparse"

    def __init__(self, index: str, k1: float = 0.9, b: float = 0.4, threads: int = 8):
        from pyserini.index.lucene import LuceneIndexReader
        from pyserini.search.lucene import LuceneSearcher
        if _is_local(index):
            path = str(resolve(index))
            self.searcher, self.reader = LuceneSearcher(path), LuceneIndexReader(path)
        else:
            self.searcher = LuceneSearcher.from_prebuilt_index(index)
            self.reader = LuceneIndexReader.from_prebuilt_index(index)
        self.k1, self.b, self.threads = k1, b, threads
        self.searcher.set_bm25(k1, b)
        from jnius import autoclass
        self._sim = autoclass("org.apache.lucene.search.similarities.BM25Similarity")(float(k1), float(b))

    def search(self, qids: list[str], texts: list[str], k: int) -> dict[str, tuple[list[str], np.ndarray]]:
        res = self.searcher.batch_search(texts, qids, k=k, threads=self.threads)
        return {q: ([h.docid for h in hs], np.array([h.score for h in hs], dtype=np.float64))
                for q, hs in res.items()}

    def score(self, text: str, docid: str) -> float:
        return float(self.reader.compute_query_document_score(docid, text, similarity=self._sim))

    def score_many(self, text: str, docids: list[str]) -> np.ndarray:
        """Exact BM25 for specific docs with ONE Lucene search: the normal bag-of-words
        query as MUST plus a non-scoring FILTER on the doc ids (unmatched docs -> 0)."""
        if not docids:
            return np.zeros(0)
        from jnius import autoclass
        if not hasattr(self, "_BQ"):
            from pyserini.analysis import get_lucene_analyzer
            self._BQ = autoclass("org.apache.lucene.search.BooleanQuery$Builder")
            self._Occur = autoclass("org.apache.lucene.search.BooleanClause$Occur")
            self._TQ = autoclass("org.apache.lucene.search.TermQuery")
            self._Term = autoclass("org.apache.lucene.index.Term")
            self._bow = autoclass("io.anserini.search.query.BagOfWordsQueryGenerator")()
            self._an = get_lucene_analyzer()
        ids = self._BQ()
        for d in docids:
            ids.add(self._TQ(self._Term("id", d)), self._Occur.SHOULD)
        b = self._BQ()
        b.add(self._bow.buildQuery("contents", self._an, text), self._Occur.MUST)
        b.add(ids.build(), self._Occur.FILTER)
        hits = {h.docid: h.score for h in self.searcher.search(b.build(), k=len(docids))}
        return np.array([hits.get(d, 0.0) for d in docids], dtype=np.float64)

    def n_query_terms(self, text: str) -> tuple[int, int]:
        toks = self.reader.analyze(text)
        return len(toks), sum(self.reader.get_term_counts(t, analyzer=None)[0] > 0 for t in toks)

    def doc(self, docid: str) -> dict:
        """{'title', 'text'} from the stored raw document (BEIR flat and our local layout)."""
        d = self.searcher.doc(docid)
        raw = d.raw() if d is not None else None
        if not raw:
            return {"title": "", "text": ""}
        try:
            j = json.loads(raw)
        except json.JSONDecodeError:
            return {"title": "", "text": raw}
        if "title" in j or "text" in j:
            return {"title": j.get("title", ""), "text": j.get("text", "")}
        c = j.get("contents", "")
        title, _, text = c.partition("\n")
        return {"title": title, "text": text}


class FaissBGE:
    name = "dense"

    def __init__(self, index: str, model: str = "BAAI/bge-base-en-v1.5", query_prefix: str = BGE_QUERY_PREFIX,
                 batch_size: int = 64, max_seq_length: int = 512, threads: int | None = None, mmap: bool = False):
        import faiss
        from sentence_transformers import SentenceTransformer
        if _is_local(index):
            d = Path(resolve(index))
        else:
            from pyserini.util import download_prebuilt_index
            d = Path(download_prebuilt_index(index, verbose=False))
        if threads:
            faiss.omp_set_num_threads(threads)
        flags = faiss.IO_FLAG_MMAP if mmap else 0
        log.info("loading FAISS index %s (mmap=%s)", d, mmap)
        self.index = faiss.read_index(str(d / "index"), flags)
        self.docids = [l.strip() for l in open(d / "docid", encoding="utf-8")]
        self.row = {x: i for i, x in enumerate(self.docids)}
        self.model = SentenceTransformer(model, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.prefix, self.batch_size = query_prefix, batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode([self.prefix + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=len(texts) > 500).astype(np.float32)

    def search(self, qids: list[str], qemb: np.ndarray, k: int) -> dict[str, tuple[list[str], np.ndarray]]:
        S, I = self.index.search(qemb, k)
        return {q: ([self.docids[i] for i in row if i >= 0], s[row >= 0].astype(np.float64))
                for q, s, row in zip(qids, S, I)}

    def score(self, qvec: np.ndarray, docids: list[str]) -> np.ndarray:
        rows = np.array([self.row[d] for d in docids], dtype=np.int64)
        V = np.stack([self.index.reconstruct(int(r)) for r in rows]) if len(rows) else np.zeros((0, len(qvec)))
        return V @ qvec


# ----------------------------------------------------------------------------- local index building

def build_local_indexes(corpus_jsonl: Path, out_dir: Path, model: str = "BAAI/bge-base-en-v1.5",
                        batch_size: int = 32, max_seq_length: int = 256, chunk: int = 512) -> None:
    """Lucene BM25 index + FAISS flat BGE index in Pyserini's prebuilt layout.
    Dense encoding is chunked and resumable (it takes hours on a laptop CPU)."""
    import faiss
    from sentence_transformers import SentenceTransformer
    out_dir = Path(out_dir)
    lucene_dir, faiss_dir, parts = out_dir / "lucene", out_dir / "faiss", out_dir / "emb_parts"
    for d in (faiss_dir, parts):
        d.mkdir(parents=True, exist_ok=True)

    if not (lucene_dir / "segments_1").exists() and not any(lucene_dir.glob("segments_*")):
        coll = out_dir / "collection"
        coll.mkdir(exist_ok=True)
        (coll / "docs.jsonl").write_bytes(Path(corpus_jsonl).read_bytes())
        cmd = [sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
               "--input", str(coll), "--index", str(lucene_dir), "--generator", "DefaultLuceneDocumentGenerator",
               "--threads", "4", "--storePositions", "--storeDocvectors", "--storeRaw"]
        log.info("building Lucene index: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, env=os.environ.copy())

    if (faiss_dir / "index").exists():
        return
    docs = [json.loads(l) for l in open(corpus_jsonl, encoding="utf-8")]
    m = SentenceTransformer(model, device="cpu")
    m.max_seq_length = max_seq_length
    n_chunks = (len(docs) + chunk - 1) // chunk
    for c in range(n_chunks):
        p = parts / f"{c:05d}.npy"
        if p.exists():
            continue
        texts = [d["contents"] for d in docs[c * chunk:(c + 1) * chunk]]   # title \n text, no prefix (BGE passages)
        np.save(p, m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32))
        log.info("encoded chunk %d/%d", c + 1, n_chunks)
    E = np.concatenate([np.load(parts / f"{c:05d}.npy") for c in range(n_chunks)])
    idx = faiss.IndexFlatIP(E.shape[1])
    idx.add(E)
    faiss.write_index(idx, str(faiss_dir / "index"))
    with open(faiss_dir / "docid", "w", encoding="utf-8") as f:
        f.writelines(d["id"] + "\n" for d in docs)
    log.info("FAISS flat index: %d x %d -> %s", *E.shape, faiss_dir)
