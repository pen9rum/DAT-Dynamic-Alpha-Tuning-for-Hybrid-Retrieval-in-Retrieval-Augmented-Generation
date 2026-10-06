"""Build the pooled corpus + Lucene/FAISS(BGE) indexes for datasets without prebuilt
indexes (MuSiQue, 2Wiki). Resumable; dense encoding takes hours on a laptop CPU.

  python -m experiments.build_local_index --config configs/musique.yaml
"""
from sedat.config import base_argparser, get_dotted, load_config, resolve, setup_logging
from sedat.data.multihop_local import build_multihop_local
from sedat.retrieval.pyserini_backend import build_local_indexes, setup_java


def main():
    args = base_argparser(__doc__).parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    setup_java(get_dotted(cfg, "paths.java_home"))
    ds = cfg["dataset"]
    out = resolve(ds["processed_dir"])
    if not (out / "questions.parquet").exists():
        build_multihop_local(ds["source"], ds["n_questions"], cfg.get("seed", 0), out)
    d = cfg["retrieval"]["dense"]
    build_local_indexes(out / "corpus.jsonl", out, model=d["model"],
                        batch_size=get_dotted(cfg, "retrieval.encode_batch_size", 32),
                        max_seq_length=get_dotted(cfg, "retrieval.passage_max_seq_length", 256))
    print(f"indexes ready in {out}")


if __name__ == "__main__":
    main()
