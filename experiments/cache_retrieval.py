"""Stage 1: load dataset, index BM25 + dense, cache union-of-top-K candidate scores."""
from sedat.config import base_argparser, load_config, set_seed, setup_logging
from sedat.pipeline import ensure_cache


def main():
    ap = base_argparser(__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    set_seed(cfg.get("seed", 0))
    cache = ensure_cache(cfg, force=args.force)
    print(f"candidate cache: {cache.dir}")
    print(cache.queries.groupby("split").size().to_string())


if __name__ == "__main__":
    main()
