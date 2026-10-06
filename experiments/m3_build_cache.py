"""Milestone 3: build the frozen-split candidate caches (cache/milestone3/<name>/).

  python -m experiments.m3_build_cache --configs configs/m3/nq.yaml configs/m3/fiqa.yaml ...
"""
import argparse

from sedat.config import load_config, setup_logging
from sedat.m3.cache import build_m3_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()
    setup_logging()
    for c in args.configs:
        cfg = load_config(c)
        cache = build_m3_cache(cfg)
        print(f"[{cfg['name']}] {cache.dir}: {cache.queries.groupby('split').size().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
