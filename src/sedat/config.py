"""Config loading, CLI overrides, cache paths, and run directories."""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- config

def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. `base:` (path relative to the file) is merged underneath."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(base, cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def get_dotted(cfg: dict, key: str, default: Any = None) -> Any:
    node = cfg
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def base_argparser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="dotted overrides, e.g. fusion.normalization=zscore")
    return ap


def alpha_grid(cfg: dict) -> np.ndarray:
    g = get_dotted(cfg, "fusion.alpha_grid", {"start": 0.0, "stop": 1.0, "step": 0.1})
    if isinstance(g, list):
        return np.round(np.asarray(g, dtype=float), 6)
    n = int(round((g["stop"] - g["start"]) / g["step"])) + 1
    return np.round(np.linspace(g["start"], g["stop"], n), 6)


# ----------------------------------------------------------------------------- paths

def fingerprint(obj: Any, n: int = 10) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def retrieval_cache_dir(cfg: dict) -> Path:
    """Cache for candidate scores. Keyed on everything that changes retrieval output."""
    key = {"dataset": cfg["dataset"], "retrieval": cfg.get("retrieval", {}), "seed": cfg.get("seed", 0)}
    d = resolve(get_dotted(cfg, "paths.cache_dir", "cache")) / cfg["name"] / fingerprint(key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def reward_table_path(cfg: dict, normalization: str | None = None) -> Path:
    norm = normalization or get_dotted(cfg, "fusion.normalization", "minmax")
    key = {"v": 2, "alpha": alpha_grid(cfg).tolist(), "norm": norm,
           "rrf_k": get_dotted(cfg, "fusion.rrf_k", 60),
           "eval": {k: get_dotted(cfg, f"evaluation.{k}") for k in ("metrics_k", "ndcg_k", "topk_store")}}
    return retrieval_cache_dir(cfg) / f"reward_table_{norm}_{fingerprint(key, 8)}.parquet"


# ----------------------------------------------------------------------------- runs

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def make_run_dir(cfg: dict, experiment: str, tag: str | None = None) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{experiment}_{stamp}" + (f"_{tag}" if tag else "")
    run = resolve(get_dotted(cfg, "paths.runs_dir", "runs")) / cfg["name"] / name
    (run / "plots").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    meta = {"experiment": experiment, "time": stamp, "git_commit": git_commit(),
            "seed": cfg.get("seed", 0), "retrieval_cache": str(retrieval_cache_dir(cfg))}
    write_json(run / "meta.json", meta)
    setup_logging(run / "logs" / "run.log")
    return run


def setup_logging(logfile: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        handlers=handlers, force=True)


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def load_dotenv(path: str | Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader (no dependency). Existing env vars win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
