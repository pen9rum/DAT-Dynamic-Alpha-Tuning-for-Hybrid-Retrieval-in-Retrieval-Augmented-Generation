"""Small-LLM client for Milestone 3B (OpenRouter), with an auditable response cache.

Cache key = (model, prompt_version, qid, memory_digest, stream_key, messages). stream_key is the
stream seed for stream-dependent conditions and "static" for conditions whose prompt does not
depend on the stream (the memory digest already pins the memory state).
Every row stores tokens, cost, latency and provider. A hard USD budget stops new calls.
The API key is read from the environment and never logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time

import numpy as np
import requests

from ..config import resolve

URL = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


def _openrouter_key() -> str:
    """OpenRouter key only (sk-or-...): env OPENROUTER_API_KEY, else the project .env
    (OPENROUTER_API_KEY or OPENAI_API_KEY). A system-wide OPENAI_API_KEY that is not an
    OpenRouter key is ignored. The key is never printed."""
    cands = [os.environ.get("OPENROUTER_API_KEY", "")]
    env = resolve(".env")
    if env.exists():
        vals = {}
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                vals[k.strip()] = v.strip().strip('"').strip("'")
        cands += [vals.get("OPENROUTER_API_KEY", ""), vals.get("OPENAI_API_KEY", "")]
    cands.append(os.environ.get("OPENAI_API_KEY", ""))
    for k in cands:
        if k and k.strip().startswith("sk-or-"):
            return k.strip()
    raise RuntimeError("no OpenRouter key (sk-or-...) found in OPENROUTER_API_KEY or .env")


class SmallLLM:
    def __init__(self, model: str, budget_usd: float, cache_path: str = "cache/m3b_llm_cache.sqlite",
                 max_tokens: int = 80, temperature: float = 0.0, seed: int = 1):   # provider requires seed >= 1
        self.model, self.max_tokens, self.temperature, self.seed = model, max_tokens, temperature, seed
        self.budget = budget_usd
        self._key = _openrouter_key()
        p = resolve(cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(p, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, model TEXT, text TEXT, "
                        "prompt_tokens INT, completion_tokens INT, cost REAL, latency REAL, provider TEXT, ts REAL)")
        self.lock = threading.Lock()
        self.spent_new = 0.0
        self.stats = {"calls_new": 0, "cache_hits": 0, "errors": 0}

    def key(self, prompt_version, qid, mem_digest, stream_key, messages) -> str:
        blob = json.dumps([self.model, prompt_version, str(qid), mem_digest, stream_key, messages,
                           self.temperature, self.max_tokens], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def lookup(self, key):
        with self.lock:
            return self.db.execute("SELECT text, prompt_tokens, completion_tokens, cost, latency, provider "
                                   "FROM calls WHERE key=?", (key,)).fetchone()

    def call(self, key: str, messages: list[dict]) -> dict:
        row = self.lookup(key)
        if row is not None:
            with self.lock:
                self.stats["cache_hits"] += 1
            return {"text": row[0], "prompt_tokens": row[1], "completion_tokens": row[2], "cost": row[3],
                    "latency": row[4], "provider": row[5], "cached": True}
        with self.lock:
            if self.spent_new >= self.budget:
                raise BudgetExceeded(f"budget ${self.budget} reached")
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "seed": self.seed, "usage": {"include": True},
                "provider": {"allow_fallbacks": True}}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(8):
            t0 = time.time()
            try:
                r = requests.post(URL, headers=headers, json=body, timeout=120)
                j = r.json()
            except Exception:
                j, r = {"error": {"message": "network"}}, None
            if r is not None and r.status_code == 200 and j.get("choices"):
                break
            with self.lock:
                self.stats["errors"] += 1
            code = (j.get("error") or {}).get("code") if isinstance(j, dict) else None
            if code in (400, 401, 402, 403, 404):
                raise RuntimeError(f"OpenRouter error {code}: {(j.get('error') or {}).get('message')}")
            time.sleep(min(60, 2 ** attempt))
        else:
            detail = json.dumps(j)[:500] if isinstance(j, dict) else str(j)[:500]
            raise RuntimeError(f"OpenRouter: too many failed attempts; last response: {detail}")
        lat = time.time() - t0
        u = j.get("usage") or {}
        out = {"text": j["choices"][0]["message"].get("content") or "", "prompt_tokens": u.get("prompt_tokens", 0),
               "completion_tokens": u.get("completion_tokens", 0), "cost": float(u.get("cost") or 0.0),
               "latency": lat, "provider": j.get("provider"), "cached": False}
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?)",
                            (key, self.model, out["text"], out["prompt_tokens"], out["completion_tokens"],
                             out["cost"], lat, out["provider"], time.time()))
            self.db.commit()
            self.spent_new += out["cost"]
            self.stats["calls_new"] += 1
        return out


_FENCE = re.compile(r"```(?:json)?", re.I)
_JSON = re.compile(r"\{.*?\}", re.S)


def _fail(g: int) -> dict:
    return {"decision": "STAY", "alpha_idx": g, "adapt_score": 0.0, "valid": False}


def parse_decision(text: str, alphas: np.ndarray, g: int) -> dict:
    """Validated decision; malformed output -> STAY at alpha_global (flagged, never retried).
    Markdown fences stripped, first JSON object used, decision validated, adapt_score clamped
    to [0, 100]; alpha accepted only within 0.05 of a legal grid value (then snapped to it)."""
    m = _JSON.search(_FENCE.sub("", text or ""))
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict):
        return _fail(g)
    dec = str(d.get("decision", "")).strip().upper()
    try:
        score = float(d.get("adapt_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = float(np.clip(score, 0.0, 100.0)) if np.isfinite(score) else 0.0
    if dec == "STAY":
        return {"decision": "STAY", "alpha_idx": g, "adapt_score": score, "valid": True}
    if dec != "ADAPT":
        return _fail(g)
    try:
        a = float(d.get("alpha"))
    except (TypeError, ValueError):
        return _fail(g)
    if not np.isfinite(a):
        return _fail(g)
    j = int(np.argmin(np.abs(alphas - a)))
    if abs(alphas[j] - a) >= 0.05 - 1e-9:
        return _fail(g)
    return {"decision": "ADAPT", "alpha_idx": j, "adapt_score": score, "valid": True}
