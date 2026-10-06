"""OpenRouter chat client (OpenAI-compatible API) with an on-disk response cache.

The API key is read from the env var named in the config (default
OPENROUTER_API_KEY, also picked up from a project-level .env). It is left empty
on purpose until LLM-based policies are needed; Milestone 1 never calls this.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from ..config import load_dotenv, resolve

log = logging.getLogger(__name__)


class LLMNotConfigured(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, model: str = "openai/gpt-4o", base_url: str = "https://openrouter.ai/api/v1",
                 api_key_env: str = "OPENROUTER_API_KEY", temperature: float = 0.0,
                 max_tokens: int = 16, cache_path: str | Path = "cache/llm_cache.sqlite",
                 max_retries: int = 5, **_):
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.base_url, self.api_key_env, self.max_retries = base_url, api_key_env, max_retries
        self._client = None
        cache_path = resolve(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")

    @classmethod
    def from_config(cls, llm_cfg: dict) -> "OpenRouterClient":
        return cls(**llm_cfg)

    def _get_client(self):
        if self._client is None:
            load_dotenv()
            key = os.environ.get(self.api_key_env, "").strip()
            if not key:
                raise LLMNotConfigured(
                    f"{self.api_key_env} is empty. Put it in .env (see .env.example) "
                    f"before running LLM-based policies.")
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=key)
        return self._client

    def _key(self, messages: list[dict]) -> str:
        blob = json.dumps({"m": self.model, "t": self.temperature, "x": self.max_tokens, "msg": messages},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages: list[dict]) -> str:
        key = self._key(messages)
        row = self.db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        if row is not None:
            return row[0]
        client = self._get_client()
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(model=self.model, messages=messages,
                                                      temperature=self.temperature,
                                                      max_tokens=self.max_tokens)
                text = resp.choices[0].message.content or ""
                break
            except Exception as e:  # rate limits / transient errors
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                log.warning("OpenRouter error (%s); retrying in %ds", e, wait)
                time.sleep(wait)
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, text))
        self.db.commit()
        return text
