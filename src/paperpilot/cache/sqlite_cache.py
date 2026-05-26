"""SQLite-backed cache for OpenAI embeddings and RAGAS judge calls.

Why: re-running ingestion or eval should be free after the first pass.
Keys are SHA-256 over (model, content) so model changes invalidate cleanly.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from paperpilot.config import settings


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class SQLiteCache:
    """Thread-safe SQLite cache. Embeddings stored as JSON-encoded float arrays.

    Three namespaces:
      - embeddings:  key = sha256(model || text)                → list[float]
      - judges:      key = sha256(model || prompt)              → arbitrary JSON dict
      - arxiv:       key = sha256(query || str(max_results))    → list[dict]
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as cx:
            cx.executescript(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key      TEXT PRIMARY KEY,
                    model    TEXT NOT NULL,
                    vec_json TEXT NOT NULL,
                    created  REAL NOT NULL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS judges (
                    key       TEXT PRIMARY KEY,
                    model     TEXT NOT NULL,
                    response  TEXT NOT NULL,
                    created   REAL NOT NULL DEFAULT (strftime('%s','now'))
                );
                CREATE TABLE IF NOT EXISTS arxiv (
                    key      TEXT PRIMARY KEY,
                    query    TEXT NOT NULL,
                    results  TEXT NOT NULL,
                    created  REAL NOT NULL DEFAULT (strftime('%s','now'))
                );
                """
            )
            cx.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        cx = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
        cx.execute("PRAGMA journal_mode=WAL")
        try:
            yield cx
        finally:
            cx.close()

    # --- Embeddings ---
    def get_embedding(self, model: str, text: str) -> list[float] | None:
        k = _key("emb", model, text)
        with self._lock, self._connect() as cx:
            row = cx.execute("SELECT vec_json FROM embeddings WHERE key = ?", (k,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_embedding(self, model: str, text: str, vec: list[float]) -> None:
        k = _key("emb", model, text)
        with self._lock, self._connect() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO embeddings(key, model, vec_json) VALUES (?, ?, ?)",
                (k, model, json.dumps(vec)),
            )
            cx.commit()

    def set_embeddings_bulk(self, model: str, texts: list[str], vecs: list[list[float]]) -> None:
        rows = [(_key("emb", model, t), model, json.dumps(v)) for t, v in zip(texts, vecs)]
        with self._lock, self._connect() as cx:
            cx.executemany(
                "INSERT OR REPLACE INTO embeddings(key, model, vec_json) VALUES (?, ?, ?)", rows
            )
            cx.commit()

    # --- Judgecalls ---
    def get_judge(self, model: str, prompt: str) -> dict[str, Any] | None:
        k = _key("judge", model, prompt)
        with self._lock, self._connect() as cx:
            row = cx.execute("SELECT response FROM judges WHERE key = ?", (k,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_judge(self, model: str, prompt: str, response: dict[str, Any]) -> None:
        k = _key("judge", model, prompt)
        with self._lock, self._connect() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO judges(key, model, response) VALUES (?, ?, ?)",
                (k, model, json.dumps(response)),
            )
            cx.commit()

    # --- ArXivsearchresults ---
    def get_arxiv(self, query: str, max_results: int) -> list[dict] | None:
        k = _key("arxiv", query, str(max_results))
        with self._lock, self._connect() as cx:
            row = cx.execute("SELECT results FROM arxiv WHERE key = ?", (k,)).fetchone()
            return json.loads(row[0]) if row else None

    def set_arxiv(self, query: str, max_results: int, results: list[dict]) -> None:
        k = _key("arxiv", query, str(max_results))
        with self._lock, self._connect() as cx:
            cx.execute(
                "INSERT OR REPLACE INTO arxiv(key, query, results) VALUES (?, ?, ?)",
                (k, query, json.dumps(results)),
            )
            cx.commit()


@lru_cache
def get_cache() -> SQLiteCache:
    return SQLiteCache(settings.cache_db_full_path)
