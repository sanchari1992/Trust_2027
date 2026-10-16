"""A disk cache for deterministic LLM calls (README section 14.11).

Only calls that ask for temperature 0 EXPLICITLY are cached: the judge (src/evalrun/judge.py) and the mutation
checks (src/mutate/verify.py). The calls whose randomness matters are never cached: the RAG's own answers (so the
`c_rerun` control still asks the RAG again), and every call that WRITES text (benchmark generation, mutation
text), which use a temperature above 0.

The key is the model, thinking mode, system text, user text, max_tokens and temperature, so any change to a prompt,
the model or the settings is a miss. A hit costs nothing and is not counted as an API call (`USAGE.cache_hits`).
What it saves: a resumed or repeated run (a judged answer is never judged twice), a `--fresh` rerun, and judging the
`c_rerun` control (the RAG usually repeats its own answer, so the judge prompts are the same as for the benchmark).
"""
import hashlib
import sqlite3
import threading
from pathlib import Path

from src.config import ROOT


class LLMCache:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()

    @staticmethod
    def key(*parts):
        h = hashlib.sha256()
        for p in parts:
            h.update(repr(p).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, k):
        with self._lock:
            row = self._db.execute("SELECT v FROM calls WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k, v):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (k, v))
            self._db.commit()

    def __len__(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]


_CACHES = {}


def get_cache(cfg):
    """The cache for this config, or None when llm.cache is false."""
    c = cfg["llm"]
    if not c.get("cache", True):
        return None
    path = ROOT / c.get("cache_path", "results/llm_cache.sqlite")
    if path not in _CACHES:
        _CACHES[path] = LLMCache(path)
    return _CACHES[path]
