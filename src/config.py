"""Loads config.yaml and the secret keys in .env, and builds folder names."""
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def corpus_tag(cfg, rag):
    c = cfg["corpora"][rag]
    if rag == "lightrag":
        return f"{c['subset']}_{c['max_docs'] or 'all'}"
    if rag == "raptor":
        return f"quality_{c['split']}_{c['max_docs'] or 'all'}"
    if rag == "medrag":
        return f"textbooks_{c['max_textbooks'] or 'all'}"
    raise ValueError(f"Unknown RAG: {rag}")


def corpus_dir(cfg, rag):
    return ROOT / "data" / "corpora" / f"{rag}_{corpus_tag(cfg, rag)}"


def index_dir(cfg, rag):
    return ROOT / "indexes" / f"{rag}_{corpus_tag(cfg, rag)}"
