"""Loads config.yaml and the secret keys in .env, and builds folder names.

PROFILES: config.yaml can contain a `profiles:` section. A profile is a set of
overrides that is merged on top of the normal settings, so one switch changes
many settings at once. Select one with the environment variable BP_PROFILE
(for example `BP_PROFILE=large python scripts/generate_benchmark.py ...`),
with `scripts/run_all.py --profile large`, or load_config(profile="large").
"""
import copy
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path=None, profile=None):
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    profiles = cfg.pop("profiles", None) or {}
    profile = profile or os.environ.get("BP_PROFILE") or None
    if profile:
        if profile not in profiles:
            raise ValueError(
                f"Unknown profile '{profile}'. Profiles in config.yaml: {sorted(profiles)}"
            )
        cfg = _deep_merge(cfg, profiles[profile])
    cfg["_profile"] = profile or "default"
    return cfg


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
