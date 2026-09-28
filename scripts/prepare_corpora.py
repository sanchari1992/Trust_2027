"""Downloads the real corpora and puts them where each RAG expects them.

Usage (from the repo root):
    python scripts/prepare_corpora.py lightrag    # UltraDomain
    python scripts/prepare_corpora.py raptor      # QuALITY articles + RAPTOR's code
    python scripts/prepare_corpora.py medrag      # MedRAG Textbooks
    python scripts/prepare_corpora.py all
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, corpus_dir, load_config

QUALITY_URL = (
    "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/"
    "QuALITY.v1.0.1.htmlstripped.{split}"
)
RAPTOR_REPO = "https://github.com/parthsarthi03/raptor.git"


def _clear_txt(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()


# ---------------------------------------------------------------- LightRAG
def unique_contexts_from_jsonl(path):
    """UltraDomain has one row per question, and many rows share the same
    'context' (the document). LightRAG's own Step_0.py keeps each context once."""
    seen, docs = set(), []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            context = json.loads(line).get("context")
            if context and context not in seen:
                seen.add(context)
                docs.append(context)
    return docs


def prepare_lightrag(cfg):
    from huggingface_hub import hf_hub_download

    c = cfg["corpora"]["lightrag"]
    print(f"Downloading UltraDomain '{c['subset']}' from Hugging Face...")
    path = hf_hub_download(
        repo_id="TommyChien/UltraDomain",
        filename=f"{c['subset']}.jsonl",
        repo_type="dataset",
    )
    docs = unique_contexts_from_jsonl(path)
    total = len(docs)
    if c["max_docs"]:
        docs = docs[: c["max_docs"]]
    out = corpus_dir(cfg, "lightrag")
    _clear_txt(out)
    for i, text in enumerate(docs):
        (out / f"ultradomain_{c['subset']}_{i:03d}.txt").write_text(
            text, encoding="utf-8"
        )
    words = sum(len(d.split()) for d in docs)
    print(f"Wrote {len(docs)} of {total} documents (~{words:,} words) to {out}")


# ------------------------------------------------------------------ RAPTOR
def articles_from_quality(path):
    seen, articles = set(), []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["article_id"] not in seen:
                seen.add(row["article_id"])
                articles.append((row["article_id"], row["article"]))
    return articles


def prepare_raptor(cfg):
    dest = ROOT / cfg["raptor"]["repo_dir"]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", RAPTOR_REPO, str(dest)], check=True)
    commit = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"RAPTOR code at {dest} (commit {commit}). Record this commit in your paper.")

    c = cfg["corpora"]["raptor"]
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    raw_file = raw / f"quality_{c['split']}.jsonl"
    if not raw_file.exists():
        print(f"Downloading QuALITY {c['split']} set from GitHub...")
        urllib.request.urlretrieve(QUALITY_URL.format(split=c["split"]), raw_file)
    articles = articles_from_quality(raw_file)
    total = len(articles)
    if c["max_docs"]:
        articles = articles[: c["max_docs"]]
    out = corpus_dir(cfg, "raptor")
    _clear_txt(out)
    for article_id, text in articles:
        (out / f"quality_{article_id}.txt").write_text(text, encoding="utf-8")
    words = sum(len(t.split()) for _, t in articles)
    print(f"Wrote {len(articles)} of {total} articles (~{words:,} words) to {out}")


# ------------------------------------------------------------------ MedRAG
def prepare_medrag(cfg):
    from huggingface_hub import HfApi, hf_hub_download

    c = cfg["corpora"]["medrag"]
    files = sorted(
        f
        for f in HfApi().list_repo_files("MedRAG/textbooks", repo_type="dataset")
        if f.startswith("chunk/") and f.endswith(".jsonl")
    )
    print(f"MedRAG Textbooks has {len(files)} textbook files.")
    if c["max_textbooks"]:
        files = files[: c["max_textbooks"]]
    out = corpus_dir(cfg, "medrag")
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        print("Downloading", f)
        hf_hub_download(
            repo_id="MedRAG/textbooks", filename=f, repo_type="dataset", local_dir=str(out)
        )
    rows = sum(
        1
        for p in (out / "chunk").glob("*.jsonl")
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    print(f"Ready: {len(files)} textbooks, {rows:,} snippets in {out}/chunk")


STEPS = {"lightrag": prepare_lightrag, "raptor": prepare_raptor, "medrag": prepare_medrag}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which != "all" and which not in STEPS:
        sys.exit(f"Usage: python scripts/prepare_corpora.py [{'|'.join(STEPS)}|all]")
    cfg = load_config()
    for name in STEPS if which == "all" else [which]:
        print(f"\n=== {name} ===")
        STEPS[name](cfg)


if __name__ == "__main__":
    main()
