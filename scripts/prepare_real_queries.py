"""Downloads real user-to-LLM queries, for comparison with our generated benchmarks.

Two sets are built, each as one user turn per row:
  general  -> the FIRST user turn of conversations sampled from WildChat-1M and
              LMSYS-Chat-1M (Zheng et al., ICLR 2024; Zhao et al., ICLR 2024).
  medical  -> the flagged health-seeking turn of conversations selected by
              HealthChat-11K (Paruchuri et al., EMNLP Findings 2025), pulled from
              the same two source datasets using HealthChat's own annotation file.

IMPORTANT - access required before this will run:
  1. Log in at https://huggingface.co and request access on BOTH dataset pages:
       https://huggingface.co/datasets/lmsys/lmsys-chat-1m
       https://huggingface.co/datasets/allenai/WildChat-1M
     Access is granted by the dataset owners, not instantly, so do this early.
  2. Create a token at https://huggingface.co/settings/tokens (read access is enough)
     and put it in .env as HF_TOKEN=...

Usage (from the repo root):
    python scripts/prepare_real_queries.py general
    python scripts/prepare_real_queries.py medical
    python scripts/prepare_real_queries.py all
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT, load_config

WILDCHAT_ID, LMSYS_ID = "allenai/WildChat-1M", "lmsys/lmsys-chat-1m"
HEALTHCHAT_ANNOTATIONS_ID = "yahskapar/HealthChat-11K"

WORD_RE = re.compile(r"\S+")


def _check_login():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        HfApi().whoami()
    except Exception:
        raise RuntimeError(
            "No working Hugging Face login found. Put HF_TOKEN=... in your .env file "
            "(see the notice at the top of this file for how to get one and request access)."
        )


def _first_user_turn(conversation):
    for msg in conversation:
        if msg.get("role") == "user":
            return (msg.get("content") or "").strip()
    return None


def _word_count(text):
    return len(WORD_RE.findall(text))


def _passes(text, min_words, max_words, languages, lang_of_row):
    if not text:
        return False
    n = _word_count(text)
    if n < min_words or n > max_words:
        return False
    if languages and lang_of_row not in languages:
        return False
    return True


def _write(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} queries to {out_path}")


# ---------------------------------------------------------------- general
def prepare_general(cfg):
    from datasets import load_dataset

    _check_login()
    rc = cfg["real_queries"]["general"]
    rng = random.Random(rc["seed"])
    rows = []

    for source in rc["sources"]:
        repo_id, turn_key, lang_key = {
            "wildchat": (WILDCHAT_ID, "conversation", "language"),
            "lmsyschat": (LMSYS_ID, "conversation", "language"),
        }[source]
        print(f"Streaming {repo_id} (this reads the dataset without downloading all of it)...")
        ds = load_dataset(repo_id, split="train", streaming=True)
        # Reservoir sampling, so every conversation in the (very long) stream has
        # an equal chance of being picked, without loading the whole dataset.
        reservoir, seen = [], 0
        target = rc["n_per_source"]
        for row in ds:
            text = _first_user_turn(row.get(turn_key) or [])
            lang = (row.get(lang_key) or "").lower()[:2] or None
            if not _passes(text, rc["min_words"], rc["max_words"], rc.get("languages"), lang):
                continue
            item = {
                "id": f"{source}-{row.get('conversation_id', seen)}",
                "text": text,
                "source": source,
                "domain": "general",
                "language": lang,
            }
            seen += 1
            if len(reservoir) < target:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen - 1)
                if j < target:
                    reservoir[j] = item
            if seen % 20000 == 0:
                print(f"  ...scanned {seen:,} matching conversations so far")
        print(f"{source}: sampled {len(reservoir)} of {seen:,} matching conversations")
        rows += reservoir

    rng.shuffle(rows)
    _write(rows, ROOT / cfg["real_queries"]["output_dir"] / "general.jsonl")


# ---------------------------------------------------------------- medical
def prepare_medical(cfg):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    _check_login()
    rc = cfg["real_queries"]["medical"]
    rng = random.Random(rc["seed"])

    print(f"Downloading HealthChat-11K annotations from {HEALTHCHAT_ANNOTATIONS_ID}...")
    ann_path = hf_hub_download(
        repo_id=HEALTHCHAT_ANNOTATIONS_ID, filename="HealthChat-11K_v2.1.0_annotations.jsonl",
        repo_type="dataset",
    )
    annotations = [json.loads(l) for l in open(ann_path, encoding="utf-8") if l.strip()]
    print(f"{len(annotations):,} annotated conversations")
    by_id = {}
    for a in annotations:
        cid = a.get("conversation_id")
        if cid:
            by_id.setdefault(a.get("dataset_source"), {})[cid] = a

    n = rc.get("n")
    if n and n < len(annotations):
        rng.shuffle(annotations)
        wanted = {a["conversation_id"] for a in annotations[:n]}
    else:
        wanted = {a["conversation_id"] for a in annotations}

    rows = []
    for source, repo_id, id_key in [("lmsys", LMSYS_ID, "conversation_id"), ("wildchat", WILDCHAT_ID, "conversation_id")]:
        ids = {cid for cid in wanted if cid in by_id.get(source, {})}
        if not ids:
            continue
        print(f"Streaming {repo_id} to find {len(ids):,} annotated conversation(s)...")
        ds = load_dataset(repo_id, split="train", streaming=True)
        found = 0
        for row in ds:
            cid = row.get(id_key)
            if cid not in ids:
                continue
            ann = by_id[source][cid]
            # The annotation records which turn index contains the health-seeking message;
            # field name mirrors HealthChat's own generate_artifacts.py.
            turn_idx = ann.get("user_turn_index", 0)
            turns = [m for m in (row.get("conversation") or []) if m.get("role") == "user"]
            text = (turns[turn_idx]["content"] if turn_idx < len(turns) else _first_user_turn(row.get("conversation") or [])) or ""
            text = text.strip()
            if not _passes(text, rc["min_words"], rc["max_words"], None, None):
                continue
            rows.append({
                "id": f"{source}-{cid}", "text": text, "source": source, "domain": "medical",
                "specialty": ann.get("specialty"),
            })
            found += 1
            ids.discard(cid)
            if not ids:
                break
        print(f"{source}: matched {found} of {len(by_id.get(source, {}))} annotated conversations")

    rng.shuffle(rows)
    _write(rows, ROOT / cfg["real_queries"]["output_dir"] / "medical.jsonl")


STEPS = {"general": prepare_general, "medical": prepare_medical}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which != "all" and which not in STEPS:
        sys.exit(f"Usage: python scripts/prepare_real_queries.py [{'|'.join(STEPS)}|all]")
    cfg = load_config()
    for name in STEPS if which == "all" else [which]:
        print(f"\n=== {name} ===")
        STEPS[name](cfg)


if __name__ == "__main__":
    main()
