"""Downloads real user-to-LLM queries, for comparison with our generated benchmarks.

Two sets are built, each as one user turn per row:
  general  -> the FIRST user turn of conversations sampled from WildChat-1M and
              LMSYS-Chat-1M (Zheng et al., ICLR 2024; Zhao et al., ICLR 2024).
  medical  -> the flagged health-seeking turn of conversations selected by
              HealthChat-11K (Paruchuri et al., EMNLP Findings 2025), pulled from
              the same two source datasets using HealthChat's own annotation file.

STREAMING vs DOWNLOADING: by default (real_queries.streaming: true in config.yaml) the two big datasets are
READ over the network without being saved; only your sample is written. Every run re-reads them. Set
real_queries.streaming: false to download them once into the Hugging Face cache and reuse them.

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


def _open(load_dataset, repo_id, cfg):
    """Open a big dataset either as a stream (nothing saved) or from a one-time local download."""
    streaming = bool(cfg["real_queries"].get("streaming", True))
    if streaming:
        print(f"Streaming {repo_id}: it is read over the network and NOT saved, so only your sample ends up on disk "
              "(but every run reads the whole dataset again; set real_queries.streaming: false to download it once).")
    else:
        print(f"Loading {repo_id}: downloaded ONCE into the Hugging Face cache (several GB; see the dataset page) "
              "and reused by later runs.")
    return load_dataset(repo_id, split="train", streaming=streaming)


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
        ds = _open(load_dataset, repo_id, cfg)
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
SPLIT_ONLY = False
REFRESH = False  # --refresh: download and sample again even if medical.jsonl exists


def _download_annotations(hf_hub_download):
    """The annotation file has been renamed on the Hub before, so try the known names and then look for the newest one."""
    from huggingface_hub import HfApi

    names = ["HealthChat-11K_master_annotations_v2.1.0.jsonl", "HealthChat-11K_v2.1.0_annotations.jsonl"]
    try:
        found = sorted(f for f in HfApi().list_repo_files(HEALTHCHAT_ANNOTATIONS_ID, repo_type="dataset")
                       if "annotations" in f and f.endswith(".jsonl"))
        names += found[::-1]
    except Exception:  # noqa: BLE001
        pass
    last = None
    for name in dict.fromkeys(names):
        try:
            return hf_hub_download(repo_id=HEALTHCHAT_ANNOTATIONS_ID, filename=name, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            last = e
    raise SystemExit(f"Could not download the HealthChat annotations ({last}).\n"
                     "If data/real_queries/medical.jsonl already exists, keep your sample and only split it:\n"
                     "    python scripts/prepare_real_queries.py medical --split-only")


def split_existing(cfg, rc):
    """No download: split the existing medical.jsonl into medical_heldout.jsonl (held out) and medical.jsonl (rest), with the
    configured seed. Your sample stays exactly the same set of queries. Running it twice does nothing the second time."""
    out = ROOT / cfg["real_queries"]["output_dir"]
    path, hold = out / "medical.jsonl", out / "medical_heldout.jsonl"
    if hold.exists():
        print(f"{hold} already exists: nothing to do. (Delete it, and put all queries back in medical.jsonl, to split again.)")
        return
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    random.Random(rc["seed"]).shuffle(rows)
    n_hold = min(int(rc.get("heldout", 300)), len(rows) // 3)
    _write(rows[:n_hold], hold)
    _write(rows[n_hold:], path)

def prepare_medical(cfg):
    out = ROOT / cfg["real_queries"]["output_dir"]
    if SPLIT_ONLY or ((out / "medical.jsonl").exists() and not REFRESH):
        # An existing sample is kept (no download), so your queries stay the same set; only the held-out split is made if missing.
        return split_existing(cfg, cfg["real_queries"]["medical"])
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    rc = cfg["real_queries"]["medical"]
    rng = random.Random(rc["seed"])
    _check_login()

    print(f"Downloading HealthChat-11K annotations from {HEALTHCHAT_ANNOTATIONS_ID}...")
    ann_path = _download_annotations(hf_hub_download)
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
        print(f"Looking for {len(ids):,} annotated conversation(s) in {repo_id}")
        ds = _open(load_dataset, repo_id, cfg)
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
    out = ROOT / cfg["real_queries"]["output_dir"]
    n_hold = min(int(rc.get("heldout", 300)), len(rows) // 3)
    # A held-out part that the mutations and the realism tables never use for design, for the paper's final closeness numbers
    # (README 35). Same seed = same split every time. medical.jsonl keeps the rest.
    _write(rows[:n_hold], out / "medical_heldout.jsonl")
    _write(rows[n_hold:], out / "medical.jsonl")


STEPS = {"general": prepare_general, "medical": prepare_medical}


def main():
    global SPLIT_ONLY, REFRESH
    SPLIT_ONLY, REFRESH = "--split-only" in sys.argv, "--refresh" in sys.argv
    sys.argv = [a for a in sys.argv if a not in ("--split-only", "--refresh")]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which != "all" and which not in STEPS:
        sys.exit(f"Usage: python scripts/prepare_real_queries.py [{'|'.join(STEPS)}|all]")
    cfg = load_config()
    for name in STEPS if which == "all" else [which]:
        print(f"\n=== {name} ===")
        STEPS[name](cfg)


if __name__ == "__main__":
    main()
