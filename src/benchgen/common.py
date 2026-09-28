"""Helpers shared by the three benchmark generators (RAGAS, ARES, DataMorgana)."""
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Questions that point at "the passage/context/document" cannot be asked to a RAG system.
LEAK_RE = re.compile(
    r"\b(provided|given|above)\s+(context|passage|text|document)\b"
    r"|\b(the|this)\s+(context|passage|excerpt)\b"
    r"|\b(according to|in) the (text|document|passage|context)\b",
    re.I,
)


# ------------------------------------------------------------------ passages
def build_passages(docs, target_words, min_words):
    """Cut the corpus into passages of about `target_words` words.
    Text is grouped per document (per textbook for MedRAG), split into
    sentences, and sentences are packed until the target size is reached."""
    groups = defaultdict(list)
    for d in docs:
        groups[d.get("group") or d["id"]].append(d["text"])

    passages = []
    for key, texts in groups.items():
        sentences = []
        for t in texts:
            sentences += [s.strip() for s in SENT_SPLIT.split(t) if s.strip()]
        buf, n, k = [], 0, 0

        def flush():
            nonlocal buf, n, k
            if buf and n >= min_words:
                passages.append(
                    {"id": f"{key}#{k}", "doc_id": key, "text": " ".join(buf), "words": n}
                )
                k += 1
            buf, n = [], 0

        for s in sentences:
            w = len(s.split())
            if buf and n + w > target_words:
                flush()
            buf.append(s)
            n += w
        flush()

    for i, p in enumerate(passages):
        p["idx"] = i
    return passages


def compute_target(size_cfg, total_words, capacity):
    """How many questions to generate, from the size settings."""
    if size_cfg["mode"] == "count":
        target = int(size_cfg["count"])
    else:
        target = round(size_cfg["questions_per_1k_words"] * total_words / 1000)
        target = max(size_cfg.get("min_questions", 1), target)
        if size_cfg.get("max_questions"):
            target = min(target, size_cfg["max_questions"])
    if target > capacity:
        print(
            f"WARNING: wanted {target} questions but the corpus only offers {capacity} "
            "(passages x max_questions_per_passage). Using that instead."
        )
        target = capacity
    return target


def sampling_order(passages, strategy, rng, per_passage):
    """The order in which passages will be tried (a list of passage indices)."""

    def one_round():
        idx = list(range(len(passages)))
        if strategy == "sequential":
            return idx
        if strategy == "random":
            rng.shuffle(idx)
            return idx
        if strategy == "spread":  # take turns between documents so all are covered
            by_doc = defaultdict(list)
            for i in idx:
                by_doc[passages[i]["doc_id"]].append(i)
            lists = list(by_doc.values())
            rng.shuffle(lists)
            for lst in lists:
                rng.shuffle(lst)
            out = []
            for r in range(max(len(l) for l in lists)):
                out += [l[r] for l in lists if r < len(l)]
            return out
        raise ValueError("sampling must be spread, random or sequential")

    order = []
    for _ in range(per_passage):
        order += one_round()
    return order


# ---------------------------------------------------------------- text tools
def clean_question(text):
    """Keep just the question if the model added a label or extra lines."""
    lines = [l.strip().strip('"').strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ""
    pick = next((l for l in lines if "?" in l), lines[0])
    return re.sub(r"^(question|query)\s*[:\-]\s*", "", pick, flags=re.I).strip()


def parse_json_objects(text):
    """Find every JSON object in a piece of text (works for JSON lines or a list)."""
    decoder = json.JSONDecoder()
    found, i = [], 0
    while True:
        i = (text or "").find("{", i)
        if i == -1:
            return found
        try:
            obj, end = decoder.raw_decode(text[i:])
            found.append(obj)
            i += end
        except json.JSONDecodeError:
            i += 1


def has_context_reference(question):
    return bool(LEAK_RE.search(question or ""))


def settings_hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def make_rng(*parts):
    return random.Random("|".join(str(p) for p in parts))


# ---------------------------------------------------------------- file store
def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def estimate_calls(target, attempts_factor, calls_per_attempt):
    return target * calls_per_attempt, math.ceil(target * attempts_factor) * calls_per_attempt
