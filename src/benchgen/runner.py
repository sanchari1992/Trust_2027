"""Runs one generator (RAGAS / ARES / DataMorgana) on one RAG's corpus.

Handles: passage building, benchmark size, sampling order, parallel calls,
filters shared by all methods, resuming an interrupted run, and cost reporting.
"""
import importlib
import json
import math
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from src.benchgen.common import (
    append_jsonl,
    build_passages,
    compute_target,
    has_context_reference,
    make_rng,
    read_jsonl,
    sampling_order,
    settings_hash,
)
from src.config import ROOT, corpus_tag
from src.llm import USAGE, get_chat_func
from src.rags.registry import get_rag

GENERATORS = {
    "ragas": ("src.benchgen.ragas_gen", "RagasGenerator"),
    "ares": ("src.benchgen.ares_gen", "AresGenerator"),
    "datamorgana": ("src.benchgen.datamorgana_gen", "DataMorganaGenerator"),
}
METHODS = list(GENERATORS)

# rough prompt sizes (tokens beyond the passage itself), used only by --dry-run
OVERHEAD = {"ragas": 180, "ares": 500, "datamorgana": 700}


def planned_calls_per_attempt(method, bc):
    if method == "ragas":
        return 2
    if method == "ares":
        return 2 + (1 if bc["ares"]["negatives"] else 0)
    return 1 + (1 if bc["datamorgana"]["filter"] else 0)


class Ctx:
    """Everything a generator may need."""

    def __init__(self, cfg, rag_name, passages, chat, out_dir):
        self.cfg, self.bc = cfg, cfg["benchmark"]
        self.rag_name, self.passages, self.chat, self.out_dir = rag_name, passages, chat, out_dir
        self._embedder, self._lock, self._pemb = None, threading.Lock(), None

    def embed(self, texts):
        with self._lock:
            if self._embedder is None:
                from src.embeddings import get_sync_embedder

                self._embedder = get_sync_embedder(self.cfg["embedding"]["model"])
            return self._embedder.encode(texts)

    def passage_embeddings(self):
        if self._pemb is None:
            cache = self.out_dir / "cache"
            cache.mkdir(parents=True, exist_ok=True)
            key = settings_hash(
                [self.cfg["embedding"]["model"], len(self.passages), self.passages[0]["id"],
                 self.passages[-1]["id"], self.passages[0]["words"]]
            )
            path = cache / f"passage_emb_{key}.npy"
            if path.exists():
                self._pemb = np.load(path)
            else:
                print(f"Embedding {len(self.passages):,} passages (saved for next time)...")
                texts = [p["text"] for p in self.passages]
                parts = [self.embed(texts[i : i + 256]) for i in range(0, len(texts), 256)]
                self._pemb = np.concatenate(parts).astype(np.float32)
                np.save(path, self._pemb)
        return self._pemb


def _safe_attempt(gen, passage, rng):
    try:
        return gen.attempt(passage, rng)
    except Exception as e:  # keep going; the error is logged with the rejects
        return {"rejected": "error", "detail": f"{type(e).__name__}: {e}"[:300]}


def run(rag_name, method, cfg, dry_run=False, fresh=False):
    bc = cfg["benchmark"]
    words = bc["passage_words"][method]
    tag = corpus_tag(cfg, rag_name)
    out_dir = ROOT / bc["output_dir"] / f"{rag_name}_{tag}"
    out_file = out_dir / f"{method}.jsonl"
    rej_file = out_dir / f"{method}.rejected.jsonl"
    meta_file = out_dir / f"{method}.meta.json"

    docs = get_rag(rag_name, cfg).load_corpus()
    if not docs:
        if dry_run:  # planning before the corpus is downloaded is normal; do not fail
            print(f"\n=== {rag_name} / {method} ===\nNo corpus on disk yet for '{rag_name}' "
                  f"(corpus folder: {corpus_tag(cfg, rag_name)}). Run: python scripts/prepare_corpora.py {rag_name}; "
                  "the plan can only be shown after that.")
            return None
        raise RuntimeError(f"No corpus for {rag_name}. Run: python scripts/prepare_corpora.py {rag_name}")
    passages = build_passages(docs, words, bc["min_passage_words"])
    if not passages:
        raise RuntimeError("No passages were built. Lower passage_words / min_passage_words.")
    total_words = sum(p["words"] for p in passages)
    per = bc["max_questions_per_passage"]
    target = compute_target(bc["size"], total_words, len(passages) * per)
    max_attempts = math.ceil(target * bc["max_attempts_factor"])
    calls = planned_calls_per_attempt(method, bc)

    print(f"\n=== {rag_name} / {method} ===")
    print(f"Corpus {rag_name}_{tag}: {len(docs):,} document(s)/snippet(s) -> {len(passages):,} passages "
          f"of ~{words} words ({total_words:,} words in total)")
    print(f"Benchmark size: {target} questions "
          f"({target / total_words * 1000:.2f} per 1,000 corpus words; size mode = {bc['size']['mode']})")
    lo, hi = target * calls, max_attempts * calls
    tok = lambda n: int(n * (words * 1.35 + OVERHEAD[method]))
    print(f"LLM calls: about {lo:,} if every attempt works, at most {hi:,}. "
          f"Rough input tokens: {tok(lo // calls):,} to {tok(hi // calls):,}")
    if dry_run:
        print("(dry run: nothing was generated)")
        return None

    # ---- settings guard + resume -------------------------------------------
    current = settings_hash({
        "rag": rag_name, "method": method, "tag": tag, "seed": bc["seed"], "words": words,
        "min_words": bc["min_passage_words"], "sampling": bc["sampling"], "per": per,
        "temperature": bc["temperature"], "method_cfg": bc[method], "model": cfg["llm"]["model"],
        "dedupe": bc["dedupe_threshold"], "leaks": bc["drop_context_references"],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for f in (out_file, rej_file, meta_file):
            f.unlink(missing_ok=True)
    elif meta_file.exists() and json.loads(meta_file.read_text())["settings_hash"] != current:
        raise RuntimeError(
            "Settings changed since the last run of this method. Re-run with --fresh to start over "
            "(this deletes the earlier results), or restore the old settings. "
            "(Changing only the benchmark SIZE is fine and resumes.)"
        )
    accepted, rejected = read_jsonl(out_file), read_jsonl(rej_file)
    if accepted:
        print(f"Resuming: {len(accepted)} accepted and {len(rejected)} rejected attempts already on disk.")

    # ---- build generator ---------------------------------------------------
    chat = partial(get_chat_func(cfg), temperature=bc["temperature"])
    ctx = Ctx(cfg, rag_name, passages, chat, out_dir)
    module, cls = GENERATORS[method]
    gen = getattr(importlib.import_module(module), cls)(ctx)

    order = sampling_order(passages, bc["sampling"], make_rng(bc["seed"], "order", method), per)
    prior = Counter(pid for it in accepted for pid in it["passage_ids"])
    prior.update(r["passage_id"] for r in rejected)
    consumed, pending = Counter(), []
    for pidx in order:
        pid = passages[pidx]["id"]
        if consumed[pid] < prior[pid]:
            consumed[pid] += 1
        else:
            pending.append(pidx)

    thr = bc["dedupe_threshold"]
    q_vecs = [ctx.embed([it["question"]])[0] for it in accepted] if thr and accepted else []

    # ---- generate ----------------------------------------------------------
    t0, usage0 = time.time(), USAGE.snapshot()
    attempts = len(accepted) + len(rejected)
    ptr, errors_in_row = 0, 0
    reasons = Counter(r["reason"] for r in rejected)
    with ThreadPoolExecutor(bc["concurrency"]) as pool:
        while len(accepted) < target and attempts < max_attempts and ptr < len(pending):
            n = min(bc["concurrency"], target - len(accepted), max_attempts - attempts, len(pending) - ptr)
            batch = pending[ptr : ptr + n]
            ptr += n
            futures = [
                pool.submit(_safe_attempt, gen, passages[i],
                            make_rng(bc["seed"], method, passages[i]["id"], attempts + j))
                for j, i in enumerate(batch)
            ]
            for pidx, fut in zip(batch, futures):
                res, p = fut.result(), passages[pidx]
                attempts += 1
                reason, question = res.get("rejected"), res.get("question")
                if reason is None and bc["drop_context_references"] and has_context_reference(question):
                    reason = "context_reference"
                qvec = None
                if reason is None and thr:
                    qvec = ctx.embed([question])[0]
                    if q_vecs and float((np.stack(q_vecs) @ qvec).max()) >= thr:
                        reason = "duplicate"
                if reason is not None:
                    row = {"passage_id": p["id"], "reason": reason,
                           "detail": res.get("detail"), "question": question}
                    append_jsonl(rej_file, row)
                    rejected.append(row)
                    reasons[reason] += 1
                    errors_in_row = errors_in_row + 1 if reason == "error" else 0
                    continue
                errors_in_row = 0
                item = {"id": f"{method}-{rag_name}-{len(accepted) + 1:04d}", "method": method,
                        "rag": rag_name, "corpus": tag, **res}
                append_jsonl(out_file, item)
                accepted.append(item)
                if qvec is not None:
                    q_vecs.append(qvec)
            if errors_in_row >= 3:
                raise RuntimeError(f"{errors_in_row} LLM calls in a row failed. Last error: {rejected[-1]['detail']}")
            print(f"  {len(accepted)}/{target} accepted after {attempts} attempts", flush=True)

    # ---- report ------------------------------------------------------------
    usage1 = USAGE.snapshot()
    meta = {
        "settings_hash": current, "rag": rag_name, "method": method, "corpus": tag,
        "n_passages": len(passages), "corpus_words": total_words, "passage_words": words,
        "target": target, "accepted": len(accepted), "attempts": attempts,
        "rejected_by_reason": dict(reasons),
        "acceptance_rate": round(len(accepted) / attempts, 3) if attempts else None,
        "llm_model": cfg["llm"]["model"], "seed": bc["seed"],
        "this_run": {"seconds": round(time.time() - t0, 1),
                     **{k: usage1[k] - usage0[k] for k in usage1}},
    }
    meta_file.write_text(json.dumps(meta, indent=2))
    print(f"Done: {len(accepted)}/{target} questions -> {out_file}")
    print(f"Rejected: {dict(reasons)}")
    print(f"This run: {meta['this_run']}")
    if len(accepted) < target:
        print("NOTE: fewer questions than the target. Raise max_attempts_factor or lower the filters.")
    return meta
