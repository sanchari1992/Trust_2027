"""Applies the three mutations (and their combination) to every question of one
(RAG, method) benchmark file, and stores each result in its own area:

  <mutation.output_dir>/mutation1_length/<rag>_<tag>/<method>.jsonl
  <mutation.output_dir>/mutation2_linguistic/...
  <mutation.output_dir>/mutation3_userinfo/...
  <mutation.output_dir>/mutation123_combined/...      (M1 + M3 text, then M2 on the whole prompt)
  ...and <method>.rejected.jsonl next to each, listing items whose mutation failed its check.

Every row keeps the SAME id as the benchmark question, so conditions can be compared item by item.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.benchgen.common import append_jsonl, make_rng, read_jsonl, settings_hash
from src.config import ROOT, corpus_tag
from src.llm import USAGE, get_chat_func
from src.mutate.common import MUTATION_DIRS, MutCtx, benchmark_file, mutation_file
from src.mutate.length import mutate_length
from src.mutate.linguistic import mutate_linguistic
from src.mutate.prompt import Prompt, merge
from src.mutate.userinfo import mutate_userinfo

CALLS_PER_ITEM = 5  # rough: ~1.5 length generations + 1 check, 1 user-info generation + 1-2 checks


def make_row(item, cond, prompt, operators, answer, behavior, changed, info, kind=None, claim=None):
    q = prompt.render()
    return {
        "id": item["id"], "source_id": item["id"], "method": item.get("method"),
        "rag": item.get("rag"), "corpus": item.get("corpus"),
        "mutation": cond, "operators": operators, "kind": kind, "claim": claim,
        "original_question": item["question"], "question": q,
        "original_answer": item["answer"], "answer": answer,
        "answer_changed": changed, "expected_behavior": behavior,
        "passage_ids": item.get("passage_ids"), "contexts": item.get("contexts"),
        "words_original": len(item["question"].split()), "words_mutated": len(q.split()),
        "meta": info,
    }


def mutate_item(item, ctx, cfg):
    """Mutate one question. Returns {"rows": {cond: row}, "rejected": {cond: reason}}."""
    mc, seed = cfg["mutation"], cfg["mutation"]["seed"]
    rng = lambda tag: make_rng(seed, item["id"], tag)  # noqa: E731
    base = Prompt(core=item["question"])
    rows, rejected = {}, {}

    r1 = mutate_length(item, ctx, mc["length"], rng("m1"))
    r2 = mutate_linguistic(base, ctx, mc["linguistic"], rng("m2"))
    r3 = mutate_userinfo(item, ctx, mc["userinfo"], rng("m3"))

    if r1["ok"]:
        rows["m1"] = make_row(item, "m1", r1["prompt"], [f"m1:{o}" for o in r1["operators"]],
                              item["answer"], "answer", False, r1["info"])
    else:
        rejected["m1"] = r1["reason"]
    if r2["ok"]:
        rows["m2"] = make_row(item, "m2", r2["prompt"], [f"m2:{o}" for o in r2["operators"]],
                              item["answer"], "answer", False, r2["info"])
    else:
        rejected["m2"] = r2["reason"]
    p3 = None
    if r3["ok"]:
        p3 = Prompt(pre=r3["pre"], core=item["question"], post=r3["post"])
        rows["m3"] = make_row(item, "m3", p3, [f"m3:{r3['operator']}"], r3["answer"], r3["behavior"],
                              r3["answer_changed"], r3["info"], kind=r3["kind"], claim=r3["claim"])
    else:
        rejected["m3"] = r3["reason"]

    if r1["ok"] and r3["ok"]:
        merged = merge(r1["prompt"], p3)
        r123 = mutate_linguistic(merged, ctx, mc["linguistic"], rng("m123"),
                                 fixed_ops=r2["operators"] if r2["ok"] else None)
        if r123["ok"]:
            ops = ([f"m1:{o}" for o in r1["operators"]] + [f"m2:{o}" for o in r123["operators"]]
                   + [f"m3:{r3['operator']}"])
            rows["m123"] = make_row(item, "m123", r123["prompt"], ops, r3["answer"], r3["behavior"],
                                    r3["answer_changed"], {"m1": r1["info"], "m2": r123["info"], "m3": r3["info"]},
                                    kind=r3["kind"], claim=r3["claim"])
        else:
            rejected["m123"] = f"m2_on_combined_failed({r123['reason']})"
    else:
        parts = []
        if not r1["ok"]:
            parts.append(f"m1:{r1['reason']}")
        if not r3["ok"]:
            parts.append(f"m3:{r3['reason']}")
        rejected["m123"] = "component_failed(" + "; ".join(parts) + ")"
    return {"rows": rows, "rejected": rejected}


def _safe(item, ctx, cfg):
    try:
        return mutate_item(item, ctx, cfg)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def run(rag, method, cfg, limit=None, fresh=False, dry_run=False):
    mc = cfg["mutation"]
    tag = corpus_tag(cfg, rag)
    src = benchmark_file(cfg, rag, method)
    if not src.exists():
        print(f"{rag}/{method}: no benchmark file at {src} (run generate_benchmark.py first). Skipping.")
        return None
    items = read_jsonl(src)
    cap = limit or mc.get("max_items_per_file")
    if cap:
        items = items[:cap]

    meta_dir = ROOT / mc["output_dir"] / "_meta" / f"{rag}_{tag}"
    progress_f, meta_f = meta_dir / f"{method}.progress.jsonl", meta_dir / f"{method}.meta.json"

    print(f"\n=== mutate {rag} / {method} ===")
    print(f"{len(items):,} benchmark questions from {src}")
    print(f"LLM calls: about {CALLS_PER_ITEM} per question, so roughly {len(items) * CALLS_PER_ITEM:,} in total "
          "(mutation 2 uses none).")
    if dry_run:
        print("(dry run: nothing was generated)")
        return None

    current = settings_hash({
        "seed": mc["seed"], "temperature": mc["temperature"], "max_retries": mc["max_retries"],
        "length": mc["length"], "linguistic": mc["linguistic"], "userinfo": mc["userinfo"],
        "model": cfg["llm"]["model"],
    })
    meta_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for c in MUTATION_DIRS:
            for f in (mutation_file(cfg, c, rag, method), mutation_file(cfg, c, rag, method).with_suffix(".rejected.jsonl")):
                f.unlink(missing_ok=True)
        progress_f.unlink(missing_ok=True)
        meta_f.unlink(missing_ok=True)
    elif meta_f.exists() and json.loads(meta_f.read_text())["settings_hash"] != current:
        raise RuntimeError(
            "Mutation settings changed since the last run for this RAG and method. Use --fresh to "
            "start over (this deletes the earlier mutated files), or restore the old settings."
        )
    for c in MUTATION_DIRS:
        mutation_file(cfg, c, rag, method).parent.mkdir(parents=True, exist_ok=True)

    done = {r["id"] for r in read_jsonl(progress_f)}
    todo = [it for it in items if it["id"] not in done]
    if done:
        print(f"Resuming: {len(done)} questions already processed, {len(todo)} to go.")

    chat = partial(get_chat_func(cfg), temperature=mc["temperature"])
    ctx = MutCtx(cfg, chat)
    t0, u0 = time.time(), USAGE.snapshot()
    errors_in_row, processed = 0, 0
    with ThreadPoolExecutor(mc["concurrency"]) as pool:
        for start in range(0, len(todo), mc["concurrency"] * 4):
            chunk = todo[start : start + mc["concurrency"] * 4]
            for item, res in zip(chunk, pool.map(lambda it: _safe(it, ctx, cfg), chunk)):
                if "error" in res:
                    errors_in_row += 1
                    print(f"  error on {item['id']}: {res['error']}")
                    if errors_in_row >= 5:
                        raise RuntimeError(f"5 questions in a row failed. Last error: {res['error']}")
                    continue
                errors_in_row = 0
                for cond, row in res["rows"].items():
                    append_jsonl(mutation_file(cfg, cond, rag, method), row)
                for cond, reason in res["rejected"].items():
                    append_jsonl(mutation_file(cfg, cond, rag, method).with_suffix(".rejected.jsonl"),
                                 {"id": item["id"], "mutation": cond, "reason": reason})
                append_jsonl(progress_f, {"id": item["id"]})
                processed += 1
            print(f"  {processed}/{len(todo)} questions mutated", flush=True)

    u1 = USAGE.snapshot()
    counts = {}
    for c in MUTATION_DIRS:
        f = mutation_file(cfg, c, rag, method)
        kept, rej = read_jsonl(f), read_jsonl(f.with_suffix(".rejected.jsonl"))
        reasons = {}
        for r in rej:
            key = r["reason"].split("(")[0].split(":")[-1][:60]
            reasons[key] = reasons.get(key, 0) + 1
        counts[c] = {"kept": len(kept), "rejected": len(rej), "rejected_reasons": reasons}
    meta = {
        "settings_hash": current, "rag": rag, "method": method, "corpus": tag,
        "n_source_questions": len(items), "counts": counts, "llm_model": cfg["llm"]["model"],
        "this_run": {"seconds": round(time.time() - t0, 1), **{k: u1[k] - u0[k] for k in u1}},
    }
    meta_f.write_text(json.dumps(meta, indent=2))
    for c, v in counts.items():
        print(f"  {c}: kept {v['kept']}, rejected {v['rejected']} {v['rejected_reasons'] or ''}")
    print(f"This run: {meta['this_run']}")
    return meta
