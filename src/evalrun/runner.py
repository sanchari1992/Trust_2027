"""Runs a benchmark and its mutated versions on a RAG, and judges every answer.

For one (RAG, method) the SAME sampled questions are used in every condition:
    baseline  the original benchmark question
    m1        + length mutation
    m2        + linguistic mutation
    m3        + user-information mutation
    m123      + all three
Only questions that survived every mutation's check are eligible, so conditions are
compared item by item. Results: <evaluation.output_dir>/<rag>_<tag>/<method>/<condition>.jsonl
"""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

from src.benchgen.common import append_jsonl, read_jsonl
from src.config import ROOT, corpus_tag, index_dir
from src.evalrun.judge import Judge
from src.llm import USAGE
from src.evalrun.controls import control_rows, retrieval_scores
from src.mutate.common import CONDITION_LABELS, CONTROL_CONDITIONS, active_conditions, condition_file
from src.rags.registry import get_rag

CALLS_PER_ANSWER = 5  # rough upper bound: 1 for the RAG, ~4 for the judge (label, statements, 1-2 verdict checks); cache hits and refusals cost less


def eval_file(cfg, cond, rag, method):
    return (ROOT / cfg["evaluation"]["output_dir"] / f"{rag}_{corpus_tag(cfg, rag)}"
            / method / f"{cond}.jsonl")


def index_ready(cfg, rag):
    d = index_dir(cfg, rag)
    return d.exists() and any(d.iterdir())


def select_questions(cfg, rag, method, conds, limit=None):
    """-> (ids, {cond: {id: row}}, usable_conditions). Only ids valid in EVERY usable condition."""
    ev = cfg["evaluation"]
    rows = {}
    for c in [c for c in conds if c not in CONTROL_CONDITIONS]:
        f = condition_file(cfg, c, rag, method)
        rows[c] = {r["id"]: r for r in read_jsonl(f)} if f.exists() else {}
    if not rows.get("baseline"):
        return None, None, None
    for c in [c for c in conds if c in CONTROL_CONDITIONS]:
        if c == "c_pad" and "m1" not in rows:  # the padding control is sized from the M1 version
            rows["m1"] = {r["id"]: r for r in read_jsonl(condition_file(cfg, "m1", rag, method))} \
                if condition_file(cfg, "m1", rag, method).exists() else {}
        rows[c] = control_rows(c, rows, cfg["evaluation"]["seed"])
    usable = [c for c in conds if rows.get(c)]
    for c in conds:
        if c not in usable:
            print(f"  note: no questions for condition '{c}' ({rag}/{method}); skipping it.")
    common = set.intersection(*[set(rows[c]) for c in usable])
    ids = sorted(common)
    random.Random(ev["seed"]).shuffle(ids)
    n = limit or ev["sample_per_file"]
    if n:
        ids = ids[:n]
    return ids, {c: rows[c] for c in usable}, usable


def evaluate(rag_name, method, cfg, rag_obj=None, conds=None, limit=None, fresh=False, dry_run=False):
    ev = cfg["evaluation"]
    conds = conds or active_conditions(cfg)
    ids, rows, usable = select_questions(cfg, rag_name, method, conds, limit)
    print(f"\n=== evaluate {rag_name} / {method} ===")
    if ids is None:
        print("  no baseline benchmark file found (run generate_benchmark.py first). Skipping.")
        return None
    n_total = len(ids) * len(usable)
    print(f"{len(ids):,} questions x {len(usable)} conditions {usable} = {n_total:,} answers to produce and judge")
    print(f"LLM calls: roughly {n_total * CALLS_PER_ANSWER:,}")
    if dry_run:
        print("(dry run: nothing was run)")
        return None
    if not index_ready(cfg, rag_name):
        raise RuntimeError(f"No index for {rag_name} at {index_dir(cfg, rag_name)}. "
                           f"Run: python scripts/build_indexes.py {rag_name}")

    rag = rag_obj or get_rag(rag_name, cfg)
    judge = Judge(cfg)
    summary = {}
    for cond in usable:
        out = eval_file(cfg, cond, rag_name, method)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            out.unlink(missing_ok=True)
        existing = read_jsonl(out)
        current = rows[cond]
        keep = [r for r in existing if r["id"] in current
                and r.get("question") == current[r["id"]]["question"]
                and r.get("reference_answer") == current[r["id"]]["answer"]]
        if len(keep) < len(existing):
            print(f"  dropped {len(existing) - len(keep)} stored result(s) whose question or reference "
                  f"answer has changed since they were evaluated; they will be redone")
            tmp = out.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
            tmp.replace(out)
        done = {r["id"] for r in keep}
        todo = [i for i in ids if i not in done]
        print(f"[{CONDITION_LABELS[cond]}] {len(done)} done, {len(todo)} to do")
        t0 = time.time()
        for start in range(0, len(todo), ev["batch_size"]):
            batch = [rows[cond][i] for i in todo[start : start + ev["batch_size"]]]
            results = rag.answer_batch([r["question"] for r in batch], ev["rag_concurrency"][rag_name])
            if all("error" in r for r in results):
                raise RuntimeError(f"Every question in a batch failed. First error: {results[0]['error']}")

            def judge_one(pair):
                row, res = pair
                if "error" in res:
                    return {"rag_error": res["error"]}
                try:
                    return judge.evaluate(row, res)
                except Exception as e:
                    return {"judge_error": True, "judge_exception": f"{type(e).__name__}: {e}"[:200]}

            with ThreadPoolExecutor(ev["judge_concurrency"]) as pool:
                judged = list(pool.map(judge_one, zip(batch, results)))
            for row, res, j in zip(batch, results, judged):
                rec = {
                    "id": row["id"], "rag": rag_name, "method": method, "condition": cond,
                    "question": row["question"], "reference_answer": row["answer"],
                    "expected_behavior": row.get("expected_behavior") or "answer",
                    "operators": row.get("operators") or [], "kind": row.get("kind"),
                    "answer_changed": row.get("answer_changed", False),
                    "response": res.get("answer"),
                    "n_contexts": len(res.get("contexts") or []),
                    "n_source_contexts": len(res.get("source_contexts") or []),
                    **dict(zip(("gold_recall", "retrieval_hit"), retrieval_scores(
                        row.get("contexts"), res.get("source_contexts") or res.get("contexts")))),
                    **j,
                }
                if ev["store_contexts"]:
                    rec["contexts"] = res.get("contexts")
                append_jsonl(out, rec)
            print(f"  {min(start + ev['batch_size'], len(todo))}/{len(todo)} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rs = read_jsonl(out)
        ok = [r for r in rs if r.get("hallucinated") is not None]
        summary[cond] = {"n": len(rs), "judged": len(ok),
                         "hallucination_rate": round(sum(r["hallucinated"] for r in ok) / len(ok), 3) if ok else None,
                         "errors": len(rs) - len(ok)}
        print(f"  -> {summary[cond]}")
    u = USAGE.snapshot()
    print(f"API calls in this run: {u['calls']:,} (plus {u['cache_hits']:,} judge calls answered from the cache); "
          f"tokens in/out: {u['prompt_tokens']:,}/{u['completion_tokens']:,}")
    if rag_obj is None:
        rag.close()
    return summary
