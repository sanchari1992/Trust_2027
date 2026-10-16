"""Run the whole pipeline in order, one command.

    python scripts/run_all.py --profile large --dry-run         # the plan, with cost estimates; spends nothing
    python scripts/run_all.py --profile large                   # everything, large scale
    python scripts/run_all.py --rags medrag --methods ragas     # a small slice, default settings
    python scripts/run_all.py --from-step mutate                # resume the sequence from a step on
    python scripts/run_all.py --steps mutate,evaluate,analyze   # only these steps
    python scripts/run_all.py --from-step mutate --fresh        # redo mutate + evaluate from scratch

STEPS, in order:
    corpora   download the corpora                      scripts/prepare_corpora.py
    indexes   build each RAG's index (slow, LLM calls)  scripts/build_indexes.py
    real      download real user queries (skipped if present) scripts/prepare_real_queries.py (medical for medrag, general for the others)
    generate  generate benchmark questions              scripts/generate_benchmark.py   (RAGAS, ARES, DataMorgana)
    union     union of the methods' questions (no LLM)  scripts/build_union.py
    db        entity + graph-template databases         scripts/create_bench.py   (real queries, and each RAG's benchmark)
    stats     corpus sizes -> results/corpus_stats.*    scripts/corpus_stats.py
    mutate    create Benchmark+ (mutations 1, 2, 3)     scripts/mutate_benchmark.py
    evaluate  run every condition on the RAGs + judge   scripts/evaluate_benchmarks.py   (baseline, m1, m2, m3 + 2 controls)
    analyze   tables, tests and figure                  scripts/analyze_eval.py
    audit     why mutated questions count as hallucinations scripts/audit_hallucinations.py
    compare   are mutated questions closer to real ones? scripts/compare_mutations.py (needs the real queries of section 11)
    report    THE paper report: real vs methods vs union vs mutations vs hallucinations   scripts/paper_report.py
    examples  example prompts, entities, graph templates and mutations for the paper     scripts/paper_examples.py

Every step is safe to re-run: finished work is skipped and interrupted work resumes.
If a step fails the run stops and prints how to continue (your same flags + --from-step <step>).
--fresh: passed to the mutate and evaluate steps only; they discard their earlier mutated questions /
evaluation results and start over (needed after you change mutation settings). It never deletes the
generated benchmark questions.
--dry-run: generate / mutate / evaluate run in their own dry-run mode (plan and LLM-call estimate,
no LLM calls); the other steps just print what they would run.
--mutate-n N / --eval-n N: questions per file for the mutate step / for the evaluate step (the evaluation samples
from the mutated questions, so mutate at least as many as you evaluate). --limit sets both.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchgen.runner import METHODS  # noqa: E402
from src.config import ROOT as _ROOT, load_config  # noqa: E402
from src.rags.registry import RAG_NAMES  # noqa: E402

STEPS = ["corpora", "indexes", "real", "generate", "union", "db", "stats", "mutate", "evaluate", "analyze", "audit",
         "compare", "report", "examples"]
DRY_RUN_CAPABLE = {"generate", "mutate", "evaluate"}


def resume_hint(step):
    args, out, skip = sys.argv[1:], [], False
    for x in args:
        if skip:
            skip = False
        elif x == "--from-step":
            skip = True
        elif not x.startswith("--from-step="):
            out.append(x)
    return "python scripts/run_all.py " + " ".join(out + ["--from-step", step])


def commands(step, rags, methods, a):
    py, S = sys.executable, (lambda n: str(ROOT / "scripts" / n))
    pairs = [(r, m) for r in rags for m in methods]
    if step == "corpora":
        return [[py, S("prepare_corpora.py"), r] for r in rags]
    if step == "indexes":
        return [[py, S("build_indexes.py"), *rags]]
    if step == "real":
        cfg = load_config(profile=a.profile)
        domains = (["medical"] if "medrag" in rags else []) + (["general"] if set(rags) - {"medrag"} else [])
        todo = []
        for d in domains:
            f = _ROOT / cfg["real_queries"]["output_dir"] / f"{d}.jsonl"
            if f.exists():
                print(f"real queries '{d}' already at {f}: skipped")
            else:
                todo.append([py, S("prepare_real_queries.py"), d])
        return todo
    if step == "union":
        return [[py, S("build_union.py"), "--rags", ",".join(rags), "--methods", ",".join(methods)]]
    if step == "db":
        cfg = load_config(profile=a.profile)
        out = []
        for d in (["medical"] if "medrag" in rags else []) + (["general"] if set(rags) - {"medrag"} else []):
            if (_ROOT / cfg["real_queries"]["output_dir"] / f"{d}.jsonl").exists():
                out.append([py, S("create_bench.py"), "--real", d])
        out += [[py, S("create_bench.py"), "--from-run", r, "--methods", ",".join(methods), "--overwrite"] for r in rags]
        return out
    if step == "generate":
        return [[py, S("generate_benchmark.py"), r, m] + (["--n", str(a.n)] if a.n else []) for r, m in pairs]
    if step == "stats":
        return [[py, S("corpus_stats.py"), "--save"]]
    fresh = ["--fresh"] if a.fresh else []
    if step == "mutate":
        n = a.mutate_n or a.limit
        return [[py, S("mutate_benchmark.py"), r, m] + (["--limit", str(n)] if n else []) + fresh for r, m in pairs]
    if step == "evaluate":
        n = a.eval_n or a.limit
        return [[py, S("evaluate_benchmarks.py"), r, m] + (["--limit", str(n)] if n else []) + fresh for r, m in pairs]
    if step == "analyze":
        return [[py, S("analyze_eval.py")]]
    if step == "audit":
        return [[py, S("audit_hallucinations.py")]]
    if step == "compare":
        return [[py, S("compare_mutations.py")] + (["--limit", str(a.limit)] if a.limit else [])]
    if step == "report":
        return [[py, S("paper_report.py"), "--rags", ",".join(rags), "--methods", ",".join(methods)]]
    if step == "examples":
        return [[py, S("paper_examples.py"), "--rags", ",".join(rags), "--methods", ",".join(methods)]]
    raise ValueError(step)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None, help="config profile, e.g. large")
    ap.add_argument("--rags", default=",".join(RAG_NAMES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--steps", default=None, help="comma-separated subset of steps")
    ap.add_argument("--from-step", default=None, choices=STEPS)
    ap.add_argument("--skip", default="", help="comma-separated steps to leave out")
    ap.add_argument("--n", type=int, default=None, help="questions per (rag, method) for the generate step")
    ap.add_argument("--limit", type=int, default=None, help="questions per file for mutate / evaluate / compare (quick trials)")
    ap.add_argument("--mutate-n", type=int, default=None, help="questions per file in the mutate step (default: --limit)")
    ap.add_argument("--eval-n", type=int, default=None, help="questions per file in the evaluate step (default: --limit)")
    ap.add_argument("--fresh", action="store_true",
                    help="mutate and evaluate steps: discard earlier results and start over (never touches generated questions)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rags, methods = a.rags.split(","), a.methods.split(",")
    for r in rags:
        if r not in RAG_NAMES:
            sys.exit(f"Unknown RAG '{r}'. Choose from {RAG_NAMES}")
    for m in methods:
        if m not in METHODS:
            sys.exit(f"Unknown method '{m}'. Choose from {METHODS}")
    steps = a.steps.split(",") if a.steps else list(STEPS)
    if a.from_step:
        steps = [s for s in steps if STEPS.index(s) >= STEPS.index(a.from_step)]
    steps = [s for s in STEPS if s in steps and s not in a.skip.split(",")]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        sys.exit(f"Unknown step(s) {bad}. Choose from {STEPS}")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")  # Windows: read and write text files as UTF-8 (medical text has symbols such as β)
    if a.profile:
        env["BP_PROFILE"] = a.profile
    print(f"profile: {a.profile or env.get('BP_PROFILE') or 'default'} | rags: {rags} | methods: {methods}")
    print("steps:", " -> ".join(steps))

    t_all = time.time()
    for step in steps:
        cmds = commands(step, rags, methods, a)
        print(f"\n{'=' * 70}\nSTEP: {step}  ({len(cmds)} command(s))\n{'=' * 70}")
        for cmd in cmds:
            shown = " ".join(Path(c).name if c.endswith(".py") else c for c in cmd[1:])
            if a.dry_run:
                if step in DRY_RUN_CAPABLE:
                    cmd = cmd + ["--dry-run"]
                    shown += " --dry-run"
                else:
                    print(f"[dry run, not executed] python {shown}")
                    continue
            print(f"\n$ python {shown}", flush=True)
            code = subprocess.call(cmd, env=env, cwd=ROOT)
            if code != 0:
                print(f"\nSTEP '{step}' FAILED (exit code {code}) on: python {shown}")
                print(f"Fix the problem, then continue with:  {resume_hint(step)}")
                sys.exit(code)
    print(f"\nAll done in {(time.time() - t_all) / 60:.1f} minutes."
          if not a.dry_run else "\nDry run finished. Nothing was spent beyond local checks.")


if __name__ == "__main__":
    main()
