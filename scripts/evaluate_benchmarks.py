"""Run the benchmark and its mutated versions on the RAGs and judge every answer (README section 14).

    python scripts/evaluate_benchmarks.py lightrag ragas --dry-run
    python scripts/evaluate_benchmarks.py lightrag ragas --limit 20      # a quick first look
    python scripts/evaluate_benchmarks.py all all

Options:
    --conditions baseline,m1,m2,m3,m123   which conditions to run (default: evaluation.conditions)
    --limit N     questions per (rag, method), instead of evaluation.sample_per_file
    --fresh       throw away earlier results of these conditions and start over
    --dry-run     print the plan and a rough LLM-call estimate, then stop
Needs: the RAG's index (scripts/build_indexes.py), the benchmark and the mutated files.
An interrupted run resumes at the last saved batch.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchgen.runner import METHODS
from src.config import load_config
from src.evalrun.runner import evaluate
from src.mutate.common import ALL_CONDITIONS as CONDITIONS, DERIVED_CONDITIONS
from src.rags.registry import RAG_NAMES, get_rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rag", choices=RAG_NAMES + ["all"])
    ap.add_argument("method", choices=METHODS + ["all"])
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    print(f"profile: {cfg['_profile']}")
    conds = args.conditions.split(",") if args.conditions else None
    for c in conds or []:
        if c in DERIVED_CONDITIONS:
            sys.exit(f"'{c}' is derived in the analysis from m1, m2 and m3; it is not run on a RAG.")
        if c not in CONDITIONS:
            sys.exit(f"Unknown condition '{c}'. Choose from {CONDITIONS}")
    for rag in RAG_NAMES if args.rag == "all" else [args.rag]:
        rag_obj = None  # one RAG object is reused for all methods of this RAG
        for method in METHODS if args.method == "all" else [args.method]:
            if rag_obj is None and not args.dry_run:
                rag_obj = get_rag(rag, cfg)
            evaluate(rag, method, cfg, rag_obj=rag_obj, conds=conds, limit=args.limit,
                     fresh=args.fresh, dry_run=args.dry_run)
        if rag_obj is not None:
            rag_obj.close()


if __name__ == "__main__":
    main()
