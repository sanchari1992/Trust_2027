"""Create Benchmark+ : mutate the generated benchmark questions (README section 13).

Run from the repo root (add BP_PROFILE=large or use run_all.py --profile large for the big set):
    python scripts/mutate_benchmark.py lightrag ragas --dry-run
    python scripts/mutate_benchmark.py lightrag ragas --limit 20     # try 20 questions first, then read them
    python scripts/mutate_benchmark.py all all

Results: <mutation.output_dir>/mutation{1_length,2_linguistic,3_userinfo,123_combined}/<rag>_<tag>/<method>.jsonl
Options:
    --limit N   only the first N benchmark questions of each file
    --fresh     delete earlier mutated files of this RAG and method and start over
    --dry-run   print the plan and a rough LLM-call estimate, then stop
An interrupted run resumes where it stopped.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchgen.runner import METHODS
from src.config import load_config
from src.mutate.runner import run
from src.rags.registry import RAG_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rag", choices=RAG_NAMES + ["all"])
    ap.add_argument("method", choices=METHODS + ["all"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    print(f"profile: {cfg['_profile']}")
    for rag in RAG_NAMES if args.rag == "all" else [args.rag]:
        for method in METHODS if args.method == "all" else [args.method]:
            run(rag, method, cfg, limit=args.limit, fresh=args.fresh, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
