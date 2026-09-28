"""Generate benchmark questions from a RAG's corpus.

Run from the repo root:
    python scripts/generate_benchmark.py lightrag ragas --dry-run     # plan + cost estimate only
    python scripts/generate_benchmark.py lightrag ragas
    python scripts/generate_benchmark.py all all                      # every RAG x every method

Optional overrides of config.yaml (for quick experiments):
    --n 100            exact number of questions
    --rate 3.0         questions per 1,000 corpus words (ratio mode)
    --words 200        passage size in words
    --sampling random  spread | random | sequential
    --seed 7
    --fresh            throw away earlier results for these settings and start over
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchgen.runner import METHODS, run
from src.config import load_config
from src.rags.registry import RAG_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rag", choices=RAG_NAMES + ["all"])
    ap.add_argument("method", choices=METHODS + ["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--n", type=int)
    ap.add_argument("--rate", type=float)
    ap.add_argument("--words", type=int)
    ap.add_argument("--sampling", choices=["spread", "random", "sequential"])
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    for rag in RAG_NAMES if args.rag == "all" else [args.rag]:
        for method in METHODS if args.method == "all" else [args.method]:
            cfg = copy.deepcopy(load_config())
            bc = cfg["benchmark"]
            if args.n is not None:
                bc["size"].update(mode="count", count=args.n)
            if args.rate is not None:
                bc["size"].update(mode="ratio", questions_per_1k_words=args.rate)
            if args.words is not None:
                bc["passage_words"][method] = args.words
            if args.sampling:
                bc["sampling"] = args.sampling
            if args.seed is not None:
                bc["seed"] = args.seed
            run(rag, method, cfg, dry_run=args.dry_run, fresh=args.fresh)


if __name__ == "__main__":
    main()
