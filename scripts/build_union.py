"""Build the UNION of the benchmark methods for each RAG (README section 31).

    python scripts/build_union.py                         # every RAG that has benchmark files
    python scripts/build_union.py --rags medrag --methods ragas,ares,datamorgana
    python scripts/build_union.py --threshold 0.85        # stricter duplicate removal

Takes the questions of RAGAS, ARES and DataMorgana in turn and drops any question whose meaning (embedding cosine
similarity, local model) is at least `union.dedupe_threshold` (default 0.9) similar to one already kept.
Writes <benchmark.output_dir>/<rag>_<tag>/union.jsonl and union.meta.json (how many each method contributed).
No LLM calls. The mutated versions and the evaluation results of a union question are those of the method it came
from, so the union is NOT mutated or evaluated again: scripts/paper_report.py selects the union's questions by id.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchdb.union import build_union
from src.benchgen.runner import METHODS
from src.config import load_config
from src.rags.registry import RAG_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rags", default=",".join(RAG_NAMES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config()
    print(f"profile: {cfg['_profile']}")
    for rag in args.rags.split(","):
        try:
            m = build_union(cfg, rag, args.methods.split(","), threshold=args.threshold)
        except FileNotFoundError as e:
            print(f"{rag}: {e}")
            continue
        print(f"{rag}: {m['total_input']:,} questions from {m['input']} -> union of {m['kept']:,} "
              f"({m['dropped_as_duplicates']:,} near-duplicates dropped, kept by method {m['kept_by_method']})")


if __name__ == "__main__":
    main()
