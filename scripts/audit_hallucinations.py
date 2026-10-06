"""Why did mutated questions count as hallucinations? (README section 14)

    python scripts/audit_hallucinations.py            # writes <evaluation.output_dir>/_report/audit_newly_failing.md/.csv
    python scripts/audit_hallucinations.py --show 20  # examples per category in the markdown (default 8)

Run it after scripts/evaluate_benchmarks.py. Read the examples before you report a number: it separates real
failures (the answer adopted a false claim, or invented something) from measurement artifacts and from
failures that a control (a plain re-run, or filler padding) produces just as well.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evalrun.audit import write_audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    res = write_audit(load_config(), a.show)
    if res:
        path, counts = res
        print(counts.to_string(index=False))
        print(f"\nExamples: {path}")


if __name__ == "__main__":
    main()
