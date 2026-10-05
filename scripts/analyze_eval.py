"""Analyse the evaluation results (README section 14).

    python scripts/analyze_eval.py

Reads <evaluation.output_dir>/<rag>_<tag>/<method>/<condition>.jsonl and writes, under
<evaluation.output_dir>/_report/ :
    report.md                     the main tables, readable as they are
    summary_by_condition.csv      hallucination / failure / unfaithful / abstain rates with 95% intervals
    paired_vs_baseline.csv        paired change vs the original benchmark, McNemar test, bootstrap interval
    coverage.csv                  hallucinations found by the benchmark alone vs benchmark + mutations
    by_operator.csv, m3_breakdown.csv, label_distribution.csv
    hallucination_by_condition.png
    results_grid.md / .csv / .tex  one table per metric (hallucination, accuracy): RAGs as rows; columns = the three
                                  methods side by side under Benchmark, +M1, +M2, +M3, +M1,2,3 (the .tex is paste-ready)
    before_after.md               BEFORE vs AFTER mutation on the same questions: all 25 structural metrics,
                                  accuracy and hallucination (+ before_after_summary.csv, before_after_deltas.csv)
Use --no-structural to skip the before/after tables (they run spaCy over every distinct question).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.evalrun.analysis import load_results, run_analysis
from src.evalrun.before_after import run_before_after


def main():
    no_structural = "--no-structural" in sys.argv
    cfg = load_config()
    print(f"profile: {cfg['_profile']}")
    out_dir, summary, paired, coverage = run_analysis(cfg)
    s = summary[summary["level"] == "overall"][["condition", "n", "hallucination_rate", "failure_rate"]]
    print("\nPooled over all RAGs and methods:")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if len(coverage):
        o = coverage[coverage["level"] == "overall"].iloc[0]
        print(f"\nHallucinations found by the benchmark alone: {o['found_by_benchmark']} of {o['n_questions']} questions")
        print(f"... by benchmark + mutations: {o['found_by_benchmark_plus_mutations']} "
              f"({o['extra_found_by_mutations']} more, x{o['fold_increase']:.2f})")
    print(f"\nFull report: {out_dir / 'report.md'}")
    if not no_structural:
        try:
            run_before_after(cfg, load_results(cfg), out_dir)
            print(f"Before/after tables: {out_dir / 'before_after.md'}")
            print(f"Paper-style grid:    {out_dir / 'results_grid.md'}  (and results_grid.tex)")
        except (OSError, ImportError) as e:  # e.g. the spaCy model is not installed
            print(f"Skipped the before/after tables ({e}). Install the spaCy model "
                  f"(python -m spacy download {cfg['real_queries']['spacy_model']}) and re-run.")


if __name__ == "__main__":
    main()
