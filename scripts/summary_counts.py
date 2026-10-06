"""One table of counts and rates per condition (README section 14).

    python scripts/summary_counts.py

Every mutated condition is the ORIGINAL benchmark question with ONLY that mutation applied:
    Benchmark + M1 only | + M2 only | + M3 only     (one mutation each, on the original question)
    Benchmark + M1/M2/M3 combined (any separate run)   DERIVED, nothing extra is run: a question counts as failing
                                                    if it fails in the benchmark or in ANY of the three runs
    Control: same question again | Control: off-topic filler   (not mutations; they show the noise floor)
(With evaluation.combined: run there is also "Benchmark + M1+M2+M3 in one prompt".)

The second table, "Combined count, step by step", is the formula
    combined = benchmark failures + new failures added by M1 + by M2 + by M3
with `sum` the plain addition (a question that newly fails under two mutations is counted twice) and `union` the
same without double counting. union is the number used in the tables above.

Columns:
    total        questions answered and judged
    correct      answers with the expected behaviour (for M3: rejecting a false claim or flagging an unsupported
                 detail counts as correct when that is what the question required)
    wrong        not correct and did not decline (incorrect / followed a false claim / accepted a detail)
    abstained    declined to answer (a failure for an answerable question, never a hallucination)
    accuracy     correct / total
    hallucinated unfaithful or wrong-behaviour answers (README section 14); hallucination_rate = hallucinated / total
    wrong_behavior, unsupported_but_correct   the two parts of the hallucination count
    retrieval_hit_rate   share of answers whose retrieved passages include the original source passage
  Paired against the unchanged benchmark, question by question (only for the mutated and control conditions):
    newly_failed   correct in the benchmark, not correct here      newly_fixed   the reverse
    failed_in_both not correct in the benchmark AND not correct here (the same hard questions fail every time)
    net_failed     newly_failed - newly_fixed
Writes <evaluation.output_dir>/_report/summary_counts.md and .csv: pooled, and per RAG and method.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.evalrun.analysis import df_to_md, load_results
from src.mutate.common import CONDITION_LABELS

KEYS = ["rag", "method", "id"]


def paired_counts(df):
    """{condition: (newly_failed, newly_fixed, failed_in_both)} versus the baseline, over questions in both."""
    wide = df.pivot_table(index=KEYS, columns="condition", values="failed", aggfunc="first", observed=True)
    out = {}
    if "baseline" not in wide:
        return out
    for c in [c for c in wide.columns if c != "baseline"]:
        sub = wide[["baseline", c]].dropna()
        b, x = sub["baseline"].to_numpy(), sub[c].to_numpy()
        out[str(c)] = (int(((b == 0) & (x == 1)).sum()), int(((b == 1) & (x == 0)).sum()),
                       int(((b == 1) & (x == 1)).sum()))
    return out


def table(df, by):
    rows = []
    for key, d in (df.groupby(by) if by else [((), df)]):
        key = key if isinstance(key, tuple) else (key,)
        pc = paired_counts(d)
        for cond, g in d.groupby("condition", observed=True):
            cond = str(cond)
            total, correct = len(g), int((g["failed"] == 0).sum())
            ab, failed = int(g["abstained"].sum()), int(g["failed"].sum())
            nf, nx, both = pc.get(cond, (np.nan, np.nan, np.nan))
            rows.append({**dict(zip(by, key)), "condition": CONDITION_LABELS.get(cond, cond), "_c": cond,
                         "total": total, "correct": correct, "wrong": failed - ab, "abstained": ab,
                         "accuracy": correct / total, "hallucinated": int(g["hallucinated"].sum()),
                         "hallucination_rate": g["hallucinated"].mean(),
                         "wrong_behavior": int(g["wrong_behavior"].fillna(0).sum()),
                         "unsupported_but_correct": int(g["unsupported_but_correct"].fillna(0).sum()),
                         "retrieval_hit_rate": g["retrieval_hit"].mean(),
                         "newly_failed": nf, "newly_fixed": nx, "failed_in_both": both,
                         "net_failed": nf - nx})
    t = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(CONDITION_LABELS)}
    t["_o"] = t["_c"].map(order)
    return t.sort_values(by + ["_o"]).drop(columns=["_c", "_o"]).reset_index(drop=True)


METRICS = {"failed": "not correct (wrong + declined)", "hallucinated": "hallucinated"}


def combined_table(df, by):
    """benchmark count + new failures of M1 + M2 + M3 = sum; union = without double counting."""
    parts = ["m1", "m2", "m3"]
    rows = []
    for key, d in (df.groupby(by) if by else [((), df)]):
        key = key if isinstance(key, tuple) else (key,)
        for metric, name in METRICS.items():
            wide = d.pivot_table(index=KEYS, columns="condition", values=metric, aggfunc="first", observed=True)
            if "baseline" not in wide or not all(p in wide for p in parts):
                continue
            w = wide[["baseline"] + parts].dropna()
            b = w["baseline"].to_numpy()
            new = {p: int(((b == 0) & (w[p].to_numpy() == 1)).sum()) for p in parts}
            union = int(np.maximum(b, w[parts].max(axis=1).to_numpy()).sum())
            total = int(b.sum()) + sum(new.values())
            rows.append({**dict(zip(by, key)), "metric": name, "questions": len(w), "benchmark": int(b.sum()),
                         "new_by_M1": new["m1"], "new_by_M2": new["m2"], "new_by_M3": new["m3"],
                         "sum": total, "union": union, "double_counted": total - union,
                         "sum_rate": total / len(w), "union_rate": union / len(w)})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    df = load_results(cfg)
    pooled = table(df, [])
    per = table(df, ["rag", "method"])
    out = ROOT / cfg["evaluation"]["output_dir"] / "_report"
    out.mkdir(parents=True, exist_ok=True)
    comb_pooled, comb_per = combined_table(df, []), combined_table(df, ["rag", "method"])
    pooled.to_csv(out / "summary_counts_pooled.csv", index=False)
    per.to_csv(out / "summary_counts.csv", index=False)
    comb_per.to_csv(out / "combined_count.csv", index=False)
    md = ["# Counts and rates per condition", "",
          "Each mutated condition is the original benchmark question with ONLY that mutation applied; "
          "`Benchmark + M1+M2+M3 (all three)` is the only combined one. `wrong` = not correct and not declined; "
          "`abstained` = declined; wrong + abstained = all failures. `accuracy` = correct / total. "
          "`newly_failed` / `newly_fixed` / `failed_in_both` compare each question with its own unchanged version; "
          "`failed_in_both` are the questions that fail whatever you do to them.", "",
          "## All RAGs and methods pooled", "", df_to_md(pooled), "", "## Per RAG and method", "", df_to_md(per), "",
          "## Combined count, step by step", "",
          "`sum` = benchmark + new failures added by M1 + M2 + M3 (a question that newly fails under two mutations "
          "is counted twice); `union` = the same without double counting (never above the number of questions); "
          "`double_counted` = sum - union. This estimates the failures of a benchmark that uses all three mutations "
          "as separate variants; it does not measure ONE prompt that contains all three mutations "
          "(`evaluation.combined: run`).", "", df_to_md(comb_pooled), "", df_to_md(comb_per), ""]
    (out / "summary_counts.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nCombined count, step by step (benchmark + new failures of M1 + M2 + M3):")
    print(comb_pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten: {out / 'summary_counts.md'}")


if __name__ == "__main__":
    main()
