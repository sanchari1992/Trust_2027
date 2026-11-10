"""Premise Injection by TYPE and by LENGTH of the added statement (paper Section 5.2, Tables prem_type / prem_len).

    python scripts/premise_length.py                     # uses BP_PROFILE or the default profile
    BP_PROFILE=paper python scripts/premise_length.py

Reads the stored evaluation results (same files as scripts/analyze_eval.py) and, for every question that has both the
unchanged version (condition `baseline`) and the Premise Injection version (condition `m3`), computes

  added_words = words(m3 prompt) - words(baseline prompt)         length of the injected statement
  type        = the m3 operator (true_premise, ..., patient_conflict)

and prints, ready for the paper:
  1. how long the injections are (median, quartiles, per type) next to the length of the original questions;
  2. hallucination, refusal and failure rates of the benchmark and of Premise Injection on the SAME questions, by
     length tercile of the added statement (pooled), and by tercile within each type that has enough prompts;
  3. a LaTeX body for Table prem_len.
Intervals are paired bootstrap (2,000 resamples, seed 42); p is the exact McNemar test (uncorrected).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_config
from src.evalrun.analysis import load_results, mcnemar_exact, paired_bootstrap_diff


def words(t):
    return len(str(t).split())


def op_name(ops):
    for x in ops or []:
        x = str(x)
        if x.startswith("m3:"):
            return x[3:]
    return None


def paired(g):
    """Rates of benchmark and Premise Injection on the same questions, with a paired interval and McNemar p."""
    x, y = g["hall_b"].to_numpy(float), g["hall_m"].to_numpy(float)
    rng = np.random.default_rng(42)
    lo, hi = paired_bootstrap_diff(x, y, 2000, rng)
    b = int(((x == 0) & (y == 1)).sum())  # fine before, hallucinating after
    c = int(((x == 1) & (y == 0)).sum())
    return {"n": len(g), "hall_bench": 100 * x.mean(), "hall_prem": 100 * y.mean(),
            "diff": 100 * (y.mean() - x.mean()), "lo": 100 * lo, "hi": 100 * hi, "p": mcnemar_exact(b, c),
            "refused_prem": 100 * g["ref_m"].mean(), "failed_prem": 100 * g["fail_m"].mean(),
            "failed_bench": 100 * g["fail_b"].mean()}


def main():
    cfg = load_config()
    print(f"profile: {cfg['_profile']}")
    df = load_results(cfg)
    df["condition"] = df["condition"].astype(str)
    key = ["rag", "method", "id"]
    b = df[df["condition"] == "baseline"].drop_duplicates(key).set_index(key)
    m = df[df["condition"] == "m3"].drop_duplicates(key).set_index(key)
    j = b[["question", "hallucinated", "refused", "failed"]].join(
        m[["question", "hallucinated", "refused", "failed", "operators"]], lsuffix="_b", rsuffix="_m", how="inner")
    j = j.rename(columns={"hallucinated_b": "hall_b", "hallucinated_m": "hall_m", "refused_b": "ref_b",
                          "refused_m": "ref_m", "failed_b": "fail_b", "failed_m": "fail_m"})
    j["type"] = j["operators"].map(op_name)
    j["q_words"] = j["question_b"].map(words)
    j["added_words"] = j["question_m"].map(words) - j["q_words"]
    j = j.dropna(subset=["type"])
    print(f"\n{len(j)} questions with both versions\n")

    print("== 1. How long are the injections? (words added to the question) ==")
    q = j["added_words"].quantile([0.25, 0.5, 0.75])
    print(f"all types: median {q[0.5]:.0f}, IQR {q[0.25]:.0f} to {q[0.75]:.0f}, mean {j['added_words'].mean():.1f}; "
          f"original question: median {j['q_words'].median():.0f}, mean {j['q_words'].mean():.1f}")
    print(j.groupby("type")["added_words"].agg(n="size", median="median", mean="mean", p25=lambda s: s.quantile(.25),
                                              p75=lambda s: s.quantile(.75)).round(1).to_string())

    print("\n== 2a. By type (this is Table prem_type) ==")
    rows = []
    for t, g in j.groupby("type"):
        r = paired(g); r["type"] = t; r["median_added_words"] = g["added_words"].median(); rows.append(r)
    print(pd.DataFrame(rows).set_index("type").round(3).to_string())

    print("\n== 2b. By length tercile of the added statement, all types pooled ==")
    j["tercile"] = pd.qcut(j["added_words"], 3, labels=["short", "medium", "long"], duplicates="drop")
    edges = j.groupby("tercile", observed=True)["added_words"].agg(["min", "max"])
    rows = []
    for t, g in j.groupby("tercile", observed=True):
        r = paired(g); r["tercile"] = t; r["words"] = f"{edges.loc[t, 'min']:.0f}-{edges.loc[t, 'max']:.0f}"; rows.append(r)
    T = pd.DataFrame(rows).set_index("tercile")
    print(T.round(3).to_string())

    print("\n== 2c. Same, within each type with at least 60 prompts (length is linked to type) ==")
    for t, g in j.groupby("type"):
        if len(g) < 60 or g["added_words"].nunique() < 3:
            continue
        g = g.copy(); g["tercile"] = pd.qcut(g["added_words"], 3, labels=["short", "medium", "long"], duplicates="drop")
        for tc, gg in g.groupby("tercile", observed=True):
            r = paired(gg)
            print(f"{t:24s} {tc:7s} n={r['n']:4d} hall {r['hall_bench']:5.1f} -> {r['hall_prem']:5.1f} "
                  f"(diff {r['diff']:+.1f} [{r['lo']:+.1f},{r['hi']:+.1f}], p={r['p']:.3f})")

    print("\n== 3. LaTeX body for Table prem_len ==")
    for t, r in T.iterrows():
        print(f"{t} & {r['words']} & {int(r['n'])} & {r['hall_bench']:.1f} $\\to$ {r['hall_prem']:.1f} & {r['refused_prem']:.1f} \\\\")


if __name__ == "__main__":
    main()
