"""Which questions change status between the benchmark and a mutated version? (README section 14.8)

    python scripts/transition_report.py

For every mutation (and the two controls) and for three outcomes, each question falls in exactly one of four cells:

    fine -> fine       fine in the benchmark, fine in the mutated version
    fine -> FAIL       fine in the benchmark, failing in the mutated version     (newly failing: the mutation exposed it)
    FAIL -> fine       failing in the benchmark, fine in the mutated version     (fixed by the mutation)
    FAIL -> FAIL       failing in the benchmark AND in the mutated version       (fails whatever you do)

Outcomes:  not_correct (wrong + declined) | hallucinated (main metric) | wrong_behavior (asserted something
wrong / adopted a false claim).  `p_mcnemar` compares fine->FAIL with FAIL->fine (exact test); `q_value` is
Benjamini-Hochberg corrected within each outcome.

A second table groups the three mutations: "any" = at least one of M1, M2, M3, "all" = all three.

Writes <evaluation.output_dir>/_report/transitions.md, transitions.csv (pooled and per RAG x method),
transitions_any_of_three.csv, and transitions_questions.csv (one row per question with its status in every
condition, so you can open the questions in each cell).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.analysis.compare import _benjamini_hochberg
from src.config import ROOT, load_config
from src.evalrun.analysis import df_to_md, load_results, mcnemar_exact
from src.mutate.common import CONDITION_LABELS

KEYS = ["rag", "method", "id"]
OUTCOMES = {"failed": "not_correct", "hallucinated": "hallucinated", "wrong_behavior": "wrong_behavior"}
MUTS = ["m1", "m2", "m3", "c_rerun", "c_pad"]
GROUPS = ["m1", "m2", "m3"]


def wide(df, col):
    return df.pivot_table(index=KEYS, columns="condition", values=col, aggfunc="first", observed=True)


def transitions(df, by):
    rows = []
    for col, name in OUTCOMES.items():
        w = wide(df, col)
        if "baseline" not in w:
            continue
        for key, g in (w.groupby(level=by) if by else [((), w)]):
            key = key if isinstance(key, tuple) else (key,)
            for c in [c for c in MUTS if c in g.columns]:
                sub = g[["baseline", c]].dropna()
                if sub.empty:
                    continue
                b, x = sub["baseline"].to_numpy(), sub[c].to_numpy()
                nf, fx = int(((b == 0) & (x == 1)).sum()), int(((b == 1) & (x == 0)).sum())
                rows.append({**dict(zip(by, key)), "outcome": name, "condition": CONDITION_LABELS[c], "_c": c,
                             "questions": len(sub), "fine_to_fine": int(((b == 0) & (x == 0)).sum()),
                             "fine_to_FAIL": nf, "FAIL_to_fine": fx, "FAIL_to_FAIL": int(((b == 1) & (x == 1)).sum()),
                             "benchmark_fails": int(b.sum()), "condition_fails": int(x.sum()),
                             "net_change": nf - fx, "p_mcnemar": mcnemar_exact(nf, fx)})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["q_value"] = np.nan
    for o in t["outcome"].unique():
        m = t["outcome"] == o
        t.loc[m, "q_value"] = _benjamini_hochberg(t.loc[m, "p_mcnemar"].to_numpy())
    return t.drop(columns="_c")


def any_of_three(df, by):
    rows = []
    for col, name in OUTCOMES.items():
        w = wide(df, col)
        if "baseline" not in w or not all(c in w for c in GROUPS):
            continue
        for key, g in (w.groupby(level=by) if by else [((), w)]):
            key = key if isinstance(key, tuple) else (key,)
            g = g[["baseline"] + GROUPS].dropna()
            if g.empty:
                continue
            b = g["baseline"].to_numpy()
            n_fail = g[GROUPS].sum(axis=1).to_numpy()  # in how many of M1, M2, M3 the question fails
            r = lambda m: int(m.sum())  # noqa: E731
            rows.append({**dict(zip(by, key)), "outcome": name, "questions": len(g), "benchmark_fails": r(b == 1),
                         "fine_in_benchmark__fails_in_at_least_one": r((b == 0) & (n_fail >= 1)),
                         "fine_in_benchmark__fails_in_all_three": r((b == 0) & (n_fail == 3)),
                         "fails_in_benchmark__fine_in_at_least_one": r((b == 1) & (n_fail <= 2)),
                         "fails_in_benchmark__fine_in_all_three": r((b == 1) & (n_fail == 0)),
                         "fails_in_benchmark__fails_in_at_least_one": r((b == 1) & (n_fail >= 1)),
                         "fails_in_benchmark__fails_in_all_three": r((b == 1) & (n_fail == 3)),
                         "fine_everywhere": r((b == 0) & (n_fail == 0)),
                         "fails_somewhere": r((b == 1) | (n_fail >= 1))})
    return pd.DataFrame(rows)


def question_level(df):
    parts = []
    for col, name in OUTCOMES.items():
        w = wide(df, col)
        w.columns = [f"{name}__{c}" for c in w.columns]
        parts.append(w)
    q = df[df["condition"].astype(str) == "baseline"].drop_duplicates(KEYS).set_index(KEYS)[["question"]]
    return q.join(parts).reset_index()


def main():
    cfg = load_config()
    df = load_results(cfg)
    df = df[~df["condition"].astype(str).isin(["m_union", "m123"])]
    out = ROOT / cfg["evaluation"]["output_dir"] / "_report"
    out.mkdir(parents=True, exist_ok=True)
    pooled, per = transitions(df, []), transitions(df, ["rag", "method"])
    g_pooled, g_per = any_of_three(df, []), any_of_three(df, ["rag", "method"])
    pd.concat([pooled.assign(rag="ALL", method="ALL"), per]).to_csv(out / "transitions.csv", index=False)
    pd.concat([g_pooled.assign(rag="ALL", method="ALL"), g_per]).to_csv(out / "transitions_any_of_three.csv", index=False)
    question_level(df).to_csv(out / "transitions_questions.csv", index=False)
    md = ["# Which questions change status between the benchmark and a mutated version?", "",
          "Each question is in exactly one cell: `fine_to_fine`, `fine_to_FAIL` (newly failing: the mutation exposed it), "
          "`FAIL_to_fine` (fixed by the mutation), `FAIL_to_FAIL` (fails whatever you do). Outcomes: `not_correct` = wrong or "
          "declined; `hallucinated` = the main metric; `wrong_behavior` = asserted something wrong / adopted a false claim. "
          "`p_mcnemar` compares `fine_to_FAIL` with `FAIL_to_fine` (exact test), `q_value` is Benjamini-Hochberg corrected "
          "within each outcome. The two Control rows show how many questions change status with NO mutation: a mutation "
          "must change more than that.", "",
          "## All RAGs and methods pooled", "", df_to_md(pooled), "",
          "## The three mutations together", "",
          "`at_least_one` = M1, M2 or M3; `all_three` = M1 and M2 and M3. `fails_somewhere` = fails in the benchmark or in "
          "any mutated version (the union).", "", df_to_md(g_pooled), "",
          "## Per RAG and method", "", df_to_md(per), "", df_to_md(g_per), "",
          "Question-level file: `transitions_questions.csv` (columns `<outcome>__<condition>`, 1 = failing).", ""]
    (out / "transitions.md").write_text("\n".join(md), encoding="utf-8")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nThe three mutations together:")
    print(g_pooled.T.to_string(header=False))
    print(f"\nWritten: {out / 'transitions.md'}")


if __name__ == "__main__":
    main()
