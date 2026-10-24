"""The paper's evaluation tables in one go: LaTeX (booktabs) + Markdown, from the finished paper run. No LLM calls.

    python scripts/final_tables.py --profile paper          # medical RAG, writes results/paper_final/final_tables.{tex,md}

Tables
  T1  How generated prompts deviate from real ones (held-out real prompts)          <- analysis_paper/deviation_deviations.csv
  T2  Does each mutation close the gap (by family, and overall)                      <- analysis_paper/deviation_{families,overall_auc}.csv
  T3  Hallucination, failure and refusal rate per condition and generator           <- eval_paper results
  T4  Paired tests against the unchanged benchmark (McNemar, BH q-values)            <- eval_paper/_report/paired_vs_baseline.csv
  T5  M3 by operator (same questions in the benchmark vs M3), with a paired test    <- eval_paper results
  T6  What the system does with a patient profile and with an unsupported detail     <- eval_paper results
  T7  Where the failures come from: refusals vs wrong answers vs retrieval         <- eval_paper results
  T8  M3 with and without the patient-conflict prompts                               <- eval_paper results
  T9  Coverage: failures found by the benchmark vs benchmark + mutations
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from src.benchdb.report import booktabs  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.evalrun.analysis import df_to_md  # noqa: E402

MUT = {"m2": "M2 Colloquial", "m3": "M3 Premise"}
GEN = {"ragas": "RAGAS", "ares": "ARES", "datamorgana": "DataMorgana"}
OPNAME = {"m3:correct_candidate": "true candidate answer", "m3:true_premise": "true premise", "m3:false_premise": "false premise",
          "m3:wrong_candidate": "wrong candidate answer", "m3:unsupported_assumption": "unsupported assumption",
          "m3:patient_consistent": "patient profile (fits)", "m3:patient_conflict": "patient profile (conflicts)"}


def load_rows(cfg):
    base = ROOT / cfg["evaluation"]["output_dir"]
    rows = []
    for f in sorted(base.glob("*/*/*.jsonl")):
        if f.stem in ("baseline", "m2", "m3"):
            rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    for c in ["hallucinated", "correct", "refused", "wrong_behavior", "unfaithful", "retrieval_hit"]:
        df[c] = df[c].astype(float)
    df["fail"] = 1 - df["correct"]
    df["op"] = df["operators"].map(lambda o: o[0] if o else "")
    return df


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n01 = int(((b == 0) & (c == 1)).sum())
    n10 = int(((b == 1) & (c == 0)).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def pct(v, d=1):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.{d}f}"


def _esc(x):
    import re

    x = str(x)
    x = re.sub(r"(?<!\\)%", r"\\%", x)
    x = re.sub(r"(?<!\\)&", r"\\&", x)
    x = re.sub(r"(?<!\\)_", r"\\_", x)
    x = x.replace("<0.001", "$<$0.001")
    return x


def tex_table(df, cols, headers, caption, label, align=None, widths=None):
    align = align or "l" + "r" * (len(cols) - 1)
    wide = len(cols) > 8
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(_esc(h) for h in headers) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_esc(r[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else ""), r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="paper")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(profile=a.profile)
    ev, an = ROOT / cfg["evaluation"]["output_dir"], ROOT / cfg["analysis"]["output_dir"]
    out = Path(a.out) if a.out else ROOT / cfg["paper"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(cfg)
    tex, md = [], []

    def add(title, table, tex_cols, headers, caption, label, align=None, note=""):
        md.append(f"## {title}\n\n{df_to_md(table[tex_cols])}\n" + (f"\n{note}\n" if note else ""))
        tex.append(tex_table(table, tex_cols, headers, caption, label, align))

    # ---- T1
    dev = pd.read_csv(an / "deviation_deviations.csv")
    t = pd.DataFrame({"Feature": dev["description"], "Family": dev["family"], "Real": dev["real_mean"].map(lambda v: f"{v:.2f}"),
                      "Generated": dev["benchmark_mean"].map(lambda v: f"{v:.2f}"),
                      "$d$ [95\\% CI]": [f"{r.d:+.2f} [{r.d_ci_low:+.2f}, {r.d_ci_high:+.2f}]" for r in dev.itertuples()],
                      "AUC": dev["single_feature_auc"].map(lambda v: f"{v:.2f}")})
    add("T1. Deviation of generated prompts from real prompts", t, list(t.columns), list(t.columns),
        "Where generated prompts differ from real (held-out) medical prompts. $d$: standardised difference (generated minus real); AUC: single-feature separability.",
        "tab:deviation", "llrrrr")

    # ---- T2
    fam = pd.read_csv(an / "deviation_families.csv")
    auc = pd.read_csv(an / "deviation_overall_auc.csv")
    t2 = auc.copy()
    t2["mean |d|"] = t2["mean_abs_d"].map(lambda v: f"{v:.3f}")
    t2["features closer"] = t2["features_closer_than_benchmark"]
    t2["Classifier AUC [95\\% CI]"] = [f"{r.auc_vs_real:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]" for r in t2.itertuples()]
    t2["Condition"] = t2["condition"]
    add("T2a. Closeness to real prompts, overall", t2, ["Condition", "mean |d|", "features closer", "Classifier AUC [95\\% CI]"],
        ["Condition", "Mean $|d|$", "Features closer", "AUC [95\\% CI]"],
        "Closeness to real prompts over 17 non-length features (lower $|d|$ and AUC are closer).", "tab:closeness", "lrrr")
    f2 = fam.copy()
    cols = ["family", "benchmark_mean_abs_d"] + [c for c in fam.columns if c.endswith("_mean_abs_d") and not c.startswith("benchmark")] + [c for c in fam.columns if c.endswith("_closure")]
    f2 = f2[cols].copy()
    for c in cols[1:]:
        f2[c] = f2[c].map(lambda v: f"{v:.3f}")
    add("T2b. Closeness by feature family (closure = 1 - |d after| / |d before|; negative = further from real)", f2, cols,
        ["Family", "Benchmark", "M2", "M3", "M2 closure", "M3 closure"][:len(cols)],
        "Mean $|d|$ per feature family and closure of the gap.", "tab:families")

    # ---- T3
    rows = []
    for cond in ["baseline", "m2", "m3"]:
        for gen in list(GEN) + ["ALL"]:
            d = df[(df.condition == cond) & ((df.method == gen) if gen != "ALL" else True)]
            n = len(d)
            h, f, r = d["hallucinated"].sum(), d["fail"].sum(), d["refused"].sum()
            lo, hi = wilson(h, n)
            rows.append({"Condition": {"baseline": "Benchmark", **MUT}[cond], "Generator": GEN.get(gen, "All"), "n": n,
                         "Hallucination \\%": f"{pct(h / n)} [{pct(lo)}, {pct(hi)}]", "Not correct \\%": pct(f / n),
                         "Refused \\%": pct(r / n), "Unfaithful \\%": pct(d['unfaithful'].mean())})
    t3 = pd.DataFrame(rows)
    add("T3. Hallucination, failure and refusal rates", t3, list(t3.columns), list(t3.columns),
        "Rates per condition and generator (Wilson 95\\% intervals for hallucination). Hallucination = unfaithful or asserts something wrong; ``not correct'' also counts refusals and abstentions.",
        "tab:rates", "llrrrrr")

    # ---- T4
    pv = pd.read_csv(ev / "_report" / "paired_vs_baseline.csv")
    pv = pv[pv.metric.isin(["hallucinated", "failed", "wrong_behavior", "refused"]) & pv.level.isin(["rag_method", "overall"])]
    names = {"hallucinated": "Hallucination", "failed": "Not correct", "wrong_behavior": "Wrong behaviour", "refused": "Refused"}
    t4 = pd.DataFrame([{"Metric": names[r.metric], "Generator": GEN.get(r.method, "All"), "Mutation": MUT[r.condition],
                        "Benchmark \\%": pct(r.baseline_rate), "Mutated \\%": pct(r.condition_rate),
                        "Diff (pp) [95\\% CI]": f"{100 * r.diff:+.1f} [{100 * r.diff_ci_low:+.1f}, {100 * r.diff_ci_high:+.1f}]",
                        "new/fixed": f"{r.newly_failing}/{r.newly_passing}", "q": "<0.001" if r.q_value < 0.001 else f"{r.q_value:.3f}"}
                       for r in pv.itertuples()])
    t4["_o"] = t4["Metric"].map({v: i for i, v in enumerate(names.values())})
    t4 = t4.sort_values(["_o", "Mutation", "Generator"], key=lambda s: s.map(lambda x: (x != "All", x)) if s.name == "Generator" else s).drop(columns="_o")
    add("T4. Paired tests against the unchanged benchmark", t4, list(t4.columns), list(t4.columns),
        "Same questions, benchmark vs mutated (McNemar test, Benjamini--Hochberg q within metric). new/fixed: questions that fail only after / only before the mutation.",
        "tab:paired", "lllrrrrr")

    # ---- T5
    H = df.pivot_table(index=["method", "id"], columns="condition", values="hallucinated")
    F = df.pivot_table(index=["method", "id"], columns="condition", values="fail")
    R = df.pivot_table(index=["method", "id"], columns="condition", values="refused")
    m3op = df[df.condition == "m3"].set_index(["method", "id"])["op"]
    rows = []
    for op in ["m3:true_premise", "m3:correct_candidate", "m3:patient_consistent", "m3:false_premise", "m3:wrong_candidate",
               "m3:unsupported_assumption", "m3:patient_conflict"]:
        idx = m3op.index[m3op == op]
        b, c = H.loc[idx, "baseline"], H.loc[idx, "m3"]
        n01, n10, p = mcnemar(b, c)
        rows.append({"M3 operator": OPNAME[op], "Expected": {"m3:true_premise": "answer", "m3:correct_candidate": "answer", "m3:patient_consistent": "answer",
                                                             "m3:false_premise": "correct premise", "m3:wrong_candidate": "correct premise",
                                                             "m3:unsupported_assumption": "flag / abstain", "m3:patient_conflict": "cannot answer"}[op],
                     "n": len(idx), "Benchmark hall. \\%": pct(b.mean()), "M3 hall. \\%": pct(c.mean()),
                     "Not correct \\% (M3)": pct(F.loc[idx, "m3"].mean()), "Refused \\% (M3)": pct(R.loc[idx, "m3"].mean()),
                     "p": "<0.001" if p < 0.001 else f"{p:.3f}"})
    t5 = pd.DataFrame(rows)
    add("T5. M3 by operator", t5, list(t5.columns), list(t5.columns),
        "M3 by operator, against the same questions in the unchanged benchmark (hall. = hallucination rate; p: McNemar, uncorrected).",
        "tab:m3ops", "llrrrrrr")

    # ---- T6
    lab = df[(df.condition == "m3") & df.op.isin(["m3:patient_conflict", "m3:patient_consistent", "m3:unsupported_assumption", "m3:false_premise", "m3:wrong_candidate"])]
    t6 = lab.groupby(["op", "label"]).size().unstack(fill_value=0)
    t6["n"] = t6.sum(axis=1)
    t6 = t6.reset_index()
    t6["M3 operator"] = t6["op"].map(OPNAME)
    cols6 = ["M3 operator"] + [c for c in t6.columns if c not in ("op", "M3 operator", "n")] + ["n"]
    for c in cols6[1:-1]:
        t6[c] = [f"{v} ({100 * v / n:.0f}\\%)" for v, n in zip(t6[c], t6["n"])]
    add("T6. What the system does", t6[cols6], cols6, cols6,
        "Judge labels of the answers. flagged / abstained: says it cannot be answered or that the detail is unsupported; answered\\_core: gives the standard answer as if the detail did not matter; accepted\\_detail: builds on the detail; corrected: corrects the false premise; consistent: answers consistently with the source without addressing the premise.",
        "tab:labels")

    # ---- T7
    nf = {}
    rows = []
    for cond in ["m2", "m3"]:
        newfail = (F["baseline"] == 0) & (F[cond] == 1)
        ref = int((R.loc[newfail, cond] == 1).sum())
        rows.append({"Mutation": MUT[cond], "Newly failing": int(newfail.sum()), "of which refused": ref,
                     "of which answered wrongly": int(newfail.sum()) - ref,
                     "Retrieval hit \\%": pct(df[df.condition == cond]["retrieval_hit"].mean()),
                     "(Benchmark)": pct(df[df.condition == "baseline"]["retrieval_hit"].mean())})
    t7 = pd.DataFrame(rows)
    add("T7. Where the new failures come from", t7, list(t7.columns), list(t7.columns),
        "New failures (pass in the benchmark, fail after the mutation), split into refusals and wrong answers, and the retrieval hit rate.",
        "tab:sources", "lrrrrr")

    # ---- T8
    rows = []
    for name, mask in [("All M3 prompts", m3op != ""), ("Without patient-conflict prompts", m3op != "m3:patient_conflict"),
                       ("Without any patient-profile prompts", ~m3op.str.contains("patient"))]:
        idx = m3op.index[mask]
        b, c = H.loc[idx, "baseline"], H.loc[idx, "m3"]
        n01, n10, p = mcnemar(b, c)
        rows.append({"Subset": name, "n": len(idx), "Benchmark \\%": pct(b.mean()), "M3 \\%": pct(c.mean()),
                     "Diff (pp)": f"{100 * (c.mean() - b.mean()):+.1f}", "new/fixed": f"{n01}/{n10}",
                     "p": "<0.001" if p < 0.001 else f"{p:.3f}"})
    t8 = pd.DataFrame(rows)
    add("T8. M3 hallucination with and without the patient-conflict prompts", t8, list(t8.columns), list(t8.columns),
        "Where the M3 increase in hallucination comes from.", "tab:m3split", "lrrrrrr")

    # ---- T9
    cov = pd.read_csv(ev / "_report" / "coverage.csv")
    cov = cov[cov.level.isin(["rag_method", "overall"])].copy()
    t9 = pd.DataFrame({"Generator": [GEN.get(m, "All") for m in cov["method"]], "Questions": cov["n_questions"],
                       "Benchmark alone": cov["found_by_benchmark"], "Benchmark + M2 + M3": cov["found_by_benchmark_plus_mutations"],
                       "Extra": cov["extra_found_by_mutations"], "Fold": cov["fold_increase"].map(lambda v: f"{v:.2f}$\\times$")})
    add("T9. Failures found", t9, list(t9.columns), list(t9.columns),
        "Hallucinated questions found by the benchmark alone and by the benchmark plus its M2 and M3 versions (a question counts once if any version fails). "
        "Without a re-run control this count includes answer-to-answer variation; the paired tests (T4, T5, T8) are the clean evidence.",
        "tab:coverage", "lrrrrr")

    (out / "final_tables.tex").write_text("% Generated by scripts/final_tables.py (requires booktabs)\n" + "\n".join(tex), encoding="utf-8")
    (out / "final_tables.md").write_text("# Final evaluation tables\n\n" + "\n".join(md), encoding="utf-8")
    print(f"Wrote {out / 'final_tables.tex'} and final_tables.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
