"""Analysis of evaluation results: do mutated benchmarks expose more hallucinations?

Outcomes per answer (see judge.py):  hallucinated, unfaithful, failed (= not correct), abstained.
All comparisons against the baseline are PAIRED: the same question, original vs mutated.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.compare import _benjamini_hochberg
from src.benchgen.common import read_jsonl
from src.config import ROOT
from src.mutate.common import (ALL_CONDITIONS as CONDITIONS, CONDITION_LABELS, CONTROL_CONDITIONS,
                               DERIVED_CONDITIONS, active_conditions)

METRICS = {"hallucinated": "Hallucination rate", "failed": "Failure rate (not correct)",
           "wrong_behavior": "Wrong-behaviour rate (asserted something wrong or adopted a false/unsupported claim)",
           "hallucinated_strict": "Hallucination rate, strict (user message NOT counted as context)",
           "unsupported_but_correct": "Unfaithful but judged correct (statements beyond the retrieved text, answer right)",
           "unfaithful_and_wrong": "Unfaithful AND judged not correct",
           "retrieval_miss": "Retrieval miss (no original source passage among the retrieved ones)"}


# ------------------------------------------------------------------ loading
UNION_MAX = ["hallucinated", "failed", "unfaithful", "abstained", "wrong_behavior", "hallucinated_strict",
             "unfaithful_strict", "unsupported_but_correct", "unfaithful_and_wrong", "retrieval_miss"]
UNION_MIN = ["faithfulness", "faithfulness_strict", "retrieval_hit", "gold_recall"]
UNION_PARTS = ["baseline", "m1", "m2", "m3"]


def add_union_condition(df):
    """DERIVED condition m_union, per question: failing if it fails in the unchanged benchmark OR in any of the
    three single-mutation runs (maximum of every bad outcome, minimum of faithfulness / retrieval). Uses only
    questions that have all four versions. Nothing is evaluated: this is the 'benchmark + added incorrect
    questions of M1 + M2 + M3' count, without double counting a question that fails under two mutations."""
    present = set(df["condition"].astype(str))
    if "m_union" in present or not all(p in present for p in UNION_PARTS):
        return df
    key = ["rag", "method", "id"]
    d = df[df["condition"].astype(str).isin(UNION_PARTS)]
    d = d[d.groupby(key)["condition"].transform("nunique") == len(UNION_PARTS)]
    if d.empty:
        return df
    d = d.copy()
    for c in UNION_MAX + UNION_MIN:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    agg = {c: "max" for c in UNION_MAX if c in d.columns}
    agg.update({c: "min" for c in UNION_MIN if c in d.columns})
    u = d.groupby(key).agg(agg).reset_index()
    base = d[d["condition"].astype(str) == "baseline"].drop_duplicates(key).set_index(key)
    u = u.merge(base[["expected_behavior"]].reset_index(), on=key, how="left")
    u["condition"], u["question"], u["kind"], u["response"] = "m_union", "(derived)", None, None
    u["operators"] = [["union"]] * len(u)
    u["correct"] = 1.0 - u["failed"]
    u["label"] = None
    return pd.concat([df, u], ignore_index=True)


def load_results(cfg):
    base = ROOT / cfg["evaluation"]["output_dir"]
    allowed = set(active_conditions(cfg))
    rows = []
    for f in sorted(base.glob("*/*/*.jsonl")):
        if f.stem in allowed or f.stem in DERIVED_CONDITIONS:  # results of conditions no longer in use are ignored
            rows += read_jsonl(f)
    if not rows:
        raise RuntimeError(f"No evaluation results under {base}. Run scripts/evaluate_benchmarks.py first.")
    df = pd.DataFrame(rows)
    for col in ("rag_error", "judge_error", "hallucinated", "correct", "unfaithful", "abstained",
                "faithfulness", "label", "kind", "operators"):
        if col not in df.columns:
            df[col] = None
    for col in ("wrong_behavior", "hallucinated_strict", "unfaithful_strict"):
        if col not in df.columns:
            df[col] = np.nan
    df["valid"] = df["hallucinated"].notna() & df["rag_error"].isna()
    n_bad = int((~df["valid"]).sum())
    if n_bad:
        print(f"NOTE: {n_bad} of {len(df)} answers have no judgement (RAG or judge errors) and are left out.")
    df = df[df["valid"]].copy()
    df["hallucinated"] = df["hallucinated"].astype(float)
    df["failed"] = 1.0 - df["correct"].astype(float)
    for col in ("retrieval_hit", "gold_recall"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df["unfaithful"] = df["unfaithful"].astype(float)
    for col in ("wrong_behavior", "hallucinated_strict", "unfaithful_strict"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df["abstained"] = df["abstained"].astype(float)
    df["unsupported_but_correct"] = df["unfaithful"] * (1.0 - df["failed"])
    df["unfaithful_and_wrong"] = df["unfaithful"] * df["failed"]
    df["retrieval_miss"] = 1.0 - df["retrieval_hit"]
    if cfg["evaluation"].get("combined", "derived") == "derived":
        df = add_union_condition(df)
    df["condition"] = pd.Categorical(df["condition"], [c for c in CONDITIONS if c in set(df["condition"])], ordered=True)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def paired_bootstrap_diff(x, y, n_samples, rng):
    """95% interval for mean(y) - mean(x) over the SAME items."""
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_samples, n))
    diffs = y[idx].mean(axis=1) - x[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_exact(b, c):
    """b = fine in baseline but failing in the mutated version; c = the reverse."""
    return float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0


# ------------------------------------------------------------------ tables
def _levels(df):
    """The same data viewed per (rag, method), per rag (pooled over methods), and overall."""
    yield "rag_method", df
    yield "rag", df.assign(method="ALL")
    yield "overall", df.assign(rag="ALL", method="ALL")


def summary_table(df):
    out = []
    for level, d in _levels(df):
        for (rag, method, cond), g in d.groupby(["rag", "method", "condition"], observed=True):
            n, k = len(g), int(g["hallucinated"].sum())
            lo, hi = wilson(k, n)
            out.append({"level": level, "rag": rag, "method": method, "condition": cond, "n": n,
                        "hallucination_rate": k / n, "hall_ci_low": lo, "hall_ci_high": hi,
                        "failure_rate": g["failed"].mean(), "unfaithful_rate": g["unfaithful"].mean(),
                        "unsupported_but_correct_rate": g["unsupported_but_correct"].mean(),
                        "unfaithful_and_wrong_rate": g["unfaithful_and_wrong"].mean(),
                        "retrieval_hit_rate": g["retrieval_hit"].mean(),
                        "abstain_rate": g["abstained"].mean(), "mean_faithfulness": g["faithfulness"].astype(float).mean()})
    return pd.DataFrame(out)


def _wide(df, metric):
    return df.pivot_table(index=["rag", "method", "id"], columns="condition", values=metric,
                          aggfunc="first", observed=True)


def paired_table(df, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for metric in METRICS:
        wide = _wide(df, metric)
        if "baseline" not in wide:
            continue
        groupings = [("rag_method", ["rag", "method"]), ("rag", ["rag"]), ("overall", [])]
        for level, keys in groupings:
            for cond in [c for c in wide.columns if c != "baseline"]:
                sub = wide[["baseline", cond]].dropna().reset_index()
                parts = sub.groupby(keys) if keys else [((), sub)]
                for key, g in parts:
                    key = key if isinstance(key, tuple) else (key,)
                    x, y = g["baseline"].to_numpy(float), g[cond].to_numpy(float)
                    b, c = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
                    lo, hi = paired_bootstrap_diff(x, y, n_boot, rng)
                    rows.append({
                        "level": level, "metric": metric,
                        "rag": key[0] if len(key) > 0 else "ALL", "method": key[1] if len(key) > 1 else "ALL",
                        "condition": cond, "n_pairs": len(g), "baseline_rate": x.mean(), "condition_rate": y.mean(),
                        "diff": y.mean() - x.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                        "newly_failing": b, "newly_passing": c, "p_mcnemar": mcnemar_exact(b, c)})
    t = pd.DataFrame(rows)
    t["q_value"] = np.nan
    for m in t["metric"].unique():  # correct for the number of tests within each metric
        mask = t["metric"] == m
        t.loc[mask, "q_value"] = _benjamini_hochberg(t.loc[mask, "p_mcnemar"].to_numpy())
    return t


def coverage_table(df):
    """How many hallucinations does the benchmark alone find, versus benchmark + mutations?"""
    wide = _wide(df, "hallucinated")
    if "baseline" not in wide:
        return pd.DataFrame()
    mut = [c for c in wide.columns if c != "baseline" and c not in CONTROL_CONDITIONS and c not in DERIVED_CONDITIONS]
    ctrl = [c for c in wide.columns if c in CONTROL_CONDITIONS]
    rows = []
    for level, keys in (("rag_method", ["rag", "method"]), ("rag", ["rag"]), ("overall", [])):
        w = wide.dropna(subset=["baseline"]).reset_index()
        parts = w.groupby(keys) if keys else [((), w)]
        for key, g in parts:
            key = key if isinstance(key, tuple) else (key,)
            base = g["baseline"].to_numpy(float)
            any_mut = g[mut].max(axis=1, skipna=True).fillna(0).to_numpy(float) if mut else np.zeros(len(g))
            union = np.maximum(base, any_mut)
            # controls: the same union logic over the control columns = hallucinations found "for free"
            any_ctl = g[ctrl].max(axis=1, skipna=True).fillna(0).to_numpy(float) if ctrl else np.zeros(len(g))
            ctl_union = np.maximum(base, any_ctl)
            rows.append({"extra_found_by_controls": int(ctl_union.sum() - base.sum()) if ctrl else np.nan,
                         "extra_found_by_mutations_beyond_controls": int(((union == 1) & (ctl_union == 0)).sum()) if ctrl else np.nan,
                "level": level, "rag": key[0] if len(key) > 0 else "ALL", "method": key[1] if len(key) > 1 else "ALL",
                "n_questions": len(g), "found_by_benchmark": int(base.sum()),
                "found_by_benchmark_plus_mutations": int(union.sum()),
                "extra_found_by_mutations": int(union.sum() - base.sum()),
                "benchmark_rate": base.mean(), "union_rate": union.mean(),
                "fold_increase": (union.sum() / base.sum()) if base.sum() else float("nan")})
    return pd.DataFrame(rows)


def operator_table(df):
    wide = _wide(df, "hallucinated")
    rows = []
    for cond in [c for c in ("m1", "m2", "m3") if c in set(df["condition"])]:
        d = df[df["condition"] == cond]
        ex = d.explode("operators").dropna(subset=["operators"])
        for op, g in ex.groupby("operators"):
            keys = list(zip(g["rag"], g["method"], g["id"]))
            base = wide["baseline"].reindex(keys).to_numpy(float) if "baseline" in wide else np.full(len(keys), np.nan)
            m = ~np.isnan(base)
            rows.append({"condition": cond, "operator": op, "n": len(g),
                         "hallucination_rate": g["hallucinated"].mean(),
                         "baseline_rate_same_questions": float(np.nanmean(base)) if m.any() else float("nan"),
                         "failure_rate": g["failed"].mean(), "mean_faithfulness": g["faithfulness"].astype(float).mean()})
    return pd.DataFrame(rows)


def m3_table(df):
    d = df[df["condition"].isin(["m3", "m123"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["m3_operator"] = d["operators"].apply(lambda o: next((x for x in (o or []) if str(x).startswith("m3:")), None))
    return (d.groupby(["condition", "kind", "m3_operator"], observed=True)
              .agg(n=("id", "size"), hallucination_rate=("hallucinated", "mean"),
                   failure_rate=("failed", "mean"), unfaithful_rate=("unfaithful", "mean"))
              .reset_index())


def retrieval_table(df):
    """Does the mutated prompt retrieve worse, and is hallucination tied to a retrieval miss?"""
    d = df.dropna(subset=["retrieval_hit"])
    if d.empty:
        return pd.DataFrame()
    rows = []
    for cond, g in d.groupby("condition", observed=True):
        hit, miss = g[g["retrieval_hit"] == 1.0], g[g["retrieval_hit"] == 0.0]
        rows.append({"condition": str(cond), "n": len(g), "retrieval_hit_rate": g["retrieval_hit"].mean(),
                     "mean_gold_recall": g["gold_recall"].mean(),
                     "hallucination_rate_when_hit": hit["hallucinated"].mean() if len(hit) else np.nan,
                     "hallucination_rate_when_miss": miss["hallucinated"].mean() if len(miss) else np.nan,
                     "n_miss": len(miss)})
    return pd.DataFrame(rows)


def label_table(df):
    return (df.groupby(["condition", "expected_behavior", "label"], observed=True)
              .size().rename("n").reset_index())


# ------------------------------------------------------------------ report
def _fmt(v):
    if isinstance(v, float):
        return "" if np.isnan(v) else (f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}")
    return str(v)


def df_to_md(df, cols=None):
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out_dir, summary, paired, coverage, operators, m3, labels, cfg, retrieval=None):
    s = summary[summary["level"] == "overall"].copy()
    s["condition"] = s["condition"].astype(str).map(CONDITION_LABELS)
    p = paired[(paired["level"] == "overall") & (paired["metric"] == "hallucinated")].copy()
    p["condition"] = p["condition"].map(CONDITION_LABELS)
    lines = ["# Benchmark vs Benchmark+ : hallucination results", "",
             f"Profile: `{cfg['_profile']}`. Faithfulness threshold: {cfg['evaluation']['faithfulness_threshold']}. "
             "All mutated results are compared with the original benchmark question by question (paired).", "",
             "## All RAGs and methods pooled", "",
             df_to_md(s, ["condition", "n", "hallucination_rate", "hall_ci_low", "hall_ci_high", "failure_rate",
                          "unfaithful_rate", "unsupported_but_correct_rate", "unfaithful_and_wrong_rate",
                          "retrieval_hit_rate", "abstain_rate", "mean_faithfulness"]), "",
             "## Paired change in hallucination rate vs the original benchmark (pooled)", "",
             "`diff` is condition minus baseline with a 95% paired-bootstrap interval; `newly_failing` = questions the RAG "
             "handled in the benchmark but hallucinated on in the mutated version; `newly_passing` = the reverse; "
             "`q_value` = McNemar exact test, Benjamini-Hochberg corrected.", "",
             df_to_md(p, ["condition", "n_pairs", "baseline_rate", "condition_rate", "diff", "diff_ci_low",
                          "diff_ci_high", "newly_failing", "newly_passing", "q_value"]), ""]
    if len(coverage):
        c = coverage[coverage["level"].isin(["overall", "rag"])]
        lines += ["## Hallucinations found: benchmark alone vs benchmark + mutations", "",
                  "A question counts as a hallucination found by the mutations if it was fine in the original "
                  "benchmark but a mutated version made the RAG hallucinate.", "",
                  df_to_md(c, ["level", "rag", "n_questions", "found_by_benchmark", "found_by_benchmark_plus_mutations",
                               "extra_found_by_mutations", "extra_found_by_controls",
                               "extra_found_by_mutations_beyond_controls", "benchmark_rate", "union_rate", "fold_increase"]), "",
                  "`extra_found_by_controls`: the same count for the control conditions (no useful change: a re-run, and "
                  "filler padding). `extra_found_by_mutations_beyond_controls`: questions that only a mutation exposed, "
                  "that neither control did. Only this last column is evidence that the mutations themselves help.", ""]
    ps = paired[(paired["level"] == "overall") & (paired["metric"].isin(
        ["wrong_behavior", "hallucinated_strict", "unsupported_but_correct", "unfaithful_and_wrong"]))].copy()
    if len(ps):
        ps["condition"] = ps["condition"].map(CONDITION_LABELS)
        lines += ["## The same change measured two other ways (pooled)", "",
                  "`wrong_behavior`: only answers that assert something wrong or adopt a false / unsupported claim "
                  "(no faithfulness involved). `hallucinated_strict`: unfaithfulness against the retrieved context "
                  "ALONE, i.e. a statement repeating what the user said is not accepted as supported. If a mutation "
                  "shows a change in `hallucinated` but not in these, check it with scripts/audit_hallucinations.py.", "",
                  df_to_md(ps, ["metric", "condition", "n_pairs", "baseline_rate", "condition_rate", "diff",
                                "diff_ci_low", "diff_ci_high", "newly_failing", "newly_passing", "q_value"]), ""]
    if retrieval is not None and len(retrieval):
        rt = retrieval.copy()
        rt["condition"] = rt["condition"].map(CONDITION_LABELS)
        lines += ["## Retrieval: do mutated prompts retrieve the original source passage?", "",
                  "`retrieval_hit_rate`: share of answers whose retrieved passages include an original source passage "
                  "of the question. `hallucination_rate_when_miss` vs `_when_hit`: whether hallucination goes with a "
                  "retrieval miss. If a condition retrieves worse, its extra hallucinations are a retrieval effect "
                  "(the query got diluted), not the model being fooled.", "", df_to_md(rt), ""]
    r = summary[summary["level"] == "rag_method"].copy()
    r["condition"] = r["condition"].astype(str)
    wide = r.pivot_table(index=["rag", "method"], columns="condition", values="hallucination_rate").reset_index()
    wide = wide[["rag", "method"] + [c for c in CONDITIONS if c in wide.columns]]
    lines += ["## Hallucination rate per RAG and method", "", df_to_md(wide), ""]
    if len(operators):
        lines += ["## By operator (hallucination rate; a question with two operators counts for both)", "",
                  df_to_md(operators), ""]
    if len(m3):
        lines += ["## Mutation 3 (user information) by kind and operator", "", df_to_md(m3), ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_figure(out_dir, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary[summary["level"] == "rag"]
    rags = sorted(d["rag"].unique()) + ["ALL"]
    d = pd.concat([d, summary[summary["level"] == "overall"]])
    conds = [c for c in CONDITIONS if c in set(d["condition"].astype(str))]
    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(rags), 4))
    width = 0.8 / len(conds)
    for i, c in enumerate(conds):
        vals, err = [], []
        for r in rags:
            row = d[(d["rag"] == r) & (d["condition"].astype(str) == c)]
            if len(row):
                v = float(row["hallucination_rate"].iloc[0])
                vals.append(v)
                err.append([v - float(row["hall_ci_low"].iloc[0]), float(row["hall_ci_high"].iloc[0]) - v])
            else:
                vals.append(np.nan)
                err.append([0, 0])
        x = np.arange(len(rags)) + i * width - 0.4 + width / 2
        ax.bar(x, vals, width, yerr=np.array(err).T, capsize=2, label=CONDITION_LABELS[c])
    ax.set_xticks(np.arange(len(rags)))
    ax.set_xticklabels(rags)
    ax.set_ylabel("Hallucination rate")
    ax.set_title("Benchmark vs Benchmark+ (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hallucination_by_condition.png", dpi=160)
    plt.close(fig)


def run_analysis(cfg):
    df = load_results(cfg)
    out_dir = ROOT / cfg["evaluation"]["output_dir"] / "_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot, seed = cfg["evaluation"]["bootstrap_samples"], cfg["evaluation"]["seed"]
    summary, paired = summary_table(df), paired_table(df, n_boot, seed)
    coverage, operators, m3, labels = coverage_table(df), operator_table(df), m3_table(df), label_table(df)
    retrieval = retrieval_table(df)
    for name, t in (("retrieval", retrieval), ("summary_by_condition", summary), ("paired_vs_baseline", paired), ("coverage", coverage),
                    ("by_operator", operators), ("m3_breakdown", m3), ("label_distribution", labels)):
        t.to_csv(out_dir / f"{name}.csv", index=False)
    write_report(out_dir, summary, paired, coverage, operators, m3, labels, cfg, retrieval)
    make_figure(out_dir, summary)
    return out_dir, summary, paired, coverage
