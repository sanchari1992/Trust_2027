"""BEFORE / AFTER tables: one place that puts together, for the original benchmark and for each
mutated version of it, ON THE SAME QUESTIONS:

  structural metrics   all 25 graph + lexical features of the question text (src/analysis), averaged
  accuracy             share of answers judged correct (for mutation 3 the "correct" behaviour can be
                       rejecting a false claim or flagging an unsupported detail, so
                       accuracy_answer_items restricts to questions whose correct behaviour is to
                       answer, which is comparable with the baseline)
  hallucination        share of answers that are unfaithful or assert a wrong / adopted-false claim
  also                 unfaithful rate, abstain rate, mean faithfulness

Changes ("delta") are paired: the same question, mutated minus original, with a 95% bootstrap interval.
"""
import numpy as np
import pandas as pd

from src.analysis.features import ALL_NUMERIC_FEATURES, compute_features, load_questions
from src.config import ROOT
from src.evalrun.analysis import _levels, df_to_md, paired_bootstrap_diff
from src.mutate.common import ALL_CONDITIONS as CONDITIONS, CONDITION_LABELS, DERIVED_CONDITIONS

OUTCOMES = {"accuracy": "acc", "hallucination_rate": "hallucinated", "unfaithful_rate": "unfaithful",
            "abstain_rate": "abstained"}
RATE_METRICS = set(OUTCOMES)


def domain_of(rag):
    return "medical" if rag == "medrag" else "general"


def add_structural_features(df, cfg):
    """Adds the 25 feature columns to every result row (computed once per distinct question text)."""
    texts = [t for t in df["question"].drop_duplicates().tolist() if t != "(derived)"]
    print(f"Computing structural features for {len(texts):,} distinct questions ...")
    F = compute_features([{"id": i, "question": t} for i, t in enumerate(texts)], "question",
                         cfg["real_queries"]["spacy_model"], show_progress=False)
    F["question"] = texts
    return df.merge(F[["question"] + ALL_NUMERIC_FEATURES], on="question", how="left")


def real_means(cfg):
    """Feature means of the real user queries, per domain, if those files exist."""
    out = {}
    for domain in ("general", "medical"):
        f = ROOT / cfg["real_queries"]["output_dir"] / f"{domain}.jsonl"
        if f.exists():
            rows = load_questions(f, "text")
            out[domain] = compute_features(rows, "text", cfg["real_queries"]["spacy_model"],
                                           show_progress=False)[ALL_NUMERIC_FEATURES].mean()
    return out


def summary_table(df):
    out = []
    for level, d in _levels(df):
        for (rag, method, cond), g in d.groupby(["rag", "method", "condition"], observed=True):
            ans = g[g["expected_behavior"] == "answer"]
            row = {"level": level, "rag": rag, "method": method, "condition": str(cond), "n": len(g),
                   "accuracy": g["acc"].mean(),
                   "accuracy_answer_items": ans["acc"].mean() if len(ans) else np.nan,
                   "hallucination_rate": g["hallucinated"].mean(),
                   "unfaithful_rate": g["unfaithful"].mean(), "abstain_rate": g["abstained"].mean(),
                   "mean_faithfulness": g["faithfulness"].astype(float).mean()}
            row.update({f: g[f].mean() for f in ALL_NUMERIC_FEATURES})
            out.append(row)
    return pd.DataFrame(out)


def delta_table(df, n_boot, seed):
    """Paired change versus the baseline, for every outcome and every structural feature."""
    rng = np.random.default_rng(seed)
    metrics = {**OUTCOMES, **{f: f for f in ALL_NUMERIC_FEATURES}}
    wides = {name: df.pivot_table(index=["rag", "method", "id"], columns="condition", values=col,
                                  aggfunc="first", observed=True) for name, col in metrics.items()}
    conds = [c for c in wides["accuracy"].columns if c != "baseline"]
    rows = []
    for level, keys in (("rag_method", ["rag", "method"]), ("rag", ["rag"]), ("domain", ["domain"]), ("overall", [])):
        for cond in conds:
            for name, w in wides.items():
                if cond not in w.columns:  # e.g. the derived condition has no structural features (all NaN)
                    continue
                sub = w[["baseline", cond]].dropna().reset_index()
                sub["domain"] = sub["rag"].map(domain_of)
                parts = sub.groupby(keys) if keys else [((), sub)]
                for key, g in parts:
                    key = key if isinstance(key, tuple) else (key,)
                    x, y = g["baseline"].to_numpy(float), g[cond].to_numpy(float)
                    lo, hi = paired_bootstrap_diff(x, y, n_boot, rng)
                    rec = {"level": level, "rag": "ALL", "method": "ALL", "domain": "ALL"}
                    if level == "rag_method":
                        rec.update(rag=key[0], method=key[1], domain=domain_of(key[0]))
                    elif level == "rag":
                        rec.update(rag=key[0], domain=domain_of(key[0]))
                    elif level == "domain":
                        rec.update(domain=key[0])
                    rows.append({**rec, "condition": cond, "metric": name,
                                 "kind": "outcome" if name in OUTCOMES else "structural",
                                 "baseline": x.mean(), "condition_value": y.mean(), "delta": y.mean() - x.mean(),
                                 "ci_low": lo, "ci_high": hi, "n_pairs": len(g)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- markdown
def _pct(v):
    return "" if pd.isna(v) else f"{100 * v:.1f}%"


def _delta(r):
    if r is None:
        return ""
    star = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else ""
    if r["metric"] in RATE_METRICS:
        return f"{100 * r['delta']:+.1f} pp{star}"
    return f"{r['delta']:+.2f}{star}"


def write_md(out_dir, df, summary, deltas, real, cfg):
    lab = lambda c: CONDITION_LABELS.get(c, c)  # noqa: E731
    present = [c for c in CONDITIONS if c in set(df["condition"].astype(str))]
    present_struct = [c for c in present if c not in DERIVED_CONDITIONS]  # derived conditions have no question text
    D = deltas.set_index(["level", "rag", "method", "domain", "condition", "metric"])

    def get(level, rag, method, domain, cond, metric):
        try:
            return D.loc[(level, rag, method, domain, cond, metric)].to_dict() | {"metric": metric}
        except KeyError:
            return None

    L = ["# Benchmark before vs after mutation", "",
         "Every number compares the SAME questions: the original benchmark question and its mutated version. "
         "`*` marks a change whose 95% paired-bootstrap interval excludes 0 (intervals are in `before_after_deltas.csv`). "
         "`pp` = percentage points.", "",
         "**Accuracy** = share of answers judged correct. For mutations 3 and 1+2+3 the correct behaviour can be to reject a "
         "false claim or flag an unsupported detail, so `accuracy (answer items)` restricts to questions whose correct "
         "behaviour is to answer; that column is comparable with the baseline. **Hallucination** = unfaithful answer or an "
         "answer that asserts a wrong / adopted-false claim (declining to answer is not counted).", ""]

    # 1) outcomes, pooled
    s = summary[summary.level == "overall"].set_index("condition")
    rows = []
    for c in present:
        r = s.loc[c]
        rows.append({"condition": lab(c), "n": int(r["n"]), "accuracy": _pct(r["accuracy"]),
                     "Δ accuracy": "" if c == "baseline" else _delta(get("overall", "ALL", "ALL", "ALL", c, "accuracy")),
                     "accuracy (answer items)": _pct(r["accuracy_answer_items"]),
                     "hallucination": _pct(r["hallucination_rate"]),
                     "Δ hallucination": "" if c == "baseline" else _delta(get("overall", "ALL", "ALL", "ALL", c, "hallucination_rate")),
                     "unfaithful": _pct(r["unfaithful_rate"]), "abstained": _pct(r["abstain_rate"]),
                     "mean faithfulness": f"{r['mean_faithfulness']:.3f}"})
    L += ["## 1. Accuracy and hallucination, all RAGs and methods pooled", "", df_to_md(pd.DataFrame(rows)), ""]

    # 2) per RAG x method
    rm = summary[summary.level == "rag_method"]
    for metric, title in (("accuracy", "Accuracy"), ("hallucination_rate", "Hallucination")):
        rows = []
        for (rag, method), g in rm.groupby(["rag", "method"]):
            g = g.set_index("condition")
            row = {"rag": rag, "method": method}
            for c in present:
                row[lab(c)] = _pct(g.loc[c, metric]) if c in g.index else ""
            rows.append(row)
        L += [f"## 2{'a' if metric == 'accuracy' else 'b'}. {title} per RAG and method", "", df_to_md(pd.DataFrame(rows)), ""]

    # 3) structural, per domain
    L += ["## 3. Structural metrics (all 25, averaged over the questions)", "",
          "Means of each feature over the evaluated questions, for the benchmark and each mutated version. "
          "`Real` = the real user queries of that domain (if `data/real_queries/` exists). "
          "'general' = LightRAG and RAPTOR; 'medical' = MedRAG. Feature definitions: `src/analysis/graph_features.py` "
          "and `lexical_features.py`.", ""]
    for domain, g in df.assign(domain=df["rag"].map(domain_of)).groupby("domain"):
        means = g.groupby(g["condition"].astype(str))[ALL_NUMERIC_FEATURES].mean()
        cols = (["Real"] if domain in real else []) + [lab(c) for c in present_struct if c in means.index]
        t = pd.DataFrame(index=ALL_NUMERIC_FEATURES, columns=cols, dtype=object)
        for c in present_struct:
            if c in means.index:
                t[lab(c)] = means.loc[c]
        if domain in real:
            t["Real"] = real[domain]
        t.insert(0, "feature", t.index)
        L += [f"### {domain}: mean of each feature ({len(g['id'].unique()):,} distinct questions)", "", df_to_md(t.reset_index(drop=True)), ""]
        rows = []
        for f in ALL_NUMERIC_FEATURES:
            row = {"feature": f}
            for c in present_struct:
                if c != "baseline":
                    row[f"{lab(c)} vs Benchmark"] = _delta(get("domain", "ALL", "ALL", domain, c, f))
            rows.append(row)
        L += [f"### {domain}: change from the benchmark (mutated minus original)", "", df_to_md(pd.DataFrame(rows)), ""]
    (out_dir / "before_after.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------- results grid (paper-style table)
METHOD_LABEL = {"ragas": "RAGAS", "ares": "ARES", "datamorgana": "DataMorgana"}
METHOD_SHORT = {"ragas": "RAGAS", "ares": "ARES", "datamorgana": "DM"}  # short headings keep the LaTeX table narrow
RAG_LABEL = {"lightrag": "LightRAG", "raptor": "RAPTOR", "medrag": "MedRAG", "ALL": "All RAGs"}
COND_SHORT = {"baseline": "", "m1": " + M1", "m2": " + M2", "m3": " + M3", "m123": " + M1,2,3 (one prompt)",
              "m_union": " + M1,2,3 (combined)",
              "c_rerun": " + re-run", "c_pad": " + padding"}
COND_GROUP = {"baseline": "Benchmark", "m1": "+ M1 (length)", "m2": "+ M2 (linguistic)",
              "m3": "+ M3 (user info)", "m123": "+ M1,2,3 (one prompt)", "m_union": "+ M1,2,3 (combined)",
              "c_rerun": "Control: re-run", "c_pad": "Control: padding"}
GRID_METRICS = [  # (key, column, title, only answer items?)
    ("hallucination", "hallucinated", "Hallucination rate", False),
    ("wrong_behavior", "wrong_behavior", "Wrong-behaviour rate (asserted wrong / adopted a false claim)", False),
    ("unfaithful_and_wrong", "unfaithful_and_wrong", "Unfaithful and wrong", False),
    ("retrieval_miss", "retrieval_miss", "Retrieval miss rate (original source passage not retrieved)", False),
    ("accuracy", "acc", "Accuracy", False),
    ("accuracy_answer_items", "acc", "Accuracy on questions whose correct behaviour is to answer", True),
]


def _grid(df, col, answer_only):
    d = df[df["expected_behavior"] == "answer"] if answer_only else df
    methods = [m for m in METHOD_LABEL if m in set(d["method"])]
    conds = [c for c in CONDITIONS if c in set(d["condition"].astype(str))]
    rags = [r for r in ("lightrag", "raptor", "medrag") if r in set(d["rag"])] + ["ALL"]
    val, cnt = {}, {}
    for r in rags:
        dr = d if r == "ALL" else d[d["rag"] == r]
        for c in conds:
            for m in methods:
                g = dr[(dr["condition"].astype(str) == c) & (dr["method"] == m)]
                val[(r, c, m)] = float(g[col].mean()) if len(g) else np.nan
                cnt[(r, c, m)] = len(g)
    return rags, conds, methods, val, cnt


def _head(c, m):
    return f"Benchmark ({METHOD_LABEL[m]}){COND_SHORT[c]}"


def write_results_grid(out_dir, df):
    """results_grid.md / .csv / .tex : for each metric, RAGs as rows and, as columns, the three
    benchmark-generation methods side by side under each condition (Benchmark, +M1, +M2, +M3, +M1,2,3)."""
    md = ["# Benchmark vs Benchmark+ : one table per metric", "",
          "Columns: the three benchmark-generation methods side by side under each condition. "
          "**M1** = length, **M2** = linguistic (typos, person, register, request form), **M3** = user-provided information "
          "(correct or incorrect), **M1,2,3** = all three. Every column covers the same questions as its benchmark column "
          "(only questions answered and judged in every condition are used).", ""]
    tex, csv_rows = [], []
    for key, col, title, answer_only in GRID_METRICS:
        rags, conds, methods, val, cnt = _grid(df, col, answer_only)
        if not conds:
            continue
        heads = [_head(c, m) for c in conds for m in methods]
        rows = []
        for r in rags:
            rows.append({"RAG": RAG_LABEL[r], **{_head(c, m): _pct(val[(r, c, m)]) for c in conds for m in methods}})
            csv_rows += [{"metric": key, "rag": r, "condition": c, "method": m, "value": val[(r, c, m)], "n": cnt[(r, c, m)]}
                         for c in conds for m in methods]
        md += [f"## {title}", "", df_to_md(pd.DataFrame(rows), ["RAG"] + heads), ""]
        if key == "hallucination" and "baseline" in conds:  # change versus the same method's benchmark
            drows = []
            for r in rags:
                drows.append({"RAG": RAG_LABEL[r], **{_head(c, m): f"{100 * (val[(r, c, m)] - val[(r, 'baseline', m)]):+.1f} pp"
                                                       for c in conds if c != "baseline" for m in methods}})
            md += ["### Change in hallucination rate versus the same method's benchmark", "",
                   df_to_md(pd.DataFrame(drows), ["RAG"] + [_head(c, m) for c in conds if c != "baseline" for m in methods]), ""]
        # LaTeX: grouped column headings, one group per condition
        ncol = len(conds) * len(methods)
        t = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
             f"\\caption{{{title} (\\%). M1: length; M2: linguistic (typos, person, register); M3: user-provided information; "
             "M1,2,3: all three. Each group shows the three benchmark-generation methods (DM: DataMorgana).}",
             f"\\label{{tab:grid-{key.replace('_', '-')}}}",
             "\\begin{tabular}{l" + "ccc".replace("ccc", "c" * len(methods)) * len(conds) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{len(methods)}}}{{c}}{{{COND_GROUP[c]}}}" for c in conds) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + i * len(methods)}-{1 + (i + 1) * len(methods)}}}" for i in range(len(conds))),
             "RAG & " + " & ".join(METHOD_SHORT[m] for _ in conds for m in methods) + " \\\\", "\\midrule"]
        for r in rags:
            if r == "ALL":
                t.append("\\midrule")
            t.append(RAG_LABEL[r] + " & " + " & ".join("" if np.isnan(val[(r, c, m)]) else f"{100 * val[(r, c, m)]:.1f}"
                                                         for c in conds for m in methods) + " \\\\")
        t += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
        tex.append("\n".join(t))
    # questions per cell
    rags, conds, methods, val, cnt = _grid(df, "acc", False)
    if conds:
        nrows = [{"RAG": RAG_LABEL[r], **{_head(c, m): str(cnt[(r, c, m)]) for c in conds for m in methods}} for r in rags]
        md += ["## Questions per cell", "", df_to_md(pd.DataFrame(nrows)), ""]
    (out_dir / "results_grid.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(out_dir / "results_grid.csv", index=False)
    (out_dir / "results_grid.tex").write_text(
        "% Requires \\usepackage{booktabs}. Generated by scripts/analyze_eval.py\n" + "\n".join(tex), encoding="utf-8")


def only_complete_questions(df):
    """Keep only questions that have a valid result in EVERY condition present for their (rag, method),
    so that the benchmark column and every mutated column describe exactly the same questions."""
    need = df.groupby(["rag", "method"])["condition"].transform("nunique")
    have = df.groupby(["rag", "method", "id"])["condition"].transform("nunique")
    kept = df[have == need]
    if len(kept) < len(df):
        print(f"Before/after tables: left out {len(df) - len(kept)} answers of questions that are missing "
              f"(RAG/judge error or not evaluated) in at least one condition.")
    return kept


def run_before_after(cfg, df, out_dir):
    df = only_complete_questions(df).copy()
    df["acc"] = df["correct"].astype(float)
    df["domain"] = df["rag"].map(domain_of)
    df = add_structural_features(df, cfg)
    real = real_means(cfg)
    summary = summary_table(df)
    deltas = delta_table(df, min(cfg["evaluation"]["bootstrap_samples"], 1000), cfg["evaluation"]["seed"])
    write_results_grid(out_dir, df)
    summary.to_csv(out_dir / "before_after_summary.csv", index=False)
    deltas.to_csv(out_dir / "before_after_deltas.csv", index=False)
    write_md(out_dir, df, summary, deltas, real, cfg)
    return summary, deltas
