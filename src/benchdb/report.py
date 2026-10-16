"""Pieces of the paper report (scripts/paper_report.py): realism + template metrics of every question set, and
helpers that tie them to the hallucination tables."""
import json

import numpy as np
import pandas as pd

from src.analysis.compare import classifier_test
from src.benchdb.metrics import features_df, sample, summarize
from src.benchgen.common import read_jsonl
from src.mutate.common import condition_file

KEY_FEATURES = ["n_sentences", "n_tokens", "starts_with_verb", "starts_lowercase", "has_first_person",
                "has_politeness_marker", "has_typo_like_repeat", "ends_with_question_mark", "n_entities"]


def mutated_records(cfg, rag, method, cond, analyzer, cache_dir):
    """Analysed records (id, template, shape, entity types, 25 features) of one mutated file; cached on disk."""
    f = condition_file(cfg, cond, rag, method)
    if not f.exists():
        return []
    cache = cache_dir / f"{rag}_{method}_{cond}.jsonl"
    stamp = f.stat().st_size
    if cache.exists():
        lines = read_jsonl(cache)
        if lines and lines[0].get("_stamp") == stamp:
            return lines[1:]
    rows = [r for r in read_jsonl(f) if r.get("question")]
    recs = []
    for r, a in zip(rows, analyzer.analyze_many([r["question"] for r in rows], "question")):
        recs.append({"id": r["id"], "template_id": a["template_id"], "shape": a["shape"],
                     "labels": [e["label"] for e in a["entities"]], "features": a["features"]})
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_stamp": stamp}) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


def auc_vs_real(real, recs, ac, max_per_set, seed):
    if len(real) < 20 or len(recs) < 20:
        return float("nan")
    a, b = features_df(sample(real, max_per_set, seed)), features_df(sample(recs, max_per_set, seed))
    return classifier_test(a, b, ac["classifier_test_size"], ac["classifier_cv_folds"], ac["seed"])["held_out_auc"]


def realism_row(label, recs, real, ac, max_per_set, seed):
    recs_s = sample(recs, max_per_set, seed)
    real_s = sample(real, max_per_set, seed)
    feats = features_df(recs_s) if recs_s else pd.DataFrame()
    row = {"label": label, "n": len(recs), **summarize(recs_s, real_s)}
    if len(feats):
        row["mean_words"] = float(feats["word_len"].mean())
        for k in KEY_FEATURES:
            row[k] = float(feats[k].mean())
    row["auc_vs_real"] = auc_vs_real(real, recs, ac, max_per_set, seed) if real else float("nan")
    return row


def top_patterns(db_or_records, n=10):
    """Most common shapes (with share) of a list of records."""
    from collections import Counter

    c = Counter(r["shape"] for r in db_or_records)
    tot = sum(c.values()) or 1
    return [(k, v, v / tot) for k, v in c.most_common(n)]


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in str(s))


def fmt(v, kind="f3"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if kind == "pct":
        return f"{100 * v:.1f}\\%"
    if kind == "int":
        return f"{int(v):,}".replace(",", "{,}")
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v:.3f}"


def booktabs(df, cols, headers, kinds, caption, label, align=None):
    align = align or "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = [tex_escape(r[cols[0]])] + [fmt(r[c], k) for c, k in zip(cols[1:], kinds[1:])]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)
