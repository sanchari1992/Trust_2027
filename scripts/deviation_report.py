"""WHERE do generated prompts deviate from real user prompts, and does each mutation close that gap?  (README 35)

This is the evidence for the paper's first two tables:
  Table A  DEVIATIONS   for every feature of a prompt (wording, user voice, sentence complexity): the mean in real prompts, the mean in
                        the generated benchmark, the standardised difference d (Cohen's d, 95% bootstrap interval), and the
                        single-feature AUC. Sorted by |d|, grouped in three families, each family naming the mutation built
                        to address it: register -> M2 Colloquial, user voice -> M3 Premise, syntactic complexity -> both.
  Table B  CLOSURE      the same features for each mutated version: its |d| and how much of the gap it closed
                        (closure = 1 - |d_after| / |d_before|; negative = the mutation moved the prompts AWAY from real ones).
  Table C  FAMILIES     mean |d| per family and condition: does each mutation reduce the family it targets, and does it
                        make another family worse?
  Table D  GENERATORS   |d| per generator (RAGAS, ARES, DataMorgana): are the deviations shared by all three generators?
  Table E  OVERALL      classifier AUC (all features together) of every condition against real prompts, with a bootstrap interval.

    python scripts/deviation_report.py --profile pilot
    python scripts/deviation_report.py --profile paper --real-file data/real_queries/medical_heldout.jsonl

Use a HELD-OUT real sample (a second sample you did not design the mutations on) for the closure numbers in the paper;
the default real file is the one the mutations were motivated by (README 11, 35).
Writes <analysis.output_dir>/deviation_*.csv / .md and deviation_tables.tex. No LLM calls.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.analysis.compare import classifier_test  # noqa: E402
from src.analysis.features import compute_features, load_questions  # noqa: E402
from src.benchdb.report import booktabs  # noqa: E402
from src.benchgen.runner import METHODS  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.evalrun.analysis import df_to_md  # noqa: E402
from src.mutate.common import CONDITION_LABELS, condition_file  # noqa: E402

# feature -> (family, mutation built for it, plain-language name)
FEATURES = {
    "starts_lowercase":        ("register", "M2", "starts with a lowercase letter"),
    "starts_with_verb":        ("register", "M2", "starts with a verb (imperative)"),
    "has_politeness_marker":   ("register", "M2", "politeness marker (please, thanks, could you)"),
    "has_typo_like_repeat":    ("register", "M2", "typo-like repeated letters"),
    "ends_with_question_mark": ("register", "M2", "ends with a question mark"),
    "has_wh_word":             ("register", "M2", "contains a wh-word"),
    "has_first_person":        ("user voice", "M3", "first person (I, my, me)"),
    "n_entities":              ("user voice", "M3", "named entities per prompt"),
    "tree_depth":              ("complexity", "M2/M3", "dependency-tree depth"),
    "mean_dep_distance":       ("complexity", "M2/M3", "mean dependency distance"),
    "n_clauses":               ("complexity", "M2/M3", "subordinate/relative clauses"),
    "fk_grade":                ("complexity", "M2/M3", "Flesch-Kincaid grade"),
    "flesch_ease":             ("complexity", "M2/M3", "Flesch reading ease"),
    "gunning_fog":             ("complexity", "M2/M3", "Gunning Fog index"),
    "syll_per_word":           ("complexity", "M2/M3", "syllables per word"),
    "long_word_ratio":         ("complexity", "M2/M3", "share of long words (7+ letters)"),
    "max_depth":               ("complexity", "M2/M3", "depth of the content-word graph"),
}
FAMILY_MUTATION = {"register": "M2 Colloquial", "user voice": "M3 Premise", "complexity": "M2 + M3"}
FAMILIES = list(FAMILY_MUTATION)
# Length (words, tokens, sentences) is reported for information in table F but is not part of any realism measure.
LENGTH_INFO = ["word_len", "n_tokens", "n_sentences"]


def cohens_d(real, gen):
    """(mean_gen - mean_real) / pooled sd. Positive = generated prompts have MORE of it than real ones."""
    a, b = np.asarray(real, float), np.asarray(gen, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / sd) if sd > 0 else (0.0 if a.mean() == b.mean() else float("nan"))


def boot_d(real, gen, n, rng):
    a, b = np.asarray(real, float), np.asarray(gen, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2 or n <= 0:
        return float("nan"), float("nan")
    ds = [cohens_d(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n)]
    ds = [x for x in ds if not np.isnan(x)]
    return (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))) if ds else (float("nan"), float("nan"))


def single_auc(real, gen):
    """P(a generated prompt has a larger value than a real one) (0.5 = no difference); orientation-free: max(p, 1-p)."""
    a, b = np.asarray(real, float), np.asarray(gen, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2 or (a.min() == a.max() == b.min() == b.max()):
        return 0.5
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    p = u / (len(a) * len(b))
    return float(max(p, 1 - p))


def boot_auc(df_real, df_gen, n, seed):
    from sklearn.metrics import roc_auc_score  # noqa: F401

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        r = df_real.sample(len(df_real), replace=True, random_state=int(rng.integers(1 << 31)))
        g = df_gen.sample(len(df_gen), replace=True, random_state=int(rng.integers(1 << 31)))
        try:
            out.append(classifier_test(r, g, 0.3, 3, int(rng.integers(1 << 31)))["held_out_auc"])
        except Exception:
            continue
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) >= 5 else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--rag", default="medrag")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--conds", default="baseline,m2,m3")
    ap.add_argument("--real-file", default=None)
    ap.add_argument("--boot", type=int, default=300, help="bootstrap resamples (0 = no intervals)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(profile=a.profile)
    ac = cfg["analysis"]
    domain = "medical" if a.rag == "medrag" else "general"
    real_path = Path(a.real_file) if a.real_file else ROOT / cfg["real_queries"]["output_dir"] / f"{domain}.jsonl"
    if not real_path.exists():
        sys.exit(f"No real queries at {real_path}. Run: python scripts/prepare_real_queries.py {domain}")
    spacy_model = cfg["real_queries"]["spacy_model"]
    out_dir = Path(a.out) if a.out else ROOT / ac["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    methods, conds = a.methods.split(","), a.conds.split(",")
    rng = np.random.default_rng(ac["seed"])

    print(f"real prompts: {real_path}")
    real = compute_features(load_questions(real_path, "text"), "text", spacy_model, show_progress=False)
    feats = {}  # (cond, method) -> DataFrame
    for c in conds:
        for m in methods:
            f = condition_file(cfg, c, a.rag, m)
            if f.exists():
                rows = load_questions(f, "question")
                if rows:
                    feats[(c, m)] = compute_features(rows, "question", spacy_model, show_progress=False)
    if ("baseline", methods[0]) not in feats and not any(k[0] == "baseline" for k in feats):
        sys.exit("No benchmark questions found. Run the generate and mutate steps first.")
    pooled = {c: pd.concat([feats[(c, m)] for m in methods if (c, m) in feats], ignore_index=True)
              for c in conds if any(k[0] == c for k in feats)}
    base = pooled["baseline"]
    print(f"{len(real):,} real prompts; benchmark {len(base)} prompts from {[m for m in methods if ('baseline', m) in feats]}")

    # ---- Table A: deviations
    rows = []
    for f, (fam, mut, name) in FEATURES.items():
        d = cohens_d(real[f], base[f])
        lo, hi = boot_d(real[f], base[f], a.boot, rng)
        rows.append({"feature": f, "description": name, "family": fam, "built_for": FAMILY_MUTATION[fam],
                     "real_mean": real[f].mean(), "benchmark_mean": base[f].mean(), "d": d, "d_ci_low": lo, "d_ci_high": hi,
                     "abs_d": abs(d), "single_feature_auc": single_auc(real[f], base[f])})
    A = pd.DataFrame(rows).sort_values("abs_d", ascending=False).reset_index(drop=True)
    A.insert(0, "rank", range(1, len(A) + 1))

    # ---- Table B: closure
    brow = []
    for f in FEATURES:
        d0 = cohens_d(real[f], base[f])
        r = {"feature": f, "family": FEATURES[f][0], "real_mean": real[f].mean(), "benchmark_mean": base[f].mean(), "abs_d_benchmark": abs(d0)}
        for c in conds:
            if c == "baseline" or c not in pooled:
                continue
            d1 = cohens_d(real[f], pooled[c][f])
            r[f"{c}_mean"], r[f"{c}_abs_d"] = pooled[c][f].mean(), abs(d1)
            r[f"{c}_closure"] = (1 - abs(d1) / abs(d0)) if abs(d0) > 1e-9 else float("nan")
        brow.append(r)
    B = pd.DataFrame(brow).merge(A[["feature", "rank"]], on="feature").sort_values("rank").drop(columns="rank")

    # ---- Table C: families
    crow = []
    for fam in FAMILIES:
        fs = [f for f, v in FEATURES.items() if v[0] == fam]
        r = {"family": fam, "built_for": FAMILY_MUTATION[fam], "benchmark_mean_abs_d": np.nanmean([abs(cohens_d(real[f], base[f])) for f in fs])}
        for c in conds:
            if c == "baseline" or c not in pooled:
                continue
            r[f"{c}_mean_abs_d"] = np.nanmean([abs(cohens_d(real[f], pooled[c][f])) for f in fs])
            r[f"{c}_closure"] = 1 - r[f"{c}_mean_abs_d"] / r["benchmark_mean_abs_d"] if r["benchmark_mean_abs_d"] else float("nan")
        crow.append(r)
    C = pd.DataFrame(crow)

    # ---- Table D: per generator
    drow = []
    for f in FEATURES:
        r = {"feature": f, "family": FEATURES[f][0]}
        for m in methods:
            if ("baseline", m) in feats:
                r[f"{m}_d"] = cohens_d(real[f], feats[("baseline", m)][f])
        drow.append(r)
    D = pd.DataFrame(drow).merge(A[["feature", "rank"]], on="feature").sort_values("rank").drop(columns="rank")

    # ---- Table E: overall AUC with interval
    erow = []
    for c in conds:
        if c not in pooled:
            continue
        auc = classifier_test(real, pooled[c], ac["classifier_test_size"], ac["classifier_cv_folds"], ac["seed"])["held_out_auc"]
        lo, hi = boot_auc(real, pooled[c], min(a.boot, 100), ac["seed"]) if a.boot else (float("nan"), float("nan"))
        ad = [abs(cohens_d(real[f], pooled[c][f])) for f in FEATURES]
        ad0 = [abs(cohens_d(real[f], base[f])) for f in FEATURES]
        closer = int(sum(1 for x, y in zip(ad, ad0) if x < y - 1e-9))
        erow.append({"condition": CONDITION_LABELS.get(c, c), "n": len(pooled[c]), "mean_words_info_only": pooled[c]["word_len"].mean(),
                     "mean_abs_d": float(np.nanmean(ad)), "features_closer_than_benchmark": f"{closer}/{len(FEATURES)}" if c != "baseline" else "-",
                     "auc_vs_real": auc, "auc_ci_low": lo, "auc_ci_high": hi})
    E = pd.DataFrame(erow)

    for name, t in [("deviations", A), ("closure", B), ("families", C), ("generators", D), ("overall_auc", E)]:
        t.to_csv(out_dir / f"deviation_{name}.csv", index=False, encoding="utf-8")
    md = ["# Where generated prompts deviate from real prompts, and what each mutation does about it", "",
          f"Real prompts: {len(real):,} ({real_path.name}). Generated: {len(base)} benchmark prompts ({', '.join(m for m in methods if ('baseline', m) in feats)}). "
          "d = standardised difference (generated minus real, in pooled standard deviations; |d| about 0.2 small, 0.5 medium, 0.8 large). "
          "Positive d: generated prompts have more of it than real ones.", "",
          "## A. Deviations of the benchmark from real prompts (largest first)", "", df_to_md(A), "",
          "## B. Does each mutation close the gap? (closure = 1 - |d after| / |d before|; negative = moved away)", "", df_to_md(B), "",
          "## C. By family", "", df_to_md(C), "",
          "## D. Is the deviation shared by all generators? (d per generator)", "", df_to_md(D), "",
          "## E. Overall: mean |d| over all realism features, how many are closer than the unchanged benchmark, and classifier AUC on all non-length features (0.5 = cannot tell apart)", "", df_to_md(E), ""]
    (out_dir / "deviation_report.md").write_text("\n".join(md), encoding="utf-8")
    tex = booktabs(A, ["description", "real_mean", "benchmark_mean", "d", "single_feature_auc"],
                   ["Feature", "Real", "Generated", "$d$", "AUC"],
                   ["s", "f2", "f2", "f2", "f2"], "Deviation of generated prompts from real user prompts, by feature.",
                   "tab:deviations", align="lrrrr")
    (out_dir / "deviation_tables.tex").write_text(tex + "\n", encoding="utf-8")
    print("\nA. Largest deviations of the benchmark from real prompts:")
    print(A[["rank", "description", "real_mean", "benchmark_mean", "d", "single_feature_auc"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nC. By family (mean |d|; closure per mutation):")
    print(C.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nE. Classifier AUC against real prompts:")
    print(E.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten to {out_dir}: deviation_report.md, deviation_*.csv, deviation_tables.tex")


if __name__ == "__main__":
    main()
