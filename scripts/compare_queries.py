"""Compares real user queries against a generated benchmark (or two arbitrary
question sets), using graph + lexical features (src/analysis/).

Usage (from the repo root):
    # real queries vs. one generated benchmark
    python scripts/compare_queries.py \\
        --a data/real_queries/general.jsonl --a-field text --a-name real \\
        --b results/benchmarks/lightrag_mix_10/ragas.jsonl --b-field question --b-name ragas

    # shortcut: compares real queries against EVERY generated benchmark found
    # under results/benchmarks/, one comparison per file
    python scripts/compare_queries.py --all-benchmarks --domain general
    python scripts/compare_queries.py --all-benchmarks --domain medical   # vs data/real_queries/medical.jsonl

Results (per comparison) go to results/analysis/<a-name>_vs_<b-name>/:
    features.csv        every question's raw feature values (for your own plots/stats)
    feature_table.csv    per-feature means, difference, 95% CI, p-value, q-value
    summary.json         classifier AUC and everything above, machine-readable
    report.md             a short human-readable write-up
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.compare import classifier_test, feature_table
from src.analysis.features import compute_features, load_questions
from src.config import ROOT, load_config


def run_one(a_path, a_field, a_name, b_path, b_field, b_name, cfg, limit=None):
    ac = cfg["analysis"]
    out_dir = ROOT / ac["output_dir"] / f"{a_name}_vs_{b_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {a_name} ({a_path.name}) vs {b_name} ({b_path.name}) ===")
    a_rows = load_questions(a_path, a_field, limit)
    b_rows = load_questions(b_path, b_field, limit)
    if len(a_rows) < 20 or len(b_rows) < 20:
        print(f"WARNING: only {len(a_rows)} / {len(b_rows)} questions available. "
              "Results with under ~50 per set are noisy; treat them as a pilot only.")

    print(f"Computing features for {len(a_rows)} '{a_name}' question(s)...")
    df_a = compute_features(a_rows, a_field, cfg["real_queries"]["spacy_model"])
    df_a["set"] = a_name
    print(f"Computing features for {len(b_rows)} '{b_name}' question(s)...")
    df_b = compute_features(b_rows, b_field, cfg["real_queries"]["spacy_model"])
    df_b["set"] = b_name

    pd_concat = __import__("pandas").concat([df_a, df_b], ignore_index=True)
    pd_concat.to_csv(out_dir / "features.csv", index=False)

    table = feature_table(df_a, df_b, a_name, b_name, ac["bootstrap_samples"], ac["seed"])
    table.to_csv(out_dir / "feature_table.csv", index=False)

    clf = classifier_test(df_a, df_b, ac["classifier_test_size"], ac["classifier_cv_folds"], ac["seed"])

    summary = {
        "a": {"name": a_name, "path": str(a_path), "n": len(df_a)},
        "b": {"name": b_name, "path": str(b_path), "n": len(df_b)},
        "classifier": clf,
        "feature_table": table.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sig = table[table["q_value"] < 0.05]
    lines = [
        f"# {a_name} vs {b_name}",
        f"\n{len(df_a)} '{a_name}' questions vs {len(df_b)} '{b_name}' questions.\n",
        f"**Classifier separability (held-out AUC): {clf['held_out_auc']:.3f}** "
        f"(5-fold CV on the training split: {clf['cv_auc_mean']:.3f} +/- {clf['cv_auc_std']:.3f}). "
        "0.5 = the two sets look the same on these features; 1.0 = a classifier can always tell them apart.\n",
        f"Top features driving that separation: "
        + ", ".join(f"`{t['feature']}`" for t in clf["top_features_by_weight"][:5]) + ".\n",
        f"\n**{len(sig)} of {len(table)} features differ significantly** (q < 0.05, Benjamini-Hochberg corrected):\n",
    ]
    for _, r in sig.iterrows():
        lines.append(
            f"- `{r['feature']}`: {a_name}={r[f'mean_{a_name}']:.3f} vs {b_name}={r[f'mean_{b_name}']:.3f} "
            f"(diff={r['diff']:.3f}, 95% CI [{r['diff_ci_low']:.3f}, {r['diff_ci_high']:.3f}], q={r['q_value']:.4f})"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"AUC={clf['held_out_auc']:.3f} | {len(sig)}/{len(table)} features significant -> {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a")
    ap.add_argument("--a-field", default="text")
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b")
    ap.add_argument("--b-field", default="question")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--limit", type=int, default=None, help="cap questions per set (for a quick trial run)")
    ap.add_argument("--all-benchmarks", action="store_true",
                     help="compare real_queries/<domain>.jsonl against every results/benchmarks/*/*.jsonl found")
    ap.add_argument("--domain", choices=["general", "medical"], default="general")
    args = ap.parse_args()
    cfg = load_config()

    if args.all_benchmarks:
        a_path = ROOT / cfg["real_queries"]["output_dir"] / f"{args.domain}.jsonl"
        if not a_path.exists():
            sys.exit(f"Missing {a_path}. Run: python scripts/prepare_real_queries.py {args.domain}")
        found = sorted((ROOT / "results" / "benchmarks").glob("*/*.jsonl"))
        found = [p for p in found if not p.name.endswith((".rejected.jsonl",))]
        if not found:
            sys.exit("No generated benchmarks found under results/benchmarks/. "
                      "Run scripts/generate_benchmark.py first.")
        for b_path in found:
            b_name = f"{b_path.parent.name}_{b_path.stem}"
            run_one(a_path, "text", "real_" + args.domain, b_path, "question", b_name, cfg, args.limit)
        return

    if not args.a or not args.b:
        sys.exit("Provide --a and --b, or use --all-benchmarks.")
    run_one(Path(args.a), args.a_field, args.a_name, Path(args.b), args.b_field, args.b_name, cfg, args.limit)


if __name__ == "__main__":
    main()
