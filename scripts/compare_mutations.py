"""Do the mutated benchmarks look MORE like real user queries than the original benchmark?

For every (RAG, method, condition) it asks: how well can a classifier tell the question set
apart from real user queries, using the graph + lexical features of section 11? AUC 0.5 = it
cannot (indistinguishable); 1.0 = always. A good mutation LOWERS the AUC relative to the
baseline. Real queries are compared in the matching domain: medrag -> medical.jsonl,
lightrag and raptor -> general.jsonl.

    python scripts/compare_mutations.py
    python scripts/compare_mutations.py --limit 500          # cap questions per set (quick look)
    python scripts/compare_mutations.py --real-file data/real_queries/general_heldout.jsonl --real-domain general

CIRCULARITY NOTE: the mutation operators were motivated by looking at a real-query sample. For
the paper, compare against real queries you did NOT look at while designing the mutations
(a fresh sample: change real_queries.general.seed, re-run prepare_real_queries.py, rename the
output, and pass it with --real-file).

Writes <analysis.output_dir>/mutation_realism.csv and mutation_realism.md, and
mutation_feature_shift.csv / .md : the mean of the key features per condition next to the REAL
value. This is the table to tune the operator weights against (config.yaml mutation.*.operators).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.compare import classifier_test, feature_table
from src.analysis.features import compute_features, load_questions
from src.benchgen.runner import METHODS
from src.config import ROOT, load_config
from src.evalrun.analysis import df_to_md
from src.mutate.common import CONDITION_LABELS, CONDITIONS, condition_file
from src.rags.registry import RAG_NAMES

# The features that separated real queries from generated questions most consistently.
KEY_FEATURES = ["n_sentences", "starts_with_verb", "starts_lowercase", "has_first_person",
                "has_politeness_marker", "has_typo_like_repeat", "ends_with_question_mark",
                "has_wh_word", "n_tokens", "n_entities", "n_components"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap questions per set")
    ap.add_argument("--real-file", default=None, help="use this real-query file instead of the default ones")
    ap.add_argument("--real-domain", choices=["general", "medical"], default=None,
                    help="with --real-file: which RAGs it is compared with (medical = medrag, general = the others)")
    args = ap.parse_args()
    cfg = load_config()
    ac, spacy_model = cfg["analysis"], cfg["real_queries"]["spacy_model"]
    out_dir = ROOT / ac["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"profile: {cfg['_profile']}")

    real_cache = {}

    def real_features(domain):
        if domain not in real_cache:
            path = Path(args.real_file) if args.real_file else ROOT / cfg["real_queries"]["output_dir"] / f"{domain}.jsonl"
            if not path.exists():
                real_cache[domain] = None
                print(f"No real queries for '{domain}' at {path} (run scripts/prepare_real_queries.py {domain}). "
                      f"Skipping the RAGs that need them.")
            else:
                rows = load_questions(path, "text", args.limit)
                print(f"Computing features for {len(rows):,} real '{domain}' queries ...")
                real_cache[domain] = compute_features(rows, "text", spacy_model, show_progress=False)
        return real_cache[domain]

    records, shift = [], []
    for rag in RAG_NAMES:
        domain = "medical" if rag == "medrag" else "general"
        if args.real_file and args.real_domain and args.real_domain != domain:
            continue
        df_real = real_features(domain)
        if df_real is None:
            continue
        if not any(r["domain"] == domain and r["condition"] == "real" for r in shift):
            shift.append({"domain": domain, "condition": "real", "n": len(df_real), **df_real[KEY_FEATURES].mean().to_dict()})
        for method in METHODS:
            for cond in CONDITIONS:
                f = condition_file(cfg, cond, rag, method)
                if not f.exists():
                    continue
                rows = load_questions(f, "question", args.limit)
                if len(rows) < 20:
                    print(f"  {rag}/{method}/{cond}: only {len(rows)} questions, skipped")
                    continue
                df_b = compute_features(rows, "question", spacy_model, show_progress=False)
                shift.append({"domain": domain, "condition": cond, "n": len(df_b), **df_b[KEY_FEATURES].mean().to_dict()})
                clf = classifier_test(df_real, df_b, ac["classifier_test_size"], ac["classifier_cv_folds"], ac["seed"])
                table = feature_table(df_real, df_b, "real", "bench", 300, ac["seed"])
                records.append({
                    "rag": rag, "method": method, "condition": cond, "domain": domain,
                    "n_real": len(df_real), "n_questions": len(df_b),
                    "auc": clf["held_out_auc"], "cv_auc": clf["cv_auc_mean"],
                    "n_sig_features": int((table["q_value"] < 0.05).sum()),
                    "mean_words": float(df_b["word_len"].mean()),
                    "top_features": ", ".join(t["feature"] for t in clf["top_features_by_weight"][:4]),
                })
                print(f"  {rag}/{method}/{cond}: AUC {clf['held_out_auc']:.3f}", flush=True)
    if not records:
        print("Nothing compared. Generate benchmarks and mutations first, and prepare the real queries.")
        return
    res = pd.DataFrame(records)
    res.to_csv(out_dir / "mutation_realism.csv", index=False)
    auc = res.pivot_table(index=["rag", "method"], columns="condition", values="auc").reset_index()
    auc = auc[["rag", "method"] + [c for c in CONDITIONS if c in auc.columns]]
    pooled = res.groupby("condition")[["auc", "n_sig_features", "mean_words"]].mean().reindex(
        [c for c in CONDITIONS if c in set(res["condition"])]).reset_index()
    pooled["condition"] = pooled["condition"].map(CONDITION_LABELS)
    md = ["# Do mutated questions look more like real queries?", "",
          "AUC of a classifier separating the question set from real user queries (0.5 = indistinguishable, "
          "lower is more realistic).", "", df_to_md(auc), "", "## Mean over RAGs and methods", "",
          df_to_md(pooled.rename(columns={"auc": "mean_auc", "n_sig_features": "mean_n_sig_features"})), ""]
    (out_dir / "mutation_realism.md").write_text("\n".join(md), encoding="utf-8")
    sh = pd.DataFrame(shift)
    sh.to_csv(out_dir / "mutation_feature_shift.csv", index=False)
    md2 = ["# Feature means: real queries vs each condition", "",
           "Mutations should move the generated questions toward the real-query values (first row of each table). "
           "Conditions are pooled over RAGs and methods (weighted by number of questions).", ""]
    for domain, g in sh.groupby("domain"):
        rows = []
        for cond, h in g.groupby("condition"):
            w = h["n"].to_numpy(float)
            rows.append({"condition": CONDITION_LABELS.get(cond, "REAL QUERIES"), "n": int(w.sum()),
                         **{f: float((h[f].to_numpy() * w).sum() / w.sum()) for f in KEY_FEATURES}})
        t = pd.DataFrame(rows)
        order = {"REAL QUERIES": -1, **{CONDITION_LABELS[c]: i for i, c in enumerate(CONDITIONS)}}
        t = t.sort_values("condition", key=lambda col: col.map(order))
        md2 += [f"## {domain}", "", df_to_md(t), ""]
    (out_dir / "mutation_feature_shift.md").write_text("\n".join(md2), encoding="utf-8")
    print("\nMean AUC by condition (lower = closer to real queries):")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWritten: {out_dir / 'mutation_realism.md'} and {out_dir / 'mutation_feature_shift.md'}")


if __name__ == "__main__":
    main()
