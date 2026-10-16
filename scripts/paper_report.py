"""THE paper report: the whole evaluation story in one place (README section 32).

    python scripts/paper_report.py                         # every RAG that has data
    python scripts/paper_report.py --rags medrag
    python scripts/paper_report.py --no-eval               # realism / template part only (no evaluation results needed)

Tells the story in this order, for each RAG (the "union" is the three methods' questions with near-duplicates removed):
  1. REAL QUERIES      size, length, entity density, the commonest graph shapes (from the real-query database)
  2. BENCHMARKS        RAGAS, ARES, DataMorgana and the UNION: size, how realistic (classifier AUC against real queries,
                       divergence of graph-shape / template / entity-type distributions), how varied (distinct templates,
                       entropy), how well entities and shapes overlap with real queries
  3. MUTATIONS         Elaborate (M1), Colloquial (M2), Premise (M3) applied to the union: how many survived their checks,
                       how realistic the mutated versions are, alone and pooled with the originals (= Benchmark+)
  4. HALLUCINATIONS    per method and for the union: accuracy and hallucination rate of the benchmark and of Benchmark+
                       (+M1, +M2, +M3, +all), paired change with tests, controls, which questions flip
  5. THE STORY TABLE   one table per RAG: real | RAGAS | ARES | DataMorgana | union | +M1 | +M2 | +M3 | Benchmark+
  6. COST, PROVENANCE  LLM calls and tokens of generation and mutation, settings, file list
Also writes a hand-audit sheet of mutated questions (to validate assumption A4 of README 13.2) and LaTeX tables.

Needs: the real queries (prepare_real_queries.py), the benchmark (+ union: build_union.py), mutated files, and, for part 4,
evaluation results. Databases are built on the fly if create_bench.py has not been run.
Writes <paper.output_dir>/paper_report.md, paper_tables.tex, story_<rag>.csv, realism_all.csv, hallucination_*.csv,
templates_top.csv, mutation_validity.csv, audit_sheet_mutations.csv, paper_numbers.json
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.benchdb.metrics import records_from_db, sample
from src.benchdb.pipeline import ensure_real_db, ensure_run_db, load_script, make_analyzer
from src.benchdb.report import booktabs, mutated_records, realism_row, top_patterns
from src.benchgen.common import read_jsonl
from src.benchgen.runner import METHODS
from src.config import ROOT, corpus_tag, load_config
from src.evalrun.analysis import df_to_md
from src.mutate.common import CONDITION_LABELS, MUTATION_INFO, condition_file
from src.rags.registry import RAG_NAMES

MUTS = ["m1", "m2", "m3"]
NAMES = {"m1": "M1 Elaborate", "m2": "M2 Colloquial", "m3": "M3 Premise"}
PRETTY = {"ragas": "RAGAS", "ares": "ARES", "datamorgana": "DataMorgana", "union": "Union"}


def build_realism(cfg, rag, methods, real, run_db, an, max_per_set, seed, cache):
    ac = cfg["analysis"]
    bench = {m: records_from_db(run_db, "benchmark", method=m) for m in methods}
    bench = {m: v for m, v in bench.items() if v}
    union = [r for m in bench for r in bench[m] if (r.get("meta") or {}).get("in_union")]
    note = None
    if not union:
        union = [r for m in bench for r in bench[m]]
        note = "no union.jsonl found (run scripts/build_union.py): the union here is ALL questions, not de-duplicated"
    uids = {r["id"] for r in union}
    mut = {m: {c: mutated_records(cfg, rag, m, c, an, cache) for c in MUTS} for m in bench}
    scopes = {m: {"B": bench[m], **{c: mut[m][c] for c in MUTS}} for m in bench}
    scopes["union"] = {"B": union, **{c: [r for m in bench for r in mut[m][c] if r["id"] in uids] for c in MUTS}}
    rows = []
    for scope, d in scopes.items():
        B = d["B"]
        rows.append({"rag": rag, "scope": scope, "set": "Benchmark", "kind": "benchmark",
                     **realism_row("Benchmark", B, real, ac, max_per_set, seed)})
        for c in MUTS:
            if d[c]:
                rows.append({"rag": rag, "scope": scope, "set": f"{NAMES[c]} (versions alone)", "kind": c,
                             **realism_row(NAMES[c], d[c], real, ac, max_per_set, seed)})
        for c in MUTS:
            if d[c]:
                rows.append({"rag": rag, "scope": scope, "set": f"Benchmark + {c.upper()}", "kind": f"B+{c}",
                             **realism_row(f"Benchmark + {c.upper()}", B + d[c], real, ac, max_per_set, seed)})
        if all(d[c] for c in MUTS):
            rows.append({"rag": rag, "scope": scope, "set": "Benchmark + M1 + M2 + M3", "kind": "B+all",
                         **realism_row("Benchmark+", B + d["m1"] + d["m2"] + d["m3"], real, ac, max_per_set, seed)})
    return pd.DataFrame(rows), scopes, uids, note


def mutation_validity(cfg, rag, methods):
    tag = corpus_tag(cfg, rag)
    rows = []
    for m in methods:
        f = ROOT / cfg["mutation"]["output_dir"] / "_meta" / f"{rag}_{tag}" / f"{m}.meta.json"
        if not f.exists():
            continue
        meta = json.loads(f.read_text(encoding="utf-8"))
        for c, v in meta["counts"].items():
            if c == "m123" and not v["kept"] and not v["rejected"]:
                continue
            tot = v["kept"] + v["rejected"]
            top = max(v["rejected_reasons"].items(), key=lambda kv: kv[1])[0] if v["rejected_reasons"] else ""
            rows.append({"rag": rag, "method": m, "mutation": c, "kept": v["kept"], "rejected": v["rejected"],
                         "kept_rate": v["kept"] / tot if tot else np.nan, "by_fallback": v.get("fallback", 0),
                         "commonest_rejection": top})
    return pd.DataFrame(rows)


def provenance(cfg, rag, methods):
    tag = corpus_tag(cfg, rag)
    rows = []
    for stage, base in (("generation", ROOT / cfg["benchmark"]["output_dir"] / f"{rag}_{tag}"),
                        ("mutation", ROOT / cfg["mutation"]["output_dir"] / "_meta" / f"{rag}_{tag}")):
        for m in methods:
            f = base / f"{m}.meta.json"
            if f.exists():
                meta = json.loads(f.read_text(encoding="utf-8"))
                tr = meta.get("this_run", {})
                rows.append({"rag": rag, "stage": stage, "method": m, "accepted": meta.get("accepted", meta.get("n_source_questions")),
                             "attempts": meta.get("attempts"), "acceptance_rate": meta.get("acceptance_rate"),
                             "llm_calls_last_run": tr.get("calls"), "prompt_tokens": tr.get("prompt_tokens"),
                             "completion_tokens": tr.get("completion_tokens"), "model": meta.get("llm_model", meta.get("model"))})
    return pd.DataFrame(rows)


def entity_overlap(real_db, run_db, uids):
    real = {r[0] for r in real_db.ent.execute("SELECT DISTINCT norm FROM entities WHERE role='question'")}
    bench = {r[0] for r in run_db.ent.execute(
        "SELECT DISTINCT e.norm FROM entities e JOIN items i ON i.item_id=e.item_id "
        "WHERE e.role='question' AND i.kind='benchmark'")}
    corpus = {r[0] for r in run_db.ent.execute("SELECT DISTINCT norm FROM passage_entities")}
    return {"distinct_real_entities": len(real), "distinct_benchmark_entities": len(bench),
            "benchmark_entities_also_in_real": len(bench & real) / len(bench) if bench else float("nan"),
            "real_entities_found_in_corpus": len(real & corpus) / len(real) if real and corpus else float("nan"),
            "benchmark_entities_found_in_corpus": len(bench & corpus) / len(bench) if bench and corpus else float("nan")}


def audit_sheet(cfg, rag, methods, uids, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for c in MUTS:
        pool = []
        for m in methods:
            f = condition_file(cfg, c, rag, m)
            if f.exists():
                pool += [r for r in read_jsonl(f) if r["id"] in uids or not uids]
        if not pool:
            continue
        pick = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
        for r in pick:
            rows.append({"rag": rag, "method": r.get("method"), "mutation": c, "id": r["id"],
                         "original_question": r["original_question"], "mutated_question": r["question"],
                         "operators": json.dumps(r["operators"]), "kind": r.get("kind"), "claim": r.get("claim"),
                         "reference_answer": r["answer"], "expected_behavior": r["expected_behavior"],
                         "passage": " ".join(r.get("contexts") or [])[:1500],
                         "same_question_asked_1_0": "", "reference_answer_correct_1_0": "",
                         "added_text_supported_by_passage_1_0": "", "natural_1_to_5": "", "notes": ""})
    return rows


def hallucination_part(cfg, rags, methods, uids_by_rag, nb, seed):
    sc = load_script("summary_counts")
    tr = load_script("transition_report")
    from src.evalrun.analysis import load_results, paired_table

    df = load_results(cfg)
    df = df[df["rag"].isin(rags) & df["method"].isin(methods)]
    df = df[~df["condition"].astype(str).isin(["m123"])]
    parts = [df]
    for rag, uids in uids_by_rag.items():
        if uids:
            parts.append(df[(df["rag"] == rag) & df["id"].isin(uids)].assign(method="union"))
    dfa = pd.concat(parts, ignore_index=True)
    dfa["condition"] = pd.Categorical(dfa["condition"].astype(str), [c for c in CONDITION_LABELS if c in set(dfa["condition"].astype(str))], ordered=True)
    bplus = sc.bplus_table(dfa, ["rag", "method"], nb, seed)
    per_cond = sc.table(dfa, ["rag", "method"])
    paired = paired_table(dfa, nb, seed)
    paired = paired[(paired["level"] == "rag_method") & paired["metric"].isin(["hallucinated", "failed", "wrong_behavior", "refused"])]
    trans = tr.transitions(dfa, ["rag", "method"])
    anyof = tr.any_of_three(dfa, ["rag", "method"])
    refus = sc.refusal_table(dfa, ["rag", "method"])
    return dfa, bplus, per_cond, paired, trans, anyof, refus


def story_table(rag, real_row, realism, bplus):
    """real | methods | union | +M1 | +M2 | +M3 | Benchmark+"""
    rows = [{"row": "Real queries", **{k: real_row.get(k) for k in
            ("n", "mean_words", "n_sentences", "distinct_shapes", "shape_entropy", "ents_per_query")}}]
    lab = {"Benchmark": "Benchmark", "Benchmark + M1": "Benchmark + M1", "Benchmark + M2": "Benchmark + M2",
           "Benchmark + M3": "Benchmark + M3", "Benchmark + M1 + M2 + M3": "Benchmark + M1 + M2 + M3"}
    order = [(s, "Benchmark", PRETTY[s]) for s in METHODS if s in set(realism["scope"])] + [("union", "Benchmark", "Union")]
    order += [("union", k, f"Union + {k.split('+', 1)[1].strip()}" if k != "Benchmark + M1 + M2 + M3" else "Union + M1 + M2 + M3 (Benchmark+)")
              for k in ("Benchmark + M1", "Benchmark + M2", "Benchmark + M3", "Benchmark + M1 + M2 + M3")]
    for scope, setname, name in order:
        r = realism[(realism["scope"] == scope) & (realism["set"] == setname)]
        if r.empty:
            continue
        r = r.iloc[0].to_dict()
        row = {"row": name, **{k: r.get(k) for k in ("n", "mean_words", "n_sentences", "distinct_shapes", "shape_entropy",
                                                    "ents_per_query", "auc_vs_real", "shape_js", "template_js", "entity_type_js")}}
        if bplus is not None and len(bplus):
            b = bplus[(bplus["rag"] == rag) & (bplus["method"] == scope) & (bplus["benchmark_plus"] == lab.get(setname, setname))]
            if len(b):
                b = b.iloc[0]
                row.update(questions=b["questions"], accuracy=b["accuracy"], hallucination_rate=b["hallucination_rate"],
                           ci_low=b["rate_ci_low"], ci_high=b["rate_ci_high"])
        rows.append(row)
    out = pd.DataFrame(rows)
    if "questions" in out.columns:
        out["questions"] = out["questions"].map(lambda v: "" if pd.isna(v) else int(v))  # not 27.000
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rags", default=",".join(RAG_NAMES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--max-per-set", type=int, default=None)
    ap.add_argument("--audit-n", type=int, default=None)
    ap.add_argument("--no-eval", action="store_true", help="skip the hallucination part")
    args = ap.parse_args()
    cfg = load_config()
    pc = cfg["paper"]
    max_set, audit_n, seed = args.max_per_set or pc["max_per_set"], args.audit_n or pc["audit_n"], pc["seed"]
    out = ROOT / pc["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    rags, methods = args.rags.split(","), args.methods.split(",")
    print(f"profile: {cfg['_profile']} | output: {out}")

    md = [f"# Benchmark+ : evaluation report", "",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} | profile `{cfg['_profile']}` | LLM `{cfg['llm']['model']}` | "
          f"embeddings `{cfg['embedding']['model']}` | seed {seed}", "",
          "Definitions: README sections 13 (mutations), 14.2 (metrics), 29-32 (databases, union, this report). "
          "`AUC` = how well a classifier tells the set from real queries (0.5 = cannot; lower = more realistic). "
          "`*_js` = Jensen-Shannon divergence in bits of the distribution of graph shapes / fine templates / entity types "
          "against the real queries (0 = identical, 1 = disjoint). `entropy` = spread of shapes (1 = even). "
          "The **union** is the three methods' questions with near-duplicates removed; its mutated versions and "
          "evaluation results are those of the method each question came from.", ""]
    all_real, all_story, all_tpl, all_valid, all_prov, all_audit, uids_by_rag = [], [], [], [], [], [], {}
    realism_all, numbers = [], {}
    for rag in rags:
        domain = "medical" if rag == "medrag" else "general"
        try:
            real_db = ensure_real_db(cfg, domain)
        except FileNotFoundError as e:
            print(f"{rag}: {e}. Skipped.")
            continue
        try:
            run_db = ensure_run_db(cfg, rag, methods)
        except (FileNotFoundError, ValueError) as e:
            print(f"{rag}: no benchmark ({e}). Skipped.")
            continue
        an = make_analyzer(cfg, run_db)
        real = records_from_db(real_db, "real")
        cache = out / "_cache"
        realism, scopes, uids, note = build_realism(cfg, rag, methods, real, run_db, an, max_set, seed, cache)
        uids_by_rag[rag] = uids
        realism_all.append(realism)
        real_row = realism_row("Real queries", real, [], cfg["analysis"], max_set, seed)
        ov = entity_overlap(real_db, run_db, uids)
        valid, prov = mutation_validity(cfg, rag, methods), provenance(cfg, rag, methods)
        all_valid.append(valid)
        all_prov.append(prov)
        all_audit += audit_sheet(cfg, rag, methods, uids, audit_n, seed)
        tpl = []
        for scope, label in [("real", "Real queries")] + [(s, PRETTY.get(s, s)) for s in scopes]:
            recs = real if scope == "real" else scopes[scope]["B"]
            c_t = Counter(r["template_id"] for r in recs)
            for (shape, n, share) in top_patterns(recs, 8):
                tpl.append({"rag": rag, "scope": label, "level": "shape", "key": shape, "n": n, "share": share})
            for t, n in c_t.most_common(5):
                ex = next((r for r in recs if r["template_id"] == t), None)
                g = (real_db if scope == "real" else run_db).graph(ex["id"], "question") if ex else None
                tpl.append({"rag": rag, "scope": label, "level": "template", "key": t, "n": n, "share": n / len(recs),
                            "pattern": g["template"] if g else "", "text_template": g["text_template"] if g else ""})
        all_tpl += tpl
        numbers[rag] = {"n_real": len(real), "n_union": len(uids), **ov,
                        **{f"n_{m}": len(scopes[m]["B"]) for m in scopes if m != "union"}}
        md += [f"## {rag} ({domain})", ""]
        if note:
            md += [f"> NOTE: {note}", ""]
        # 1 real queries
        md += ["### 1. Real user queries", "",
               f"{len(real):,} real {domain} queries. Mean {real_row['mean_words']:.1f} words, "
               f"{real_row['n_sentences']:.2f} sentences, {real_row['ents_per_query']:.2f} entities per query; "
               f"{real_row['distinct_shapes']} distinct graph shapes (entropy {real_row['shape_entropy']:.2f}). "
               "Commonest shapes: " + ", ".join(f"`{s}` {100 * p:.1f}%" for s, _, p in top_patterns(real, 5)) + ".", ""]
        # 2 benchmarks
        cols = ["scope", "n", "mean_words", "n_sentences", "auc_vs_real", "shape_js", "template_js", "entity_type_js",
                "distinct_templates", "template_entropy", "distinct_shapes", "shape_entropy", "ents_per_query",
                "real_shape_coverage", "shape_in_real"]
        b = realism[realism["set"] == "Benchmark"]
        md += ["### 2. Benchmarks: how realistic and how varied", "", df_to_md(b, [c for c in cols if c in b.columns]), "",
               "Entity overlap: " + "; ".join(f"{k.replace('_', ' ')} {v:.3f}" if isinstance(v, float) else f"{k.replace('_', ' ')} {v:,}"
                                              for k, v in ov.items()) + ".", ""]
        um = ROOT / cfg["benchmark"]["output_dir"] / f"{rag}_{corpus_tag(cfg, rag)}" / "union.meta.json"
        if um.exists():
            u = json.loads(um.read_text(encoding="utf-8"))
            md += [f"Union: {u['total_input']:,} questions from {u['input']} -> {u['kept']:,} after removing "
                   f"{u['dropped_as_duplicates']:,} near-duplicates (cosine >= {u['dedupe_threshold']}); kept by method "
                   f"{u['kept_by_method']}.", ""]
        # 3 mutations
        u = realism[realism["scope"] == "union"]
        md += ["### 3. Mutations (applied to the union)", "", "Checks passed (kept) and rejected. `by_fallback` = how many of the kept ones come from a fallback instead of the full operator (`mutation.fill_all`, README 13.12; they are included in `kept`):", "",
               df_to_md(valid) if len(valid) else "(no mutation metadata found)", "",
               "Realism of the mutated versions alone and pooled with the originals (Benchmark+):", "",
               df_to_md(u, [c for c in ["set", "n", "mean_words", "n_sentences", "auc_vs_real", "shape_js", "template_js",
                                        "entity_type_js", "template_entropy", "shape_entropy", "starts_lowercase",
                                        "has_first_person", "starts_with_verb"] if c in u.columns]), ""]
        md += [""]
    realism_all = pd.concat(realism_all, ignore_index=True) if realism_all else pd.DataFrame()
    realism_all.to_csv(out / "realism_all.csv", index=False)
    pd.DataFrame(all_tpl).to_csv(out / "templates_top.csv", index=False)
    pd.concat(all_valid, ignore_index=True).to_csv(out / "mutation_validity.csv", index=False) if all_valid else None
    pd.concat(all_prov, ignore_index=True).to_csv(out / "provenance.csv", index=False) if all_prov else None
    pd.DataFrame(all_audit).to_csv(out / "audit_sheet_mutations.csv", index=False)

    # 4 hallucinations
    bplus = None
    if not args.no_eval and uids_by_rag:
        try:
            nb = min(cfg["evaluation"]["bootstrap_samples"], 1000)
            dfa, bplus, per_cond, paired, trans, anyof, refus = hallucination_part(
                cfg, list(uids_by_rag), methods, uids_by_rag, nb, cfg["evaluation"]["seed"])
            bplus.to_csv(out / "hallucination_benchmark_plus.csv", index=False)
            per_cond.to_csv(out / "hallucination_per_condition.csv", index=False)
            paired.to_csv(out / "hallucination_paired.csv", index=False)
            trans.to_csv(out / "hallucination_transitions.csv", index=False)
            anyof.to_csv(out / "hallucination_any_of_three.csv", index=False)
            refus.to_csv(out / "refusals.csv", index=False)
            for _, r in refus.iterrows():  # the numbers for the text of the paper
                numbers.setdefault(r["rag"], {}).setdefault("refusals", {})[f"{r['method']}|{r['condition']}"] = {
                    k: (None if pd.isna(r[k]) else float(r[k])) for k in
                    ("total", "refused", "refused_rate", "became_refused", "stopped_refusing", "refused_in_both")}
            md += ["## 4. Hallucinations and accuracy", "",
                   "Benchmark+ = the questions plus their mutated versions (N, 2N, 4N questions; README 14.7). "
                   "`hallucination_rate` is the main metric (README 14.2); the interval resamples original questions. "
                   "Rows for `union` pool the questions of the three methods that are in the union.", "",
                   df_to_md(bplus, [c for c in ["rag", "method", "benchmark_plus", "questions", "added_questions", "correct", "wrong",
                                                "abstained", "refused", "accuracy", "hallucinated", "hallucination_rate", "rate_ci_low",
                                                "rate_ci_high", "hallucinated_in_added"] if c in bplus.columns]), "",
                   "### Does each mutation expose more? (paired against the unchanged question)", "",
                   "`diff` = condition rate - benchmark rate on the SAME questions; the two Control rows are the noise "
                   "floor and the off-topic-filler reference (README 14.1, 16).", "",
                   "(Union only here; all methods are in `hallucination_paired.csv`. `m_union` is the derived condition: "
                   "a question fails if it fails in the benchmark or in any of M1, M2, M3.)", "",
                   df_to_md(paired[(paired["method"] == "union") & paired["metric"].isin(["hallucinated", "failed", "refused"])],
                            [c for c in ["rag", "metric", "condition", "n_pairs", "baseline_rate", "condition_rate",
                                         "diff", "diff_ci_low", "diff_ci_high", "newly_failing", "newly_passing", "q_value"]
                             if c in paired.columns]), "",
                   "### Which questions change status? (hallucinated)", "",
                   df_to_md(trans[(trans["outcome"] == "hallucinated") & (trans["method"] == "union")]) if len(trans) else "", "",
                   "### The three mutations together (union, hallucinated)", "",
                   df_to_md(anyof[(anyof["outcome"] == "hallucinated") & (anyof["method"] == "union")]) if len(anyof) else "", "",
                   "### Refusals: \"Cannot be answered from given information.\" (README 14.10)", "",
                   "The RAG is told to give exactly this answer when its documents lack the information. `refused` counts such "
                   "answers per condition. `became_refused` = answered the original question but refuses this version "
                   "(for M3: the patient information added to the question makes it unanswerable from the documents); "
                   "`stopped_refusing` = the reverse; `refused_in_both` = the documents never held the answer. "
                   "`abstained_by_judge` also counts declines in other words (refused <= abstained_by_judge). "
                   "A refusal is a failure to answer, never a hallucination.", "",
                   df_to_md(refus[refus["method"] == "union"]) if len(refus) else "", "",
                   "(All methods: `refusals.csv`.)", ""]
        except RuntimeError as e:
            md += ["## 4. Hallucinations and accuracy", "", f"(not available: {e})", ""]
    # 5 story
    md += ["## 5. The story table", "",
           "One row per question set. Realism columns: lower AUC / JS is closer to real queries. Hallucination columns: "
           "from the pooled Benchmark+ definition (the `+M` rows contain the originals AND the mutated versions).", ""]
    tex = []
    for rag in uids_by_rag:
        rr = realism_row("Real", records_from_db(ensure_real_db(cfg, "medical" if rag == "medrag" else "general"), "real"),
                         [], cfg["analysis"], max_set, seed)
        st = story_table(rag, rr, realism_all[realism_all["rag"] == rag], bplus)
        st.insert(0, "rag", rag)
        st.to_csv(out / f"story_{rag}.csv", index=False)
        all_story.append(st)
        md += [f"### {rag}", "", df_to_md(st, ["row"] + [c for c in ("n", "mean_words", "n_sentences", "ents_per_query", "distinct_shapes",
                                                                       "shape_entropy", "auc_vs_real", "shape_js", "template_js", "entity_type_js",
                                                                       "questions", "accuracy", "hallucination_rate", "ci_low", "ci_high") if c in st.columns]), ""]
        t = st.copy()
        for c in ("mean_words", "auc_vs_real", "shape_js", "accuracy", "hallucination_rate"):
            if c not in t:
                t[c] = np.nan
        tex.append(booktabs(t, ["row", "n", "mean_words", "auc_vs_real", "shape_js", "accuracy", "hallucination_rate"],
                            ["Question set", "N", "Words", "AUC$\\downarrow$", "Shape JS$\\downarrow$", "Acc.", "Halluc."],
                            ["", "int", "f2", "f3", "f3", "pct", "pct"],
                            f"Benchmark and Benchmark+ on {rag}: realism against real queries (AUC, shape JS divergence), "
                            f"accuracy and hallucination rate.", f"tab:story-{rag}"))
    (out / "paper_tables.tex").write_text("\n\n".join(tex), encoding="utf-8")
    # 6 cost
    if all_prov:
        p = pd.concat(all_prov, ignore_index=True)
        md += ["## 6. Cost and provenance", "", "LLM calls and tokens of the LAST run of each stage (a resumed run only "
               "counts its own part):", "", df_to_md(p), "",
               "Hand-audit sheet for assumption A4 (README 13.2): `audit_sheet_mutations.csv` "
               f"({len(all_audit)} mutated questions; fill the last five columns and report the shares).", ""]
    md += ["## Files", "", f"All in `{out}`: " + ", ".join(f"`{f.name}`" for f in sorted(out.glob('*')) if f.is_file()), ""]
    (out / "paper_report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "paper_numbers.json").write_text(json.dumps(numbers, indent=2, default=float), encoding="utf-8")
    print(f"\nReport: {out / 'paper_report.md'}")
    for st in all_story:
        print(f"\n{st['rag'].iloc[0]}:")
        print(st.drop(columns=["rag"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
