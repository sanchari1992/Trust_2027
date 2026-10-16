"""Example prompts for the paper (README section 32.4).

    python scripts/paper_examples.py                          # every RAG with data
    python scripts/paper_examples.py --rags medrag --n 3 --seed 7

Pulls, for each RAG:
  A. a SIMPLE benchmark query from each method (RAGAS, ARES, DataMorgana), with its reference answer
  B. REALISTIC queries: real user queries of the same domain (what the benchmark should look like)
  C. NAMED ENTITIES and GRAPH TEMPLATES of a few prompts: entities (type, text), the question graph, its template, the
     shape, the text template with entity slots, and the template of the reference answer; plus a figure
  D. MUTATED queries for each method: the same question after Elaborate (M1), Colloquial (M2), Premise (M3), with the
     operators, the reference answer for the mutated prompt, the expected behaviour and, when evaluation results exist,
     what the RAG did with the original and the mutated prompt (examples where the mutation changed the outcome come first)
  E. REFUSALS: how often the RAG answered "Cannot be answered from given information." for the original benchmark and
     for each mutated version (counts per method and for the union), examples of original questions the RAG refuses, and
     examples of mutated questions it refuses although it answered the original (typically Premise: the patient
     information added to the question makes it unanswerable from the documents)
Writes <paper.output_dir>/paper_examples.md, paper_examples.tex (booktabs/tabularx-free, paste-ready), paper_examples.json
(including "refusals"), refusal_counts_<rag>.csv,
example_graphs.png / .pdf
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.benchdb.analyze import ascii_tree, contract_entities, graph_from_json
from src.benchdb.pipeline import ensure_real_db, ensure_run_db, make_analyzer
from src.benchdb.report import tex_escape
from src.benchgen.common import read_jsonl
from src.benchgen.runner import METHODS
from src.config import ROOT, load_config
from src.evalrun.runner import eval_file
from src.mutate.common import MUTATION_INFO, condition_file
from src.rags.base import is_refusal, refusal_phrase
from src.rags.registry import RAG_NAMES

MUTS = ["m1", "m2", "m3"]


def short(s, n=260):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def pick_simple(items, rng, k):
    """Questions of typical length (around the median) so the example is not an outlier."""
    lens = np.array([len(i["question"].split()) for i in items])
    med = np.median(lens)
    order = np.argsort(np.abs(lens - med) + rng.random(len(lens)))
    return [items[i] for i in order[:k]]


def draw_graphs(entries, path_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.8), squeeze=False)
    colors = {"VERB": "#4c78a8", "NOUN": "#54a24b", "PROPN": "#54a24b", "ADJ": "#b279a2", "ADV": "#b279a2"}
    for ax, e in zip(axes[0], entries):
        g = contract_entities(graph_from_json(e["graph"]))
        depth = {}
        for r in [x for x in g if g.in_degree(x) == 0]:
            for node, d in nx.single_source_shortest_path_length(g, r).items():
                depth[node] = min(depth.get(node, 99), d)
        levels = {}
        for node, d in sorted(depth.items()):
            levels.setdefault(d, []).append(node)
        pos, col = {}, 0.0
        for d, nodes in levels.items():
            for i, node in enumerate(nodes):
                pos[node] = ((i - (len(nodes) - 1) / 2) * 1.5 + (0.0 if d == 0 else 0.0), -d)
        labels = {x: (g.nodes[x]["ent"] and f"{g.nodes[x]['lemma']}\n<{g.nodes[x]['ent']}>") or f"{g.nodes[x]['lemma']}\n({g.nodes[x]['pos']})"
                  for x in g}
        nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, edge_color="#888888", node_size=1500, arrowsize=10)
        nx.draw_networkx_edge_labels(g, pos, edge_labels={(a, b): d["dep"] for a, b, d in g.edges(data=True)},
                                     ax=ax, font_size=7, font_color="#555555")
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1500,
                               node_color=["#f58518" if g.nodes[x].get("ent") else colors.get(g.nodes[x]["pos"], "#bbbbbb") for x in g])
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
        ax.set_title(short(e["text"], 60) + f"\n{e['shape']}", fontsize=8)
        ax.margins(0.2)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    fig.savefig(str(path_png).replace(".png", ".pdf"))
    plt.close(fig)


def eval_rows(cfg, cond, rag, method, cache):
    key = (cond, method)
    if key not in cache:
        f = eval_file(cfg, cond, rag, method)
        cache[key] = {r["id"]: r for r in read_jsonl(f)} if f.exists() else {}
    return cache[key]


def was_refused(cfg, r):
    """The stored flag, or (results written before the refusal phrase existed) a check of the stored response."""
    if r.get("refused") is not None:
        return bool(r["refused"])
    return is_refusal(r.get("response"), refusal_phrase(cfg))


def outcome(cfg, cond, rag, method, iid, cache):
    r = eval_rows(cfg, cond, rag, method, cache).get(iid)
    if not r or r.get("hallucinated") is None:
        return None
    return {"hallucinated": bool(r["hallucinated"]), "correct": bool(r.get("correct")), "label": r.get("label"),
            "refused": was_refused(cfg, r), "response": short(r.get("response"), 320)}


def refusal_part(cfg, rag, methods, uids, k, rng, cache):
    """-> (counts rows, original-question refusals per method, mutation-induced refusals per method and mutation)."""
    conds = ["baseline"] + MUTS
    counts, original, induced = [], {}, {}
    for m in methods:
        base = eval_rows(cfg, "baseline", rag, m, cache)
        for scope in (m, "union"):
            for c in conds:
                rows = eval_rows(cfg, c, rag, m, cache)
                if scope == "union":
                    rows = {i: r for i, r in rows.items() if i in uids}
                rows = {i: r for i, r in rows.items() if r.get("hallucinated") is not None}
                counts.append({"rag": rag, "scope": scope, "method": m if scope != "union" else "union-part-of-" + m,
                               "condition": c, "evaluated": len(rows),
                               "refused": sum(was_refused(cfg, r) for r in rows.values())})
        refused0 = [r for r in base.values() if r.get("hallucinated") is not None and was_refused(cfg, r)]
        rng.shuffle(refused0)
        original[m] = [{"id": r["id"], "question": r["question"], "reference_answer": short(r.get("reference_answer"), 200),
                        "response": short(r.get("response"), 200)} for r in refused0[:k]]
        induced[m] = {}
        for c in MUTS:
            rows = eval_rows(cfg, c, rag, m, cache)
            cand = [r for i, r in rows.items() if r.get("hallucinated") is not None and was_refused(cfg, r)
                    and i in base and base[i].get("hallucinated") is not None and not was_refused(cfg, base[i])]
            rng.shuffle(cand)
            induced[m][c] = [{"id": r["id"], "original": base[r["id"]]["question"], "original_response": short(base[r["id"]].get("response"), 200),
                              "prompt": r["question"], "operators": r.get("operators"), "kind": r.get("kind"),
                              "expected_behavior": r.get("expected_behavior"), "response": short(r.get("response"), 200)}
                             for r in cand[:k]]
    return counts, original, induced


def mutated_examples(cfg, rag, method, uids, k, rng, cache):
    rows = {c: {r["id"]: r for r in read_jsonl(condition_file(cfg, c, rag, method))}
            if condition_file(cfg, c, rag, method).exists() else {} for c in MUTS}
    base = {r["id"]: r for r in read_jsonl(condition_file(cfg, "baseline", rag, method))}
    ids = [i for i in base if all(i in rows[c] for c in MUTS) and (not uids or i in uids)]
    if not ids:
        return []
    score = {}
    for i in ids:
        o0 = outcome(cfg, "baseline", rag, method, i, cache)
        flips = 0
        for c in MUTS:
            o = outcome(cfg, c, rag, method, i, cache)
            if o and o0 and o["hallucinated"] and not o0["hallucinated"]:
                flips += 1
        kind = rows["m3"][i].get("kind"), rows["m3"][i].get("expected_behavior")
        score[i] = (flips, kind)
    order = sorted(ids, key=lambda i: (-score[i][0], rng.random()))
    chosen, seen = [], set()
    for i in order:  # prefer examples where a mutation changed the outcome, and different M3 behaviours
        beh = rows["m3"][i].get("expected_behavior")
        if beh not in seen:
            chosen.append(i)
            seen.add(beh)
        if len(chosen) >= k:
            break
    for i in order:
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    out = []
    for i in chosen:
        ex = {"id": i, "method": method, "original": base[i]["question"], "reference_answer": base[i]["answer"],
              "baseline_outcome": outcome(cfg, "baseline", rag, method, i, cache), "mutations": {}}
        for c in MUTS:
            r = rows[c][i]
            ex["mutations"][c] = {"name": MUTATION_INFO[c]["name"], "prompt": r["question"], "operators": r["operators"],
                                  "kind": r.get("kind"), "claim": r.get("claim"), "reference_answer": r["answer"],
                                  "expected_behavior": r["expected_behavior"], "outcome": outcome(cfg, c, rag, method, i, cache)}
        out.append(ex)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rags", default=",".join(RAG_NAMES))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n", type=int, default=3, help="examples per kind")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    seed = args.seed if args.seed is not None else cfg["paper"]["seed"]
    out = ROOT / cfg["paper"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    methods = args.methods.split(",")
    result, md, tex = {}, ["# Examples for the paper", ""], []
    for rag in args.rags.split(","):
        domain = "medical" if rag == "medrag" else "general"
        try:
            real_db = ensure_real_db(cfg, domain)
            run_db = ensure_run_db(cfg, rag, methods)
        except (FileNotFoundError, ValueError) as e:
            print(f"{rag}: {e}. Skipped.")
            continue
        rng = np.random.default_rng(seed)
        an = make_analyzer(cfg, run_db)
        items = run_db.items(kind="benchmark")
        uids = {i["id"] for i in items if (i["meta"] or {}).get("in_union")}
        res = {"simple": {}, "real": [], "entities": [], "mutated": {}}
        md += [f"## {rag}", "", "### A. Simple benchmark queries (one method each)", ""]
        for m in methods:
            pool = [i for i in items if i["method"] == m]
            for it in pick_simple(pool, rng, args.n) if pool else []:
                res["simple"].setdefault(m, []).append({"id": it["id"], "question": it["question"], "answer": short(it["answer"])})
                md += [f"- **{m}** `{it['id']}`: {it['question']}", f"  - reference answer: {short(it['answer'])}"]
        real_items = [r for r in real_db.items(kind="real") if 8 <= len(r["question"].split()) <= 40]
        real_pick = [real_items[i] for i in rng.choice(len(real_items), min(args.n, len(real_items)), replace=False)] if real_items else []
        md += ["", "### B. Realistic queries (real users)", ""]
        for r in real_pick:
            res["real"].append({"id": r["id"], "question": r["question"]})
            md.append(f"- {r['question']}")
        # C entities + graph templates: diverse templates from the union, plus one real query
        md += ["", "### C. Named entities and graph templates", ""]
        pool = [i for i in items if i["id"] in uids] or items
        groups = {}
        for i in pool:
            g = run_db.graph(i["id"], "question")
            if g and g["n_entities"] >= 1 and 6 <= len(i["question"].split()) <= 30:
                groups.setdefault(g["template_id"], []).append(i)
        keys = list(groups)
        rng.shuffle(keys)
        sel = [groups[k][0] for k in keys[: args.n]]
        gentries = []
        for kind, db_, itm in [("benchmark", run_db, s) for s in sel] + [("real", real_db, r) for r in real_pick[:1]]:
            q = db_.graph(itm["id"], "question")
            ents = db_.entities(itm["id"], "question")
            a = db_.graph(itm["id"], "answer") if kind == "benchmark" and itm.get("answer") else None
            aents = db_.entities(itm["id"], "answer") if a else []
            e = {"kind": kind, "id": itm["id"], "text": itm["question"], "entities": [f"{x['text']} <{x['label']}>" for x in ents],
                 "template": q["template"], "template_id": q["template_id"], "shape": q["shape"],
                 "text_template": q["text_template"], "slots": q["slots"], "graph": q["graph_json"],
                 "answer": short(itm.get("answer"), 200) if a else None,
                 "answer_template": a["template"] if a else None, "answer_shape": a["shape"] if a else None,
                 "answer_entities": [f"{x['text']} <{x['label']}>" for x in aents][:8]}
            res["entities"].append(e)
            gentries.append(e)
            md += [f"**{kind}** `{itm['id']}`: {itm['question']}", "",
                   f"- entities: {', '.join(e['entities']) or '(none)'}",
                   f"- text template: `{e['text_template']}`", f"- graph template (`{e['template_id']}`): `{e['template']}`",
                   f"- shape: `{e['shape']}`"]
            if a:
                md += [f"- reference answer: {e['answer']}", f"- answer entities: {', '.join(e['answer_entities']) or '(none)'}",
                       f"- answer template (first sentence): `{e['answer_template']}`; shape `{e['answer_shape']}`"]
            md += ["", "```", ascii_tree(graph_from_json(e["graph"])), "```", ""]
        if gentries:
            draw_graphs([{"text": g["text"], "shape": g["shape"], "graph": g["graph"]} for g in gentries], out / f"example_graphs_{rag}.png")
        # D mutated
        md += ["### D. Mutated queries", ""]
        cache = {}
        for m in methods:
            exs = mutated_examples(cfg, rag, m, uids, args.n, rng, cache)
            res["mutated"][m] = exs
            for ex in exs:
                md += [f"**{m}** `{ex['id']}`", "", f"- original: {ex['original']}"]
                if ex["baseline_outcome"]:
                    md.append(f"  - RAG on the original: hallucinated={ex['baseline_outcome']['hallucinated']}, correct={ex['baseline_outcome']['correct']}, refused={ex['baseline_outcome']['refused']}")
                for c in MUTS:
                    x = ex["mutations"][c]
                    md += [f"- {x['name']} ({c}; operators {', '.join(x['operators'])}"
                           + (f"; {x['kind']} claim" if x['kind'] else "") + f"): {x['prompt']}",
                           f"  - reference answer for this prompt: {short(x['reference_answer'], 160)} (expected: {x['expected_behavior']})"]
                    if x["outcome"]:
                        md.append(f"  - RAG: hallucinated={x['outcome']['hallucinated']}, correct={x['outcome']['correct']}, refused={x['outcome']['refused']}; "
                                  f"response: {x['outcome']['response']}")
                md.append("")
        # E refusals
        md += ["### E. Refusals (\"" + refusal_phrase(cfg) + "\")", ""]
        counts, orig_ref, ind_ref = refusal_part(cfg, rag, methods, uids, args.n, rng, cache)
        res["refusals"] = {"counts": counts, "original": orig_ref, "induced": ind_ref}
        if any(c["evaluated"] for c in counts):
            import csv
            with open(out / f"refusal_counts_{rag}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(counts[0]))
                w.writeheader()
                w.writerows(counts)
            md += ["Answers that are exactly the refusal phrase, per method and for the questions of that method that are in the union:", "",
                   "| scope | condition | evaluated | refused | % |", "|---|---|---:|---:|---:|"]
            for c in counts:
                if c["evaluated"] and c["scope"] == c["method"]:
                    md.append(f"| {c['method']} | {c['condition']} | {c['evaluated']} | {c['refused']} | {100 * c['refused'] / c['evaluated']:.1f} |")
            tot = {}
            for c in counts:
                if c["scope"] == "union":
                    t = tot.setdefault(c["condition"], [0, 0])
                    t[0] += c["evaluated"]
                    t[1] += c["refused"]
            for cond, (n_, r_) in tot.items():
                if n_:
                    md.append(f"| union | {cond} | {n_} | {r_} | {100 * r_ / n_:.1f} |")
            md.append("")
        else:
            md += ["(no evaluation results yet: run the evaluate step first)", ""]
        for m in methods:
            for r in orig_ref.get(m, []):
                md += [f"- **{m}, original question refused** `{r['id']}`: {r['question']}",
                       f"  - reference answer: {r['reference_answer']}; RAG: {r['response']}"]
            for c in MUTS:
                for r in ind_ref.get(m, {}).get(c, []):
                    md += [f"- **{m}, {MUTATION_INFO[c]['name']} ({c}) made it refuse** `{r['id']}`",
                           f"  - original: {r['original']}", f"    - RAG answered: {r['original_response']}",
                           f"  - mutated ({', '.join(r['operators'] or [])}" + (f"; {r['kind']} claim" if r['kind'] else "") + f"): {r['prompt']}",
                           f"    - RAG: {r['response']}"]
        md.append("")
        result[rag] = res
        # LaTeX
        def row(*cells):
            return " & ".join(tex_escape(c) for c in cells) + r" \\"
        t1 = [r"\begin{table*}[t]", r"\centering\small", r"\begin{tabular}{p{0.14\textwidth}p{0.80\textwidth}}", r"\toprule",
              r"Source & Query \\", r"\midrule"]
        for m, lst in res["simple"].items():
            t1.append(row(f"{m} (benchmark)", lst[0]["question"]))
        for r in res["real"][:2]:
            t1.append(row("real user", r["question"]))
        t1 += [r"\bottomrule", r"\end{tabular}", rf"\caption{{Simple benchmark queries and real user queries ({rag}).}}",
               rf"\label{{tab:examples-simple-{rag}}}", r"\end{table*}"]
        t2 = [r"\begin{table*}[t]", r"\centering\small",
              r"\begin{tabular}{p{0.26\textwidth}p{0.22\textwidth}p{0.26\textwidth}p{0.16\textwidth}}", r"\toprule",
              r"Query & Entities & Graph template & Shape \\", r"\midrule"]
        for e in res["entities"]:
            t2.append(row(e["text"], "; ".join(e["entities"]), e["template"], e["shape"]))
        t2 += [r"\bottomrule", r"\end{tabular}", rf"\caption{{Named entities and graph templates of example prompts ({rag}).}}",
               rf"\label{{tab:examples-templates-{rag}}}", r"\end{table*}"]
        t3 = [r"\begin{table*}[t]", r"\centering\small",
              r"\begin{tabular}{p{0.10\textwidth}p{0.22\textwidth}p{0.22\textwidth}p{0.20\textwidth}p{0.20\textwidth}}", r"\toprule",
              r"Method & Original & M1 Elaborate & M2 Colloquial & M3 Premise \\", r"\midrule"]
        for m, exs in res["mutated"].items():
            for ex in exs[:1]:
                t3.append(row(m, ex["original"], ex["mutations"]["m1"]["prompt"], ex["mutations"]["m2"]["prompt"],
                              ex["mutations"]["m3"]["prompt"]))
        t3 += [r"\bottomrule", r"\end{tabular}", rf"\caption{{One benchmark question and its three mutated versions per method ({rag}).}}",
               rf"\label{{tab:examples-mutated-{rag}}}", r"\end{table*}"]
        t4 = []
        have = [c for c in counts if c["scope"] == c["method"] and c["evaluated"]]
        if have:
            t4 = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
                  r"Method & Benchmark & M1 Elaborate & M2 Colloquial & M3 Premise \\", r"\midrule"]
            for m in methods:
                cells = [m]
                for cnd in ["baseline"] + MUTS:
                    x = next((c for c in have if c["method"] == m and c["condition"] == cnd), None)
                    cells.append(f"{x['refused']}/{x['evaluated']}" if x else "-")
                t4.append(row(*cells))
            t4 += [r"\bottomrule", r"\end{tabular}",
                   rf"\caption{{Answers ``{tex_escape(refusal_phrase(cfg))}'' (refused / evaluated) for the original and the mutated questions ({rag}).}}",
                   rf"\label{{tab:refusals-{rag}}}", r"\end{table}"]
        tex += ["\n".join(t1), "\n".join(t2), "\n".join(t3)] + (["\n".join(t4)] if t4 else [])
    (out / "paper_examples.md").write_text("\n".join(md), encoding="utf-8")
    (out / "paper_examples.tex").write_text("\n\n".join(tex), encoding="utf-8")
    (out / "paper_examples.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"Written: {out / 'paper_examples.md'}, paper_examples.tex, paper_examples.json, example_graphs_<rag>.png/.pdf")


if __name__ == "__main__":
    main()
