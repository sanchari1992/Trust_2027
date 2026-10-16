"""Generating test prompts from a benchmark database (scripts/gen_bench.py).

1. SELECT items from the database (how many, which entities / templates, spread over many templates or matched to
   the example prompts the benchmark creator supplied).
2. MUTATE each selected item with the chosen mutations (README section 13): Elaborate (M1), Colloquial (M2),
   Premise (M3), optionally all in one prompt (Composite). Tunable: how many operators are stacked (`depth`),
   which operators, randomness (`random`), probability of a correct user premise, target length.
3. WRITE one row per (item, mutation): the prompt, the reference answer FOR THAT PROMPT, the expected behaviour,
   which operators ran, the entities and graph template of the original question and of the mutated prompt.
Items without a source passage or reference answer can only get checks that need neither (see ASSUMPTIONS in
README section 30.5): Elaborate with `instruction` / `restate` (unverified), Colloquial in full; Premise needs a passage.
"""
import copy
import json
import time
from functools import partial

import numpy as np

from src.benchgen.common import make_rng
from src.mutate.common import MUTATION_INFO, MutCtx, resolve_mutations
from src.mutate.length import mutate_length
from src.mutate.linguistic import check_resources, mutate_linguistic
from src.mutate.prompt import Prompt, merge
from src.mutate.runner import make_row
from src.mutate.userinfo import mutate_userinfo

CALLS = {"m1": 3, "m2": 1, "m3": 3}  # rough LLM calls per item


def parse_yes_no(v):
    s = str(v).strip().lower()
    if s in ("y", "yes", "true", "1", "t"):
        return True
    if s in ("n", "no", "false", "0", "f"):
        return False
    raise ValueError(f"Expected Y or N, got '{v}'")


class DBContext(MutCtx):
    """MutCtx whose "real query lengths" come from the database's real/example prompts, if it has any."""

    def __init__(self, cfg, chat, lengths=None):
        super().__init__(cfg, chat)
        self._db_lengths = lengths or []

    def real_lengths(self, domain):
        return self._db_lengths or super().real_lengths(domain)


# --------------------------------------------------------------------------- selection
def select_items(db, n, mode="diverse", seed=42, kind="benchmark", methods=None, entity_types=None, entities=None,
                 template_ids=None, shapes=None, in_union=False, need_answer=True):
    rng = np.random.default_rng(seed)
    items = [i for i in db.items(kind=kind) if (not methods or i["method"] in methods)]
    if in_union:
        items = [i for i in items if (i["meta"] or {}).get("in_union")]
    if need_answer:
        items = [i for i in items if (i.get("answer") or "").strip()]
    g = {r["item_id"]: r for r in db.graph_rows(kind=kind, role="question")}
    if entity_types or entities:
        keep = set()
        if entity_types:
            q = ",".join("?" * len(entity_types))
            keep |= {r[0] for r in db.ent.execute(
                f"SELECT DISTINCT item_id FROM entities WHERE role='question' AND label IN ({q})",
                [t.upper() for t in entity_types])}
        if entities:
            q = ",".join("?" * len(entities))
            keep |= {r[0] for r in db.ent.execute(
                f"SELECT DISTINCT item_id FROM entities WHERE role='question' AND norm IN ({q})",
                [e.lower() for e in entities])}
        items = [i for i in items if i["id"] in keep]
    if template_ids:
        items = [i for i in items if g.get(i["id"], {}).get("template_id") in set(template_ids)]
    if shapes:
        items = [i for i in items if g.get(i["id"], {}).get("shape") in set(shapes)]
    if not items:
        return []
    n = min(n, len(items)) if n else len(items)
    if mode == "first":
        return items[:n]
    if mode == "random":
        return [items[i] for i in rng.choice(len(items), n, replace=False)]
    if mode == "diverse":  # take turns between graph templates (shuffled), so many different templates appear
        groups = {}
        for i in items:
            groups.setdefault(g.get(i["id"], {}).get("template_id"), []).append(i)
        for v in groups.values():
            rng.shuffle(v)
        order = list(groups)
        rng.shuffle(order)
        out, depth = [], 0
        while len(out) < n:
            added = False
            for t in order:
                if depth < len(groups[t]) and len(out) < n:
                    out.append(groups[t][depth])
                    added = True
            if not added:
                break
            depth += 1
        return out
    if mode == "match-examples":  # sample so that the shapes match the example / real prompts of the database
        ex = db.template_counts(level="shape", kind="example") + db.template_counts(level="shape", kind="real")
        if not ex:
            raise ValueError("--select match-examples needs example or real prompts in the database "
                             "(create_bench.py --add_graphs examples.csv).")
        bench = db.template_counts(level="shape", kind=kind)
        sx, sb = sum(ex.values()), sum(bench.values())
        w = np.array([max((ex.get(g.get(i["id"], {}).get("shape"), 0) / sx) /
                          max(bench.get(g.get(i["id"], {}).get("shape"), 1) / sb, 1e-9), 0.0) + 0.02 for i in items])
        keys = rng.random(len(items)) ** (1.0 / w)  # weighted sampling without replacement
        return [items[i] for i in np.argsort(-keys)[:n]]
    raise ValueError(f"Unknown selection mode '{mode}'")


# --------------------------------------------------------------------------- mutation settings
def build_mutation_cfg(cfg, plan, depth=None, p_correct=None, target_length=None, no_llm=False):
    """Copy of cfg['mutation'] restricted to the chosen operators and with depth / p_correct applied."""
    mc = copy.deepcopy(cfg["mutation"])
    for mid, ops in plan.items():
        fam = MUTATION_INFO[mid]["family"]
        if mid in ("m1", "m2"):
            w = mc[fam]["operators"]
            for op in list(w):
                if ops is not None and op not in ops:
                    w[op] = 0.0
            if mid == "m1" and no_llm:
                for op in ("restate", "background", "appositive"):
                    w[op] = 0.0
            if mid == "m2" and no_llm:
                w["synonyms"] = 0.0
        else:
            c, i = mc[fam]["correct_operators"], mc[fam]["incorrect_operators"]
            if ops is not None:
                for op in list(c):
                    c[op] = c[op] if op in ops else 0.0
                for op in list(i):
                    i[op] = i[op] if op in ops else 0.0
                has_c, has_i = any(v > 0 for v in c.values()), any(v > 0 for v in i.values())
                if has_c and not has_i:
                    mc[fam]["p_correct"] = 1.0
                elif has_i and not has_c:
                    mc[fam]["p_correct"] = 0.0
    if p_correct is not None and (plan.get("m3") is None):
        mc["userinfo"]["p_correct"] = p_correct
    if depth is not None:
        mc["length"]["n_ops"] = depth
        mc["length"]["target"]["mode"] = "none"
        mc["linguistic"]["n_ops"] = depth
    elif target_length:
        if target_length == "real":
            mc["length"]["target"]["mode"] = "real"
        elif target_length == "none":
            mc["length"]["target"]["mode"] = "none"
        elif target_length.startswith("range:"):
            lo, hi = target_length[6:].split("-")
            mc["length"]["target"].update(mode="range", range=[int(lo), int(hi)])
    return mc


def _restrict_no_passage(mc):
    """Without a passage / answer the LLM-written, passage-based operators cannot be checked: use only the ones
    that can (instruction, restate) and switch the LLM check off."""
    mc = copy.deepcopy(mc)
    for op in ("background", "appositive"):
        mc["length"]["operators"][op] = 0.0
    mc["length"]["verify"] = False
    return mc


# --------------------------------------------------------------------------- generation
def generate_item(item, plan, ctx, mc, seed, combine, analyzer=None):
    rng = lambda tag: make_rng(seed, item["id"], tag)  # noqa: E731
    has_src = bool(item.get("contexts")) and bool((item.get("answer") or "").strip())
    mci = mc if has_src else _restrict_no_passage(mc)
    base = Prompt(core=item["question"])
    rows, skipped = {}, {}
    r1 = r2 = r3 = None
    if "m1" in plan:
        r1 = mutate_length(item, ctx, mci["length"], rng("m1"))
        if r1["ok"]:
            rows["m1"] = make_row(item, "m1", r1["prompt"], [f"m1:{o}" for o in r1["operators"]], item.get("answer"),
                                  "answer", False, {**r1["info"], "verified": has_src and mci["length"]["verify"]})
        else:
            skipped["m1"] = r1["reason"]
    if "m2" in plan:
        r2 = mutate_linguistic(base, ctx, mci["linguistic"], rng("m2"))
        if r2["ok"]:
            rows["m2"] = make_row(item, "m2", r2["prompt"], [f"m2:{o}" for o in r2["operators"]], item.get("answer"),
                                  "answer", False, r2["info"])
        else:
            skipped["m2"] = r2["reason"]
    p3 = None
    if "m3" in plan:
        if not has_src:
            skipped["m3"] = "needs_passage_and_reference_answer"
        else:
            r3 = mutate_userinfo(item, ctx, mci["userinfo"], rng("m3"))
            if r3["ok"]:
                p3 = Prompt(pre=r3["pre"], core=item["question"], post=r3["post"])
                rows["m3"] = make_row(item, "m3", p3, [f"m3:{r3['operator']}"], r3["answer"], r3["behavior"],
                                      r3["answer_changed"], r3["info"], kind=r3["kind"], claim=r3["claim"])
            else:
                skipped["m3"] = r3["reason"]
    if combine and len(plan) > 1:  # Composite: the M1 and M3 sentences around the question, M2 on the whole prompt
        merged = Prompt(core=item["question"])
        ops, behavior, answer, changed, kind, claim = [], "answer", item.get("answer"), False, None, None
        if r1 and r1["ok"]:
            merged = merge(merged, r1["prompt"])
            ops += [f"m1:{o}" for o in r1["operators"]]
        if p3 is not None:
            merged = merge(merged, p3)
            ops.append(f"m3:{r3['operator']}")
            behavior, answer, changed, kind, claim = r3["behavior"], r3["answer"], r3["answer_changed"], r3["kind"], r3["claim"]
        if "m2" in plan:
            rc = mutate_linguistic(merged, ctx, mci["linguistic"], rng("m123"),
                                   fixed_ops=r2["operators"] if r2 and r2["ok"] else None)
            if rc["ok"]:
                merged = rc["prompt"]
                ops += [f"m2:{o}" for o in rc["operators"]]
            else:
                skipped["m123"] = f"m2_on_combined_failed({rc['reason']})"
        if "m123" not in skipped and ops:
            rows["m123"] = make_row(item, "m123", merged, ops, answer, behavior, changed, {}, kind=kind, claim=claim)
    return rows, skipped


def run_generation(db, items, plan, cfg, mc, seed, combine=False, no_llm=False, analyzer=None, progress=True):
    check_resources(mc["linguistic"])
    lengths = db.example_lengths()
    if no_llm:
        def chat(*a, **k):
            raise RuntimeError("an LLM call was needed but --no-llm is set")
    else:
        from src.llm import get_chat_func

        chat = partial(get_chat_func(cfg), temperature=mc["temperature"])
    ctx = DBContext(cfg, chat, lengths)
    out, skipped_all = [], {}
    for k, item in enumerate(items, 1):
        rows, skipped = generate_item(dict(item), plan, ctx, mc, seed, combine)
        g = db.graph(item["id"], "question") or {}
        ents = [e["text"] + "/" + e["label"] for e in db.entities(item["id"], "question")]
        for cond, r in rows.items():
            rec = {
                "prompt_id": f"{item['id']}:{cond}", "source_id": item["id"], "mutation": cond,
                "mutation_name": MUTATION_INFO[cond]["name"] if cond in MUTATION_INFO else "Composite",
                "prompt": r["question"], "original_question": r["original_question"],
                "reference_answer": r["answer"], "original_answer": r["original_answer"],
                "expected_behavior": r["expected_behavior"], "answer_changed": r["answer_changed"],
                "operators": r["operators"], "kind": r["kind"], "claim": r["claim"],
                "words_original": r["words_original"], "words_prompt": r["words_mutated"],
                "entities": ents, "template_id": g.get("template_id"), "template": g.get("template"),
                "shape": g.get("shape"), "text_template": g.get("text_template"),
                "method": item.get("method"), "passage_ids": item.get("passage_ids"), "seed": seed, "meta": r["meta"],
            }
            if analyzer is not None:
                a = analyzer.analyze(r["question"])
                rec.update(prompt_template_id=a["template_id"], prompt_shape=a["shape"],
                           prompt_text_template=a["text_template"],
                           prompt_entities=[e["text"] + "/" + e["label"] for e in a["entities"]])
            out.append(rec)
        for cond, why in skipped.items():
            skipped_all.setdefault(cond, {}).setdefault(why.split("(")[0].split(":")[-1][:60], 0)
            skipped_all[cond][why.split("(")[0].split(":")[-1][:60]] += 1
        if progress and k % 25 == 0:
            print(f"  {k}/{len(items)} items", flush=True)
    return out, skipped_all


def write_rows(rows, path, fmt=None):
    path = str(path)
    fmt = fmt or ("csv" if path.endswith(".csv") else "jsonl")
    if fmt == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        import csv

        cols = list(rows[0]) if rows else []
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                            for v in (r[c] for c in cols)])
    return path


def default_seed(random, seed, cfg):
    if seed is not None:
        return int(seed)
    return int(time.time()) % 2**31 if random else int(cfg["mutation"]["seed"])
