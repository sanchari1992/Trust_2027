"""Generate test prompts from a benchmark database (README section 30).

For people who TEST a RAG (software engineers, auditors): take questions from a database made by create_bench.py and
turn them into harder, more realistic prompts with the mutations of README section 13. Every output row carries the
reference answer for the mutated prompt and the behaviour that counts as correct.

    python scripts/gen_bench.py --db medical.benchmark --n 200 --mutation all --out prompts.jsonl
    python scripts/gen_bench.py --db medical.benchmark --n 100 --depth=3 --mutation=elaborate,colloquial --random=Y --out p.jsonl
    python scripts/gen_bench.py --db medical.benchmark --n 50 --mutation premise --entity-types PERSON,ORG --out p.csv
    python scripts/gen_bench.py --db medical.benchmark --n 50 --mutation typos,casual,restate --no-llm --out p.jsonl
    python scripts/gen_bench.py --list-mutations

MUTATIONS (--mutation, comma separated; names, ids, or single operators; 'all'):
    elaborate (m1, length)       restate | background | instruction | appositive      longer, on-topic prompts
    colloquial (m2, linguistic)  typos | casual | first_person | politeness | imperative | synonyms
    premise (m3, userinfo)       true_premise | correct_candidate | false_premise | wrong_candidate | unsupported_assumption | patient_consistent | patient_conflict
    --combine                    also write the Composite prompt: every chosen mutation in ONE prompt
TUNING:
    --depth N        operators stacked per prompt in Elaborate and Colloquial (Premise always adds one claim)
    --random Y|N     Y = new randomness each run (seed from the clock unless --seed); N = reproducible (seed from config)
    --seed S         fix the seed (overrides --random's clock seed)
    --p-correct P    Premise: share of correct user claims (the rest wrong or unsupported)
    --target-length  real (default) | none | range:25-80 : how long Elaborate should make the prompt (ignored with --depth)
SELECTION (which items are mutated):
    --n N            number of items       --select diverse | random | first | match-examples
    --entity-types A,B   items with a question entity of these types (PERSON, ORG, GPE, CONCEPT, ...)
    --entities x,y       items whose question mentions these entities     --template ID / --shape S   graph template / shape
    --methods ragas,ares   --in-union   items of these generator methods / of the union only
OTHER:
    --no-llm         only rule-based operators (instruction, all of Colloquial except synonyms): no API key needed
    --dry-run        show the plan and the LLM-call estimate
    --format jsonl|csv   (default from the file extension)
Needs: the database, `python -m spacy download en_core_web_sm`; DEEPSEEK_API_KEY in .env unless --no-llm.
Items without a passage and reference answer get only the checks that need neither (README section 30.5).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchdb.analyze import Analyzer
from src.benchdb.generate import (CALLS, build_mutation_cfg, default_seed, parse_yes_no, run_generation,
                                  select_items, write_rows)
from src.benchdb.store import BenchDB
from src.config import load_config
from src.mutate.common import MUTATION_INFO, resolve_mutations


def list_mutations():
    for mid, i in MUTATION_INFO.items():
        print(f"{i['name']:<11} ({mid}, {i['family']}): {i['what']}\n    operators: {', '.join(i['operators'])}")
    print("Composite   (m123): every chosen mutation in one prompt (--combine)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db")
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mutation", default="all")
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--random", default="N")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--p-correct", type=float, default=None)
    ap.add_argument("--target-length", default=None)
    ap.add_argument("--select", choices=["diverse", "random", "first", "match-examples"], default="diverse")
    ap.add_argument("--entity-types", default=None)
    ap.add_argument("--entities", default=None)
    ap.add_argument("--template", default=None)
    ap.add_argument("--shape", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--in-union", action="store_true")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--format", choices=["jsonl", "csv"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-mutations", action="store_true")
    args = ap.parse_args()
    if args.list_mutations:
        return list_mutations()
    if not args.db:
        sys.exit("Give --db <database>.benchmark (see scripts/create_bench.py), or --list-mutations.")
    try:
        plan = resolve_mutations(args.mutation)
        rnd = parse_yes_no(args.random)
    except ValueError as e:
        sys.exit(str(e))
    if not plan:
        sys.exit("No mutation chosen.")
    cfg = load_config()
    if args.no_llm and "m3" in plan:
        if args.mutation.strip().lower() == "all":
            plan.pop("m3")
            print("note: --no-llm cannot write Premise claims; Premise left out.")
        else:
            sys.exit("--no-llm cannot be used with Premise (m3): its claims are written and checked by an LLM.")
    seed = default_seed(rnd, args.seed, cfg)
    db = BenchDB.open(args.db)
    splitc = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
    items = select_items(db, args.n, args.select, seed, "benchmark", splitc(args.methods),
                         splitc(args.entity_types), splitc(args.entities), splitc(args.template), splitc(args.shape),
                         args.in_union, need_answer=False)
    if not items:
        sys.exit("No items match the selection (check --entity-types / --template / --methods, and that the "
                 "database holds benchmark prompts).")
    mc = build_mutation_cfg(cfg, plan, args.depth, args.p_correct, args.target_length, args.no_llm)
    names = ", ".join(f"{MUTATION_INFO[m]['name']}" + (f"[{','.join(o)}]" if o else "") for m, o in plan.items())
    calls = 0 if args.no_llm else sum(CALLS[m] for m in plan)
    print(f"database: {db.path} ({db.counts().get('benchmark', 0)} benchmark prompts)")
    print(f"selected {len(items)} items ({args.select}); mutations: {names}"
          f"{' + Composite' if args.combine and len(plan) > 1 else ''}")
    print(f"depth: {args.depth if args.depth is not None else 'config'} | random: {'Y' if rnd else 'N'} | seed: {seed}")
    print(f"LLM calls: about {calls * len(items):,}")
    if args.dry_run:
        print("(dry run: nothing generated)")
        return
    an = Analyzer(db.manifest.get("spacy_model") or cfg["real_queries"]["spacy_model"],
                  concepts=db.manifest.get("concepts", True), answer_chars=cfg["benchdb"]["answer_chars"])
    rows, skipped = run_generation(db, items, plan, cfg, mc, seed, args.combine, args.no_llm, an)
    if not rows:
        sys.exit(f"Nothing was generated. Why: {skipped}")
    write_rows(rows, args.out, args.format)
    by = {}
    for r in rows:
        by[r["mutation_name"]] = by.get(r["mutation_name"], 0) + 1
    print(f"\nWrote {len(rows)} prompts to {args.out}: {by}")
    if skipped:
        print("Not generated (item, reason):", skipped)


if __name__ == "__main__":
    main()
