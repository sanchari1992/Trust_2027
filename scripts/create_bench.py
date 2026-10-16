"""Turn benchmark prompts and responses into a DATABASE of named entities and graph templates (README section 29).

For benchmark CREATORS (doctors, domain experts, anyone who has a set of questions and answers):

    # 1. a file of questions (and optionally answers + source passages) -> a benchmark database
    python scripts/create_bench.py --in_file questions.csv --out medical.benchmark

    # 2. add example prompts (e.g. what real patients ask) so their graph templates are in the database too
    python scripts/create_bench.py --in medical.benchmark --add_graphs examples.csv

    # look inside
    python scripts/create_bench.py --info medical.benchmark
    python scripts/create_bench.py --in medical.benchmark --export          # entities.csv, graphs.jsonl, templates.csv

For the PIPELINE of this repository:

    python scripts/create_bench.py --real medical                           # real user queries -> real_medical.benchmark
    python scripts/create_bench.py --from-run medrag                        # the generated benchmark (all methods) of the medical RAG
    python scripts/create_bench.py --from-run medrag --methods ragas,ares   # only some methods

A database is a folder: entities.sqlite (prompts, answers, passages, named entities) and graphs.sqlite (one dependency
graph + template per prompt and per answer, and a template library). Input columns are detected (question / prompt /
query / text; answer / response / reference_answer; context / passage / document; id) or named with --question-col,
--answer-col, --context-col, --id-col. Accepts .csv .tsv .json .jsonl.

Options:
    --out PATH         database folder (default: <benchdb.dir>/<input name>.benchmark)
    --kind KIND        benchmark (default) | real | example : what the prompts are
    --overwrite        start the database over instead of adding to it
    --spacy-model M    entity / graph model (default: benchdb.spacy_model or real_queries.spacy_model)
    --no-concepts      store named entities only, not noun-phrase concepts
    --no-passages      do not analyse the source passages
    --export           also write the CSV / JSONL exports
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchdb.analyze import Analyzer
from src.benchdb.pipeline import real_db_path, run_db_path
from src.benchdb.readers import read_prompts, read_real, read_run
from src.benchdb.store import BenchDB
from src.benchgen.runner import METHODS
from src.config import ROOT, load_config


def show_info(db):
    c = db.counts()
    print(f"\n{db.path}")
    print("  counts:", c)
    print("  history:", db.manifest.get("history", [])[-3:])
    for kind in ("benchmark", "real", "example"):
        if c.get(kind):
            sh = db.template_counts(level="shape", kind=kind)
            print(f"  [{kind}] {c[kind]} prompts, {len(db.template_counts(kind=kind))} distinct templates, "
                  f"{len(sh)} shapes; commonest shapes: {sh.most_common(3)}")
    print("  entity types (questions):", dict(db.entity_label_counts("question").most_common(8)))
    print("  most frequent entities:", [(e["norm"], e["label"], e["n"]) for e in db.top_entities(n=8)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_file", "--in-file", "--in", dest="in_file", default=None,
                    help="a questions file (.csv .tsv .json .jsonl) to turn into a database, or an existing .benchmark folder to add to")
    ap.add_argument("--add_graphs", "--add-graphs", dest="add_graphs", default=None,
                    help="a file of example prompts whose graph templates are added to the database in --in")
    ap.add_argument("--out", default=None)
    ap.add_argument("--kind", choices=["benchmark", "real", "example"], default="benchmark")
    ap.add_argument("--real", choices=["general", "medical"], default=None, help="build the real-query database")
    ap.add_argument("--from-run", dest="from_run", default=None, metavar="RAG", help="build from a generated benchmark")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--info", default=None, metavar="DB")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--question-col")
    ap.add_argument("--answer-col")
    ap.add_argument("--context-col")
    ap.add_argument("--id-col")
    ap.add_argument("--spacy-model")
    ap.add_argument("--no-concepts", action="store_true")
    ap.add_argument("--no-passages", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    args = ap.parse_args()
    cfg = load_config()
    bc = cfg["benchdb"]

    if args.info:
        db = BenchDB.open(args.info)
        show_info(db)
        return

    def analyzer():
        return Analyzer(args.spacy_model or bc.get("spacy_model") or cfg["real_queries"]["spacy_model"],
                        concepts=bc["concepts"] and not args.no_concepts, answer_chars=bc["answer_chars"])

    use_passages = bc["passage_entities"] and not args.no_passages
    in_path = Path(args.in_file) if args.in_file else None
    for label, f in (("--in_file", in_path), ("--add_graphs", Path(args.add_graphs) if args.add_graphs else None)):
        if f is not None and not f.exists():
            sys.exit(f"{label} {f}: no such file. Input files are never created; they are YOUR files (CSV/JSONL with a question column).\n"
                     "To build from the benchmarks this project generated, use:  python scripts/create_bench.py --from-run medrag")

    # ---- step 2: add example prompts to an existing database
    if args.add_graphs:
        if not in_path or not in_path.is_dir():
            sys.exit("--add_graphs needs --in <database>.benchmark (the database to add the example templates to).")
        db = BenchDB.open(in_path)
        rows = read_prompts(args.add_graphs, args.question_col, None, None, args.id_col, prefix="ex")
        rows = [{**r, "id": "example-" + r["id"], "answer": None, "contexts": []} for r in rows]
        if args.limit:
            rows = rows[:args.limit]
        n = db.add_items(rows, analyzer(), kind="example", passage_entities=False)
        print(f"Added {n} example prompts to {db.path}.")
        show_info(db)
        if args.export:
            print("Exported to", db.export())
        return

    # ---- step 1: build a database
    if args.real:
        out = Path(args.out) if args.out else real_db_path(cfg, args.real)
        rows, kind = read_real(cfg, args.real), "real"
    elif args.from_run:
        out = Path(args.out) if args.out else run_db_path(cfg, args.from_run)
        rows, kind = read_run(cfg, args.from_run, args.methods.split(",")), "benchmark"
    elif in_path and in_path.is_dir():
        sys.exit(f"{in_path} is already a database. To add prompts to it use --add_graphs examples.csv.")
    elif in_path:
        out = Path(args.out) if args.out else ROOT / bc["dir"] / f"{in_path.stem}.benchmark"
        rows, kind = read_prompts(in_path, args.question_col, args.answer_col, args.context_col, args.id_col), args.kind
    else:
        sys.exit("Nothing to do. Give --in_file FILE, --real DOMAIN, --from-run RAG, or --info DB (see --help).")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("No prompts found.")
    print(f"{len(rows):,} prompts ({kind}) -> {out}")
    if out.exists() and not args.overwrite:
        db = BenchDB.open(out)
        print("Database exists: adding the prompts that are not in it yet (use --overwrite to start over).")
    else:
        db = BenchDB.create(out, overwrite=args.overwrite, info={"source": str(in_path or args.real or args.from_run)})
    db.add_items(rows, analyzer(), kind=kind, passage_entities=use_passages)
    show_info(db)
    if args.export:
        print("Exported to", db.export())
    print(f"\nDatabase: {db.path}")
    print("Next: python scripts/create_bench.py --in", db.path, "--add_graphs examples.csv   (optional)")
    print("      python scripts/gen_bench.py --db", db.path, "--n 100 --mutation all --out prompts.jsonl")


if __name__ == "__main__":
    main()
