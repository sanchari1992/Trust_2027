# Command cheat sheet

Run from the repository root. `DEEPSEEK_API_KEY` must be in `.env`; install the spaCy model once with
`python -m spacy download en_core_web_sm`. Details: README sections 29 to 33.

## A. Medical RAG (MedRAG): the building blocks

```bash
# 0. one-time setup: corpus and index of the medical RAG (local, no LLM calls; a GPU helps)
python scripts/prepare_corpora.py medrag
python scripts/build_indexes.py medrag
python scripts/test_rag.py medrag                                   # smoke test

# 1. benchmark creators: generate questions with the three methods (prompt + response)
python scripts/generate_benchmark.py medrag ragas --n 200
python scripts/generate_benchmark.py medrag ares --n 200
python scripts/generate_benchmark.py medrag datamorgana --n 200

# 2. benchmark creators: store prompts and responses as a database of named entities + graph templates
python scripts/create_bench.py --in_file questions.csv --out medical.benchmark        # your own csv (question, answer, context)
python scripts/create_bench.py --from-run medrag                                      # or: the benchmark generated above
python scripts/prepare_real_queries.py medical
python scripts/create_bench.py --real medical                                         # database of REAL patient queries
python scripts/create_bench.py --in medical.benchmark --add_graphs examples.csv       # add example prompts' graph templates
python scripts/create_bench.py --info medical.benchmark                               # look inside
python scripts/create_bench.py --in medical.benchmark --export                        # entities.csv, graphs.jsonl, templates.csv

# 3. testers: generate prompts from the database (mutations: elaborate, colloquial, premise)
python scripts/gen_bench.py --list-mutations
python scripts/gen_bench.py --db medical.benchmark --n 200 --mutation all --out prompts.jsonl
python scripts/gen_bench.py --db medical.benchmark --n 100 --depth=3 --mutation=elaborate,colloquial --random=Y --out p.jsonl
python scripts/gen_bench.py --db medical.benchmark --n 100 --mutation premise --combine --out p.csv
python scripts/gen_bench.py --db medical.benchmark --n 100 --mutation typos,casual,first_person --no-llm --out p.jsonl   # no API key
python scripts/gen_bench.py --db medical.benchmark --n 100 --entity-types CONCEPT --select match-examples --mutation all --out p.jsonl
python scripts/gen_bench.py --db medical.benchmark --n 50 --mutation all --dry-run                                      # plan + cost only

# 4. run those mutations on the medical RAG and measure hallucinations (the pipeline's own questions)
python scripts/mutate_benchmark.py medrag ragas --limit 200
python scripts/evaluate_benchmarks.py medrag ragas --limit 200
python scripts/analyze_eval.py && python scripts/summary_counts.py
```

## B. The paper run: 10,000 questions per method, union, mutations, one report

```bash
# 0. one-time: corpus, index, real queries (medical = patients' queries; general only if you also run LightRAG / RAPTOR)
export BP_PROFILE=paper                                  # or add --profile paper to run_all.py
python scripts/prepare_corpora.py medrag
python scripts/build_indexes.py medrag
python scripts/prepare_real_queries.py medical

# 1. see the plan and the cost BEFORE spending anything (also shows how many questions each method can produce)
python scripts/run_all.py --profile paper --rags medrag --dry-run

# 2. a small pilot first, profile `pilot` (its own folders results/*_pilot: it never mixes with earlier runs or the paper run)
python scripts/run_all.py --profile pilot --rags medrag --n 100 --limit 50 --steps real,generate,union,db,mutate,evaluate,analyze,audit,report,examples

# 3. the paper run, step by step (every step resumes if interrupted)
python scripts/run_all.py --profile paper --rags medrag --steps generate          # 10,000 per method (RAGAS, ARES, DataMorgana)
python scripts/run_all.py --profile paper --rags medrag --steps union,db,stats    # union (no LLM), entity+graph databases, corpus sizes
python scripts/run_all.py --profile paper --rags medrag --steps mutate --mutate-n 3000   # Elaborate, Colloquial, Premise
python scripts/run_all.py --profile paper --rags medrag --steps evaluate --eval-n 1000   # baseline, m1, m2, m3 + 2 controls
#    (the RAG now answers "Cannot be answered from given information." when its documents lack the information; results from an
#     OLDER evaluation of the same folder used another prompt: re-run with --fresh or use a new evaluation.output_dir. README 14.10)
python scripts/run_all.py --profile paper --rags medrag --steps analyze,audit
python scripts/summary_counts.py && python scripts/transition_report.py && python scripts/plot_transitions.py
python scripts/run_all.py --profile paper --rags medrag --steps report,examples   # THE report + examples for the paper

# or everything in one command
python scripts/run_all.py --profile paper --rags medrag --mutate-n 3000 --eval-n 1000

# all three RAGs: drop --rags (LightRAG and RAPTOR need their own indexes and the general real queries)
```
Outputs: `results/paper_final/` (`paper_report.md`, `paper_tables.tex`, `story_medrag.csv`, `paper_examples.md/.tex`,
`example_graphs_medrag.pdf`, `audit_sheet_mutations.csv`, ...), `results/eval_paper/_report/` (the evaluation tables and figures).
For the realism numbers in the paper use real queries you did not design the mutations on:
`python scripts/compare_mutations.py --real-file data/real_queries/medical_heldout.jsonl --real-domain medical`.

Refusals (README 14.10): the counts of "Cannot be answered from given information." are in `summary_counts.md` (Refusals table), `refusals.csv`, `paper_report.md` (section 4) and `paper_examples.md` (part E).
To switch the sentence off: `medrag.refusal_instruction: false` in `config.yaml`.

Mutation web page / API (README 34): `python scripts/serve_mutator.py` then open http://127.0.0.1:8000 (API docs at /docs). Needs `pip install fastapi uvicorn`.

Patient profiles (README 13.13): the `paper` and `pilot` profiles now include the `patient_consistent` / `patient_conflict` Premise operators. Mutate with `--fresh` after upgrading. Generate only these from a database: `python scripts/gen_bench.py --db my.benchmark --mutation patient_conflict`. In the web page: "Premise adds ... a patient profile".

Two LLMs (README 14.12): DeepSeek generates (`llm:`), GPT-6 Luna judges (`evaluation.judge_llm`). Put both keys in `.env`: `DEEPSEEK_API_KEY=...` and `OPENAI_API_KEY=...`. Check the judge against a reference first: `python scripts/judge_agreement.py --profile pilot --rag medrag --n 200` (run the evaluate step with `evaluation.store_contexts: true` before). Changing the judge redoes the stored evaluation results. `judge_llm: null` = judge with DeepSeek as before.

Update-37 (README 35): M1 and the re-ask/padding controls are off; conditions are baseline, m2, m3. New complexity features; length is no longer a realism measure. Deviation tables: `python scripts/deviation_report.py --profile pilot` (add `--real-file <held-out>` for the paper). Rerun the pilot: delete `results\*_pilot`, then `bash run_paper.sh --pilot --n 20`.

Update-38 (README 36, docs/WEBPAGE.md): the web page shows the prompt graph and what each mutation changes. Patient-conflict scoring fixed; patient profiles only on questions. Delete `results\\*_pilot` before re-running.

Update-39: `python scripts/prepare_real_queries.py medical` now also writes data/real_queries/medical_heldout.jsonl (300 queries, removed from medical.jsonl). run_paper.sh runs it for you; then use `--heldout data/real_queries/medical_heldout.jsonl`.

Update-39b: `prepare_real_queries.py medical` no longer downloads when data/real_queries/medical.jsonl exists: it only makes medical_heldout.jsonl from it (same seed). `--refresh` forces a new download (the HealthChat file name is tried under its new name too).
