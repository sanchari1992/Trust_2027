# benchmark-plus

Code for a study of RAG benchmarks. This repository currently contains two building blocks:

1. **Three published RAG systems**, each on its own real corpus: **LightRAG** (knowledge graph), **RAPTOR** (tree of summaries) and **MedRAG** (medical textbooks, flat retrieval).
2. **Three published benchmark-generation methods** that create questions from those corpora: **RAGAS**, **ARES** and **DataMorgana**.

All language-model calls go to DeepSeek; all embeddings run locally. The later steps of the project (comparing generated questions with real user queries, mutating them into Benchmark+, and DSRS compression) are not in this repository yet.

---

## 1. Quick start

```bash
# 1. Set up (once)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # then open .env and paste your DeepSeek key

# 2. Download the corpora (once, or again after changing the corpora: settings)
python scripts/prepare_corpora.py all

# 3. Check that each RAG works (builds its index, then asks one question)
python scripts/test_rag.py lightrag
python scripts/test_rag.py raptor
python scripts/test_rag.py medrag

# 4. Generate benchmark questions
python scripts/generate_benchmark.py lightrag ragas --dry-run    # plan + rough cost, calls nothing
python scripts/generate_benchmark.py lightrag ragas --n 10       # a first small batch: read the questions!
python scripts/generate_benchmark.py all all                     # every RAG x every method
```

Notes:

- **Do not** install the `requirements.txt` files inside RAPTOR's or MedRAG's own repositories: they pin old library versions that conflict with LightRAG. Only this repository's `requirements.txt` is needed.
- The first run downloads models automatically: the embedding model, the two MedCPT encoders (MedRAG), and a tokenizer vocabulary (LightRAG and RAPTOR). This needs internet access and some disk space.
- Building an index calls DeepSeek many times (LightRAG and RAPTOR) or runs a local model over every snippet (MedRAG, which uses a GPU if one is available). Start with the small default sizes in `config.yaml`.
- Everything is driven by `config.yaml`. You should rarely need to edit the code.

---

## 2. Project layout

```
benchmark-plus/
├── config.yaml                  every setting (section 4); profiles: at the bottom (section 12)
├── .env                         your API keys (create it from .env.example; never commit it)
├── requirements.txt
├── configs/                     inputs of the benchmark methods (section 6)
│   ├── ares_fewshot_general.json / ares_fewshot_medical.json
│   └── datamorgana_general.json / datamorgana_medical.json
├── scripts/
│   ├── prepare_corpora.py       downloads the corpora
│   ├── build_indexes.py         builds each RAG's index
│   ├── test_rag.py              smoke test of one RAG
│   ├── generate_benchmark.py    creates benchmark questions (RAGAS, ARES, DataMorgana)
│   ├── corpus_stats.py          corpus sizes
│   ├── prepare_real_queries.py  downloads real user queries
│   ├── compare_queries.py       real queries vs one question set
│   ├── mutate_benchmark.py      creates Benchmark+ (mutations 1, 2, 3)         section 13
│   ├── evaluate_benchmarks.py   runs every condition on the RAGs and judges    section 14
│   ├── analyze_eval.py          tables, tests, figure                          section 14
│   ├── compare_mutations.py     are mutated questions closer to real ones?     section 14.5
│   └── run_all.py               the whole pipeline in order                    section 15
├── src/
│   ├── config.py                reads config.yaml (+ profiles), builds folder names
│   ├── llm.py                   the ONLY place that talks to DeepSeek
│   ├── embeddings.py            local embedding models
│   ├── rags/                    one file per RAG, all with the same functions
│   │   ├── base.py, registry.py
│   │   └── lightrag_rag.py, raptor_rag.py, medrag_rag.py
│   ├── benchgen/                the benchmark generators (ragas_gen, ares_gen, datamorgana_gen, runner, common)
│   ├── analysis/                graph + lexical features, real-vs-generated comparison
│   ├── mutate/                  the three mutations: length.py, linguistic.py, userinfo.py, verify.py, runner.py
│   └── evalrun/                 judge.py (faithfulness, correctness), runner.py, analysis.py
│
│  Created while you work (all ignored by git):
├── data/corpora/<rag>_<tag>/    the downloaded corpora
├── data/real_queries/           real user queries
├── data/raw/                    raw downloads (QuALITY)
├── third_party/raptor/          RAPTOR's own code
├── indexes/<rag>_<tag>/         the built RAG indexes
└── results/
    ├── benchmarks/<rag>_<tag>/  generated questions   (benchmarks_large/ for the large profile)
    ├── mutations/               mutation1_length/ mutation2_linguistic/ mutation3_userinfo/ mutation123_combined/
    ├── eval/                    per-question results; _report/ holds the analysis
    └── analysis/                real-vs-generated comparisons
```

`<tag>` describes the corpus and its size, for example `lightrag_mix_10` (UltraDomain "mix", 10 documents), `raptor_quality_dev_10` or `medrag_textbooks_3`. When a size setting is `null`, the tag says `all` instead of a number. Different corpus settings therefore get different folders automatically.

---

## 3. Commands

### `scripts/prepare_corpora.py`

```bash
python scripts/prepare_corpora.py lightrag     # or: raptor | medrag | all
```

| Argument | What it downloads | From |
|---|---|---|
| `lightrag` | UltraDomain documents, one text file each (duplicates removed) | Hugging Face, `TommyChien/UltraDomain` |
| `raptor` | QuALITY articles, one text file each, plus RAPTOR's own code into `third_party/raptor` (its exact commit is printed: record it in your paper) | GitHub, `nyu-mll/quality` and `parthsarthi03/raptor` |
| `medrag` | The MedRAG Textbooks corpus (short snippets, in `chunk/*.jsonl`) | Hugging Face, `MedRAG/textbooks` |
| `all` | All of the above | |

How much is kept is set by the `corpora:` section. Re-running with different `corpora:` settings creates a new folder and keeps the old one.

### `scripts/test_rag.py`

```bash
python scripts/test_rag.py raptor
python scripts/test_rag.py medrag --question "Which nerve supplies the diaphragm?"
```

Builds the index if it does not exist yet (this is the slow, and for LightRAG/RAPTOR the costly, part; it is skipped on later runs), asks one question, and prints the answer, the context the model saw, and the original source passages among them. It prints `TEST PASSED` if the answer is non-empty and something was retrieved.

With RAPTOR, a broad question can retrieve only LLM-written summaries and no original text. The script then prints a note. This is normal; a more specific question retrieves source passages too.

### `scripts/generate_benchmark.py`

```bash
python scripts/generate_benchmark.py RAG METHOD [options]
```

`RAG` is `lightrag`, `raptor`, `medrag` or `all`. `METHOD` is `ragas`, `ares`, `datamorgana` or `all`.

| Option | Meaning |
|---|---|
| `--dry-run` | Print the plan (number of passages, benchmark size, expected LLM calls and a rough token estimate) and stop. Costs nothing. |
| `--n 100` | Exactly 100 questions (sets `size.mode: count` for this run). |
| `--rate 3.0` | 3.0 questions per 1,000 corpus words (sets `size.mode: ratio` for this run). |
| `--words 200` | Passage size in words, for the chosen method(s). |
| `--sampling random` | `spread`, `random` or `sequential`. |
| `--seed 7` | Random seed. |
| `--fresh` | Delete the earlier results of this RAG and method and start over (needed after changing most settings, see section 4.9). |

These options only affect the current run; they do not edit `config.yaml`. A run that is interrupted (Ctrl+C, network error, closed laptop) can simply be started again with the same command: it continues where it stopped.

---

## 4. Configuration reference (`config.yaml`)

The defaults below are read from `config.yaml` when this README is generated.

### 4.1 `llm`: the language model

| Setting | Default | What it does |
|---|---|---|
| `llm.model` | `deepseek-flash` | The DeepSeek model used for every LLM call: building indexes, answering questions, generating benchmark questions. DeepSeek renames models from time to time; if calls fail with a "model not found" error, check https://api-docs.deepseek.com/updates/. |
| `llm.base_url` | `https://api.deepseek.com` | Address of the (OpenAI-compatible) API. Change it, together with `model` and `api_key_env`, to run the "different LLM" experiment with another provider. An environment variable `DEEPSEEK_BASE_URL`, if set, overrides it (handy for testing against a fake server). |
| `llm.api_key_env` | `DEEPSEEK_API_KEY` | The NAME of the variable in your `.env` file that holds the API key (not the key itself). |
| `llm.thinking` | `disabled` | DeepSeek's newer models "think" before answering by default, which is slower, costlier and changes the response format. `disabled` switches that off; `enabled` turns it back on. |
| `llm.temperature` | `0.0` | Randomness of the LLM calls made while building indexes and answering questions (0 = as repeatable as possible). Benchmark generation has its own `benchmark.temperature`. |

### 4.2 `embedding`: the local embedding model

| Setting | Default | What it does |
|---|---|---|
| `embedding.model` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers model used by LightRAG, RAPTOR and the benchmark generators (ARES round-trip filter, duplicate check). Downloaded automatically on first use. DeepSeek has no embedding API, hence local. MedRAG does not use it (it has its own retriever, MedCPT). If you change it, check its input-length limit against the chunk sizes below and rebuild the indexes. |

### 4.3 `corpora`: which data each RAG uses

| Setting | Default | What it does |
|---|---|---|
| `corpora.lightrag.subset` | `mix` | Which UltraDomain subset LightRAG indexes: `mix`, `agriculture`, `cs` or `legal` (the four used in the LightRAG paper). |
| `corpora.lightrag.max_docs` | `10` | Keep only the first N documents of that subset. `null` = all of them (slow and costly to index on the big subsets). |
| `corpora.raptor.split` | `dev` | Which QuALITY split the RAPTOR articles come from: `dev` (115 articles) or `train`. |
| `corpora.raptor.max_docs` | `10` | Keep only the first N articles. `null` = all. |
| `corpora.medrag.max_textbooks` | `3` | How many of the 18 MedRAG textbooks to use (the first N in alphabetical order). `null` = all 18. |

### 4.4 `lightrag`

| Setting | Default | What it does |
|---|---|---|
| `lightrag.query_mode` | `hybrid` | How LightRAG retrieves when answering. `local` = focus on specific entities; `global` = focus on broader relations and themes; `hybrid` = both; `naive` = plain vector search over text chunks (no graph); `mix` = knowledge graph plus vector search (LightRAG's own default). Query-time only: no rebuild needed. |
| `lightrag.chunk_token_size` | `400` | Size (in tokens) of the text chunks LightRAG cuts documents into before extracting its graph. LightRAG's own default is 1200; we use less so that chunks fit the 512-token window of the embedding model. Changing it needs a rebuild. |
| `lightrag.chunk_overlap_token_size` | `50` | Tokens shared between neighbouring chunks. Changing it needs a rebuild. |

### 4.5 `raptor`

| Setting | Default | What it does |
|---|---|---|
| `raptor.repo_dir` | `third_party/raptor` | Where RAPTOR's own code lives (cloned there by `prepare_corpora.py raptor`). |
| `raptor.chunk_tokens` | `100` | Maximum tokens per bottom-level text chunk. Rebuild after changing. |
| `raptor.num_layers` | `5` | Maximum number of summary layers built on top of the chunks. RAPTOR stops earlier when there are too few nodes left to cluster. Rebuild after changing. |
| `raptor.summary_tokens` | `100` | Maximum length of each LLM-written summary. Rebuild after changing. |
| `raptor.top_k` | `10` | How many tree nodes (from any layer, chunks or summaries) are handed to the answering model. Query-time only. |
| `raptor.max_context_tokens` | `3500` | Cap on the total size of that context; nodes beyond the budget are dropped. Query-time only. |

### 4.6 `medrag`

| Setting | Default | What it does |
|---|---|---|
| `medrag.top_k` | `32` | How many textbook snippets are retrieved per question. Query-time only. |
| `medrag.batch_size` | `32` | How many snippets are embedded at once while building the index. Only affects speed and memory; lower it if you run out of memory. |

### 4.7 `tests`

| Setting | Default | What it does |
|---|---|---|
| `tests.lightrag` | `What are the main topics covered in these documents?` | Question `test_rag.py` asks LightRAG (override on the command line with `--question`). |
| `tests.raptor` | `What happens in the first story?` | Same, for RAPTOR. |
| `tests.medrag` | `What is the function of the phrenic nerve?` | Same, for MedRAG. |

### 4.8 `benchmark`: generating questions

**General settings (all three methods):**

| Setting | Default | What it does |
|---|---|---|
| `benchmark.seed` | `42` | Random seed for choosing passages. Same seed and same settings pick the same passages. (The LLM itself cannot be seeded, so keep `temperature` at 0 for repeatable output.) |
| `benchmark.output_dir` | `results/benchmarks` | Folder where benchmark files are written (one sub-folder per RAG and corpus). |
| `benchmark.passage_words.<method>` | ragas: `250`<br>ares: `100`<br>datamorgana: `300` | Size, in words, of the text unit a question is generated from, per method. The corpus is cut into passages of about this size (whole sentences are packed together until the size is reached). The defaults are guesses from the papers: RAGAS used Wikipedia intro sections, ARES short KILT-style passages, DataMorgana whole documents (here capped to one passage). Smaller passages give more specific questions and more passages to draw from. |
| `benchmark.min_passage_words` | `40` | Passages shorter than this (leftovers at the end of a document) are dropped. |
| `benchmark.sampling` | `spread` | In which order passages are tried. `spread` = take turns between documents so every document is covered; `random` = fully random (big documents dominate); `sequential` = in reading order (only the start of the corpus gets covered). |
| `benchmark.max_questions_per_passage` | `1` | `1` = every question comes from a different passage. Raise it to allow several questions per passage, which also raises the largest possible benchmark. |
| `benchmark.max_attempts_factor` | `3` | The run gives up after (target size x this number) attempts. Needed because filters reject some attempts. If you see "fewer questions than the target", raise it. |
| `benchmark.concurrency` | `8` | How many LLM calls run in parallel. Lower it if DeepSeek returns rate-limit errors; raise it to go faster. |
| `benchmark.temperature` | `0.0` | Randomness of the LLM when generating questions. `0` = repeatable. Raise (e.g. 0.7) for more varied questions. |
| `benchmark.dedupe_threshold` | `0.9` | A new question is dropped if its meaning is at least this similar (cosine similarity of embeddings, 0 to 1) to an already accepted one. `null` switches the check off. Lower = stricter. |
| `benchmark.drop_context_references` | `true` | Drop questions that refer to their source, such as "in the passage" or "given context", because they cannot be asked to a RAG system. This filter is ours, not from the papers, and applies to all three methods. |

**Benchmark size (`benchmark.size`):**

| Setting | Default | What it does |
|---|---|---|
| `benchmark.size.mode` | `ratio` | How the benchmark size is decided. `ratio` = proportional to the corpus size (see below). `count` = exactly `size.count` questions. |
| `benchmark.size.questions_per_1k_words` | `2.0` | Used in `ratio` mode: number of questions per 1,000 words of corpus. |
| `benchmark.size.count` | `100` | Used in `count` mode: the exact number of questions. |
| `benchmark.size.min_questions` | `20` | `ratio` mode only: never fewer than this many questions. |
| `benchmark.size.max_questions` | `300` | `ratio` mode only: never more than this many questions (a safety cap on cost). |

*How the size is decided.* In `ratio` mode:

```
corpus words   = total words in the passages (after dropping fragments shorter than min_passage_words)
target         = round(questions_per_1k_words x corpus words / 1000)
target         = at least min_questions, at most max_questions
target         = at most (number of passages x max_questions_per_passage)   <- a warning is printed if this bites
```

Example with the defaults: a corpus of 50,000 words gives 100 questions; a corpus of 400,000 words would give 800, but the cap of 300 applies; a tiny corpus still gets at least 20. In `count` mode the target is `size.count` (default 100) and only the passage limit above applies. The corpus size is measured per method, because each method cuts the corpus into passages of a different size.

**RAGAS (`benchmark.ragas`):**

| Setting | Default | What it does |
|---|---|---|
| `benchmark.ragas` | `{}` | No settings: the RAGAs paper's question and answer prompts are fixed. (Passage size and everything above still apply.) |

**ARES (`benchmark.ares`):**

| Setting | Default | What it does |
|---|---|---|
| `benchmark.ares.fewshot.<rag>` | lightrag: `configs/ares_fewshot_general.json`<br>raptor: `configs/ares_fewshot_general.json`<br>medrag: `configs/ares_fewshot_medical.json` | JSON file with the few-shot examples ARES shows the LLM, per RAG. They should be in-domain: replace them with examples written from your own corpus if you can (format in section 6). |
| `benchmark.ares.n_fewshot` | `5` | How many examples go into each prompt (drawn at random from the file). If the file has fewer, all are used. |
| `benchmark.ares.roundtrip_filter` | `true` | ARES's quality filter: keep a generated question only if searching the corpus with it returns its own passage at the top. `false` = keep everything (faster, lower quality). |
| `benchmark.ares.roundtrip_top_k` | `1` | How high the passage must rank for the filter above. `1` is what the paper requires; raise it to be more lenient (many rejections usually mean this is too strict for your corpus). |
| `benchmark.ares.negatives` | `false` | Also create ARES's negative examples (a random passage, a same-document passage, a contradictory answer) for training judges. Costs one extra LLM call per question. Not needed just to get benchmark questions. |

**DataMorgana (`benchmark.datamorgana`):**

| Setting | Default | What it does |
|---|---|---|
| `benchmark.datamorgana.categorizations.<rag>` | lightrag: `configs/datamorgana_general.json`<br>raptor: `configs/datamorgana_general.json`<br>medrag: `configs/datamorgana_medical.json` | JSON file that defines who asks (user categories) and what kinds of questions are asked (question categories), with probabilities. Edit freely (format in section 6). |
| `benchmark.datamorgana.candidates_per_call` | `3` | How many candidate question-answer pairs the LLM writes per attempt (`k` in the paper); the filter then picks one. |
| `benchmark.datamorgana.filter` | `true` | The paper's quality filter: check that candidates follow the constraints and categories and that answers are faithful to the passage. `false` = skip it and take a random candidate (one LLM call fewer per attempt). The filter prompt is ours; the paper does not publish it. |

### 4.9 Which changes need a rebuild or `--fresh`

Folder names only track the `corpora:` settings. Other settings change how an index or a benchmark is built, and old files are not updated automatically.

| You change | Delete first | Why |
|---|---|---|
| `embedding.model` | the RAG's folder in `indexes/` (LightRAG, RAPTOR) | Stored vectors come from the old model and have a different shape. |
| `lightrag.chunk_token_size`, `lightrag.chunk_overlap_token_size` | `indexes/lightrag_<tag>` | The graph is built from the chunks. |
| `raptor.chunk_tokens`, `num_layers`, `summary_tokens` | `indexes/raptor_<tag>` | The tree is built from them. |
| `llm.model` | `indexes/lightrag_<tag>` and `indexes/raptor_<tag>` if you want the index built by the new model | The graph and the summaries were written by the old model. |
| `lightrag.query_mode`, `raptor.top_k`, `raptor.max_context_tokens`, `medrag.top_k` | nothing | These are used only when answering. |
| `benchmark.seed`, `passage_words`, `min_passage_words`, `sampling`, `max_questions_per_passage`, `temperature`, `dedupe_threshold`, `drop_context_references`, anything inside `ares:` or `datamorgana:`, `llm.model` | nothing, but run with `--fresh` | The run refuses to continue with different settings, so results are never mixed. |
| `benchmark.size.*`, `max_attempts_factor`, `concurrency` | nothing | Safe: a larger size simply continues the earlier run. |
| the **contents** of a few-shot or category JSON file (not its path) | nothing, but run with `--fresh` | The safety check compares settings, not file contents. |

---

## 5. What the outputs look like

### Generated questions

`results/benchmarks/<rag>_<tag>/<method>.jsonl`: one JSON object per line.

```json
{
  "id": "ragas-raptor-0001",
  "method": "ragas",
  "rag": "raptor",
  "corpus": "quality_dev_10",
  "question": "...",
  "answer": "...reference answer written from the passage...",
  "passage_ids": ["quality_63523#6"],
  "contexts": ["...the passage the question came from..."],
  "meta": {}
}
```

- `passage_ids` are `<document>#<number>`: the file name for UltraDomain and QuALITY documents, the textbook name for MedRAG.
- `contexts` is the passage the question was made from (the "gold" context).
- `meta` depends on the method:
  - ARES: `roundtrip_rank`, and if `negatives` is on, `negatives` with `context_weak`, `context_strong` and `answer_strong_contradictory`.
  - DataMorgana: `user_categories`, `question_categories`, `candidates_generated`, `candidates_passed_filter`. This records which kind of question each one is.

`<method>.rejected.jsonl`: one line per rejected attempt (`passage_id`, `reason`, `detail`, `question`). Reasons:

| Reason | Meaning |
|---|---|
| `roundtrip_filter` | ARES: the question did not retrieve its own passage at the required rank. |
| `all_candidates_failed_filter` | DataMorgana: none of the candidates passed the quality filter. |
| `no_valid_candidates` | DataMorgana: the model's reply contained no usable question-answer pair. |
| `context_reference` | The question mentioned "the passage", "given context" and the like. |
| `duplicate` | Too similar to an earlier question (`dedupe_threshold`). |
| `empty_question`, `empty_query`, `empty_answer` | The model returned nothing usable. |
| `error` | An LLM call failed (details in `detail`). Three or more failures in a row stop the run with the last error message, so a wrong API key does not burn through all attempts. |

`<method>.meta.json`: a summary of the run: sizes, `target`, `accepted`, `attempts`, `acceptance_rate`, `rejected_by_reason`, model, seed, and for the latest run the seconds, LLM calls and tokens used. Use it for the cost and time numbers in the paper.

### What a RAG returns when asked a question

`rag.answer(question)` returns a dictionary:

| Field | Meaning |
|---|---|
| `answer` | The answer text. |
| `contexts` | **Everything the model was shown**, in pieces. Faithfulness is judged against this by default, which is what RAGAS and ARES assume. |
| `source_contexts` | Only the **original text** among those pieces. Can be empty (RAPTOR retrieving only summaries). Use it for a stricter check. |

Extra fields: LightRAG adds `graph_entities` and `graph_relationships`; RAPTOR adds `summaries` and `layers` (0 = original chunk, higher = summary layer); MedRAG adds `context_meta` (snippet id, textbook, score).

---

## 6. Editing the inputs of the benchmark methods

**ARES few-shot examples** (`configs/ares_fewshot_*.json`): a list of examples. The ARES paper uses in-domain examples written by humans; writing 5 from your own corpus is the most faithful setup.

```json
[
  {
    "document": "a short passage from your corpus",
    "query": "a question that passage answers",
    "answer": "the correct answer, based on the passage",
    "contradictory_answer": "a plausible but wrong answer (only used when negatives is on)"
  }
]
```

**DataMorgana categories** (`configs/datamorgana_*.json`): two lists, `user_categorizations` (who asks) and `question_categorizations` (what is asked). Each categorization has categories with a `name`, a `probability` and a `description`.

```json
{
  "user_categorizations": [
    {"name": "expertise", "categories": [
      {"name": "expert", "probability": 0.5, "description": "a specialized user with deep understanding of the corpus."},
      {"name": "novice", "probability": 0.5, "description": "a regular user with no understanding of specialized terms."}
    ]}
  ],
  "question_categorizations": [ ... same structure ... ]
}
```

- For every question, one category is drawn from each categorization according to the probabilities (they are relative weights and do not need to add up to 1).
- The description is pasted into the prompt as it is: user categories after "They must be ..." and question categories after "It must be ...". Write descriptions that complete those sentences.
- Keys starting with `_` (like `_note`) are ignored.

**Adding another RAG:** create `src/rags/<name>_rag.py` with a class that has `load_corpus()`, `build_index()` and `answer()` (see `base.py`), register it in `src/rags/registry.py`, and add its entries under `corpora:`, `benchmark.ares.fewshot` and `benchmark.datamorgana.categorizations` in `config.yaml` and in `corpus_tag()` in `src/config.py`.

---

## 7. Recipes

| I want to... | Do this |
|---|---|
| Try a method cheaply | `python scripts/generate_benchmark.py lightrag ares --dry-run`, then `--n 10`, and read `ares.jsonl` and `ares.rejected.jsonl`. |
| A bigger benchmark from the same corpus | Rerun with a larger `--n`. The earlier questions are kept and only the missing ones are generated. |
| Benchmarks of several sizes for the compression experiment | Generate the largest, then take subsets of it. Or rerun with `--fresh` for each size, first copying the folder (or changing `benchmark.output_dir`) so the earlier files are not deleted. |
| More varied questions | Raise `benchmark.temperature` (e.g. 0.7) and use `--sampling random`. |
| Cheaper generation | Lower the size; set `benchmark.datamorgana.filter: false`; keep `ares.negatives: false`. |
| More questions from a small corpus | Raise `max_questions_per_passage`, or lower `passage_words`. |
| Use a different LLM | Change `llm.model`, `llm.base_url` and `llm.api_key_env` (and put the key in `.env`). |
| Use a bigger corpus | Raise `corpora.*.max_docs` or `max_textbooks` (or set `null`), rerun `prepare_corpora.py`, then the tests. Indexing cost grows with corpus size. |

---

## 8. Troubleshooting

| Message or symptom | Likely cause and fix |
|---|---|
| `No API key found` | `.env` is missing or the variable name does not match `llm.api_key_env`. |
| `model not found` or a 400/404 error from DeepSeek | The model name changed. Check https://api-docs.deepseek.com/updates/ and update `llm.model`. |
| Rate-limit errors (429) or timeouts | Lower `benchmark.concurrency`. |
| `N LLM calls in a row failed` | The last error is printed. Usually a wrong key, no credit, or no internet. |
| `Settings changed since the last run ... use --fresh` | You changed a setting that affects the questions (section 4.9). Restore it, or use `--fresh` (this deletes the earlier results of that RAG and method). |
| `fewer questions than the target` | Filters rejected too many attempts. Raise `max_attempts_factor`, relax a filter, or look at `.rejected.jsonl` to see why. |
| ARES rejects most attempts (`roundtrip_filter`) | The retrieval check is strict. Try a larger `roundtrip_top_k`, smaller `passage_words`, or in-domain few-shot examples. |
| `wanted N questions but the corpus only offers M` | Size is limited by the number of passages. Raise `max_questions_per_passage`, lower `passage_words`, or use a bigger corpus. |
| RAPTOR test prints `TEST PASSED` but no original passages | Normal for broad questions, see section 3. |
| Errors after changing an embedding or chunk setting | Delete the affected folder in `indexes/` (section 4.9). |
| `No corpus found` | Run `python scripts/prepare_corpora.py <rag>` first. |
| MedRAG indexing is slow | It embeds every snippet with MedCPT. It is faster on a GPU, uses fewer textbooks with a lower `max_textbooks`, and can be interrupted and resumed. |

---

## 9. Differences from the original papers (for your Methods section)

- **LLM:** DeepSeek replaces the models used in the papers (ChatGPT for RAGAs, FLAN-T5 XXL for ARES, Claude 3.5 Sonnet for DataMorgana).
- **Embeddings:** local `bge-small-en-v1.5` instead of OpenAI embeddings. Where LightRAG or RAPTOR used OpenAI embeddings, results are not an exact reproduction.
- **LightRAG:** smaller chunks (`chunk_token_size`) than LightRAG's default so they fit the embedding model.
- **RAPTOR:** the paper builds one tree per document; here all documents of the corpus go into one tree, so a question can be answered from any document.
- **MedRAG:** a re-implementation of its pipeline (same Textbooks corpus, MedCPT retriever, k = 32, prompt adapted from MedRAG's), not a call into its released code, which pins old library versions.
- **RAGAS:** only the two prompts described in the RAGAs paper are used. The newer test-set generator of the `ragas` software library is a different method and is not used.
- **ARES:** the round-trip filter uses a local embedding model with exact search instead of OpenAI embeddings. The few-shot examples in `configs/` are generic or medical examples written for this project, not in-domain human-written ones. Weak answer negatives (an answer taken from another question) are not stored; they can be sampled from the finished benchmark. Strong context negatives fall back to embedding neighbours where the paper uses BM25.
- **DataMorgana:** the paper does not publish its filter prompt or the probabilities of its question categories; the filter prompt is ours and the probabilities are uniform.
- **`drop_context_references`:** an extra filter that is not in any of the three papers.
- **Passage size:** the papers generate questions from different kinds of text units; here every method works on passages of `passage_words` words.

---

## 10. Corpus sizes

```bash
python scripts/corpus_stats.py                    # every downloaded RAG, plain text
python scripts/corpus_stats.py lightrag raptor     # just these
python scripts/corpus_stats.py --markdown          # a table to paste into a paper or README
python scripts/corpus_stats.py --save              # also writes results/corpus_stats.{json,md}
```

For each RAG it reports: document/snippet count, how many source files (documents for LightRAG/RAPTOR, textbooks for MedRAG) those group into, total words, total characters, an approximate token count (words x 1.35, a rule of thumb — not a real tokenizer count, so report it as approximate), and words per document (average, min, max). Numbers depend on your `corpora:` settings, so re-run it after changing `max_docs` or `max_textbooks`, and after re-running `prepare_corpora.py`.

## 11. Real queries and comparing question sets

### Downloading real user-to-LLM queries

```bash
python scripts/prepare_real_queries.py general    # WildChat-1M + LMSYS-Chat-1M, first user turn
python scripts/prepare_real_queries.py medical     # HealthChat-11K's flagged health turns
python scripts/prepare_real_queries.py all
```

**Before this works**, both source datasets are gated on Hugging Face:
1. Log in at huggingface.co and request access on both dataset pages: `lmsys/lmsys-chat-1m` and `allenai/WildChat-1M`. Access is granted by the dataset owners, so request it early; it is not instant.
2. Create a read-access token at huggingface.co/settings/tokens and add `HF_TOKEN=...` to `.env`.

Both datasets are streamed (not fully downloaded), and `general` uses reservoir sampling so every conversation has an equal chance of being picked without loading the whole stream. Results go to `data/real_queries/general.jsonl` and `medical.jsonl`, one user turn per line: `{"id", "text", "source", "domain", ...}`.

**On `medical`:** it pulls HealthChat-11K's own annotation file (`yahskapar/HealthChat-11K` on Hugging Face) and pulls each annotated conversation's flagged turn from the same two source datasets, following the approach in HealthChat's own `generate_artifacts.py`. I have not been able to confirm the exact field names in that annotation file (`user_turn_index`, `dataset_source`, `specialty`), so **check the first run's output against a few rows of the annotation file itself** before trusting it; the field names may need a small fix.

| `real_queries.*` setting | Default | What it does |
|---|---|---|
| `real_queries.output_dir` | `data/real_queries` | Where the two files are written. |
| `real_queries.spacy_model` | `en_core_web_sm` | Model used both here and by the comparison step below. Run `python -m spacy download <name>` for a different one. |
| `real_queries.general.sources` | `[wildchat, lmsyschat]` | Which source datasets to draw from. |
| `real_queries.general.n_per_source` | `2000` | How many conversations to sample from EACH source. |
| `real_queries.general.languages` | `[en]` | Keep only these languages; `null` = keep all. |
| `real_queries.general.min_words` / `max_words` | `3` / `200` | Drop near-empty turns and unusually long pastes (code dumps, whole documents). |
| `real_queries.medical.n` | `2000` | How many HealthChat-11K conversations to use; `null` = all (~11,000). |

### Comparing question sets

```bash
# real queries vs. every generated benchmark found under results/benchmarks/
python scripts/compare_queries.py --all-benchmarks --domain general
python scripts/compare_queries.py --all-benchmarks --domain medical

# any two question sets
python scripts/compare_queries.py \
    --a data/real_queries/general.jsonl --a-field text --a-name real \
    --b results/benchmarks/lightrag_mix_10/ragas.jsonl --b-field question --b-name ragas \
    --limit 500   # optional: cap per set, for a quick trial
```

Each comparison computes, for every question, **13 graph features** (from a lightweight dependency+entity graph built with spaCy: node/edge counts, density, depth, components, entity counts, wh-word, whether the first word is a verb — see `src/analysis/graph_features.py`) and **12 lexical features** (length, lexical diversity, hedging words, first-person, politeness markers, punctuation habits — see `src/analysis/lexical_features.py`). No LLM calls are made, so this scales to tens of thousands of real queries cheaply.

It reports two things:
- **Per feature:** the mean in each set, their difference with a bootstrapped 95% confidence interval, and a Mann-Whitney U test p-value, corrected for testing 25 features at once (Benjamini-Hochberg `q_value`).
- **A classifier two-sample test:** a logistic regression trained on these features to tell the two sets apart, reported as held-out AUC (0.5 = indistinguishable, 1.0 = perfectly separable) plus 5-fold cross-validation on the training split and the top features driving the separation. This is the graph-independent check for circularity discussed for Benchmark+: once questions are mutated to look more like real queries, this AUC dropping toward 0.5 is evidence the mutation worked, independent of the graph features used to design the mutations.

Results go to `results/analysis/<a-name>_vs_<b-name>/`: `features.csv` (every question's raw feature values), `feature_table.csv`, `summary.json`, and a human-readable `report.md`.

| `analysis.*` setting | Default | What it does |
|---|---|---|
| `analysis.output_dir` | `results/analysis` | Where comparison results are written. |
| `analysis.classifier_test_size` | `0.3` | Fraction of each set held out to report the classifier's AUC on. |
| `analysis.classifier_cv_folds` | `5` | Cross-validation folds on the remaining training part. |
| `analysis.bootstrap_samples` | `2000` | Resamples used for each feature's confidence interval. |
| `analysis.seed` | `42` | Random seed (data split, cross-validation, bootstrap). |

A warning is printed if either set has fewer than 20 questions; treat results that small as a pilot, not a finding.

**Known limitation of the graph:** a content word whose only path to another kept word runs through a stopword that is the sentence root (e.g. "keep" in "keep crashing") ends up in its own component instead of connected to its sibling. This is a simplification of the collapsing rule, not a crash or missing feature. Because it is applied identically to every question in both sets, it does not favor one set over the other in a comparison, but it means `n_components` slightly overcounts true structural pieces. Full constituency parsing would avoid it, at the cost of a heavier dependency.

---

## 12. Large-scale runs (profiles)

A **profile** is a named set of overrides in `config.yaml` (`profiles:` at the bottom) that is merged on top of the normal settings. One switch changes many settings, and every script honours it.

```bash
python scripts/run_all.py --profile large --dry-run      # the plan, with cost estimates (see section 15)
BP_PROFILE=large python scripts/generate_benchmark.py lightrag ragas     # or set it for one command
export BP_PROFILE=large                                  # or for the whole shell session
```

The `large` profile exists to produce **thousands of questions per RAG x method** (default 2000, far more than the default small run) while keeping its files apart from the small run:

| Setting | Default profile | `large` profile | Why |
|---|---|---|---|
| `corpora.lightrag` | subset: `mix`<br>max_docs: `10` | max_docs: `30` | UltraDomain documents indexed by LightRAG (and used as the question source). |
| `corpora.raptor` | split: `dev`<br>max_docs: `10` | max_docs: `40` | QuALITY articles indexed by RAPTOR. |
| `corpora.medrag` | max_textbooks: `3` | max_textbooks: `3` | MedRAG textbooks used. |
| `benchmark.output_dir` | `results/benchmarks` | `results/benchmarks_large` | Large and small runs never mix. |
| `benchmark.size` | mode: `ratio`<br>questions_per_1k_words: `2.0`<br>count: `100`<br>min_questions: `20`<br>max_questions: `300` | mode: `count`<br>count: `2000` | Exactly this many questions per RAG x method. Change the number here or with `--n`. |
| `benchmark.max_questions_per_passage` | `1` | `4` | Several questions per passage, so a modest corpus still yields thousands. |
| `benchmark.max_attempts_factor` | `3` | `4` | Give up after target x this many attempts. |
| `benchmark.temperature` | `0.0` | `0.7` | Must be above 0: otherwise several questions from one passage come out identical and the duplicate filter rejects them. |
| `benchmark.concurrency` | `8` | `16` | Parallel LLM calls. Lower it if DeepSeek rate-limits you. |
| `mutation.output_dir` | `results/mutations` | `results/mutations_large` | Mutated areas of the large run. |
| `evaluation.output_dir` | `results/eval` | `results/eval_large` | Results of the large run. |
| `analysis.output_dir` | `results/analysis` | `results/analysis_large` | Realism-comparison outputs of the large run. |

**How many questions can a corpus give?** At most `passages x max_questions_per_passage`, where passages are `passage_words` long (RAGAS 250, ARES 100, DataMorgana 300 words by default). Measured on real QuALITY text (30 articles, 131,000 words): 542 RAGAS passages, 1,371 ARES passages and 455 DataMorgana passages, so at 4 questions per passage RAGAS and ARES reached 2000 but DataMorgana stopped at 1,820. The large profile uses 40 articles (about a third more text, so roughly a third more passages, which should lift DataMorgana past 2000; this scaling is an estimate, the `--dry-run` shows the exact number). When the corpus is too small the run prints `WARNING: wanted N questions but the corpus only offers M` and uses M. To get more: raise `corpora.<rag>.max_docs` (QuALITY dev has 115 articles), raise `max_questions_per_passage`, or lower `passage_words`. A bigger corpus is a **new corpus folder with a new index**, which costs LLM calls for LightRAG (30 UltraDomain documents) and RAPTOR; MedRAG indexing is local (a GPU helps).

To change only the number: `python scripts/generate_benchmark.py all all --n 5000` with `BP_PROFILE=large`, or edit `benchmark.size.count` in the profile. Keep it between about 1,000 and 10,000: below that the compression experiments have little to compress, above it generation cost grows without much benefit.

---

## 13. Mutations: from Benchmark to Benchmark+

The measured differences between real user queries and generated benchmark questions (section 11) are turned into **three mutations** of each benchmark question. Each is stored in its own area, always with the reference answer that is correct *for the mutated question*. All keep the question's `id`, so any two conditions can be compared question by question.

| Condition | Area | What changes | Reference answer |
|---|---|---|---|
| `baseline` | the benchmark itself | nothing | original |
| `m1` | `mutation1_length/` | the prompt gets **longer** | unchanged |
| `m2` | `mutation2_linguistic/` | **how it is written**: typos, register, person, request form | unchanged |
| `m3` | `mutation3_userinfo/` | the **user states something**: correct (answer unchanged) or incorrect (the right response changes) | unchanged, or rewritten (see 13.5) |
| `m123` | `mutation123_combined/` | all three together | that of `m3` |

### 13.1 Why these three

Section 11 found the same differences in every RAG x method: real queries are longer (about 2.8 sentences against 1.0), start with a verb far more often, and are more often lowercase, first-person, polite, stretched ("whyyy", "???", "..."), and less often end in `?`. Mutation 1 targets length and sentence count, mutation 2 the writing style, and mutation 3 something the section-11 features cannot see but real conversations do contain: users telling the system things, some of them wrong.

### 13.2 Guarantees: definitions, assumptions, propositions

*Notation.* A benchmark item is `(q, c, a)`: question, the passage `c` it was generated from, reference answer. `c ⊨ s` means the passage states or clearly implies the claim `s`; `c ⊨ ¬s` means it states something that conflicts with `s`; otherwise `s` is *not mentioned*. `Ans(q, c)` is the set of answers to `q` that `c` supports. `K(q)` is the set of constraints an answer must satisfy (entities, relations, values, the asked slot).

*Assumptions.* The guarantees hold relative to these, which the paper should state:

- **A1 Closed-world context.** An answer is judged only against `c`; the reference answer `a` is the single element of `Ans(q, c)` up to paraphrase.
- **A2 Constraint locality.** `Ans(q, c)` depends on `q` only through `K(q)`. Text that leaves `K` unchanged leaves `Ans` unchanged.
- **A3 Robust reading.** Surface changes (typos, register, person, request form) do not change `K`.
- **A4 Verifier soundness.** The LLM checks below are correct for the items they pass. This is *empirical*, not a theorem: report the pass rates (`_meta/.../*.meta.json`) and audit a random sample by hand.

*Propositions* (proof sketches; `q ⊕ s` = `q` with text `s` added):

- **P1 Content-free text.** If `s` contains no domain content, `K(q ⊕ s) = K(q)`, so `Ans` is unchanged. *Proof:* A2. Covers `instruction`, `politeness`, the first-person lead-ins.
- **P2 Restatement.** If `p` is a paraphrase of `q` with `K(p) = K(q)`, then `K(q ⊕ p) = K(q)`. Covers `restate`.
- **P3 Entailed additions.** If `c ⊨ s` and `s` does not state `a`, then `s` adds no constraint that `c` does not already satisfy: every answer `c` supports for `q` still satisfies `s`, and none becomes newly supported, so `Ans(q ⊕ s, c) = Ans(q, c)`. Covers `background`, `appositive`, `true_premise`, `correct_candidate`.
- **P4 Surface transformations.** Each mutation-2 operator preserves `K` by construction: a typo changes one word at edit distance exactly 1 and never a number, acronym, negation or quantifier; `casual` uses fixed meaning-identical tables; `first_person` replaces a *generic* subject ("the patient") by "I" (this needs the assumption that the question concerns a generic person); `imperative` embeds the same WH-question after a request verb; `politeness` adds closed-class words. By A3, `K` is unchanged.
- **P5 Contradicted claims.** If `c ⊨ ¬s` (`false_premise`, `wrong_candidate`), any answer relying on `s` is unsupported by `c` (A1). The only supported responses do not adopt `s` and give an answer in `Ans(q, c)`. So the reference answer becomes *rejection + original answer*, expected behaviour `reject_premise`.
- **P6 Unsupported claims.** If `c ⊭ s`, `c ⊭ ¬s`, and the question presupposes `s`, `c` supports no answer to the modified question: `Ans(q ⊕ s, c) = ∅`. Expected behaviour `abstain` (flag the unsupported detail).
- **P7 Composition.** Mutations compose: M2 preserves `K`; M1 adds only content-free or `c`-entailed text; M3 adds one controlled claim. So P3, P5 or P6 applies to the combined prompt with the same label as the standalone M3. *Caveat:* this assumes the M1 text and the M3 claim do not interact; M1 text is content-free or entailed by `c`, which is what makes that plausible, but it is not proved.

**What is proved and what is checked.** P1 and P4 hold by construction and are confirmed by rule-based checks (numbers and negations unchanged, edit distance 1). P2, P3, P5 and P6 need a semantic judgement about each added statement; they are enforced per item by an LLM check (A4), and items that fail are dropped, not kept.

### 13.3 Mutation 1: length

Each question gets `length.n_ops` different operators.

| Operator | What it adds | Guarantee | Example (question: *What is the function of the phrenic nerve?*) |
|---|---|---|---|
| `restate` | a restatement of the question in other words, after it | P2 | + *In other words: what does the phrenic nerve do?* |
| `background` | one sentence of background using only facts of the passage, never the answer, before the question | P3 | *The phrenic nerve runs through the chest.* + the question |
| `instruction` | a content-free answer-style instruction from a fixed list (no LLM) | P1 | + *Be specific rather than general.* |
| `appositive` | a short passage-based description right after one term of the question (the term is copied from the question, the insertion is done by the code) | P3 | *What is the function of the phrenic nerve (a nerve supplying the diaphragm)?* |

**Check** (when any LLM-written text was added): one LLM call confirms that the modified question asks the same thing, the reference answer is still correct, every added statement is supported by the passage, and the answer is not given away. Also required: the prompt really is longer.

### 13.4 Mutation 2: linguistic

Rule-based, seeded, no LLM calls. Each question gets `linguistic.n_ops` different operators; they act on each piece of the prompt (text before the question, the question, text after).

| Operator | What it does | Example |
|---|---|---|
| `typos` | one character error (swap neighbours / drop / double / neighbouring key) in a few words of at least `min_word_len` letters, first letter kept. Never in numbers, acronyms, negations, quantifier words; optionally never in named entities | *What is the primary function of the phrenic **nreve**?* |
| `casual` | contractions, informal equivalents ("utilize" -> "use"), lowercase first letter, dropped or stretched final punctuation, dropped apostrophes ("whats"), filler opener | *ok so whats the primary function of the phrenic nerve* |
| `first_person` | generic third-person subject -> "I" (`Does the patient need X?` -> `Do I need X?`, `the user's` -> `my`); if no rule applies, a first-person lead-in sentence | *I have a quick question. What is ...?* |
| `politeness` | a polite lead-in or closing | *Could you help with a question? What is ...?* |
| `imperative` | a WH-question embedded after a request verb (the sentence now starts with a verb) | *Explain what is the primary function of the phrenic nerve.* |

**Check:** numbers identical before and after, number of negation markers identical, text changed. For typos the code also asserts edit distance 1.

### 13.5 Mutation 3: user-provided information

Each question is assigned `correct` with probability `p_correct`, otherwise `incorrect`, then one operator of that kind. Example question: *Who founded the Harborview Library?* (reference answer: *Elena Marquez founded it in 1987.*)

| Kind | Operator | The user says | Reference answer | Expected behaviour | Check: passage vs claim |
|---|---|---|---|---|---|
| correct | `true_premise` | a true fact from the passage, not the answer: *I read somewhere that the library opened in 1987.* | unchanged | `answer` | entailed, does not give the answer |
| correct | `correct_candidate` | the correct answer, asking to confirm: *I think that Elena Marquez founded it. Is that right?* | unchanged | `answer` | entailed |
| incorrect | `false_premise` | a fact that contradicts the passage: *My understanding is that the city council founded it.* | *That is not correct. [correction from the passage]. [original answer]* | `reject_premise` | contradicted; the correction is entailed |
| incorrect | `wrong_candidate` | a wrong answer, asking to confirm: *My guess is that Priya Nair founded it. Am I right?* | *No, that is not correct. [original answer]* | `reject_premise` | contradicted |
| incorrect | `unsupported_assumption` | a detail the passage never mentions: *Note that this is about the 2023 renovation.* | *The provided information does not mention this detail, so the question cannot be answered as stated.* | `abstain` | not mentioned |

Guarantees: P3 for the correct kinds, P5 for `false_premise` and `wrong_candidate`, P6 for `unsupported_assumption`. A claim whose passage relation is not the intended one is regenerated, then dropped.

### 13.6 The algorithm

```
MUTATE(item = (q, c, a)):
  # --- Mutation 1 (length) -------------------------------------------------
  repeat up to 1 + max_retries times:
      ops1    <- sample n_ops distinct operators by weight
      prompt1 <- apply ops1 to q          # LLM operators see only q and c; the code does the insertion
      if prompt1 is not longer than q: continue
      if any LLM operator was used and verify:
          ask the verifier: same information? answer still correct? additions supported? answer revealed?
          if the check fails: continue
      accept; stop
  # --- Mutation 2 (linguistic, rule-based) -----------------------------------
  ops2    <- sample n_ops distinct operators by weight
  prompt2 <- apply ops2 to q              # an operator that changes nothing is replaced by an unused one
  check: numbers equal, negations equal, text changed
  # --- Mutation 3 (user information) ------------------------------------------
  kind <- "correct" with probability p_correct, else "incorrect"
  repeat up to 1 + max_retries times:
      op3 <- sample one operator of that kind; have the LLM write the claim
      verify: passage ⊨ claim / ⊨ ¬claim / not mentioned, as the operator intends
      (false_premise: its correction must also be entailed)
  reference answer, expected behaviour <- by operator
  # --- Combined (needs M1 and M3 accepted) -----------------------------------
  prompt123 <- M1's sentences + M3's sentences around the core question
  apply M2's operators (falling back to unused ones) to the whole prompt; same check
  reference answer, expected behaviour <- those of M3
```

All randomness comes from one generator seeded with (`seed`, question id, mutation name), so reruns, resumed runs and different `concurrency` values give identical files.

### 13.7 Output layout and row format

```
<mutation.output_dir>/
├── mutation1_length/<rag>_<tag>/<method>.jsonl        (+ <method>.rejected.jsonl)
├── mutation2_linguistic/...
├── mutation3_userinfo/...
├── mutation123_combined/...
└── _meta/<rag>_<tag>/<method>.meta.json               counts, rejection reasons, LLM usage, settings hash
```

```json
{
  "id": "ragas-lightrag-0007",            "source_id": "ragas-lightrag-0007",
  "method": "ragas", "rag": "lightrag", "corpus": "mix_30",
  "mutation": "m3",
  "operators": ["m3:false_premise"],       "kind": "incorrect",
  "claim": "the city council founded the library",
  "original_question": "Who founded the Harborview Library?",
  "question": "My understanding is that the city council founded the library. Who founded the Harborview Library?",
  "original_answer": "Elena Marquez founded it in 1987.",
  "answer": "That is not correct. Elena Marquez founded the library. Elena Marquez founded it in 1987.",
  "answer_changed": true,                  "expected_behavior": "reject_premise",
  "passage_ids": ["doc#12"], "contexts": ["...the passage..."],
  "words_original": 5, "words_mutated": 17, "meta": { "...what each operator added, verifier replies..." }
}
```

`<method>.rejected.jsonl` lists `{id, mutation, reason}` for every question whose mutation failed its check (`verification_failed`, `generation_failed`, `not_longer`, `component_failed(...)` for the combined condition, ...). Rejected questions never appear in the kept files, and the evaluation only uses questions that survived all mutations.

### 13.8 Parameters (`mutation:` in `config.yaml`)

**General**

| Setting | Default | What it does |
|---|---|---|
| `mutation.output_dir` | `results/mutations` | Where the four mutated areas are written (sub-folders `mutation1_length/`, `mutation2_linguistic/`, `mutation3_userinfo/`, `mutation123_combined/`, plus `_meta/` for progress and counts). |
| `mutation.seed` | `42` | Random seed. Every random choice for a question depends only on (seed, question id, mutation), so results are identical across runs and across `concurrency` values. |
| `mutation.temperature` | `0.3` | LLM randomness when WRITING mutation text. The checks always run at 0. |
| `mutation.concurrency` | `8` | Questions mutated in parallel. Lower it if DeepSeek returns rate-limit errors. |
| `mutation.max_retries` | `2` | A mutation whose check fails is redone up to this many EXTRA times (new random choices), then dropped for that question. |
| `mutation.max_items_per_file` | `null` | Only the first N benchmark questions of each file; `null` = all. `--limit` on the command line does the same. |

**Mutation 1 (`mutation.length`)**

| Setting | Default | What it does |
|---|---|---|
| `mutation.length.n_ops` | `2` | How many DIFFERENT length operators each question gets. More = longer prompts. |
| `mutation.length.operators` | restate: `1.0`<br>background: `1.0`<br>instruction: `1.0`<br>appositive: `1.0` | Relative sampling weight of each operator; `0` switches one off. |
| `mutation.length.verify` | `true` | LLM check (one call) that the modified question still asks the same thing, the reference answer still holds, the added statements are supported by the passage, and the answer is not given away. `false` skips it (cheaper, no guarantee). |

**Mutation 2 (`mutation.linguistic`)**

| Setting | Default | What it does |
|---|---|---|
| `mutation.linguistic.n_ops` | `2` | How many DIFFERENT linguistic operators each question gets (an operator that would change nothing, e.g. first-person on a text that is already first-person, is replaced by an unused one). |
| `mutation.linguistic.operators` | typos: `1.0`<br>casual: `1.0`<br>first_person: `0.7`<br>politeness: `0.5`<br>imperative: `1.0` | Relative sampling weight of each operator. These are the knobs for matching how often real queries show each property (section 13.9). |
| `mutation.linguistic.typo_rate` | `0.15` | Fraction of eligible words that get a typo. |
| `mutation.linguistic.min_typos` | `1` | At least this many typos (if any word is eligible). |
| `mutation.linguistic.max_typos` | `3` | At most this many typos. |
| `mutation.linguistic.min_word_len` | `5` | Only words with at least this many letters can get a typo (short words would become unrecognisable). |
| `mutation.linguistic.protect_entities` | `false` | `true` = never put a typo inside a named entity (safer for retrieval-by-name; real users do mistype names). |
| `mutation.linguistic.casual_lowercase_prob` | `0.8` | `casual`: chance of lowercasing the first letter (never for "I", acronyms or names). |
| `mutation.linguistic.casual_drop_punct_prob` | `0.7` | `casual`: chance of dropping the final `?` or `.`. |
| `mutation.linguistic.casual_elongate_prob` | `0.15` | `casual`: chance of stretching the final punctuation instead (`?` -> `???`, `.` -> `...`). |
| `mutation.linguistic.casual_filler_prob` | `0.4` | `casual`: chance of a filler opener such as "ok so" or "hey,". |
| `mutation.linguistic.verify` | `true` | Rule-based check that numbers and negations are identical before and after and that the text changed. No LLM. |

**Mutation 3 (`mutation.userinfo`)**

| Setting | Default | What it does |
|---|---|---|
| `mutation.userinfo.p_correct` | `0.5` | Fraction of questions that get CORRECT user information (the rest get incorrect). `0.5` gives a balanced set. |
| `mutation.userinfo.correct_operators` | true_premise: `1.0`<br>correct_candidate: `1.0` | Sampling weights among the correct variants. |
| `mutation.userinfo.incorrect_operators` | false_premise: `1.0`<br>wrong_candidate: `1.0`<br>unsupported_assumption: `1.0` | Sampling weights among the incorrect variants. |
| `mutation.userinfo.verify` | `true` | LLM check that the passage entails / contradicts / does not mention each added claim, as intended. `false` skips it (cheaper, no guarantee). |

Changing `seed`, `temperature`, `max_retries`, anything under `length`, `linguistic` or `userinfo`, or `llm.model` makes `mutate_benchmark.py` refuse to continue an earlier run until you pass `--fresh`, so mutated files made with different settings are never mixed. (`concurrency`, `max_items_per_file` and `output_dir` are safe to change.)

### 13.9 Matching how often real queries show each property

`python scripts/compare_mutations.py` writes `mutation_feature_shift.md`: for each condition, the mean of the features that separate real from generated queries, next to the real-query value. The operator weights (`mutation.linguistic.operators`, `mutation.length.operators`) and `n_ops` are the knobs. If, say, `has_first_person` is 0.45 in `m2` but 0.31 in your real queries, lower the `first_person` weight; if `starts_with_verb` is too low, raise `imperative`. Aim to match the prevalence in the real sample you use for the paper, and report the final table.

Note on `starts_with_verb` (called `root_is_verb` in the first comparison reports): it is 1 when the **first word** of the question is a verb. It was misnamed; the numbers in earlier reports are the same quantity.

### 13.10 Commands

```bash
python scripts/mutate_benchmark.py lightrag ragas --dry-run      # plan and rough LLM-call estimate (about 5 per question)
python scripts/mutate_benchmark.py lightrag ragas --limit 20     # try 20 questions, then READ the four output files
python scripts/mutate_benchmark.py all all                       # everything
python scripts/mutate_benchmark.py lightrag ragas --fresh        # throw away earlier mutated files and redo
```

**Read the first 20 of each before scaling up**, and read the rejected files: a high rejection rate for one operator tells you the prompt or the operator needs attention.

### 13.11 Limits to keep in mind

- The checks use the same LLM that wrote the mutation. Audit a random sample of kept items by hand (a few dozen per mutation) and report the agreement; that, plus the pass rates, is the evidence for A4.
- `first_person` assumes the question is about a generic person; apply it to questions about named individuals with care (the rules only fire on generic subjects such as "the patient", "the user", "people").
- `unsupported_assumption` is judged on whether the passage mentions the detail. A fact the RAG could legitimately know from elsewhere in its corpus is outside A1.
- The reference answer for adversarial items states the expected *behaviour*; the judge (section 14) compares responses against it with an LLM, which is an approximation.

---

## 14. Evaluating Benchmark vs Benchmark+

### 14.1 Design

For one (RAG, method), the **same** questions are answered in five conditions (`baseline`, `m1`, `m2`, `m3`, `m123`) by the RAG, and every answer is judged. Only questions that survived every mutation are eligible, and `evaluation.sample_per_file` (default 500) of them are drawn at random, once, for all conditions. This makes every comparison **paired**: the same question, original versus mutated.

```bash
python scripts/build_indexes.py                              # once; slow for LightRAG and RAPTOR
python scripts/evaluate_benchmarks.py lightrag ragas --dry-run
python scripts/evaluate_benchmarks.py lightrag ragas --limit 20     # a quick first look
python scripts/evaluate_benchmarks.py all all
python scripts/analyze_eval.py
```

Options of `evaluate_benchmarks.py`: `--conditions baseline,m1,...`, `--limit N`, `--fresh`, `--dry-run`. An interrupted run resumes at the last saved batch. A RAG error on one question is recorded in that row (`rag_error`) and does not stop the run.

### 14.2 Metrics

**Faithfulness** follows the RAGAs paper (Es et al., EACL 2024, Section 3): an LLM splits the answer into short statements, an LLM then decides for each statement whether the retrieved context supports it, and `F = supported / total statements`. The two instructions are the paper's; the worked example in the first step and the answer format in the second are ours (the paper does not print them), see `src/evalrun/judge.py`. The context judged against is `contexts`: everything the RAG showed its generator (for LightRAG that includes graph descriptions, for RAPTOR possibly summaries). Answers that decline or flag a problem make no claims and score 1.0.

**Correctness / expected behaviour** is judged against the reference answer, with a label that depends on the item:

| Expected behaviour | Labels | Counts as failure (not correct) | Counts as hallucination |
|---|---|---|---|
| `answer` (baseline, m1, m2, correct m3) | correct, incorrect, abstained | incorrect, abstained | incorrect |
| `reject_premise` (false claim from the user) | corrected, consistent, followed_false_claim, abstained | followed_false_claim, abstained | followed_false_claim |
| `abstain` (detail the passage never mentions) | flagged, answered_core, accepted_detail | accepted_detail | accepted_detail |

*corrected* = explicitly says the claim is wrong; *consistent* = does not adopt it and is consistent with the reference; *flagged* = says the detail is not available or cannot be verified; *answered_core* = answers the underlying question without treating the detail as established.

**Outcomes per answer** (in every result row):

- `unfaithful`: faithfulness below `faithfulness_threshold`.
- `correct`: the label is not a failure label. Declining to answer an answerable question is a failure (it did not answer correctly)...
- `hallucinated`: `unfaithful` **or** the label is a hallucination label (the answer asserts something wrong, or adopts what the user falsely stated). ...but declining is **not** a hallucination.
- `abstained`: the label is `abstained`.

### 14.3 Parameters (`evaluation:` in `config.yaml`)

| Setting | Default | What it does |
|---|---|---|
| `evaluation.output_dir` | `results/eval` | Where per-question results are written (`<rag>_<tag>/<method>/<condition>.jsonl`); the analysis goes to `_report/` inside it. |
| `evaluation.conditions` | `[baseline, m1, m2, m3, m123]` | Which conditions to run. `baseline` is required for the paired analysis. |
| `evaluation.sample_per_file` | `500` | Questions evaluated per (RAG, method), THE SAME ones in every condition (chosen at random with `seed` from the questions that survived every mutation). `null` = all. Evaluation costs about 5 LLM calls per answer, so start small. |
| `evaluation.seed` | `42` | Seed of that random choice. |
| `evaluation.batch_size` | `32` | Questions answered between saves; an interrupted run resumes at the last save. |
| `evaluation.rag_concurrency` | lightrag: `8`<br>raptor: `4`<br>medrag: `8` | Questions a RAG answers at once, per RAG. Lower these if you hit rate limits or run out of memory. |
| `evaluation.judge_concurrency` | `8` | Judge calls in parallel. |
| `evaluation.judge_llm` | `null` | Judge with a different model than the one inside the RAGs, e.g. `{model: ..., base_url: ..., api_key_env: ...}`. `null` = the model in `llm:`. A judge that is also the generator can favour its own answers; a separate judge is advisable for final results. |
| `evaluation.faithfulness_threshold` | `1.0` | An answer is UNFAITHFUL if its faithfulness score is below this. `1.0` = any unsupported statement counts. |
| `evaluation.max_context_chars` | `60000` | The retrieved context shown to the faithfulness judge is cut at this length. |
| `evaluation.also_source_faithfulness` | `false` | Also score faithfulness against ORIGINAL source text only (`faithfulness_source`; 1 extra judge call per answer). Useful for RAPTOR, whose context can be LLM-written summaries. |
| `evaluation.store_contexts` | `false` | Keep the retrieved contexts in every result row (large files). |
| `evaluation.bootstrap_samples` | `2000` | Resamples for the paired confidence intervals. |

### 14.4 Outputs and how to read them

Per-question results: `<evaluation.output_dir>/<rag>_<tag>/<method>/<condition>.jsonl`. The analysis (`analyze_eval.py`) writes, in `<evaluation.output_dir>/_report/`:

| File | Content |
|---|---|
| `report.md` | the main tables, readable as they are |
| `summary_by_condition.csv` | per RAG x method, per RAG, and overall: n, hallucination / failure / unfaithful / abstain rate with 95% Wilson intervals, mean faithfulness |
| `paired_vs_baseline.csv` | per condition: baseline rate, condition rate, difference with a 95% paired-bootstrap interval, `newly_failing` (fine in the benchmark, hallucinated in the mutated version), `newly_passing` (the reverse), exact McNemar p-value, Benjamini-Hochberg `q_value` |
| `coverage.csv` | hallucinations found by the benchmark alone vs benchmark + all mutations, and the fold increase |
| `by_operator.csv` | outcome rates per operator (a question with two operators counts for both) next to the baseline rate on the same questions |
| `m3_breakdown.csv`, `label_distribution.csv` | mutation 3 by kind and operator; every label by condition |
| `hallucination_by_condition.png` | grouped bars with intervals |

**The claim to test**: `diff > 0` with an interval above zero and a small `q_value` means the mutated benchmark exposes more hallucinations than the original on the *same* questions; `coverage.csv` states the same as "found by the benchmark alone vs found with the mutations". Report `failure_rate` next to `hallucination_rate`, because some mutations (declining to answer) change behaviour without causing hallucination.

**Caution when comparing `m3`/`m123` with the baseline.** Their expected behaviour differs (a RAG can fail by following a false claim, which cannot happen in the baseline), so a higher rate there is partly *new kinds of failure*, not only more of the same. Use `m1` and `m2` for "same task, harder wording", and `by_operator.csv` / `m3_breakdown.csv` to separate the kinds.

### 14.5 Realism check

```bash
python scripts/compare_mutations.py              # needs data/real_queries/*.jsonl (section 11)
```

For each RAG x method x condition it reports the AUC of a classifier separating the questions from real user queries (0.5 = indistinguishable; lower = more realistic), and `mutation_feature_shift.md` (section 13.9). Real queries are matched by domain: MedRAG against `medical.jsonl`, LightRAG and RAPTOR against `general.jsonl`. **Use real queries you did not look at while designing the mutations** for the numbers in the paper (see the note at the top of the script and `--real-file`).

---

## 15. Running everything: `scripts/run_all.py`

```bash
python scripts/run_all.py --profile large --dry-run      # the plan: what would run, rough LLM-call counts
python scripts/run_all.py --profile large                # everything, large scale
python scripts/run_all.py --rags medrag --methods ragas --limit 20     # a small slice to test the whole chain
python scripts/run_all.py --from-step mutate             # continue a sequence from a step
python scripts/run_all.py --steps mutate,evaluate,analyze
python scripts/run_all.py --skip compare
```

| # | Step | Runs | Cost |
|---|---|---|---|
| 1 | `corpora` | `prepare_corpora.py` | downloads |
| 2 | `indexes` | `build_indexes.py` | many LLM calls for LightRAG and RAPTOR; local for MedRAG |
| 3 | `generate` | `generate_benchmark.py` for each RAG x method | about 2-4 LLM calls per question, more with rejections |
| 4 | `stats` | `corpus_stats.py --save` | free |
| 5 | `mutate` | `mutate_benchmark.py` | about 5 LLM calls per question (mutation 2 is free) |
| 6 | `evaluate` | `evaluate_benchmarks.py` | about 5 LLM calls per answer, 5 conditions |
| 7 | `analyze` | `analyze_eval.py` | free |
| 8 | `compare` | `compare_mutations.py` | free (needs the real queries) |

Every step is safe to re-run (finished work is skipped, interrupted work resumes). If a step fails, the run stops and prints the command to continue (`--from-step <step>`). `--dry-run` executes `generate`, `mutate` and `evaluate` in their own dry-run mode (plan and LLM-call estimate, nothing spent) and only prints the other steps.

Options: `--profile`, `--rags`, `--methods` (comma-separated), `--steps`, `--from-step`, `--skip`, `--n` (questions per RAG x method in `generate`), `--limit` (questions per file in `mutate`, `evaluate`, `compare`), `--dry-run`.

**A sensible order for a first run:** (1) `--profile large --dry-run`; (2) a slice with `--rags medrag --methods ragas --limit 20` to read the questions, mutated questions and judge labels with your own eyes; (3) the full large run. Evaluate with a modest `sample_per_file` first (default 500); evaluation, not generation, is where most LLM calls go.
