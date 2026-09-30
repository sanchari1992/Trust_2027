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
├── config.yaml                  every setting (section 4)
├── .env                         your API key (create it from .env.example; never commit it)
├── requirements.txt
├── configs/                     inputs of the benchmark methods (section 6)
│   ├── ares_fewshot_general.json / ares_fewshot_medical.json
│   └── datamorgana_general.json / datamorgana_medical.json
├── scripts/
│   ├── prepare_corpora.py       downloads the data
│   ├── test_rag.py              smoke test of one RAG
│   └── generate_benchmark.py    creates benchmark questions
├── src/
│   ├── config.py                reads config.yaml, builds folder names
│   ├── llm.py                   the ONLY place that talks to DeepSeek
│   ├── embeddings.py            local embedding models
│   ├── rags/                    one file per RAG, all with the same 3 functions
│   │   ├── base.py, registry.py
│   │   └── lightrag_rag.py, raptor_rag.py, medrag_rag.py
│   └── benchgen/                the benchmark generators
│       ├── ragas_gen.py, ares_gen.py, datamorgana_gen.py
│       ├── runner.py            sizing, sampling, filters, resume, cost report
│       └── common.py            shared helpers
│
│  Created while you work (all ignored by git):
├── data/corpora/<rag>_<tag>/    the downloaded corpora
├── data/raw/                    raw downloads (QuALITY)
├── third_party/raptor/          RAPTOR's own code
├── indexes/<rag>_<tag>/         the built RAG indexes
└── results/benchmarks/<rag>_<tag>/   generated questions
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

## 10. Real queries and comparing question sets

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

Each comparison computes, for every question, **13 graph features** (from a lightweight dependency+entity graph built with spaCy: node/edge counts, density, depth, components, entity counts, wh-word, whether the root is a verb — see `src/analysis/graph_features.py`) and **12 lexical features** (length, lexical diversity, hedging words, first-person, politeness markers, punctuation habits — see `src/analysis/lexical_features.py`). No LLM calls are made, so this scales to tens of thousands of real queries cheaply.

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
