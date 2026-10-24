# The Benchmark+ web page and API

A small page (and JSON API) for trying the mutations on ONE prompt. It uses exactly the same code as the benchmark pipeline
(`src/benchdb/generate.py`), so what you see here is what the paper's mutated questions are made of.

## Run it

```
pip install fastapi uvicorn
python scripts/serve_mutator.py            # http://127.0.0.1:8000   (API docs at /docs)
python scripts/serve_mutator.py --host 0.0.0.0 --token SECRET    # reachable by others: a token is then required
```
Needs the DeepSeek key in `.env` (`DEEPSEEK_API_KEY`). The graph view also needs the spaCy model from `requirements.txt`
(`python -m spacy download en_core_web_sm`); without it the page still works, only the graphs are missing.

## What you do
1. Type a **question**. As you type, the page draws the **graph of your prompt** (no LLM call).
2. Optionally give the **expected answer** (needed for Premise) and the **evidence** (the passage the answer comes from).
3. Tick the mutations: **Elaborate** (M1, off in the paper), **Colloquial** (M2), **Premise** (M3), and/or **Composite** (all chosen together).
   Under "Premise adds..." choose a statement by the user (true / false / unsupported) or a **patient profile** that fits or conflicts with the
   question, or type your own profile. "More options": pick single operators, how many to stack, a seed.
4. Press **Mutate**. For each mutation you get the mutated prompt, the **new expected answer**, whether the answer changed, what the system
   should do (answer, correct the premise, or say it cannot be answered), the operators used, and the **graph view**.

## The graph view
The graph is the same structure the paper uses to describe how a question is built (`src/analysis/graph_features.py`): one node for every
content word (noun, verb, adjective, adverb) and named entity, one edge for every dependency relation between them (the label on the edge
is the relation, e.g. `nsubj`, `dobj`). Entities have a blue outline. Nodes hang below their head word, so the depth of the drawing is the
depth of the question.

After a mutation the page shows two graphs side by side, **original** and **mutated**, with the differences marked:
* green = a node that is new in the mutated prompt (words added by Elaborate or Premise, a respelled word from a typo);
* red dashed = a node of the original that was changed or removed (a typo or a synonym replaces the word);
* a sentence under the graphs says what happened, e.g. "4 new nodes; 2 new separate parts (the added sentence is not linked to the question)".

A small table gives the numbers before and after: nodes, edges, graph depth, separate parts, entities, and the sentence-complexity
measures used in the paper (parse-tree depth, mean dependency distance, Flesch-Kincaid grade, reading ease, Gunning Fog).

### Why this view (and why it is in the paper)
* It is the paper's own representation. The deviation and realism tables compare prompts by features of this graph (depth, separate parts,
  entities) and by sentence complexity; the page lets a user see those features for their own prompt and watch what each mutation does to them.
* It makes the mutations inspectable. M2 changes spelling and style but leaves the structure (often the node set) alone; M3 adds a separate
  part to the graph (the user's statement or profile is not linked to the question), which is what "added user context" looks like
  structurally; M1 adds nodes under the question. A user can check that a mutation did what it claims.
* It is cheap and deterministic: no LLM call, so it can be shown instantly and gives the same graph every time.
* Limits: it is a structural proxy built from a dependency parse, not a knowledge graph or a meaning representation; parse errors show as
  odd edges. Nodes are matched between the two graphs by lemma, so a respelled word appears as one removed and one added node.

## API
| Method and path | Body | Returns |
|---|---|---|
| `GET /` | | the page |
| `GET /api/mutations` | | the mutations and their operators |
| `POST /api/graph` | `{"text": "..."}` | `nodes` (id, label, lemma, pos, entity, depth, component, status), `edges` (source, target, dep), `stats`, `complexity` |
| `POST /api/mutate` | `question`, `answer`, `evidence`, `mutations`, `composite`, `premise_kind`, `patient_profile`, `profile_relation`, `depth`, `seed`, `randomize` | `graph_before`, and per result: `prompt`, `expected_answer`, `answer_changed`, `expected_behavior`, `operators`, `graph_before`, `graph`, `graph_summary`, ... |

With `--token`, send it as the header `X-Token`. Example:
```
curl -s localhost:8000/api/graph -H "Content-Type: application/json" -d '{"text": "What nerve supplies the diaphragm?"}'
```

## Notes
* Without evidence the answer is the only source ("answer only"); results are better checked with evidence.
* A patient profile is only added to a real question, not to a statement.
* Every result says whether a simple fallback was used and whether an LLM checked it.
