# benchmark-plus

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste your DeepSeek key into .env
```
Do NOT install the requirements.txt files inside RAPTOR's or MedRAG's own repos:
they pin old library versions that conflict with LightRAG.

## Get the data
```
python scripts/prepare_corpora.py all        # or: lightrag | raptor | medrag
```

## Test each RAG (builds its index, then asks one question)
```
python scripts/test_rag.py lightrag
python scripts/test_rag.py raptor
python scripts/test_rag.py medrag
```
Sizes (how many documents, which UltraDomain subset, how many textbooks) are in config.yaml.

## Generate benchmark questions
```
python scripts/generate_benchmark.py lightrag ragas --dry-run   # plan and cost estimate, calls nothing
python scripts/generate_benchmark.py lightrag ragas             # or ares | datamorgana | all
python scripts/generate_benchmark.py all all
```
Results go to `results/benchmarks/<rag>_<corpus>/<method>.jsonl` (plus `.rejected.jsonl` and
`.meta.json` with counts, tokens and time). An interrupted run resumes where it stopped.
All knobs (benchmark size vs. corpus size, passage size, sampling, filters, ...) are in the
`benchmark:` section of `config.yaml`.
