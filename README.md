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
