"""Smoke test for any RAG: build its index, ask ONE question.

Run from the repo root:
    python scripts/test_rag.py lightrag
    python scripts/test_rag.py raptor
    python scripts/test_rag.py medrag
    python scripts/test_rag.py medrag --question "Your own question here"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.rags.registry import RAG_NAMES, get_rag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rag", choices=RAG_NAMES)
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    cfg = load_config()
    question = args.question or cfg["tests"][args.rag]
    rag = get_rag(args.rag, cfg)

    docs = rag.load_corpus()
    print(f"1) Loaded {len(docs):,} document(s)/snippet(s)")
    if not docs:
        sys.exit("No corpus found. Run: python scripts/prepare_corpora.py " + args.rag)

    print("2) Building the index (skipped if it already exists)...")
    rag.build_index()

    print(f"3) Asking: {question}")
    result = rag.answer(question)

    print("\nANSWER:\n", result["answer"])
    print(f"\nCONTEXT PIECES THE MODEL SAW: {len(result['contexts'])}")
    for i, text in enumerate(result["contexts"][:5], 1):
        print(f"  [{i}] {text[:150].strip()!r}...")
    print(f"ORIGINAL SOURCE PASSAGES AMONG THEM: {len(result['source_contexts'])}")
    for key in ("summaries", "graph_entities", "graph_relationships"):
        if key in result:
            print(f"{key.upper()}: {len(result[key])}")
    if not result["source_contexts"]:
        print("NOTE: no original text was retrieved, only summaries/graph text (normal for broad "
              "questions with RAPTOR). Ask a more specific question to see source passages.")

    ok = bool(result["answer"].strip()) and len(result["contexts"]) > 0
    print("\nTEST PASSED" if ok else "\nTEST FAILED: empty answer or nothing retrieved")
    rag.close()


if __name__ == "__main__":
    main()
