"""Build the index of each RAG (the slow, LLM-heavy step for LightRAG and RAPTOR).

    python scripts/build_indexes.py              # all three
    python scripts/build_indexes.py lightrag     # just one
An index that already exists is reused, and an interrupted build can be resumed.
Use BP_PROFILE=large (or run_all.py --profile large) to index the large corpora.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import index_dir, load_config
from src.rags.registry import RAG_NAMES, get_rag


def main():
    wanted = sys.argv[1:] or RAG_NAMES
    bad = [w for w in wanted if w not in RAG_NAMES]
    if bad:
        sys.exit(f"Unknown RAG(s) {bad}. Choose from {RAG_NAMES}")
    cfg = load_config()
    print(f"profile: {cfg['_profile']}")
    for name in wanted:
        rag = get_rag(name, cfg)
        n = len(rag.load_corpus())
        if not n:
            print(f"{name}: no corpus on disk. Run: python scripts/prepare_corpora.py {name}")
            continue
        print(f"\n=== {name}: indexing {n:,} document(s)/snippet(s) -> {index_dir(cfg, name)} ===")
        t0 = time.time()
        rag.build_index()
        rag.close()
        print(f"{name}: done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
