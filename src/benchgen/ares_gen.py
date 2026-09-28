"""ARES synthetic query/answer generation (Saad-Falcon et al., NAACL 2024), Section 3.1.

Steps taken from the paper:
  1. few-shot prompt "Example #k / Document / Query" turns a passage into a query;
  2. few-shot prompt "Query / Document / Answer" turns (query, passage) into an answer;
  3. FILTER: keep a query only if it retrieves its own passage as the top result
     (round-trip check, as in Promptagator / ARES);
  4. optional NEGATIVES for judge training: weak (random passage) and strong
     (passage from the same document; else nearest neighbours) context negatives,
     and a contradictory answer made with the same few-shot prompt.
Differences from the paper, to disclose: DeepSeek instead of FLAN-T5 XXL; a local
embedding model + exact search (same ranking as FAISS IndexFlatL2 on unit vectors)
instead of OpenAI ada embeddings; few-shot examples are the generic/medical ones in
configs/ (the paper uses in-domain, human-written ones: replace them if you can).
Weak ANSWER negatives (an answer taken from a different item) are not stored: they
can be sampled from the finished benchmark at judge-training time.
"""
import json
import threading

import numpy as np

from src.benchgen.common import clean_question
from src.config import ROOT

QUERY_SYSTEM = (
    "Complete the final example, following the pattern of the earlier examples exactly. "
    "Output only the text that comes after the last label."
)
CONTRADICT_SYSTEM = (
    "Complete the final example, following the pattern of the earlier examples exactly. "
    "The answer you write must contradict the document (it must be wrong). "
    "Output only the text that comes after the last label."
)


class AresGenerator:
    name = "ares"
    overhead_tokens = 500

    def __init__(self, ctx):
        self.ctx = ctx
        self.chat = ctx.chat
        ac = ctx.bc["ares"]
        path = ROOT / ac["fewshot"][ctx.rag_name]
        self.examples = json.loads(path.read_text(encoding="utf-8"))
        self.n = min(ac["n_fewshot"], len(self.examples))
        if self.n < ac["n_fewshot"]:
            print(f"NOTE: only {len(self.examples)} few-shot examples available; using {self.n}.")
        self.roundtrip = ac["roundtrip_filter"]
        self.top_k = ac["roundtrip_top_k"]
        self.negatives = ac["negatives"]
        self.calls_per_attempt = 2 + (1 if self.negatives else 0)

        self.lock = threading.Lock()
        if self.roundtrip or self.negatives:
            import faiss

            vectors = ctx.passage_embeddings()
            self.vectors = vectors
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)

    # ---- helpers ------------------------------------------------------
    def _search(self, vector, k):
        with self.lock:
            _, ids = self.index.search(vector.reshape(1, -1), k)
        return [int(i) for i in ids[0] if i >= 0]

    def _query_prompt(self, shots, passage):
        parts = [
            f"Example #{i}\nDocument: {e['document']}\nQuery: {e['query']}\n"
            for i, e in enumerate(shots, 1)
        ]
        parts.append(f"Example #{len(shots) + 1}\nDocument: {passage['text']}\nQuery:")
        return "\n".join(parts)

    def _answer_prompt(self, shots, query, passage, field):
        parts = [
            f"Example #{i}\nQuery: {e['query']}\nDocument: {e['document']}\nAnswer: {e[field]}\n"
            for i, e in enumerate(shots, 1)
        ]
        parts.append(f"Example #{len(shots) + 1}\nQuery: {query}\nDocument: {passage['text']}\nAnswer:")
        return "\n".join(parts)

    # ---- one attempt --------------------------------------------------
    def attempt(self, passage, rng):
        shots = rng.sample(self.examples, self.n)
        query = clean_question(self.chat(QUERY_SYSTEM, self._query_prompt(shots, passage)))
        if not query:
            return {"rejected": "empty_query"}

        meta = {}
        if self.roundtrip:
            vec = self.ctx.embed([query])[0]
            top = self._search(vec, self.top_k)
            if passage["idx"] not in top:
                return {"rejected": "roundtrip_filter", "detail": f"top-{self.top_k} was {top}"}
            meta["roundtrip_rank"] = top.index(passage["idx"]) + 1

        answer = self.chat(QUERY_SYSTEM, self._answer_prompt(shots, query, passage, "answer"))
        if not answer:
            return {"rejected": "empty_answer"}

        if self.negatives:
            meta["negatives"] = self._negatives(query, answer, passage, shots, rng)

        return {
            "question": query,
            "answer": answer,
            "passage_ids": [passage["id"]],
            "contexts": [passage["text"]],
            "meta": meta,
        }

    def _negatives(self, query, answer, passage, shots, rng):
        passages = self.ctx.passages
        others = [p for p in passages if p["doc_id"] != passage["doc_id"]] or [
            p for p in passages if p["idx"] != passage["idx"]
        ]
        weak = rng.choice(others) if others else None

        same_doc = [p for p in passages if p["doc_id"] == passage["doc_id"] and p["idx"] != passage["idx"]]
        if same_doc:
            strong = rng.choice(same_doc)
        else:  # the paper uses BM25 top-10 here; we use nearest neighbours by embedding
            near = [i for i in self._search(self.vectors[passage["idx"]], 11) if i != passage["idx"]]
            strong = passages[rng.choice(near)] if near else None

        contradictory = self.chat(
            CONTRADICT_SYSTEM, self._answer_prompt(shots, query, passage, "contradictory_answer")
        )
        return {
            "context_weak": weak["text"] if weak else None,
            "context_strong": strong["text"] if strong else None,
            "answer_strong_contradictory": contradictory,
        }
