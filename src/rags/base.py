"""The shared shape every RAG in this project follows."""
import re
from concurrent.futures import ThreadPoolExecutor

# What the RAG must answer when its documents do not contain what the question needs (README section 14.10).
# One place for the wording: the RAG prompt, the judge and the counts all read it from here / from the config.
DEFAULT_REFUSAL = "Cannot be answered from given information."


def refusal_phrase(cfg):
    return ((cfg or {}).get("medrag") or {}).get("refusal_phrase") or DEFAULT_REFUSAL


def refusal_instruction(cfg):
    """Sentence added to the RAG's system prompt ("" when medrag.refusal_instruction is false)."""
    if not ((cfg or {}).get("medrag") or {}).get("refusal_instruction", True):
        return ""
    return (" If the documents do not contain the information needed to answer the question, do not guess and do "
            f'not use outside knowledge: reply with exactly "{refusal_phrase(cfg)}" and nothing else.')


def _norm(t):
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", str(t or "").lower())).strip()


def is_refusal(text, phrase=DEFAULT_REFUSAL):
    """True when the answer IS the refusal phrase: the phrase, ignoring case, quotes and final punctuation, with at
    most a few extra words around it. An answer that contains the phrase inside a longer answer is not counted."""
    t, ph = _norm(text), _norm(phrase)
    return bool(t) and ph in t and len(t) <= len(ph) + 40


class BaseRAG:
    name = "base"

    def load_corpus(self):
        """Return the documents this RAG indexes: [{"id": "...", "text": "..."}].
        The benchmark generators (RAGAS, ARES, DataMorgana) read this."""
        raise NotImplementedError

    def build_index(self):
        """One-time step: index the corpus and save it under indexes/."""
        raise NotImplementedError

    def answer(self, question):
        """Answer one question. Return a dict with:
          "answer":          str
          "contexts":        [str, ...]  EVERYTHING the generating model was shown, in
                                         pieces. Faithfulness is judged against this
                                         by default (this is what RAGAS/ARES assume).
          "source_contexts": [str, ...]  only the ORIGINAL source text among those pieces
                                         (may be empty, e.g. when RAPTOR retrieves only
                                         summaries). Use it for a stricter check.
        Some RAGs add extra fields (summaries, graph_entities, ...)."""
        raise NotImplementedError

    def answer_batch(self, questions, concurrency=4):
        """Answer many questions, several at a time. Returns one dict per question,
        in order. A question that fails gives {"error": "..."} instead of raising,
        so one bad question never stops a long evaluation run."""
        if not questions:
            return []

        def one(i):
            try:
                return self.answer(questions[i])
            except Exception as e:
                return {"error": f"{type(e).__name__}: {e}"[:300]}

        results = [None] * len(questions)
        # The first question runs alone: it loads models/indexes once, before any
        # threads start (two threads loading the same model would waste memory).
        results[0] = one(0)
        if len(questions) > 1:
            with ThreadPoolExecutor(max(1, concurrency)) as pool:
                for i, r in zip(range(1, len(questions)), pool.map(one, range(1, len(questions)))):
                    results[i] = r
        return results

    def close(self):
        pass
