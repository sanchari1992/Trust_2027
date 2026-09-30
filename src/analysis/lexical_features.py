"""Simple, non-graph properties of a question. Complements the graph features
with things a graph does not capture well: length, phrasing, hedging.
"""
import re

HEDGE_WORDS = {"maybe", "possibly", "i think", "not sure", "might", "could it be", "wondering"}
FIRST_PERSON = {"i", "i'm", "im", "my", "me", "we", "our"}
POLITE = {"please", "could you", "can you", "would you"}


def _words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def extract(text):
    words = _words(text)
    n = len(words) or 1
    lower = text.lower()
    unique = len(set(words))
    return {
        "char_len": len(text),
        "word_len": len(words),
        "lexical_diversity": unique / n,
        "avg_word_len": sum(len(w) for w in words) / n,
        "n_question_marks": text.count("?"),
        "n_sentences_punct": max(1, len(re.findall(r"[.!?]+", text))),
        "starts_lowercase": int(bool(text) and text[0].islower()),
        "has_typo_like_repeat": int(bool(re.search(r"(.)\1{2,}", text))),  # e.g. "whyyyy", "sooo"
        "has_hedge": int(any(h in lower for h in HEDGE_WORDS)),
        "has_first_person": int(any(w in FIRST_PERSON for w in words)),
        "has_politeness_marker": int(any(p in lower for p in POLITE)),
        "ends_with_question_mark": int(text.strip().endswith("?")),
    }


NUMERIC_FEATURES = [
    "char_len", "word_len", "lexical_diversity", "avg_word_len", "n_question_marks",
    "n_sentences_punct", "starts_lowercase", "has_typo_like_repeat", "has_hedge",
    "has_first_person", "has_politeness_marker", "ends_with_question_mark",
]
