"""Sentence-complexity measures from the readability and syntax literature (no LLM, no extra packages).

  tree_depth          depth of the full dependency parse (longest root-to-leaf path), a standard syntactic-complexity
                      measure (Yngve/Frazier-style depth; used in L2 writing and parsing studies)
  mean_dep_distance   mean dependency distance: average |position(token) - position(head)| (Liu 2008; Futrell et al. 2015)
  n_clauses           clause markers per prompt: subordinate / relative / complement clauses (advcl, relcl, ccomp, xcomp, acl)
  fk_grade            Flesch-Kincaid grade level (Kincaid et al. 1975)
  flesch_ease         Flesch Reading Ease (Flesch 1948); higher = easier
  gunning_fog         Gunning Fog index (Gunning 1952)
  syll_per_word       mean syllables per word
  long_word_ratio     share of words with 7+ letters (the LIX / Gunning "long word" notion)
"""
import re

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_CLAUSE_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl", "csubj", "pcomp"}

NUMERIC_FEATURES = ["tree_depth", "mean_dep_distance", "n_clauses", "fk_grade", "flesch_ease", "gunning_fog",
                    "syll_per_word", "long_word_ratio"]


def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def _depth(token):
    d = 0
    while token.head.i != token.i and d < 100:
        token, d = token.head, d + 1
    return d


def extract(text, doc):
    words = _WORD.findall(text)
    nw = max(len(words), 1)
    sents = max(len(list(doc.sents)), 1)
    syl = [syllables(w) for w in words]
    ns = sum(syl)
    complex_words = sum(1 for s in syl if s >= 3)
    toks = [t for t in doc if not t.is_punct and not t.is_space]
    dist = [abs(t.i - t.head.i) for t in toks if t.head.i != t.i]
    return {
        "tree_depth": max((_depth(t) for t in toks), default=0),
        "mean_dep_distance": sum(dist) / len(dist) if dist else 0.0,
        "n_clauses": sum(1 for t in doc if t.dep_ in _CLAUSE_DEPS),
        "fk_grade": 0.39 * nw / sents + 11.8 * ns / nw - 15.59,
        "flesch_ease": 206.835 - 1.015 * nw / sents - 84.6 * ns / nw,
        "gunning_fog": 0.4 * (nw / sents + 100 * complex_words / nw),
        "syll_per_word": ns / nw,
        "long_word_ratio": sum(1 for w in words if len(w) >= 7) / nw,
    }
