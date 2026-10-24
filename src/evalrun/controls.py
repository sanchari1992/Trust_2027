"""CONTROL conditions: what would happen WITHOUT a mutation doing anything useful?

A mutated benchmark that finds more hallucinations than the original proves little if the same
increase also appears when nothing meaningful changed, because LLM answers are not deterministic and
every extra attempt is another chance to hallucinate. Two controls make that visible:

  c_rerun  the ORIGINAL benchmark question, answered and judged a second time.
           Its change from the baseline is the noise floor of the whole pipeline.
  c_pad    the original question + content-free filler sentences, as many words as the M1 version added.
           If M1 finds more hallucinations than c_pad, the cause is the content M1 added, not length alone.

c_rerun2, c_rerun3  two more repeats, so that the derived c_union (fails in any of 4 runs) is the matched noise
           control of m_union (fails in baseline or any of 3 mutations).

All keep the reference answer, so they are judged exactly like the baseline.
"""
from src.benchgen.common import make_rng

FILLER = [
    "Thanks for taking a look at this.",
    "I would appreciate a careful answer.",
    "Take your time with this one.",
    "I am asking because I am curious about it.",
    "Please keep the answer easy to follow.",
    "This has been on my mind for a while.",
    "Any help with this is welcome.",
    "I want to be sure I understand it properly.",
    "It would be great to get this right.",
    "I will read your answer closely.",
    "Thank you in advance for your help.",
    "I am hoping for a clear response.",
]


def pad_row(base_row, m1_row, seed):
    """base_row + filler as long as (m1_row - base_row) words. Returns None if there is no M1 version."""
    if m1_row is None:
        return None
    extra = len(m1_row["question"].split()) - len(base_row["question"].split())
    if extra <= 0:
        return None
    rng = make_rng(seed, base_row["id"], "c_pad")
    pool, add, n = FILLER[:], [], 0
    rng.shuffle(pool)
    for s in pool:
        if n >= extra:
            break
        add.append(s)
        n += len(s.split())
    row = dict(base_row)
    row.update(question=" ".join([base_row["question"]] + add), mutation="c_pad", operators=["c_pad:filler"],
               words_mutated=len((base_row["question"] + " " + " ".join(add)).split()))
    return row


def control_rows(cond, rows, seed):
    """rows: {cond: {id: row}} of the conditions already loaded. Returns {id: row} for a control."""
    base = rows["baseline"]
    if cond in ("c_rerun", "c_rerun2", "c_rerun3"):
        return {i: {**r, "mutation": cond, "operators": [cond]} for i, r in base.items()}
    if cond == "c_pad":
        m1 = rows.get("m1") or {}
        out = {}
        for i, r in base.items():
            p = pad_row(r, m1.get(i), seed)
            if p:
                out[i] = p
        return out
    raise ValueError(cond)


# ---------------------------------------------------------------- retrieval check
import re as _re

_TOK = _re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_TOK.findall(str(text).lower()))


def retrieval_scores(gold_passages, retrieved, min_cover=0.6):
    """Did the RAG retrieve the passage(s) the benchmark question was written from?
    A gold passage counts as retrieved if some retrieved text contains at least `min_cover` of its words
    (retrieved chunks and benchmark passages are cut differently, so exact matching would miss).
    Returns (gold_recall, hit) or (None, None) if the question has no gold passages."""
    gold = [g for g in (gold_passages or []) if g and str(g).strip()]
    if not gold:
        return None, None
    ret = [_tokens(r) for r in (retrieved or [])]
    found = 0
    for g in gold:
        gt = _tokens(g)
        if gt and any(len(gt & rt) / len(gt) >= min_cover for rt in ret):
            found += 1
    return found / len(gold), 1.0 if found else 0.0
