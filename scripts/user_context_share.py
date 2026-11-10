"""Share of prompts that state something about the person asking (paper Section 2.4, 'Patient information').

    python scripts/user_context_share.py data/real_queries/medical.jsonl            # real queries (field "text")
    python scripts/user_context_share.py results/benchmarks/.../questions.jsonl     # generated questions (field "question")
    python scripts/user_context_share.py file.jsonl --field question

Rule-based and deliberately conservative (a lower bound): an age, a statement about the speaker or a relative, or a
condition, pregnancy, allergy or diagnosis. Run it on the 300 held-out real queries and on the 1,500 generated
questions and report both numbers with the same patterns.
"""
import argparse
import json
import re
import statistics

AGE = re.compile(r"\b(\d{1,3})[- ]?(year|yr|y/o|yo)s?[- ]?old\b|\b(i am|i'm|im)\s+(a\s+)?\d{1,3}\b|\bage[d]? \d{1,3}\b|\b(\d{1,3})\s?(m|f|yo)\b", re.I)
SELF = re.compile(r"\b(i am|i'm|i have|i've|i was|i take|i'm taking|i am taking|my (son|daughter|mother|mom|father|dad|husband|wife|baby|child|doctor|blood|pain|symptoms?))\b", re.I)
COND = re.compile(r"\b(i (am|'m) (pregnant|diabetic|allergic)|i have (diabetes|asthma|cancer|hypertension|a (history|condition))|allerg|pregnan|diagnosed)\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--field", default=None, help='text field; default: "text", else "question"')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    field = a.field or ("text" if "text" in rows[0] else "question")
    T = [str(r[field]) for r in rows]
    n = len(T)
    for name, rx in [("age", AGE), ("statement about the speaker or a relative", SELF), ("condition / pregnancy / allergy / diagnosis", COND)]:
        print(f"{name:45s} {100 * sum(bool(rx.search(t)) for t in T) / n:5.1f}%")
    anyc = sum(bool(AGE.search(t) or SELF.search(t) or COND.search(t)) for t in T)
    w = [len(t.split()) for t in T]
    print(f"{'any of the three':45s} {100 * anyc / n:5.1f}%   (n={n}; words: median {statistics.median(w):.0f}, mean {statistics.mean(w):.1f})")


if __name__ == "__main__":
    main()
