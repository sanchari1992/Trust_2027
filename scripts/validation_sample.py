"""Human verification of the mutations (paper Section 5.4, Table 'validation').

Step 1  make the sheets (one run, fixed seed):
    $env:BP_PROFILE="paper"; python scripts/validation_sample.py sample --n 25 --out results/validation

  writes three CSV files you fill in by hand (open in Excel; fill the empty column, 1 = yes, 0 = no):
    prem_check.csv   the added statement + expected behavior   -> fill  behavior_ok
    coll_check.csv   original vs reworded question             -> fill  need_preserved
    fail_check.csv   hallucinating answers to mutated prompts  -> fill  verdict  (invented / adopted / artifact)
  Rows are stratified: n Premise Injection items per expected behavior (answer, reject_premise, abstain), n Colloquialization
  items, and n hallucinating answers per mutation.

Step 2  after filling the sheets:
    python scripts/validation_sample.py score --out results/validation
  prints the share of 1s with a Wilson 95% interval for the table, and the split of the failure verdicts.

For 'behavior_ok' read the evidence passage (column evidence if present, otherwise look the id up in the benchmark file) and
decide whether the expected behavior is right: answer = statement is supported and does not change the answer; reject_premise =
the passage contradicts it; abstain = the passage does not mention it or the profile does not fit the passage.
For 'need_preserved' decide whether the reworded question still asks for exactly the same information (same numbers, names,
negations, same slot).
For 'verdict' read the response: invented = states something the sources do not support; adopted = builds on the user's false
or unsupported statement; artifact = the answer is acceptable and the judge was wrong (or the question is broken).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_config
from src.evalrun.analysis import load_results


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def op_name(ops, prefix):
    for x in ops or []:
        x = str(x)
        if x.startswith(prefix):
            return x[len(prefix):]
    return ""


def take(g, n, seed):
    return g.sample(min(n, len(g)), random_state=seed)


def sample(args):
    cfg = load_config()
    df = load_results(cfg)
    df["condition"] = df["condition"].astype(str)
    key = ["rag", "method", "id"]
    get = lambda c: df[df["condition"] == c].drop_duplicates(key).set_index(key)
    b, p, c = get("baseline"), get("m3"), get("m2")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ev = [col for col in ("evidence", "passage", "context") if col in df.columns]

    pj = b[["question"]].join(p[["question", "expected_behavior", "operators", "reference_answer", "hallucinated"] + ev],
                              lsuffix="_orig", rsuffix="_prem", how="inner").reset_index()
    pj["statement_type"] = pj["operators"].map(lambda o: op_name(o, "m3:"))
    pj = pd.concat([take(g, args.n, args.seed) for _, g in pj.groupby("expected_behavior")])
    pj["behavior_ok"] = ""
    pj[["rag", "method", "id", "question_orig", "question_prem", "statement_type", "expected_behavior",
        "reference_answer"] + ev + ["behavior_ok"]].to_csv(out / "prem_check.csv", index=False)

    cj = b[["question"]].join(c[["question", "operators"]], lsuffix="_orig", rsuffix="_coll", how="inner").reset_index()
    cj["operators_used"] = cj["operators"].map(lambda o: ",".join(str(x) for x in (o or [])))
    cj = take(cj, args.n * 3, args.seed)
    cj["need_preserved"] = ""
    cj[["rag", "method", "id", "question_orig", "question_coll", "operators_used", "need_preserved"]].to_csv(
        out / "coll_check.csv", index=False)

    rows = []
    for cond, name in (("m3", "prem"), ("m2", "coll")):
        m = get(cond).reset_index()
        m = m[m["hallucinated"] == 1]
        m = take(m, args.n * 2, args.seed)
        m["mutation"] = name
        rows.append(m)
    fj = pd.concat(rows)
    fj["verdict"] = ""
    cols = [x for x in ["mutation", "rag", "method", "id", "question", "expected_behavior", "reference_answer", "response"]
            if x in fj.columns]
    fj[cols + ["verdict"]].to_csv(out / "fail_check.csv", index=False)
    print(f"wrote prem_check.csv ({len(pj)} rows), coll_check.csv ({len(cj)}), fail_check.csv ({len(fj)}) to {out}")


def rate(s, label):
    s = pd.to_numeric(s, errors="coerce").dropna()
    n, k = len(s), int(s.sum())
    lo, hi = wilson(k, n)
    print(f"{label}: {100 * k / max(n, 1):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]  (n={n})")


def score(args):
    out = Path(args.out)
    pc = pd.read_csv(out / "prem_check.csv")
    rate(pc["behavior_ok"], "Premise Injection: expected behavior right (all)")
    for beh, g in pc.groupby("expected_behavior"):
        rate(g["behavior_ok"], f"   {beh}")
    cc = pd.read_csv(out / "coll_check.csv")
    rate(cc["need_preserved"], "Colloquialization: information need preserved")
    fc = pd.read_csv(out / "fail_check.csv")
    fc["verdict"] = fc["verdict"].astype(str).str.strip().str.lower()
    print("\nFailure verdicts (share of checked hallucinations):")
    print((fc.groupby("mutation")["verdict"].value_counts(normalize=True) * 100).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "score"])
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/validation")
    a = ap.parse_args()
    sample(a) if a.mode == "sample" else score(a)
