"""Audit: WHY did a mutated question count as a hallucination when the original did not?

For every question that was fine in the benchmark but hallucinated in a mutated condition ("newly failing"),
the cause is classified automatically:

  behaviour_failure     the answer asserted something wrong or adopted a false / unsupported claim
                        (label incorrect / followed_false_claim / accepted_detail). A real failure.
  unsupported_but_correct  the answer was judged correct, but some statement is not in the retrieved context
                        (even counting the user's message). Usually the model's own knowledge, often true: not a
                        harmful hallucination. Also check `retrieval_hit`: a miss explains it.
  unfaithful_and_wrong  unfaithful AND judged not correct: the strongest candidate for a real hallucination.
  judge_artifact        flagged ONLY by the strict faithfulness score (hallucinated_strict) and not by the
                        adjusted one: the statement only repeated what the user said.
  also_in_control       the same question also hallucinated in a control (re-run or padding), so the mutation
                        is not needed to explain it: noise or length, not the mutation.

A question can be both a behaviour_failure and also_in_control. The category order above decides the primary
one, `also_in_control` is a separate yes/no column.
"""
import numpy as np
import pandas as pd

from src.config import ROOT
from src.evalrun.analysis import df_to_md, load_results
from src.mutate.common import CONDITION_LABELS, CONTROL_CONDITIONS, DERIVED_CONDITIONS


def _classify(r, in_control):
    if r.get("wrong_behavior") == 1.0:
        cat = "behaviour_failure"
    elif r["hallucinated"] == 1.0 and r["unfaithful"] == 1.0:
        cat = "unsupported_but_correct" if r.get("failed") == 0.0 else "unfaithful_and_wrong"
    else:
        cat = "other"
    return cat


def build_audit(df):
    key = ["rag", "method", "id"]
    base = df[df["condition"] == "baseline"].set_index(key)
    ctrl = df[df["condition"].isin(CONTROL_CONDITIONS)].groupby(key)["hallucinated"].max()
    rows = []
    for _, r in df[~df["condition"].isin(["baseline"] + CONTROL_CONDITIONS + DERIVED_CONDITIONS)].iterrows():
        k = (r["rag"], r["method"], r["id"])
        if k not in base.index or base.loc[k, "hallucinated"] != 0.0 or r["hallucinated"] != 1.0:
            continue
        in_ctl = bool(ctrl.get(k, 0.0) == 1.0)
        rows.append({"rag": r["rag"], "method": r["method"], "id": r["id"], "condition": str(r["condition"]),
                     "category": _classify(r, in_ctl), "also_in_control": in_ctl, "label": r.get("label"),
                     "faithfulness": r.get("faithfulness"), "faithfulness_strict": r.get("faithfulness_strict"),
                     "retrieval_hit": r.get("retrieval_hit"),
                     "operators": ",".join(r["operators"]) if isinstance(r["operators"], list) else r["operators"],
                     "original_question": base.loc[k, "question"], "question": r["question"],
                     "response": r.get("response")})
    # items the STRICT score would also have counted but the adjusted one does not: judge artifacts
    s = df[(df["hallucinated_strict"] == 1.0) & (df["hallucinated"] == 0.0)
           & ~df["condition"].isin(["baseline"] + CONTROL_CONDITIONS + DERIVED_CONDITIONS)]
    art = []
    for _, r in s.iterrows():
        k = (r["rag"], r["method"], r["id"])
        if k in base.index and base.loc[k, "hallucinated_strict"] == 0.0:
            art.append({"rag": r["rag"], "method": r["method"], "id": r["id"], "condition": str(r["condition"]),
                        "category": "judge_artifact", "also_in_control": bool(ctrl.get(k, 0.0) == 1.0),
                        "label": r.get("label"), "faithfulness": r.get("faithfulness"),
                        "faithfulness_strict": r.get("faithfulness_strict"),
                        "operators": ",".join(r["operators"]) if isinstance(r["operators"], list) else r["operators"],
                        "original_question": base.loc[k, "question"], "question": r["question"],
                        "response": r.get("response")})
    return pd.DataFrame(rows + art)


def write_audit(cfg, per_category=8):
    df = load_results(cfg)
    out_dir = ROOT / cfg["evaluation"]["output_dir"] / "_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    a = build_audit(df)
    if a.empty:
        print("No newly failing questions: nothing to audit.")
        return None
    a.to_csv(out_dir / "audit_newly_failing.csv", index=False)
    a["condition_label"] = a["condition"].map(CONDITION_LABELS)
    counts = (a.groupby(["condition_label", "category"]).size().unstack(fill_value=0)
              .reindex(columns=["behaviour_failure", "unfaithful_and_wrong", "unsupported_but_correct",
                                "judge_artifact", "other"], fill_value=0))
    counts["also_in_control"] = a[a["category"] != "judge_artifact"].groupby("condition_label")["also_in_control"].sum()
    counts = counts.fillna(0).astype(int).reset_index()
    L = ["# Audit: why are mutated questions counted as hallucinations?", "",
         "Questions that were fine in the original benchmark but hallucinated in a mutated condition, by cause "
         "(definitions in the header of `src/evalrun/audit.py`). `judge_artifact` rows are NOT counted as hallucinations "
         "in the main tables; they are shown so you can see what the strict score would have added.", "",
         df_to_md(counts), ""]
    for cat in ["behaviour_failure", "unfaithful_and_wrong", "unsupported_but_correct", "judge_artifact", "other"]:
        sub = a[a["category"] == cat]
        if sub.empty:
            continue
        L += [f"## {cat} ({len(sub)})", ""]
        for _, r in sub.head(per_category).iterrows():
            L += [f"**{r['condition_label']}** | {r['rag']}/{r['method']} | {r['id']} | label `{r['label']}` | "
                  f"faithfulness {r['faithfulness']} (strict {r['faithfulness_strict']}) | retrieval_hit {r.get('retrieval_hit')} | operators {r['operators']}"
                  f"{' | ALSO HALLUCINATED IN A CONTROL' if r['also_in_control'] else ''}", "",
                  f"- original: {r['original_question']}", f"- mutated: {r['question']}",
                  f"- response: {str(r['response'])[:600]}", ""]
    (out_dir / "audit_newly_failing.md").write_text("\n".join(L), encoding="utf-8")
    return out_dir / "audit_newly_failing.md", counts
