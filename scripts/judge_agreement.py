"""How much does the judge (evaluation.judge_llm, GPT-6 Luna) agree with a second, reference judge?  (README 14.12)

Takes answers that are ALREADY stored by the evaluate step, judges a sample of them twice (with the judge in
`evaluation.judge_llm`, and with a reference judge: by default the generation model `llm:`, or any other model you
give with --reference-llm) and reports, per expected behaviour:
  * label agreement and Cohen's kappa (correct / incorrect / abstained, corrected / ..., flagged / ...)
  * agreement on "correct" and, if the contexts were stored, on "unfaithful" and "hallucinated"
and writes every disagreement to a CSV so you can read them and decide which judge is right.

    python scripts/judge_agreement.py --profile pilot --rag medrag --n 150
    python scripts/judge_agreement.py --profile paper --rag medrag --n 300 --conds baseline,m1,m2,m3
    # a stronger reference judge from another provider:
    python scripts/judge_agreement.py --profile pilot --reference-llm "{\"model\": \"claude-haiku-4-5\", \"base_url\": \"https://api.anthropic.com/v1\", \"api_key_env\": \"ANTHROPIC_API_KEY\", \"thinking\": null}"

For the faithfulness part the retrieved contexts must have been stored: run the evaluate step with
`evaluation.store_contexts: true` for the validation run. Without them only the labels are compared.
Rule of thumb for the paper: report kappa; below about 0.6 the cheap judge is not a safe choice.
Cost: about as many judge calls as judging those answers twice (4 or 5 calls per answer per judge).
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.benchgen.common import read_jsonl  # noqa: E402
from src.benchgen.runner import METHODS  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.evalrun.judge import Judge  # noqa: E402
from src.evalrun.runner import eval_file  # noqa: E402
from src.mutate.common import condition_file  # noqa: E402
from src.rags.base import is_refusal  # noqa: E402


def kappa(a, b):
    """Cohen's kappa of two label lists."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def collect(cfg, rag, methods, conds, n, seed):
    items = []
    for m in methods:
        for c in conds:
            f = eval_file(cfg, c, rag, m)
            cf = condition_file(cfg, c, rag, m)
            if not f.exists() or not cf.exists():
                continue
            src = {r["id"]: r for r in read_jsonl(cf)}
            for r in read_jsonl(f):
                if r.get("response") and r["id"] in src and not r.get("judge_error"):
                    items.append((m, c, src[r["id"]], r))
    random.Random(seed).shuffle(items)
    return items[:n] if n else items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--rag", default="medrag")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--conds", default="baseline,m1,m2,m3")
    ap.add_argument("--n", type=int, default=150, help="answers to judge twice (default 150; 0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference-llm", default=None,
                    help='JSON like {"model": ..., "base_url": ..., "api_key_env": ..., "thinking": null}; default: the generation model llm:')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(profile=args.profile)
    items = collect(cfg, args.rag, args.methods.split(","), args.conds.split(","), args.n, args.seed)
    if not items:
        sys.exit("No stored answers found. Run the evaluate step first (same --profile).")
    ref_cfg = {**cfg, "evaluation": {**cfg["evaluation"], "judge_llm": json.loads(args.reference_llm) if args.reference_llm else None}}
    dec, ref = Judge(cfg), Judge(ref_cfg)  # dec = the judge under test (evaluation.judge_llm)
    if dec.id == ref.id:
        sys.exit(f"Both judges are {dec.id}. Set evaluation.judge_llm to a different model, or give --reference-llm.")
    print(f"{len(items)} answers; judge under test {dec.id}; reference judge {ref.id}")
    phrase = ref.phrase

    def one(it):
        m, c, row, rec = it
        resp = rec["response"]
        if is_refusal(resp, phrase):
            return None  # both judges skip refusals: nothing to compare
        out = {"method": m, "condition": c, "id": rec["id"], "behavior": rec.get("expected_behavior") or "answer",
               "question": rec["question"], "response": resp, "reference": rec.get("reference_answer")}
        out["label_ref"], out["label_dec"] = ref.label(row, resp), dec.label(row, resp)
        ctx = rec.get("contexts")
        if ctx:
            res = {"answer": resp, "contexts": ctx}
            a, b = ref.evaluate(row, res), dec.evaluate(row, res)
            for k in ("unfaithful", "hallucinated", "correct"):
                out[k + "_ref"], out[k + "_dec"] = a.get(k), b.get(k)
        return out

    with ThreadPoolExecutor(cfg["evaluation"]["judge_concurrency"]) as pool:
        rows = [r for r in pool.map(one, items) if r]
    df = pd.DataFrame(rows).dropna(subset=["label_ref", "label_dec"])
    out_dir = Path(args.out) if args.out else ROOT / "results" / "judge_agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "judged_twice.csv", index=False, encoding="utf-8")
    dis = df[df["label_ref"] != df["label_dec"]]
    dis.to_csv(out_dir / "label_disagreements.csv", index=False, encoding="utf-8")

    lines = [f"# Judge agreement: {dec.id} (under test)  vs  {ref.id} (reference)", "",
             f"{len(df)} answers judged twice (refusals excluded: both judges skip them). Disagreements: `label_disagreements.csv`.", "",
             "| behaviour | n | label agreement | Cohen's kappa |", "|---|---|---|---|"]
    for beh, g in list(df.groupby("behavior")) + [("ALL", df)]:
        lines.append(f"| {beh} | {len(g)} | {(g.label_ref == g.label_dec).mean():.3f} | {kappa(g.label_ref.tolist(), g.label_dec.tolist()):.3f} |")
    if "unfaithful_ref" in df and df["unfaithful_ref"].notna().any():
        f = df.dropna(subset=["unfaithful_ref", "unfaithful_dec"])
        lines += ["", "Faithfulness (needs stored contexts):", "", "| measure | n | agreement | kappa | rate (reference) | rate (judge under test) |", "|---|---|---|---|---|---|"]
        for k in ("unfaithful", "hallucinated", "correct"):
            a, b = f[k + "_ref"].astype(bool).tolist(), f[k + "_dec"].astype(bool).tolist()
            lines.append(f"| {k} | {len(f)} | {sum(x == y for x, y in zip(a, b)) / len(a):.3f} | {kappa(a, b):.3f} | {sum(a) / len(a):.3f} | {sum(b) / len(b):.3f} |")
    else:
        lines += ["", "Faithfulness not compared: the contexts were not stored. Rerun the evaluate step with `evaluation.store_contexts: true`."]
    conf = defaultdict(Counter)
    for _, r in df.iterrows():
        conf[r.behavior][(r.label_ref, r.label_dec)] += 1
    lines += ["", "Confusion (rows: reference judge, columns: judge under test):", ""]
    for beh, c in conf.items():
        lines.append(f"- {beh}: " + ", ".join(f"{a}->{b}: {n}" for (a, b), n in sorted(c.items())))
    (out_dir / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
