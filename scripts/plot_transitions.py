"""Which mutation exposes failures, which one fixes some, which one is consistent? (README section 14.9)

    python scripts/plot_transitions.py                        # all three outcomes
    python scripts/plot_transitions.py --outcome hallucinated # one of: hallucinated | wrong_behavior | not_correct

One figure per outcome, three panels (PNG and PDF, in <evaluation.output_dir>/_report/):

  A  What each mutation does to the benchmark's questions.
     Right (orange): questions fine in the benchmark that now FAIL (newly failing).
     Left (green):   questions failing in the benchmark that are now fine (fixed).
     Black diamond:  net change.  Hatched bars: the controls (no mutation): that is the noise to beat.
  B  Overlap of the newly failing questions: among questions fine in the benchmark, how many fail under
     ONLY M1, only M2, only M3, M1+M2, ... all three (an UpSet plot). Big single bars = each mutation finds its
     own failures; a big "all three" bar = a group of fragile questions that any rewording breaks.
  C  Consistency: rows = benchmark fine / failing; columns = in how many of the three mutated versions
     the question fails (0 to 3). Top-right = fragile, bottom-right = fails everywhere (hard questions).

and a table `transition_summary.md` (+ csv): per mutation newly failing, fixed, failing in both, flip rate,
and how much of its newly failing set is unique to it vs shared with the other two mutations.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.evalrun.analysis import df_to_md, load_results
from src.mutate.common import CONDITION_LABELS

KEYS = ["rag", "method", "id"]
OUTCOMES = {"hallucinated": "Hallucinated", "wrong_behavior": "Wrong behaviour", "failed": "Not correct (wrong or declined)"}
SLUG = {"hallucinated": "hallucinated", "wrong_behavior": "wrong_behavior", "failed": "not_correct"}
MUT = ["m1", "m2", "m3"]
NAME = {"m1": "M1 length", "m2": "M2 linguistic", "m3": "M3 user info",
        "c_rerun": "Control: same question again", "c_rerun2": "Control: same question, 3rd time", "c_rerun3": "Control: same question, 4th time", "c_pad": "Control: off-topic filler"}
BAD, GOOD, NEUTRAL = "#D55E00", "#009E73", "#6B7280"  # colour-blind safe: vermillion / bluish green / grey


def wide(df, col):
    return df.pivot_table(index=KEYS, columns="condition", values=col, aggfunc="first", observed=True)


def stats(w):
    """Per condition: newly failing, fixed, failing in both, fine in both, plus the overlap with the other mutations."""
    rows = []
    mut = [c for c in MUT if c in w]
    for c in [c for c in MUT + ["c_rerun", "c_rerun2", "c_rerun3", "c_pad"] if c in w]:
        s = w[["baseline", c]].dropna()
        b, x = s["baseline"].to_numpy(), s[c].to_numpy()
        new = (b == 0) & (x == 1)
        row = {"condition": NAME[c], "_c": c, "questions": len(s), "newly_failing": int(new.sum()),
               "fixed": int(((b == 1) & (x == 0)).sum()), "failing_in_both": int(((b == 1) & (x == 1)).sum()),
               "fine_in_both": int(((b == 0) & (x == 0)).sum())}
        row["net_change"] = row["newly_failing"] - row["fixed"]
        row["flip_rate"] = (row["newly_failing"] + row["fixed"]) / len(s)
        if c in MUT and len(mut) == 3:
            others = w.loc[s.index[new], [m for m in mut if m != c]]
            row["newly_failing_unique_to_it"] = int((others.sum(axis=1) == 0).sum())
            row["newly_failing_shared_with_another"] = int((others.sum(axis=1) >= 1).sum())
            row["share_unique"] = row["newly_failing_unique_to_it"] / row["newly_failing"] if row["newly_failing"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def subset_counts(w):
    """Among questions fine in the benchmark: how many fail under exactly each subset of {M1, M2, M3}."""
    g = w[["baseline"] + MUT].dropna()
    g = g[g["baseline"] == 0]
    out = []
    for r in range(1, 4):
        for sub in itertools.combinations(MUT, r):
            m = np.ones(len(g), bool)
            for c in MUT:
                m &= (g[c].to_numpy() == 1) if c in sub else (g[c].to_numpy() == 0)
            out.append((sub, int(m.sum())))
    return out, len(g)


def heat(w):
    g = w[["baseline"] + MUT].dropna()
    n_fail = g[MUT].sum(axis=1).astype(int).to_numpy()
    b = g["baseline"].astype(int).to_numpy()
    return np.array([[int(((b == r) & (n_fail == k)).sum()) for k in range(4)] for r in (0, 1)])


def figure(w, st, outcome, path):
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 0.8], wspace=0.32)
    # ---- A
    ax = fig.add_subplot(gs[0])
    st = st.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(st))
    ctrl = st["_c"].str.startswith("c_")
    for i, r in st.iterrows():
        h = "///" if ctrl[i] else None
        ax.barh(i, r["newly_failing"], color=BAD, alpha=0.45 if ctrl[i] else 0.95, hatch=h, edgecolor="white")
        ax.barh(i, -r["fixed"], color=GOOD, alpha=0.45 if ctrl[i] else 0.95, hatch=h, edgecolor="white")
        if r["newly_failing"]:
            ax.text(r["newly_failing"] + 0.3, i, str(r["newly_failing"]), va="center", fontsize=9)
        if r["fixed"]:
            ax.text(-r["fixed"] - 0.3, i, str(r["fixed"]), va="center", ha="right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\nnet {n:+d}" for c, n in zip(st["condition"], st["net_change"])], fontsize=9)
    lim = max(st[["newly_failing", "fixed"]].max()) * 1.25 + 1
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("questions   (green = fixed by the mutation  |  orange = newly failing)")
    ax.set_title("A  What each mutation does", loc="left", fontsize=11)
    ax.text(0.99, 0.01, "hatched = controls (no mutation)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=NEUTRAL)
    # ---- B (UpSet)
    sets, n_fine = subset_counts(w)
    sets = sorted(sets, key=lambda t: -t[1])
    sets = [t for t in sets if t[1] > 0] or sets  # hide empty combinations
    sub = gs[1].subgridspec(2, 1, height_ratios=[3, 1.1], hspace=0.05)
    axb, axm = fig.add_subplot(sub[0]), fig.add_subplot(sub[1], sharex=None)
    xs = np.arange(len(sets))
    axb.bar(xs, [c for _, c in sets], color=[BAD if len(s) == 1 else NEUTRAL for s, _ in sets])
    for x, (_, c) in zip(xs, sets):
        axb.text(x, c + 0.2, str(c), ha="center", fontsize=9)
    axb.set_xticks([])
    axb.set_ylabel(f"questions (of {n_fine} fine in the benchmark)")
    axb.set_title("B  Which mutations break the same questions", loc="left", fontsize=11)
    axb.text(0.99, 0.97, "orange = one mutation only\ngrey = shared by two or three", transform=axb.transAxes,
             ha="right", va="top", fontsize=8, color=NEUTRAL)
    for i, name in enumerate(MUT):
        axm.scatter(xs, [i] * len(xs), s=40, color="#D1D5DB", zorder=1)
        on = [x for x, (s, _) in zip(xs, sets) if name in s]
        axm.scatter(on, [i] * len(on), s=40, color="black", zorder=2)
    for x, (s, _) in zip(xs, sets):
        ys = [MUT.index(c) for c in s]
        if len(ys) > 1:
            axm.plot([x, x], [min(ys), max(ys)], color="black", lw=2, zorder=1.5)
    axm.set_yticks(range(3))
    axm.set_yticklabels([NAME[m] for m in MUT], fontsize=9)
    axm.set_xticks([])
    axm.set_xlim(axb.get_xlim())
    axm.set_ylim(-0.6, 2.6)
    axm.invert_yaxis()
    for a in (axb, axm):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    # ---- C
    axc = fig.add_subplot(gs[2])
    H = heat(w)
    axc.imshow(np.log1p(H), cmap="Greys", vmin=0, vmax=np.log1p(H.max()) * 1.6, aspect="auto")
    for (r, k), v in np.ndenumerate(H):
        axc.text(k, r, str(v), ha="center", va="center", fontsize=12, color="white" if np.log1p(v) > np.log1p(H.max()) * 0.8 else "black")
    axc.set_xticks(range(4))
    axc.set_xticklabels(["0\n(none)", "1", "2", "3\n(all)"])
    axc.set_yticks([0, 1])
    axc.set_yticklabels(["fine in the\nbenchmark", "failing in the\nbenchmark"])
    axc.set_xlabel("in how many of M1, M2, M3 the question fails")
    axc.set_title("C  Consistency", loc="left", fontsize=11)
    fig.suptitle(f"Outcome: {OUTCOMES[outcome]}. Each question compared with its own mutated versions", fontsize=12, y=1.02)
    fig.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", choices=list(SLUG.values()) + ["all"], default="all")
    a = ap.parse_args()
    cfg = load_config()
    df = load_results(cfg)
    df = df[~df["condition"].astype(str).isin(["m_union", "c_union", "m123"])]
    out = ROOT / cfg["evaluation"]["output_dir"] / "_report"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Which mutation exposes failures, which fixes some, which is consistent?", "",
          "Figures: `transitions_<outcome>.png` / `.pdf` (A: newly failing vs fixed per mutation and control; "
          "B: overlap of the newly failing questions; C: how many of the three mutations break a question).", ""]
    for col, title in OUTCOMES.items():
        if a.outcome not in ("all", SLUG[col]):
            continue
        w = wide(df, col)
        if "baseline" not in w or not any(m in w for m in MUT) or not all(m in w for m in MUT):
            print(f"skipped {title}: needs baseline, m1, m2, m3")
            continue
        st = stats(w)
        figure(w, st, col, out / f"transitions_{SLUG[col]}")
        st.drop(columns="_c").to_csv(out / f"transition_summary_{SLUG[col]}.csv", index=False)
        sets, n_fine = subset_counts(w)
        md += [f"## {title}", "", df_to_md(st.drop(columns="_c")), "",
               f"Among the {n_fine} questions fine in the benchmark, failing under exactly:", "",
               df_to_md(pd.DataFrame([{"mutations that fail": " + ".join(NAME[c].split()[0] for c in s), "questions": n}
                                      for s, n in sorted(sets, key=lambda t: -t[1])])), ""]
        print(f"\n{title}\n" + st.drop(columns="_c").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    (out / "transition_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nFigures and tables in {out}")


if __name__ == "__main__":
    main()
