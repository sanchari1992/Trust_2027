"""Numbers that describe a set of prompts by its entities and graph templates, and compare it with real queries.

A "record" is {"template_id", "shape", "labels": [entity types], "features": {25 features}}; records come from the
database (records_from_db) or straight from the analyzer (records_from_analyses).

  distinct_templates / distinct_shapes   how many different graph templates / shapes occur
  template_entropy / shape_entropy       Shannon entropy divided by log2(distinct): 1 = evenly spread, 0 = all the same
  top1_template_share                    share of the commonest template (lower = more varied)
  singleton_share                        templates that occur once / distinct templates
  ents_per_query, share_with_entity      named entities + concepts per prompt; prompts with at least one
  entity_type_entropy                    spread of entity types
Against a real-query set (lower JS = closer; JS is Jensen-Shannon divergence in bits, 0 to 1):
  shape_js, template_js, entity_type_js  divergence of the distributions of shapes, fine templates, entity types
  real_shape_coverage                    share of REAL prompts whose shape also occurs in this set
  shape_in_real                          share of THIS set whose shape also occurs among the real prompts
  template_in_real                       the same for fine templates
"""
import math
from collections import Counter

import numpy as np
import pandas as pd

from src.analysis.features import ALL_NUMERIC_FEATURES


def records_from_analyses(analyses):
    return [{"template_id": a["template_id"], "shape": a["shape"],
             "labels": [e["label"] for e in a["entities"]], "features": a["features"]} for a in analyses]


def records_from_db(db, kind=None, method=None, role="question", ids=None):
    rows = db.graph_rows(kind=kind, role=role, method=method)
    if ids is not None:
        ids = set(ids)
        rows = [r for r in rows if r["item_id"] in ids]
    labels = {}
    for r in db.ent.execute("SELECT item_id, label FROM entities WHERE role=?", (role,)):
        labels.setdefault(r["item_id"], []).append(r["label"])
    return [{"id": r["item_id"], "template_id": r["template_id"], "shape": r["shape"],
             "labels": labels.get(r["item_id"], []), "features": r["features"], "meta": r["meta"],
             "method": r["method"]} for r in rows]


def features_df(records):
    df = pd.DataFrame([r["features"] for r in records])
    for c in ALL_NUMERIC_FEATURES:
        if c not in df:
            df[c] = np.nan
    return df[ALL_NUMERIC_FEATURES].astype(float)


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits between two Counters (0 identical, 1 disjoint)."""
    keys = set(p) | set(q)
    sp, sq = sum(p.values()), sum(q.values())
    if not keys or sp == 0 or sq == 0:
        return float("nan")
    a = np.array([p.get(k, 0) / sp for k in keys])
    b = np.array([q.get(k, 0) / sq for k in keys])
    m = (a + b) / 2

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())

    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def norm_entropy(c):
    total, k = sum(c.values()), len(c)
    if k <= 1 or total == 0:
        return 0.0
    h = -sum((v / total) * math.log2(v / total) for v in c.values())
    return h / math.log2(k)


def describe(records):
    tpl = Counter(r["template_id"] for r in records)
    shp = Counter(r["shape"] for r in records)
    lab = Counter(l for r in records for l in r["labels"])
    n = len(records)
    return {
        "n": n, "distinct_templates": len(tpl), "template_entropy": norm_entropy(tpl),
        "top1_template_share": (max(tpl.values()) / n) if n else float("nan"),
        "singleton_share": (sum(1 for v in tpl.values() if v == 1) / len(tpl)) if tpl else float("nan"),
        "distinct_shapes": len(shp), "shape_entropy": norm_entropy(shp),
        "ents_per_query": float(np.mean([len(r["labels"]) for r in records])) if n else float("nan"),
        "share_with_entity": float(np.mean([bool(r["labels"]) for r in records])) if n else float("nan"),
        "distinct_entity_types": len(lab), "entity_type_entropy": norm_entropy(lab),
    }


def against_real(records, real):
    if not records or not real:
        return {}
    t, s = Counter(r["template_id"] for r in records), Counter(r["shape"] for r in records)
    rt, rs = Counter(r["template_id"] for r in real), Counter(r["shape"] for r in real)
    lab = Counter(l for r in records for l in r["labels"])
    rlab = Counter(l for r in real for l in r["labels"])
    n, nr = len(records), len(real)
    return {
        "shape_js": js_divergence(s, rs), "template_js": js_divergence(t, rt), "entity_type_js": js_divergence(lab, rlab),
        "real_shape_coverage": sum(v for k, v in rs.items() if k in s) / nr,
        "shape_in_real": sum(v for k, v in s.items() if k in rs) / n,
        "template_in_real": sum(v for k, v in t.items() if k in rt) / n,
    }


def summarize(records, real=None):
    return {**describe(records), **against_real(records, real)}


def sample(records, k, seed=42):
    if k and len(records) > k:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(records), k, replace=False)
        return [records[i] for i in sorted(idx)]
    return records
