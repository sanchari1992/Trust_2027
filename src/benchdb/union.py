"""The UNION of the questions several benchmark methods generated for the same RAG.

The three methods (RAGAS, ARES, DataMorgana) sometimes produce near-identical questions from the same passage.
The union keeps one of each group: questions are taken in turn from every method (so no method dominates) and a
question is dropped if its meaning (cosine similarity of local embeddings) is at least `union.dedupe_threshold`
similar to one already kept. Result: <benchmark.output_dir>/<rag>_<tag>/union.jsonl (same row format as a method
file, plus meta.union_from) and union.meta.json. Nothing is mutated or evaluated twice: a union question IS the
question of its method, so its mutated versions and evaluation results are those of that method (the reports
select them by id).
"""
import json

import numpy as np

from src.benchgen.common import read_jsonl
from src.config import ROOT, corpus_tag


def interleave(lists):
    out, i = [], 0
    while any(i < len(l) for l in lists):
        for l in lists:
            if i < len(l):
                out.append(l[i])
        i += 1
    return out


def dedupe(texts, vectors, threshold, block=512):
    """Greedy: indices to keep, in order, dropping any text too similar to an earlier kept one."""
    kept = []
    kept_vecs = np.zeros((0, vectors.shape[1]), dtype=np.float32)
    for s in range(0, len(texts), block):
        v = vectors[s:s + block]
        sim_prev = (v @ kept_vecs.T) if len(kept_vecs) else np.zeros((len(v), 0))
        add = []
        for j in range(len(v)):
            if sim_prev.shape[1] and sim_prev[j].max() >= threshold:
                continue
            if add and (v[add] @ v[j]).max() >= threshold:
                continue
            add.append(j)
        kept += [s + j for j in add]
        if add:
            kept_vecs = np.vstack([kept_vecs, v[add]])
    return kept


def build_union(cfg, rag, methods, embed=None, threshold=None):
    base = ROOT / cfg["benchmark"]["output_dir"] / f"{rag}_{corpus_tag(cfg, rag)}"
    threshold = threshold if threshold is not None else cfg.get("union", {}).get("dedupe_threshold", 0.9)
    per = {m: read_jsonl(base / f"{m}.jsonl") for m in methods if (base / f"{m}.jsonl").exists()}
    if not per:
        raise FileNotFoundError(f"No benchmark files for {rag} in {base}. Run generate_benchmark.py first.")
    rows = interleave(list(per.values()))
    if embed is None:
        from src.embeddings import get_sync_embedder

        embed = get_sync_embedder(cfg["embedding"]["model"]).encode
    texts = [r["question"] for r in rows]
    vecs = np.vstack([embed(texts[i:i + 256]) for i in range(0, len(texts), 256)]).astype(np.float32)
    keep = dedupe(texts, vecs, threshold) if threshold else list(range(len(rows)))
    out = []
    for i in keep:
        r = dict(rows[i])
        r["meta"] = {**(r.get("meta") or {}), "union_from": r.get("method")}
        out.append(r)
    with open(base / "union.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {"rag": rag, "methods": list(per), "dedupe_threshold": threshold,
            "input": {m: len(v) for m, v in per.items()}, "total_input": len(rows), "kept": len(out),
            "dropped_as_duplicates": len(rows) - len(out),
            "kept_by_method": {m: sum(1 for r in out if r.get("method") == m) for m in per}}
    (base / "union.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
