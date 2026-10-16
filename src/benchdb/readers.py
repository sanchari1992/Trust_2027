"""Reading benchmarks / prompts from files, and from a pipeline run, into the row format of the database."""
import csv
import json
from pathlib import Path

from src.benchgen.common import read_jsonl
from src.config import ROOT, corpus_tag

Q_COLS = ["question", "prompt", "query", "text", "q", "user_query", "input"]
A_COLS = ["answer", "response", "reference_answer", "ground_truth", "reference", "a", "output"]
C_COLS = ["context", "contexts", "passage", "document", "source_text", "evidence"]
ID_COLS = ["id", "qid", "question_id", "item_id", "uid"]


def _read_raw(path):
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".jsonl", ".ndjson"):
        return read_jsonl(path)
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("data") or data.get("items") or [data]
    delim = "\t" if suf in (".tsv", ".tab") else ","
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=delim))


def _pick(row, names, explicit=None):
    if explicit:
        return row.get(explicit)
    low = {k.lower(): k for k in row}
    for n in names:
        if n in low and row[low[n]] not in (None, ""):
            return row[low[n]]
    return None


def read_prompts(path, question_col=None, answer_col=None, context_col=None, id_col=None, prefix="q"):
    """A .csv / .tsv / .json / .jsonl file -> database rows. Needs one column with the prompt (question); the
    answer, the source passage and an id are optional. Column names are detected (question / prompt / query / text;
    answer / response / reference_answer; context / passage / document), or given explicitly."""
    raw = _read_raw(path)
    if not raw:
        raise ValueError(f"{path} has no rows.")
    rows, seen = [], set()
    for i, r in enumerate(raw):
        q = _pick(r, Q_COLS, question_col)
        if not q or not str(q).strip():
            continue
        a = _pick(r, A_COLS, answer_col)
        c = _pick(r, C_COLS, context_col)
        if isinstance(c, str):
            try:
                v = json.loads(c)
                c = v if isinstance(v, list) else [c]
            except (ValueError, TypeError):
                c = [c]
        iid = str(_pick(r, ID_COLS, id_col) or f"{prefix}{i + 1:06d}")
        while iid in seen:
            iid += "_dup"
        seen.add(iid)
        rows.append({"id": iid, "question": str(q).strip(), "answer": (str(a).strip() if a else None),
                     "contexts": [str(x) for x in (c or [])], "passage_ids": r.get("passage_ids") or [],
                     "rag": r.get("rag"), "method": r.get("method") or Path(path).stem, "corpus": r.get("corpus"),
                     "meta": r.get("meta") if isinstance(r.get("meta"), dict) else {}})
    if not rows:
        raise ValueError(f"No prompt column found in {path}. Columns were {sorted(raw[0])}; "
                         f"use --question-col to name the column that holds the prompt.")
    return rows


def read_run(cfg, rag, methods):
    """The generated benchmark of one RAG: every method's questions, each marked with whether it is in the union."""
    base = ROOT / cfg["benchmark"]["output_dir"] / f"{rag}_{corpus_tag(cfg, rag)}"
    union_ids = set()
    uf = base / "union.jsonl"
    if uf.exists():
        union_ids = {r["id"] for r in read_jsonl(uf)}
    rows = []
    for m in methods:
        f = base / f"{m}.jsonl"
        if not f.exists():
            print(f"  (no benchmark file for {rag}/{m} at {f}; skipped)")
            continue
        for r in read_jsonl(f):
            meta = dict(r.get("meta") or {})
            meta["in_union"] = (r["id"] in union_ids) if union_ids else None
            rows.append({**{k: r.get(k) for k in ("id", "question", "answer", "contexts", "passage_ids", "rag",
                                                  "method", "corpus")}, "meta": meta})
    return rows


def read_real(cfg, domain):
    f = ROOT / cfg["real_queries"]["output_dir"] / f"{domain}.jsonl"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Run: python scripts/prepare_real_queries.py {domain}")
    rows = []
    for r in read_jsonl(f):
        if r.get("text"):
            rows.append({"id": f"real-{domain}-{r.get('id', len(rows))}", "question": r["text"], "answer": None,
                         "contexts": [], "passage_ids": [], "rag": None, "method": r.get("source") or domain,
                         "corpus": domain, "meta": {k: r[k] for k in ("source", "domain") if k in r}})
    return rows
