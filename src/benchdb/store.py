"""The benchmark database: a folder `<name>.benchmark/` with

    entities.sqlite   items (question, answer, passages, meta) + entities of every question / answer / passage
    graphs.sqlite     one dependency graph per question and per answer, its template and shape, a template library
    manifest.json     what is inside, which spaCy model made it, when
    exports/          (on request) entities.csv, graphs.jsonl, templates.csv

Item kinds:  benchmark (generated question + reference answer),  real (a real user query, no answer),
             example (prompts the benchmark creator supplied with --add_graphs, e.g. what real users in this field ask).
Plain sqlite3, no server. Everything can also be read with any SQLite tool.
"""
import csv
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import sqlite3

VERSION = 1

ENTITY_SQL = """
CREATE TABLE IF NOT EXISTS items(
    item_id TEXT PRIMARY KEY, kind TEXT, rag TEXT, method TEXT, corpus TEXT,
    question TEXT, answer TEXT, passage_ids TEXT, contexts TEXT, meta TEXT);
CREATE TABLE IF NOT EXISTS entities(
    id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT, role TEXT, text TEXT, norm TEXT, label TEXT, kind TEXT,
    start INTEGER, end INTEGER);
CREATE TABLE IF NOT EXISTS passages(passage_id TEXT PRIMARY KEY, text TEXT);
CREATE TABLE IF NOT EXISTS passage_entities(
    id INTEGER PRIMARY KEY AUTOINCREMENT, passage_id TEXT, text TEXT, norm TEXT, label TEXT, kind TEXT);
CREATE INDEX IF NOT EXISTS ix_ent_item ON entities(item_id);
CREATE INDEX IF NOT EXISTS ix_ent_norm ON entities(norm);
CREATE INDEX IF NOT EXISTS ix_ent_label ON entities(label);
CREATE INDEX IF NOT EXISTS ix_items_kind ON items(kind, method);
CREATE INDEX IF NOT EXISTS ix_pent_norm ON passage_entities(norm);
"""
GRAPH_SQL = """
CREATE TABLE IF NOT EXISTS graphs(
    item_id TEXT, role TEXT, kind TEXT, template_id TEXT, template TEXT, shape TEXT, text_template TEXT,
    n_nodes INTEGER, n_edges INTEGER, max_depth REAL, n_sentences INTEGER, n_entities INTEGER,
    slots TEXT, features TEXT, graph_json TEXT, PRIMARY KEY(item_id, role));
CREATE TABLE IF NOT EXISTS template_library(
    template_id TEXT, role TEXT, template TEXT, n_items INTEGER, n_benchmark INTEGER, n_real INTEGER,
    n_example INTEGER, example_text TEXT, PRIMARY KEY(template_id, role));
CREATE INDEX IF NOT EXISTS ix_g_tpl ON graphs(template_id);
CREATE INDEX IF NOT EXISTS ix_g_shape ON graphs(shape);
"""


def _j(v):
    return json.dumps(v, ensure_ascii=False)


class BenchDB:
    def __init__(self, path):
        self.path = Path(path)
        self.ent = sqlite3.connect(self.path / "entities.sqlite")
        self.gr = sqlite3.connect(self.path / "graphs.sqlite")
        self.ent.row_factory = self.gr.row_factory = sqlite3.Row

    # ------------------------------------------------------------ create / open
    @classmethod
    def create(cls, path, overwrite=False, info=None):
        path = Path(path)
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} exists. Open it to add to it, or pass --overwrite to start over.")
            shutil.rmtree(path)
        path.mkdir(parents=True)
        db = cls(path)
        db.ent.executescript(ENTITY_SQL)
        db.gr.executescript(GRAPH_SQL)
        db.manifest = {"version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "history": [], **(info or {})}
        db._save_manifest()
        return db

    @classmethod
    def open(cls, path):
        path = Path(path)
        if not (path / "entities.sqlite").exists():
            raise FileNotFoundError(f"{path} is not a benchmark database (no entities.sqlite). "
                                    "Create one with scripts/create_bench.py.")
        db = cls(path)
        db.manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return db

    def _save_manifest(self):
        self.manifest["counts"] = self.counts()
        (self.path / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def close(self):
        self.ent.close()
        self.gr.close()

    # ------------------------------------------------------------ writing
    def add_items(self, rows, analyzer, kind="benchmark", passage_entities=True, batch=256, progress=True,
                  skip_existing=True):
        """rows: dicts with id, question (the prompt text), optional answer, contexts, passage_ids, rag, method,
        corpus, meta. Analyses the prompt (role question) and the answer (role answer); passages once each."""
        have = {r[0] for r in self.ent.execute("SELECT item_id FROM items")} if skip_existing else set()
        todo = [r for r in rows if r["id"] not in have]
        n_new = 0
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            q_an = list(analyzer.analyze_many([r["question"] for r in chunk], "question"))
            with_ans = [(i, r) for i, r in enumerate(chunk) if (r.get("answer") or "").strip()]
            a_an = dict(zip([i for i, _ in with_ans],
                            analyzer.analyze_many([r["answer"] for _, r in with_ans], "answer"))) if with_ans else {}
            new_pass = {}
            if passage_entities:
                for r in chunk:
                    ids = r.get("passage_ids") or []
                    for k, ctx in enumerate(r.get("contexts") or []):
                        pid = ids[k] if k < len(ids) else "ctx_" + hashlib.sha1(ctx.encode("utf-8")).hexdigest()[:12]
                        if pid not in new_pass and not self.ent.execute(
                                "SELECT 1 FROM passages WHERE passage_id=?", (pid,)).fetchone():
                            new_pass[pid] = ctx
            p_ids = list(new_pass)
            p_an = list(analyzer.analyze_many([new_pass[p] for p in p_ids], "passage")) if p_ids else []
            for i, r in enumerate(chunk):
                self.ent.execute(
                    "INSERT OR REPLACE INTO items VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (r["id"], kind, r.get("rag"), r.get("method"), r.get("corpus"), r["question"], r.get("answer"),
                     _j(r.get("passage_ids") or []), _j(r.get("contexts") or []), _j(r.get("meta") or {})))
                for role, an in (("question", q_an[i]), ("answer", a_an.get(i))):
                    if an is None:
                        continue
                    self.ent.executemany(
                        "INSERT INTO entities(item_id,role,text,norm,label,kind,start,end) VALUES(?,?,?,?,?,?,?,?)",
                        [(r["id"], role, e["text"], e["norm"], e["label"], e["kind"], e["start"], e["end"])
                         for e in an["entities"]])
                    st = an["stats"]
                    self.gr.execute(
                        "INSERT OR REPLACE INTO graphs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r["id"], role, kind, an["template_id"], an["template"], an["shape"], an["text_template"],
                         st["n_nodes"], st["n_edges"], st["max_depth"], st["n_sentences"], st["n_entities"],
                         _j(an["slots"]), _j(an["features"]), _j(an["graph"])))
            for pid, an in zip(p_ids, p_an):
                self.ent.execute("INSERT OR REPLACE INTO passages VALUES(?,?)", (pid, new_pass[pid]))
                self.ent.executemany(
                    "INSERT INTO passage_entities(passage_id,text,norm,label,kind) VALUES(?,?,?,?,?)",
                    [(pid, e["text"], e["norm"], e["label"], e["kind"]) for e in an["entities"]])
            self.ent.commit()
            self.gr.commit()
            n_new += len(chunk)
            if progress:
                print(f"  stored {n_new}/{len(todo)} items", flush=True)
        self.manifest["history"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind,
                                         "added": n_new, "skipped_existing": len(rows) - len(todo)})
        self.manifest.update(spacy_model=analyzer.model_name, concepts=analyzer.concepts)
        self.finalize()
        return n_new

    def finalize(self):
        """Rebuild the template library and the manifest counts."""
        self.gr.execute("DELETE FROM template_library")
        rows = self.gr.execute(
            "SELECT template_id, role, template, kind, COUNT(*) n, MIN(item_id) first_item FROM graphs "
            "GROUP BY template_id, role, kind").fetchall()
        lib = {}
        for r in rows:
            d = lib.setdefault((r["template_id"], r["role"]), {"template": r["template"], "n": Counter(), "ex": r["first_item"]})
            d["n"][r["kind"]] += r["n"]
        for (tid, role), d in lib.items():
            ex = self.ent.execute("SELECT question, answer FROM items WHERE item_id=?", (d["ex"],)).fetchone()
            text = (ex["answer"] if role == "answer" else ex["question"]) if ex else None
            n = d["n"]
            self.gr.execute("INSERT INTO template_library VALUES(?,?,?,?,?,?,?,?)",
                            (tid, role, d["template"], sum(n.values()), n["benchmark"], n["real"], n["example"],
                             (text or "")[:300]))
        self.gr.commit()
        self._save_manifest()

    # ------------------------------------------------------------ reading
    def counts(self):
        c = {r["kind"]: r["n"] for r in self.ent.execute("SELECT kind, COUNT(*) n FROM items GROUP BY kind")}
        c["entities"] = self.ent.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        c["passages"] = self.ent.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
        c["graphs"] = self.gr.execute("SELECT COUNT(*) FROM graphs").fetchone()[0]
        c["templates"] = self.gr.execute("SELECT COUNT(DISTINCT template_id) FROM graphs WHERE role='question'").fetchone()[0]
        return c

    @staticmethod
    def _item(r):
        d = dict(r)
        for k in ("passage_ids", "contexts", "meta"):
            d[k] = json.loads(d[k]) if d.get(k) else ([] if k != "meta" else {})
        d["id"] = d.pop("item_id")
        return d

    def items(self, kind=None, method=None, limit=None):
        q, args = "SELECT * FROM items WHERE 1=1", []
        if kind:
            q += " AND kind=?"
            args.append(kind)
        if method:
            q += " AND method=?"
            args.append(method)
        q += " ORDER BY rowid" + (f" LIMIT {int(limit)}" if limit else "")
        return [self._item(r) for r in self.ent.execute(q, args)]

    def item(self, item_id):
        r = self.ent.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
        return self._item(r) if r else None

    def entities(self, item_id, role="question"):
        return [dict(r) for r in self.ent.execute(
            "SELECT text,norm,label,kind,start,end FROM entities WHERE item_id=? AND role=? ORDER BY start",
            (item_id, role))]

    def graph(self, item_id, role="question"):
        r = self.gr.execute("SELECT * FROM graphs WHERE item_id=? AND role=?", (item_id, role)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("slots", "features", "graph_json"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        return d

    def graph_rows(self, kind=None, role="question", method=None):
        """Graph rows (without the big graph_json), optionally restricted to a kind / method."""
        q = ("SELECT g.item_id, g.role, g.kind, g.template_id, g.shape, g.n_nodes, g.n_edges, g.max_depth, "
             "g.n_sentences, g.n_entities, g.features, i.method, i.meta FROM graphs g JOIN items i ON i.item_id=g.item_id "
             "WHERE g.role=?")
        args = [role]
        if kind:
            q += " AND g.kind=?"
            args.append(kind)
        if method:
            q += " AND i.method=?"
            args.append(method)
        return self._graph_rows_attached(q, args)

    def _graph_rows_attached(self, q, args):
        # graphs.sqlite and entities.sqlite are separate files: join through ATTACH on a throw-away connection
        con = sqlite3.connect(self.path / "graphs.sqlite")
        con.row_factory = sqlite3.Row
        con.execute("ATTACH DATABASE ? AS e", (str(self.path / "entities.sqlite"),))
        q = q.replace("JOIN items i", "JOIN e.items i")
        rows = []
        for r in con.execute(q, args):
            d = dict(r)
            d["features"] = json.loads(d["features"]) if d.get("features") else {}
            d["meta"] = json.loads(d["meta"]) if d.get("meta") else {}
            rows.append(d)
        con.close()
        return rows

    def template_counts(self, role="question", level="template", kind=None, method=None):
        col = "template_id" if level == "template" else "shape"
        return Counter(r[col] for r in self.graph_rows(kind=kind, role=role, method=method))

    def entity_label_counts(self, role="question", kind=None):
        q = "SELECT e.label, COUNT(*) n FROM entities e JOIN items i ON i.item_id=e.item_id WHERE e.role=?"
        args = [role]
        if kind:
            q += " AND i.kind=?"
            args.append(kind)
        q += " GROUP BY e.label"
        return Counter({r["label"]: r["n"] for r in self.ent.execute(q, args)})

    def top_entities(self, label=None, role="question", n=20):
        q, args = "SELECT norm, label, COUNT(*) n FROM entities WHERE role=?", [role]
        if label:
            q += " AND label=?"
            args.append(label)
        q += " GROUP BY norm, label ORDER BY n DESC LIMIT ?"
        args.append(n)
        return [dict(r) for r in self.ent.execute(q, args)]

    def passage_entity_pool(self, label=None):
        q, args = "SELECT DISTINCT norm, label FROM passage_entities", []
        if label:
            q += " WHERE label=?"
            args.append(label)
        return [dict(r) for r in self.ent.execute(q, args)]

    def example_lengths(self):
        """Word counts of the real and example prompts in this database (used as M1's target lengths)."""
        return sorted(len(r["question"].split()) for r in self.ent.execute(
            "SELECT question FROM items WHERE kind IN ('real','example')"))

    # ------------------------------------------------------------ export
    def export(self, out_dir=None):
        out = Path(out_dir or self.path / "exports")
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "entities.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "kind", "method", "role", "text", "norm", "label", "entity_kind"])
            for r in self.ent.execute("SELECT e.item_id, i.kind, i.method, e.role, e.text, e.norm, e.label, e.kind "
                                      "FROM entities e JOIN items i ON i.item_id=e.item_id"):
                w.writerow(list(r))
        with open(out / "graphs.jsonl", "w", encoding="utf-8") as f:
            for r in self.gr.execute("SELECT * FROM graphs"):
                d = dict(r)
                for k in ("slots", "features", "graph_json"):
                    d[k] = json.loads(d[k]) if d.get(k) else None
                d["graph"] = d.pop("graph_json")
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        with open(out / "templates.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["template_id", "role", "template", "n_items", "n_benchmark", "n_real", "n_example", "example_text"])
            for r in self.gr.execute("SELECT * FROM template_library ORDER BY n_items DESC"):
                w.writerow(list(r))
        return out
