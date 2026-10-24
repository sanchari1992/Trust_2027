"""Prompt graphs for the web page (README 34, docs/WEBPAGE.md).

The graph is the SAME one the paper uses to describe how a question is built (src/analysis/graph_features.py): one node per
content word or named entity, one edge per dependency relation. The page draws it for the prompt the user typed and, after a
mutation, for the mutated prompt with the differences marked, so a user can see WHAT a mutation changed in the structure of the
question (a new disconnected part for a premise, new nodes for a background sentence, renamed nodes for typos or synonyms).
No LLM is called here.
"""
from collections import defaultdict

from src.analysis.complexity_features import extract as complexity_extract
from src.analysis.graph_features import get_nlp, graph_features, question_graph_from_doc

SHOWN_STATS = ["n_nodes", "n_edges", "max_depth", "n_components", "n_entities"]
SHOWN_COMPLEXITY = ["tree_depth", "mean_dep_distance", "fk_grade", "flesch_ease", "gunning_fog"]


def _depths(g):
    """Depth of every node = number of steps from the root of its component (roots have no incoming edge)."""
    roots = [n for n in g.nodes if g.in_degree(n) == 0] or list(g.nodes)[:1]
    depth, stack = {}, [(r, 0) for r in roots]
    while stack:
        n, d = stack.pop()
        if n in depth and depth[n] <= d:
            continue
        depth[n] = d
        stack.extend((c, d + 1) for c in g.successors(n))
    for n in g.nodes:
        depth.setdefault(n, 0)
    return depth


def build(text, nlp):
    """-> {"nodes": [...], "edges": [...], "stats": {...}, "complexity": {...}} for one prompt."""
    doc = nlp(text)
    g = question_graph_from_doc(doc)
    depth = _depths(g)
    comp = {}
    for i, cc in enumerate(sorted(__import__("networkx").connected_components(g.to_undirected()), key=min)):
        for n in cc:
            comp[n] = i
    nodes = [{"id": int(n), "label": doc[n].text, "lemma": g.nodes[n]["lemma"], "pos": g.nodes[n]["pos"],
              "entity": g.nodes[n].get("ent"), "depth": int(depth[n]), "component": comp.get(n, 0), "status": "kept"}
             for n in sorted(g.nodes)]
    edges = [{"source": int(a), "target": int(b), "dep": d.get("dep", "")} for a, b, d in g.edges(data=True)]
    st = graph_features(g)
    stats = {k: (None if st[k] != st[k] else (int(st[k]) if float(st[k]).is_integer() else round(float(st[k]), 2))) for k in SHOWN_STATS}
    cx = complexity_extract(text, doc)
    complexity = {k: round(float(cx[k]), 2) for k in SHOWN_COMPLEXITY}
    return {"nodes": nodes, "edges": edges, "stats": stats, "complexity": complexity,
            "wh_word": g.graph.get("wh_word"), "n_words": len(text.split())}


def diff(before, after):
    """Mark the nodes of both graphs. Nodes are matched by lemma (each node of `before` can match at most one of `after`).
    before: nodes that disappeared get status 'removed' (a typo or synonym changes the lemma, so it shows as removed + added).
    after:  nodes that did not exist before get status 'added'. Returns (before, after) with `status` filled in."""
    pool = defaultdict(list)
    for n in before["nodes"]:
        pool[n["lemma"]].append(n)
    for n in after["nodes"]:
        if pool[n["lemma"]]:
            pool[n["lemma"]].pop(0)["status"] = "kept"
            n["status"] = "kept"
        else:
            n["status"] = "added"
    for lst in pool.values():
        for n in lst:
            n["status"] = "removed"
    return before, after


def summary(before, after):
    """Short plain-language sentence about the change, shown under the graphs."""
    a = sum(1 for n in after["nodes"] if n["status"] == "added")
    r = sum(1 for n in before["nodes"] if n["status"] == "removed")
    parts = []
    if a:
        parts.append(f"{a} new node{'s' if a != 1 else ''}")
    if r:
        parts.append(f"{r} node{'s' if r != 1 else ''} changed or removed")
    dc = int((after["stats"]["n_components"] or 0) - (before["stats"]["n_components"] or 0))
    if dc > 0:
        parts.append(f"{dc} new separate part{'s' if dc != 1 else ''} (the added sentence is not linked to the question)")
    dd = int((after["stats"]["max_depth"] or 0) - (before["stats"]["max_depth"] or 0))
    if dd:
        parts.append(f"graph depth {'+' if dd > 0 else ''}{dd}")
    return "; ".join(parts) if parts else "the content-word structure is unchanged (only wording, spelling or style changed)"


class GraphService:
    def __init__(self, cfg):
        self.model = cfg["real_queries"]["spacy_model"]
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = get_nlp(self.model)
        return self._nlp

    def graph(self, text):
        return build(text, self.nlp)

    def compare(self, original, mutated):
        b, a = build(original, self.nlp), build(mutated, self.nlp)
        diff(b, a)
        return b, a, summary(b, a)
