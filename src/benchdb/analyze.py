"""Turns a text into (1) named entities, (2) a dependency graph, (3) graph templates.

ENTITIES   spaCy named entities (PERSON, ORG, GPE, ...) plus, optionally, noun phrases that are not named
           entities ("concept" entities, label CONCEPT). General spaCy models know little medical vocabulary, so
           for a medical corpus the concepts carry most of the information; a biomedical model such as
           scispaCy's en_core_sci_md can be used instead (benchdb.spacy_model in config.yaml).

GRAPH      the same lightweight graph as the realism analysis (src/analysis/graph_features.py): one node per
           content word or entity token, one edge per dependency between kept tokens.

TEMPLATES  two levels, both computed without any LLM call:
           template  the graph with the words removed: every node is replaced by its part of speech (or by
                     ENT:<type> for a named entity), every edge by its dependency label, written as a canonical
                     bracketed string that does not depend on word order, e.g.
                         WH:what ; VERB[nsubj>NOUN[amod>ADJ],dobj>ENT:ORG]
                     template_id = first 10 hex characters of the SHA-1 of that string.
           shape     a coarse bucketed description of the graph: lead word, sentences, nodes, depth, entities, e.g.
                         what|S1|N4-6|D2|E1
                     Coarse enough that real queries and benchmark questions share shapes.
           text_template   the text with each entity replaced by its type: "What does {CONCEPT} do in {GPE}?"
           For a RESPONSE (answer) the template is taken from its first sentence; shape and graph cover the
           whole answer (cut at `answer_chars`).
"""
import hashlib
import re

from src.analysis.graph_features import graph_features, question_graph_from_doc
from src.analysis.lexical_features import extract as lexical_extract

WH = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
DROP_LEAD = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "his", "her", "its", "their",
             "some", "any", "each", "every", "no"}


def _norm(text):
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t.strip(" .,;:!?\"'()[]")


# --------------------------------------------------------------------------- entities
def extract_entities(doc, concepts=True):
    """[{text, norm, label, kind, start, end}] ; kind is "ner" or "concept"."""
    out, taken = [], set()
    for e in doc.ents:
        if not any(c.isalnum() for c in e.text):
            continue
        out.append({"text": e.text, "norm": _norm(e.text), "label": e.label_, "kind": "ner",
                    "start": e.start_char, "end": e.end_char})
        taken.update(range(e.start, e.end))
    if concepts:
        try:
            chunks = list(doc.noun_chunks)
        except (ValueError, NotImplementedError):  # model without a parser
            chunks = []
        for ch in chunks:
            toks = [t for t in ch if not (t.lower_ in DROP_LEAD or t.is_punct or t.pos_ in ("DET", "PRON"))
                    or t.i > ch.start + 1]
            while toks and (toks[0].lower_ in DROP_LEAD or toks[0].pos_ in ("DET", "PRON")):
                toks = toks[1:]
            if not toks or any(t.i in taken for t in toks):
                continue
            if ch.root.pos_ not in ("NOUN", "PROPN") or ch.root.is_stop or ch.root.lower_ in WH:
                continue
            start, end = toks[0].idx, toks[-1].idx + len(toks[-1])
            text = doc.text[start:end]
            if not any(c.isalpha() for c in text) or len(text) < 3:
                continue
            out.append({"text": text, "norm": _norm(text), "label": "CONCEPT", "kind": "concept",
                        "start": start, "end": end})
            taken.update(t.i for t in toks)
    out.sort(key=lambda e: e["start"])
    return out


# --------------------------------------------------------------------------- templates
def _label(g, n):
    d = g.nodes[n]
    return f"ENT:{d['ent']}" if d.get("ent") else d["pos"]


def contract_entities(g):
    """Merge neighbouring tokens that belong to the same entity type into one node ("New" + "York")."""
    g = g.copy()
    changed = True
    while changed:
        changed = False
        for p, c in list(g.edges):
            ep, ec = g.nodes[p].get("ent"), g.nodes[c].get("ent")
            if ep and ep == ec:
                for _, gc, data in list(g.out_edges(c, data=True)):
                    g.add_edge(p, gc, **data)
                g.remove_node(c)
                changed = True
                break
    return g


def canonical_pattern(g, wh=None):
    """Canonical bracketed string of a graph (a forest): children sorted, so word order does not matter."""
    g = contract_entities(g)

    def canon(n, seen):
        if n in seen:
            return _label(g, n)
        seen = seen | {n}
        kids = sorted(f"{g.edges[n, c]['dep']}>{canon(c, seen)}" for c in g.successors(n))
        return _label(g, n) + (f"[{','.join(kids)}]" if kids else "")

    roots = sorted(canon(n, frozenset()) for n in g if g.in_degree(n) == 0)
    body = " | ".join(roots) if roots else "EMPTY"
    return (f"WH:{wh} ; " if wh else "") + body


def template_id(pattern):
    return hashlib.sha1(pattern.encode("utf-8")).hexdigest()[:10]


def _bucket(v, edges, labels):
    for e, lab in zip(edges, labels):
        if v <= e:
            return lab
    return labels[-1]


def shape_of(stats, wh, first_pos):
    lead = wh if wh in WH else ("WH" if wh else
                                ("VERB" if first_pos == "VERB" else "AUX" if first_pos == "AUX" else "OTHER"))
    s = min(int(stats["n_sentences"]), 3)
    nb = _bucket(stats["n_nodes"], [3, 6, 10], ["1-3", "4-6", "7-10", "11+"])
    d = stats["max_depth"]
    d = 0 if d != d else d  # NaN
    db = _bucket(d, [1, 2, 3], ["0-1", "2", "3", "4+"])
    eb = _bucket(stats["n_entities"], [0, 1, 2], ["0", "1", "2", "3+"])
    return f"{lead}|S{s}{'+' if s == 3 else ''}|N{nb}|D{db}|E{eb}"


def text_template(text, entities):
    out, pos = [], 0
    for e in entities:  # sorted by start, non-overlapping
        if e["start"] < pos:
            continue
        out.append(text[pos:e["start"]])
        out.append("{" + e["label"] + "}")
        pos = e["end"]
    out.append(text[pos:])
    return "".join(out)


def graph_to_json(g):
    return {
        "nodes": [{"id": int(n), "lemma": d.get("lemma"), "pos": d.get("pos"), "ent": d.get("ent")}
                  for n, d in g.nodes(data=True)],
        "edges": [{"source": int(a), "target": int(b), "dep": d.get("dep")} for a, b, d in g.edges(data=True)],
        "wh_word": g.graph.get("wh_word"), "n_tokens": g.graph.get("n_tokens"),
        "n_sentences": g.graph.get("n_sentences"),
    }


def graph_from_json(obj):
    import networkx as nx

    g = nx.DiGraph()
    for n in obj["nodes"]:
        g.add_node(n["id"], lemma=n["lemma"], pos=n["pos"], ent=n["ent"])
    for e in obj["edges"]:
        g.add_edge(e["source"], e["target"], dep=e["dep"])
    g.graph.update(wh_word=obj.get("wh_word"), n_tokens=obj.get("n_tokens"), n_sentences=obj.get("n_sentences"))
    return g


def ascii_tree(g):
    """A readable text drawing of a graph (for papers and logs)."""
    g = contract_entities(g)
    lines = []

    def walk(n, prefix, last, dep):
        d = g.nodes[n]
        tag = f"{d['lemma']}" + (f" <{d['ent']}>" if d.get("ent") else f" ({d['pos']})")
        lines.append(f"{prefix}{'`-- ' if last else '|-- '}{('[' + dep + '] ') if dep else ''}{tag}")
        kids = sorted(g.successors(n))
        for i, c in enumerate(kids):
            walk(c, prefix + ("    " if last else "|   "), i == len(kids) - 1, g.edges[n, c]["dep"])

    roots = sorted(n for n in g if g.in_degree(n) == 0)
    for i, r in enumerate(roots):
        walk(r, "", i == len(roots) - 1, None)
    return "\n".join(lines)


# --------------------------------------------------------------------------- the analyzer
class Analyzer:
    def __init__(self, spacy_model, concepts=True, answer_chars=1200):
        from src.analysis.graph_features import get_nlp

        self.nlp = get_nlp(spacy_model)
        self.model_name = spacy_model
        self.concepts = concepts
        self.answer_chars = answer_chars

    def _one(self, doc, text, role):
        ents = extract_entities(doc, self.concepts)
        g = question_graph_from_doc(doc)
        stats = graph_features(g)
        wh, first_pos = g.graph.get("wh_word"), g.graph.get("first_pos")
        if role == "answer" and len(list(doc.sents)) > 1:
            first = next(doc.sents).as_doc()
            gt = question_graph_from_doc(first)
            pattern = canonical_pattern(gt, gt.graph.get("wh_word"))
        else:
            gt = g
            pattern = canonical_pattern(g, wh)
        slots = []
        cg = contract_entities(gt)
        for n in sorted(cg.nodes):
            if cg.nodes[n].get("ent"):
                span = next((e for e in doc.ents if e.start <= n < e.end), None)
                slots.append({"slot": f"ENT_{len(slots)}", "type": cg.nodes[n]["ent"],
                              "text": span.text if span is not None else cg.nodes[n]["lemma"]})
        feats = {**{k: v for k, v in stats.items() if k != "wh_word"}, **lexical_extract(text)}
        return {
            "role": role, "text": text, "entities": ents, "graph": graph_to_json(g),
            "template": pattern, "template_id": template_id(pattern),
            "shape": shape_of(stats, wh, first_pos), "text_template": text_template(text, ents),
            "slots": slots, "stats": {k: (None if v != v else v) for k, v in stats.items() if k != "wh_word"},
            "features": {k: (None if (isinstance(v, float) and v != v) else v) for k, v in feats.items()},
        }

    def prep(self, text, role):
        text = (text or "").strip()
        return text[: self.answer_chars] if role == "answer" else text

    def analyze_many(self, texts, role="question", batch_size=64):
        texts = [self.prep(t, role) for t in texts]
        for doc, text in zip(self.nlp.pipe(texts, batch_size=batch_size), texts):
            yield self._one(doc, text, role)

    def analyze(self, text, role="question"):
        return next(self.analyze_many([text], role))
