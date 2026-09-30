"""Turns one question into a graph and a handful of numbers describing its shape.

The graph has one node per token that is a content word (noun, verb, adjective,
adverb, proper noun) or a named entity, and one edge per dependency relation
between two such tokens (a path through the parse tree is collapsed to a direct
edge when the tokens in between were dropped). This is a lightweight structural
proxy for "how the question is built", not a semantic knowledge graph: it needs
no LLM calls, is deterministic, and is cheap enough to run on tens of thousands
of real user queries.
"""
import networkx as nx
import spacy

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

_nlp_cache = {}


def get_nlp(model_name):
    if model_name not in _nlp_cache:
        _nlp_cache[model_name] = spacy.load(model_name)
    return _nlp_cache[model_name]


def question_graph(text, nlp):
    """Build a directed graph for one question. Returns a networkx.DiGraph."""
    doc = nlp(text)
    keep = {t.i for t in doc if t.pos_ in CONTENT_POS and not t.is_stop}
    ent_spans = {t.i: ent.label_ for ent in doc.ents for t in ent}

    g = nx.DiGraph()
    for i in keep:
        t = doc[i]
        g.add_node(i, lemma=t.lemma_.lower(), pos=t.pos_, ent=ent_spans.get(i))

    def nearest_kept_ancestor(token):
        node = token.head
        seen = set()
        while node.i != node.head.i and node.i not in keep:
            if node.i in seen:
                return None
            seen.add(node.i)
            node = node.head
        return node.i if node.i in keep else None

    for i in keep:
        t = doc[i]
        if t.head.i == t.i:  # root
            continue
        parent = t.head.i if t.head.i in keep else nearest_kept_ancestor(t)
        if parent is not None and parent != i:
            g.add_edge(parent, i, dep=t.dep_)

    g.graph["n_tokens"] = len(doc)
    g.graph["n_sentences"] = len(list(doc.sents))
    g.graph["root_pos"] = doc[0].pos_ if len(doc) else None
    g.graph["wh_word"] = next((t.text.lower() for t in doc if t.tag_ in ("WP", "WRB", "WDT")), None)
    g.graph["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
    return g


def graph_features(g):
    """Numeric summary of one question graph. Every value is a plain float/int,
    ready to go into a table or a classifier."""
    n, m = g.number_of_nodes(), g.number_of_edges()
    und = g.to_undirected()
    components = list(nx.connected_components(und)) if n else []
    largest = max((len(c) for c in components), default=0)

    if n >= 2:
        try:
            depth = nx.dag_longest_path_length(g)
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            depth = float("nan")
        degrees = [d for _, d in g.degree()]
        avg_degree = sum(degrees) / n
        density = nx.density(g)
    else:
        depth = 0
        avg_degree = 0.0
        density = 0.0

    ent_labels = [label for _, label in g.graph.get("entities", [])]
    return {
        "n_tokens": g.graph.get("n_tokens", 0),
        "n_sentences": g.graph.get("n_sentences", 1),
        "n_nodes": n,
        "n_edges": m,
        "graph_density": density,
        "avg_degree": avg_degree,
        "max_depth": depth,
        "n_components": len(components),
        "largest_component_frac": (largest / n) if n else 0.0,
        "n_entities": len(ent_labels),
        "n_entity_types": len(set(ent_labels)),
        "has_wh_word": int(g.graph.get("wh_word") is not None),
        "wh_word": g.graph.get("wh_word"),
        "root_is_verb": int(g.graph.get("root_pos") == "VERB"),
    }


NUMERIC_FEATURES = [
    "n_tokens", "n_sentences", "n_nodes", "n_edges", "graph_density", "avg_degree",
    "max_depth", "n_components", "largest_component_frac", "n_entities",
    "n_entity_types", "has_wh_word", "root_is_verb",
]


def extract(text, nlp):
    return graph_features(question_graph(text, nlp))
