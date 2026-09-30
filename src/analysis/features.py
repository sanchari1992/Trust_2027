"""Loads a set of questions and computes graph + lexical features for all of them."""
import json
from pathlib import Path

import pandas as pd

from src.analysis.graph_features import NUMERIC_FEATURES as GRAPH_FEATURES
from src.analysis.graph_features import extract as graph_extract
from src.analysis.graph_features import get_nlp
from src.analysis.lexical_features import NUMERIC_FEATURES as LEXICAL_FEATURES
from src.analysis.lexical_features import extract as lexical_extract

ALL_NUMERIC_FEATURES = GRAPH_FEATURES + LEXICAL_FEATURES


def load_questions(path, text_field="text", limit=None):
    """Reads a .jsonl file and returns [{"id", "text", ...other columns kept as-is}].
    Works for BOTH real-query files (text_field="text") and our generated
    benchmark files (text_field="question"), so the same code compares either."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get(text_field)
            if text:
                rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def compute_features(rows, text_field, spacy_model, show_progress=True):
    """rows -> a DataFrame with one row per question and one column per feature."""
    nlp = get_nlp(spacy_model)
    out = []
    for i, row in enumerate(rows):
        text = row[text_field]
        feats = {**graph_extract(text, nlp), **lexical_extract(text)}
        feats["id"] = row.get("id", i)
        out.append(feats)
        if show_progress and (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(rows)} questions processed")
    return pd.DataFrame(out)
