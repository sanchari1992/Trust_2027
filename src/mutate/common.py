"""Shared helpers for the mutation step: file locations, sampling, the LLM context."""
import threading

from src.benchgen.common import make_rng, parse_json_objects, read_jsonl  # noqa: F401  (re-exported)
from src.config import ROOT, corpus_tag

MUTATION_DIRS = {
    "m1": "mutation1_length",
    "m2": "mutation2_linguistic",
    "m3": "mutation3_userinfo",
    "m123": "mutation123_combined",
}
CONDITIONS = ["baseline", "m1", "m2", "m3", "m123"]
CONDITION_LABELS = {
    "baseline": "Benchmark",
    "m1": "+M1 length",
    "m2": "+M2 linguistic",
    "m3": "+M3 user info",
    "m123": "+M1+M2+M3",
}


def benchmark_file(cfg, rag, method):
    return ROOT / cfg["benchmark"]["output_dir"] / f"{rag}_{corpus_tag(cfg, rag)}" / f"{method}.jsonl"


def mutation_file(cfg, cond, rag, method):
    return (
        ROOT / cfg["mutation"]["output_dir"] / MUTATION_DIRS[cond]
        / f"{rag}_{corpus_tag(cfg, rag)}" / f"{method}.jsonl"
    )


def condition_file(cfg, cond, rag, method):
    """Where the questions of one condition live (baseline = the original benchmark)."""
    if cond == "baseline":
        return benchmark_file(cfg, rag, method)
    return mutation_file(cfg, cond, rag, method)


def weighted_sample(rng, weights, k):
    """Pick k DIFFERENT names from {name: weight}, by weight. Weight 0 = never."""
    pool = {n: float(w) for n, w in weights.items() if float(w) > 0}
    chosen = []
    for _ in range(min(k, len(pool))):
        names = list(pool)
        pick = rng.choices(names, weights=[pool[n] for n in names], k=1)[0]
        chosen.append(pick)
        del pool[pick]
    return chosen


class MutCtx:
    """What every mutation operator needs: the LLM, a spaCy pipeline, the settings."""

    def __init__(self, cfg, chat):
        self.cfg = cfg
        self.chat = chat  # chat(system, user, max_tokens=None, temperature=None) -> text
        self.max_retries = cfg["mutation"]["max_retries"]
        self._nlp, self._lock = None, threading.Lock()

    @property
    def nlp(self):
        with self._lock:
            if self._nlp is None:
                from src.analysis.graph_features import get_nlp

                self._nlp = get_nlp(self.cfg["real_queries"]["spacy_model"])
            return self._nlp

    def ask_json(self, system, user, max_tokens=300, temperature=None):
        # Only pass temperature when given: passing None would override the
        # mutation temperature that is already set on self.chat.
        kwargs = {"max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        raw = self.chat(system, user, **kwargs)
        for obj in parse_json_objects(raw):
            if isinstance(obj, dict):
                return obj
        return None


def passage_text(item):
    return "\n\n".join(item.get("contexts") or [])
