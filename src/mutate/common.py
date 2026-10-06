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
# CONTROLS are not mutations. They show how much of an increase in hallucinations is NOT caused by the
# mutations: c_rerun = the unchanged benchmark question answered a second time (the RAG and the LLM are not
# deterministic), c_pad = the benchmark question + content-free filler as long as the M1 version (length alone).
CONTROL_CONDITIONS = ["c_rerun", "c_pad"]
# DERIVED conditions are computed in the analysis from other conditions, never run on a RAG.
# m_union = "Benchmark + M1/M2/M3 combined from the separate runs": a question counts as failing if it fails in the
# unchanged benchmark OR in any of the three single-mutation runs (no extra evaluation, see README section 14.6).
DERIVED_CONDITIONS = ["m_union"]
ALL_CONDITIONS = CONDITIONS + DERIVED_CONDITIONS + CONTROL_CONDITIONS


def active_conditions(cfg):
    """Conditions that are RUN on the RAGs, from evaluation.conditions and evaluation.combined:
    combined: derived (default)  m123 is not run; the combined number is derived from m1, m2, m3
    combined: run                m123 is run too (the three mutations applied to one prompt)"""
    ev = cfg["evaluation"]
    conds = [c for c in ev.get("conditions", ["baseline", "m1", "m2", "m3"]) if c not in DERIVED_CONDITIONS and c != "m123"]
    if ev.get("combined", "derived") == "run":
        i = conds.index("m3") + 1 if "m3" in conds else len(conds)
        conds.insert(i, "m123")
    return conds
CONDITION_LABELS = {
    "baseline": "Benchmark (unchanged)",
    "m1": "Benchmark + M1 only",
    "m2": "Benchmark + M2 only",
    "m3": "Benchmark + M3 only",
    "m123": "Benchmark + M1+M2+M3 in one prompt",
    "m_union": "Benchmark + M1/M2/M3 combined (any separate run)",
    "c_rerun": "Control: same question again",
    "c_pad": "Control: off-topic filler",
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
        self._embedder, self._emb_lock = None, threading.Lock()
        self._parse_lock = threading.Lock()

    @property
    def nlp(self):
        with self._lock:
            if self._nlp is None:
                from src.analysis.graph_features import get_nlp

                self._nlp = get_nlp(self.cfg["real_queries"]["spacy_model"])
            return self._nlp

    def parse(self, text):
        """spaCy parse, one at a time (a spaCy pipeline is not guaranteed to be thread-safe)."""
        nlp = self.nlp
        with self._parse_lock:
            return nlp(text)

    @property
    def embedder(self):
        with self._lock:
            if self._embedder is None:
                from src.embeddings import get_sync_embedder

                self._embedder = get_sync_embedder(self.cfg["embedding"]["model"])
            return self._embedder

    def similarity(self, reference, texts):
        """Cosine similarity of each text with the reference (local embedding model)."""
        emb = self.embedder
        with self._emb_lock:  # one encode at a time: the model is shared between worker threads
            vecs = emb.encode([reference] + list(texts))
        return (vecs[1:] @ vecs[0]).tolist()

    def real_lengths(self, domain):
        """Word counts of the real user queries of a domain ("general" or "medical"), sorted. [] if the
        file does not exist (run scripts/prepare_real_queries.py)."""
        with self._lock:
            cache = self.__dict__.setdefault("_real_len", {})
            if domain not in cache:
                import json

                f = ROOT / self.cfg["real_queries"]["output_dir"] / f"{domain}.jsonl"
                lens = []
                if f.exists():
                    with open(f, encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                t = json.loads(line).get("text") or ""
                                if t:
                                    lens.append(len(t.split()))
                cache[domain] = sorted(lens)
            return cache[domain]

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
