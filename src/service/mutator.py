"""Mutate ONE prompt on request: the engine behind the web page and the API (scripts/serve_mutator.py).

Input:  a question, its expected answer, optionally the evidence (source text) the answer comes from,
        and which mutations to apply.
Output: for every chosen mutation the mutated prompt and the NEW expected answer / behaviour.

It reuses the same code as the benchmark pipeline (src/benchdb/generate.py), so a prompt mutated here is built
exactly like the ones in the paper. Two differences, because there is no database and no retrieval here:
  * Evidence. Premise (and the LLM checks of Elaborate) need a source text to write and verify claims. If you give
    none, the expected ANSWER is used as the only source ("answer_only"): claims are then checked against the answer,
    nothing else. This is weaker than checking against the textbook passage; the response says so.
  * Fallbacks (mutation.fill_all) are always on: a mutation whose checks fail is replaced by the simple fallback and
    marked, instead of returning nothing.
"""
import copy
import threading
import time
from functools import partial

from src.benchdb.generate import DBContext, build_mutation_cfg, generate_item
from src.benchgen.common import make_rng
from src.config import load_config
from src.mutate.common import MUTATION_INFO, resolve_mutations
from src.mutate.length import fallback_length
from src.mutate.linguistic import check_resources, fallback_linguistic
from src.mutate.prompt import Prompt
from src.mutate.runner import _fill_userinfo, make_row

MAX_QUESTION_CHARS = 2000
MAX_ANSWER_CHARS = 6000

BEHAVIOR_TEXT = {
    "answer": "The RAG should answer; the reference answer is unchanged.",
    "reject_premise": "The RAG should say the user's statement is wrong (and give the correct answer).",
    "abstain": "The RAG should say it cannot be answered from the given information.",
}
CORRECT_OPS = ["true_premise", "correct_candidate"]
INCORRECT_OPS = ["false_premise", "wrong_candidate", "unsupported_assumption"]


class RequestError(ValueError):
    """A problem with what the user sent (shown to them as is)."""


class MutationService:
    def __init__(self, cfg=None, chat=None):
        self.cfg = cfg or load_config()
        self._chat = chat  # tests can pass a fake chat(system, user, max_tokens=None, temperature=None)
        self._ctx = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ catalog (for the page)
    def catalog(self):
        mc = self.cfg["mutation"]
        ops = {"m1": list(mc["length"]["operators"]), "m2": list(mc["linguistic"]["operators"]),
               "m3": CORRECT_OPS + INCORRECT_OPS}
        return [{"id": m, "name": i["name"], "what": i["what"], "operators": ops[m]}
                for m, i in MUTATION_INFO.items()]

    # ------------------------------------------------------------------ context (made once)
    def _context(self, mc):
        with self._lock:
            if self._ctx is None:
                if self._chat is None:
                    from src.llm import get_chat_func

                    self._chat = get_chat_func(self.cfg)
                chat = partial(self._chat, temperature=mc["temperature"])
                self._ctx = DBContext(self.cfg, chat, lengths=None)
            return self._ctx

    # ------------------------------------------------------------------ the request
    def mutate(self, question, answer="", evidence="", mutations=("elaborate", "colloquial", "premise"),
               composite=False, premise_kind="either", depth=None, seed=None, randomize=True):
        question, answer, evidence = (question or "").strip(), (answer or "").strip(), (evidence or "").strip()
        if not question:
            raise RequestError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS or len(answer) > MAX_ANSWER_CHARS or len(evidence) > MAX_ANSWER_CHARS:
            raise RequestError("The text is too long (question up to 2,000 characters, answer and evidence up to 6,000).")
        if isinstance(mutations, str):
            mutations = [m for m in mutations.split(",")]
        try:
            plan = resolve_mutations(",".join(mutations))
        except ValueError as e:
            raise RequestError(str(e))
        if not plan:
            raise RequestError("Choose at least one mutation.")
        if "m3" in plan and plan["m3"] is None and premise_kind in ("correct", "incorrect"):
            plan["m3"] = CORRECT_OPS if premise_kind == "correct" else INCORRECT_OPS
        if depth is not None and not 1 <= int(depth) <= 4:
            raise RequestError("Depth must be between 1 and 4.")

        notes = []
        if evidence:
            grounding, contexts = "evidence", [evidence]
        elif answer:
            grounding, contexts = "answer_only", [answer]
            if any(m in plan for m in ("m1", "m3")):
                notes.append("No evidence text was given, so the expected answer is the only source: added "
                             "sentences and premises are checked against it alone. Paste the passage the answer "
                             "comes from for better results.")
        else:
            grounding, contexts = "none", []
        if "m3" in plan and not answer:
            notes.append("Premise needs the expected answer; it was skipped.")
            plan.pop("m3")
            if not plan:
                raise RequestError("Premise needs the expected answer. Add it, or choose another mutation.")

        mc = build_mutation_cfg(self.cfg, plan, depth=depth)
        mc["fill_all"] = True
        if mc["length"]["target"].get("mode") == "real" and not self._real_lengths_available(mc):
            mc["length"]["target"].update(mode="range", range=[20, 45])  # no real-query file: a sensible length range
        try:
            check_resources(mc["linguistic"])
        except Exception as e:  # missing WordNet etc.
            raise RequestError(f"A required resource is missing: {e}")
        ctx = self._context(mc)

        seed = int(time.time() * 1000) % 2_000_000_000 if (seed is None and randomize) else int(seed or mc["seed"])
        item = {"id": "user", "question": question, "answer": answer, "contexts": contexts,
                "rag": "medrag", "method": "user", "corpus": None, "passage_ids": None}
        rows, skipped = generate_item(copy.deepcopy(item), plan, ctx, mc, seed, combine=composite)
        self._fill(item, plan, rows, skipped, ctx, mc, seed)

        results = []
        order = [m for m in ("m1", "m2", "m3", "m123") if m in rows]
        for m in order:
            results.append(self._present(m, rows[m], answer))
        failed = {m: skipped[m] for m in skipped if m not in rows}
        return {"question": question, "expected_answer": answer, "grounding": grounding, "seed": seed,
                "results": results, "not_generated": failed, "notes": notes}

    # ------------------------------------------------------------------ helpers
    def _real_lengths_available(self, mc):
        return bool(self._context(mc).real_lengths("medical"))

    def _fill(self, item, plan, rows, skipped, ctx, mc, seed):
        """Fallbacks for mutations whose checks failed (same ones as the pipeline's fill_all)."""
        rng = lambda tag: make_rng(seed, item["id"], tag)  # noqa: E731
        if "m1" in plan and "m1" not in rows:
            r = fallback_length(item)
            first = skipped.pop("m1", "unknown")
            r["info"]["first_failure"] = first
            rows["m1"] = make_row(item, "m1", r["prompt"], [f"m1:{o}" for o in r["operators"]],
                                  item["answer"], "answer", False, r["info"])
        if "m2" in plan and "m2" not in rows:
            r = fallback_linguistic(Prompt(core=item["question"]), ctx, mc["linguistic"], rng("m2fill"))
            if r["ok"]:
                skipped.pop("m2", None)
                rows["m2"] = make_row(item, "m2", r["prompt"], [f"m2:{o}" for o in r["operators"]],
                                      item["answer"], "answer", False, r["info"])
        if "m3" in plan and "m3" not in rows and item["answer"] and item["contexts"]:
            u = copy.deepcopy(mc["userinfo"])
            if plan["m3"] is not None:  # keep to the chosen kinds of premise
                has_c = any(o in plan["m3"] for o in CORRECT_OPS)
                has_i = any(o in plan["m3"] for o in INCORRECT_OPS)
                if has_c != has_i:
                    u["p_correct"] = 1.0 if has_c else 0.0
            r = _fill_userinfo(item, ctx, u, seed)
            if r["ok"]:
                skipped.pop("m3", None)
                p3 = Prompt(pre=r["pre"], core=item["question"], post=r["post"])
                rows["m3"] = make_row(item, "m3", p3, [f"m3:{r['operator']}"], r["answer"], r["behavior"],
                                      r["answer_changed"], r["info"], kind=r["kind"], claim=r["claim"])

    @staticmethod
    def _present(cond, row, original_answer):
        info = MUTATION_INFO.get(cond)
        meta = row.get("meta") or {}
        fb = meta.get("fallback") or (meta.get("m1") or {}).get("fallback")
        return {
            "mutation": cond, "name": info["name"] if info else "Composite",
            "prompt": row["question"],
            "expected_answer": row["answer"] if row["answer"] else None,
            "answer_changed": bool(row["answer_changed"]),
            "expected_behavior": row["expected_behavior"],
            "expected_behavior_text": BEHAVIOR_TEXT.get(row["expected_behavior"], ""),
            "operators": [o.split(":", 1)[-1] for o in row["operators"]],
            "premise_kind": row.get("kind"), "premise": row.get("claim"),
            "words": {"before": row["words_original"], "after": row["words_mutated"]},
            "fallback": bool(fb),
            "checked_by_llm": bool(meta.get("verification") or meta.get("verified")),
        }
