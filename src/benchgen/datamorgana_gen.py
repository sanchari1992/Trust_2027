"""DataMorgana generation (Filice et al., 2025), Section 3.

Configuration stage: a JSON file lists user categorizations and question
categorizations; each has categories with a probability and a description.
Generation stage, per Q&A pair:
  (1) pick one category per categorization by its probability;
  (2) take a document (here: a passage);
  (3) ask the LLM for k candidate Q&A pairs (k = 3 in the paper) with the paper's prompt;
  (4) FILTER: keep candidates that satisfy the prompt's constraints, adhere to the
      chosen categories, and have answers faithful to the document; if several
      pass, sample one.
The paper does not publish its filter prompt, so step (4) uses our own prompt
(one LLM call for all k candidates). The paper used Claude 3.5 Sonnet; we use DeepSeek.
"""
import json

from src.benchgen.common import parse_json_objects
from src.config import ROOT

GEN_SYSTEM = "You are a helpful assistant."

GEN_PROMPT = """You are a user simulator that should generate {k} candidate questions for starting a conversation.

The {k} questions must be about facts discussed in the documents you will now receive. When generating the questions, assume that the real users you must simulate, as well as the readers of the questions, do not have access to these documents. Therefore, never refer to the author of the documents or the documents themselves. Also, assume that whoever reads the questions will read each question independently. The {k} questions must be diverse and different from each other. Return only the questions without any preamble. Write each pair in a new line, in the following JSON format: '{{"question": <question>, "answer": <answer>}}.'

### The generated questions should be about facts from the following document:
{document}

### Each of the generated questions must reflect a user with the following characteristics:
{user_lines}
### Each of the generated questions must have the following characteristics:
{question_lines}"""

FILTER_PROMPT = """You are a strict quality checker for synthetic question-answer pairs.

### Document:
{document}

### The questions were supposed to reflect a user with these characteristics:
{user_lines}
### The questions were supposed to have these characteristics:
{question_lines}
### General constraints: each question must make sense on its own for a reader who cannot see the document, and must never refer to the document, its author, or phrases such as "the text" or "the passage".

### Candidates:
{candidates}

For EACH candidate decide three things:
- "constraints_ok": the question satisfies the general constraints;
- "categories_ok": the question adheres to ALL the user and question characteristics above;
- "answer_faithful": the answer is fully supported by the document (nothing added, nothing contradicted).
Return only JSON, in this format: {{"results": [{{"index": 0, "constraints_ok": true, "categories_ok": true, "answer_faithful": true}}]}}"""


class DataMorganaGenerator:
    name = "datamorgana"
    overhead_tokens = 700

    def __init__(self, ctx):
        self.chat = ctx.chat
        dc = ctx.bc["datamorgana"]
        path = ROOT / dc["categorizations"][ctx.rag_name]
        spec = json.loads(path.read_text(encoding="utf-8"))
        self.user_cats = spec["user_categorizations"]
        self.question_cats = spec["question_categorizations"]
        self.k = dc["candidates_per_call"]
        self.use_filter = dc["filter"]
        self.calls_per_attempt = 2 if self.use_filter else 1

    @staticmethod
    def _pick(categorizations, rng):
        chosen = []
        for z in categorizations:
            cats = z["categories"]
            weights = [c.get("probability", 1.0 / len(cats)) for c in cats]
            chosen.append((z["name"], rng.choices(cats, weights=weights, k=1)[0]))
        return chosen

    def attempt(self, passage, rng):
        users = self._pick(self.user_cats, rng)
        questions = self._pick(self.question_cats, rng)
        user_lines = "".join(f"    - They must be {c['description']}\n" for _, c in users)
        question_lines = "".join(f"    - It must be {c['description']}\n" for _, c in questions)

        raw = self.chat(
            GEN_SYSTEM,
            GEN_PROMPT.format(
                k=self.k,
                document=passage["text"],
                user_lines=user_lines,
                question_lines=question_lines,
            ),
        )
        candidates = [
            {"question": str(o["question"]).strip(), "answer": str(o["answer"]).strip()}
            for o in parse_json_objects(raw)
            if isinstance(o, dict) and o.get("question") and o.get("answer")
        ]
        if not candidates:
            return {"rejected": "no_valid_candidates"}

        passing = candidates
        if self.use_filter:
            listing = json.dumps(
                [{"index": i, **c} for i, c in enumerate(candidates)], ensure_ascii=False, indent=1
            )
            verdict_raw = self.chat(
                GEN_SYSTEM,
                FILTER_PROMPT.format(
                    document=passage["text"],
                    user_lines=user_lines,
                    question_lines=question_lines,
                    candidates=listing,
                ),
            )
            passing = []
            for o in parse_json_objects(verdict_raw):
                for r in o.get("results", []) if isinstance(o, dict) else []:
                    i = r.get("index")
                    if (
                        isinstance(i, int)
                        and 0 <= i < len(candidates)
                        and r.get("constraints_ok")
                        and r.get("categories_ok")
                        and r.get("answer_faithful")
                    ):
                        passing.append(candidates[i])
            if not passing:
                return {"rejected": "all_candidates_failed_filter"}

        chosen = rng.choice(passing)
        return {
            "question": chosen["question"],
            "answer": chosen["answer"],
            "passage_ids": [passage["id"]],
            "contexts": [passage["text"]],
            "meta": {
                "user_categories": {n: c["name"] for n, c in users},
                "question_categories": {n: c["name"] for n, c in questions},
                "candidates_generated": len(candidates),
                "candidates_passed_filter": len(passing),
            },
        }
