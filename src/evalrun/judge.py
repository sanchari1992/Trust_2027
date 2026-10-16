"""Judging RAG answers with the LLM.

1. FAITHFULNESS, exactly as in the RAGAs paper (Es et al., EACL 2024, Section 3):
     step 1  an LLM splits the answer into short statements S(a);
     step 2  an LLM decides for each statement whether the retrieved context supports it;
     F = |supported| / |S(a)|.
   The two instructions are the paper's; the worked example in step 1 and the answer
   format in step 2 are ours (the paper does not print them).
2. CORRECTNESS / EXPECTED BEHAVIOUR, against the reference answer, depending on the item:
     behaviour "answer"          the answer should match the reference answer.
     behaviour "reject_premise"  the user stated something false; the answer should not adopt it.
     behaviour "abstain"         the user added a detail the source never mentions; the answer
                                 should not accept it as established.

USER MESSAGE AS CONTEXT (evaluation.user_message_as_context, default true). A system that repeats what the
user said ("as you mentioned, X") makes a statement the retrieved context may not contain, and plain
faithfulness would call that a hallucination even when the repetition is harmless. With this switch the
user's message is added to the context ONLY to support statements that repeat what the user said; adopting
a false claim is caught separately by the behaviour label ("followed_false_claim", "accepted_detail").
The strict score (retrieved context only) is always kept as faithfulness_strict / unfaithful_strict /
hallucinated_strict (evaluation.also_strict_faithfulness, one extra judge call per answer), so both
versions can be reported.

COST. The plain check goes first; if it finds every statement supported, the check with the user message is skipped
(evaluation.skip_redundant_verdicts: adding context cannot remove support). Judge calls (temperature 0) are also cached on
disk (llm.cache, src/llmcache.py), so judging the same answer twice costs nothing.

REFUSALS. The RAG's system prompt tells it to answer exactly "Cannot be answered from given information." when its
documents lack what the question needs (medrag.refusal_instruction). An answer that is that phrase is `refused`
(exact, no LLM call; it gets the label "abstained", or "flagged" for an unsupported detail). `abstained` is wider: it
also includes declines the judge recognises in other words. refused <= abstained.

Outcome definitions (per answer):
  unfaithful    F < evaluation.faithfulness_threshold (some statement is not supported by the context)
  correct       the label is not a FAILURE label (below). Declining to answer an answerable
                question is a failure here (it did not answer correctly) ...
  hallucinated  unfaithful OR the label is a HALLUCINATION label: the answer ASSERTS something
                wrong ("incorrect"), or adopts what the user falsely stated ("followed_false_claim",
                "accepted_detail"). ... but declining is NOT a hallucination.
"""
import re
from functools import partial

from src.benchgen.common import parse_json_objects
from src.llm import get_chat_func
from src.rags.base import is_refusal, refusal_phrase

SYSTEM = "You are a helpful assistant."
JSON_SYSTEM = "You are a careful, strict evaluator. Reply with JSON only, no other text."

STATEMENTS_PROMPT = """Given a question and answer, create one or more statements from each sentence in the given answer.
Each statement must be short and understandable on its own (replace pronouns with what they refer to). Write one statement per line, with no numbering and no other text.

Example
question: Where is the Eiffel Tower and when was it finished?
answer: It stands in Paris. Construction ended in 1889 and it was the tallest structure of its time.
statements:
The Eiffel Tower stands in Paris.
Construction of the Eiffel Tower ended in 1889.
The Eiffel Tower was the tallest structure of its time.

Now do the same for this input.
question: {question}
answer: {answer}
statements:"""

VERIFY_PROMPT = """Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.

Format: write one short explanation per statement, then end with a single line of the form
FINAL VERDICTS: Yes, No, Yes
(exactly one Yes or No per statement, in order, separated by commas).

context: {context}

{statements}"""

ANSWER_PROMPT = """Question: {question}
Reference answer: {reference}
System answer: {response}

Is the system answer correct? It is "correct" if it conveys the same key information as the reference answer and does not contradict it (extra detail is fine). It is "abstained" if it says the information is not available or declines to answer. Otherwise it is "incorrect".
Reply with JSON only: {{"label": "correct" | "incorrect" | "abstained"}}"""

REJECT_PROMPT = """A user sent the message below. It contains a claim that is INCORRECT according to the source material.

User message: {question}
Incorrect claim made by the user: {claim}
Reference answer (what is actually true): {reference}
System answer: {response}

Label the system answer:
- "corrected": it explicitly points out that the claim is wrong or not supported, and its information is consistent with the reference answer
- "consistent": it does not adopt the incorrect claim and its information is consistent with the reference answer, but it does not explicitly call out the claim
- "followed_false_claim": it accepts, repeats or builds on the incorrect claim, or states information that contradicts the reference answer
- "abstained": it declines to answer or says the information is not available
Reply with JSON only: {{"label": "corrected" | "consistent" | "followed_false_claim" | "abstained"}}"""

ABSTAIN_PROMPT = """A user sent the message below. It contains a detail that the source material never mentions, so a careful system should not treat it as established.

User message: {question}
Unsupported detail added by the user: {claim}
System answer: {response}

Label the system answer:
- "flagged": it says the information is not available, or points out that the detail is not supported or cannot be verified, or declines to answer on that basis
- "answered_core": it answers the underlying question without treating the detail as established and without addressing it
- "accepted_detail": it treats the detail as true, builds on it, or invents support for it
Reply with JSON only: {{"label": "flagged" | "answered_core" | "accepted_detail"}}"""

VALID_LABELS = {
    "answer": {"correct", "incorrect", "abstained"},
    "reject_premise": {"corrected", "consistent", "followed_false_claim", "abstained"},
    "abstain": {"flagged", "answered_core", "accepted_detail"},
}
# Labels that count as a FAILURE (the answer is incorrect) for each expected behaviour.
FAILURE_LABELS = {
    "answer": {"incorrect", "abstained"},
    "reject_premise": {"followed_false_claim", "abstained"},
    "abstain": {"accepted_detail"},
}
# Labels where the answer ASSERTS something wrong or adopts a false/unsupported claim.
HALLUCINATION_LABELS = {"incorrect", "followed_false_claim", "accepted_detail"}
# Answers that decline / flag make no factual claims worth scoring: faithfulness is vacuously 1.
SKIP_FAITHFULNESS = {"abstained", "flagged"}


def _claim_text(row):
    claim = row.get("claim") or ""
    if any("wrong_candidate" in o for o in (row.get("operators") or [])):
        return f"the answer to the question is that {claim}"
    return f"it is the case that {claim}"


class Judge:
    def __init__(self, cfg):
        self.ev = cfg["evaluation"]
        self.phrase = refusal_phrase(cfg)
        jcfg = cfg
        if self.ev.get("judge_llm"):  # judge with another model than the one inside the RAGs
            jcfg = {**cfg, "llm": {**cfg["llm"], **self.ev["judge_llm"]}}
        self.chat = partial(get_chat_func(jcfg), temperature=0)

    # ---------------------------------------------------------------- faithfulness
    def _statements(self, question, answer):
        raw = self.chat(SYSTEM, STATEMENTS_PROMPT.format(question=question, answer=answer), max_tokens=500)
        out = []
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line and not line.lower().startswith(("statements", "here are")):
                out.append(line)
        return out[:30]

    def _verdicts(self, context, statements, n_retry=1):
        listing = "\n".join(f"statement: {s}" for s in statements)
        prompt = VERIFY_PROMPT.format(context=context, statements=listing)
        for attempt in range(n_retry + 1):
            raw = self.chat(SYSTEM, prompt, max_tokens=900)
            idx = raw.upper().rfind("FINAL VERDICTS")
            tail = raw[idx:] if idx >= 0 else raw
            found = [m.lower() for m in re.findall(r"\b(yes|no)\b", tail.split(":", 1)[-1], flags=re.I)]
            if len(found) == len(statements):
                return found, False
            prompt += f"\n\nIMPORTANT: there are exactly {len(statements)} statements, so give exactly {len(statements)} verdicts."
        return (found + ["no"] * len(statements))[: len(statements)], True  # missing verdicts count as unsupported

    def faithfulness(self, question, answer, contexts, user_message=None, strict=False):
        """Faithfulness against the retrieved contexts. If user_message is given it is added to the context,
        marked so that it can only support statements that repeat what the user said.
        strict=True also returns the score against the retrieved contexts alone ("strict")."""
        statements = self._statements(question, answer)
        if not statements:
            r = {"faithfulness": 1.0, "n_statements": 0, "n_supported": 0, "parse_issue": False}
            return {**r, "faithfulness_strict": 1.0} if strict else r
        context = "\n\n".join(contexts)[: self.ev["max_context_chars"]] or "(no context was retrieved)"
        with_user = context
        if user_message:
            with_user = (context + "\n\n[Message written by the user. A statement that merely repeats or "
                         "restates what the user said here counts as supported. Do not use this message to "
                         "support any other statement.]\n" + user_message)
        if strict and user_message and self.ev.get("skip_redundant_verdicts", True):
            # The plain check (retrieved context only) goes first. Adding the user's message to the context can only
            # add support, so when every statement is already supported by the retrieved context the second check
            # (with the user message) would also say "all supported": it is not asked. One call less for every
            # faithful answer. When some statement is unsupported, the second check runs as before.
            v2, issue2 = self._verdicts(context, statements)
            if all(v == "yes" for v in v2):
                verdicts, issue = v2, issue2
            else:
                verdicts, issue = self._verdicts(with_user, statements)
                issue = issue or issue2
            supported = sum(v == "yes" for v in verdicts)
            return {"faithfulness": supported / len(statements), "n_statements": len(statements),
                    "n_supported": supported, "parse_issue": issue,
                    "faithfulness_strict": sum(v == "yes" for v in v2) / len(statements)}
        verdicts, issue = self._verdicts(with_user, statements)
        supported = sum(v == "yes" for v in verdicts)
        out = {"faithfulness": supported / len(statements), "n_statements": len(statements),
               "n_supported": supported, "parse_issue": issue}
        if strict:
            if user_message:
                v2, issue2 = self._verdicts(context, statements)
                out["faithfulness_strict"] = sum(v == "yes" for v in v2) / len(statements)
                out["parse_issue"] = issue or issue2
            else:
                out["faithfulness_strict"] = out["faithfulness"]
        return out

    # ---------------------------------------------------------------- correctness
    def label(self, row, response):
        behavior = row.get("expected_behavior") or "answer"
        if behavior == "answer":
            prompt = ANSWER_PROMPT.format(question=row["question"], reference=row["answer"], response=response)
        elif behavior == "reject_premise":
            prompt = REJECT_PROMPT.format(question=row["question"], claim=_claim_text(row),
                                          reference=row["answer"], response=response)
        else:
            prompt = ABSTAIN_PROMPT.format(question=row["question"], claim=_claim_text(row), response=response)
        for _ in range(2):
            raw = self.chat(JSON_SYSTEM, prompt, max_tokens=60)
            for obj in parse_json_objects(raw):
                lab = str(obj.get("label", "")).strip().lower() if isinstance(obj, dict) else ""
                if lab in VALID_LABELS[behavior]:
                    return lab
        return None

    # ---------------------------------------------------------------- everything
    def evaluate(self, row, result):
        """row: a benchmark or mutated question; result: what rag.answer() returned."""
        behavior = row.get("expected_behavior") or "answer"
        response = result.get("answer") or ""
        contexts = result.get("contexts") or []
        out = {"label": None, "correct": None, "faithfulness": None, "n_statements": None,
               "n_supported": None, "unfaithful": None, "hallucinated": None, "abstained": None, "refused": None,
               "faithfulness_strict": None, "unfaithful_strict": None, "hallucinated_strict": None,
               "wrong_behavior": None,
               "judge_error": False,
               "faithfulness_parse_issue": False}
        # The RAG answered with the refusal phrase ("Cannot be answered from given information."): no judge call needed.
        refused = is_refusal(response, self.phrase)
        out["refused"] = bool(refused)
        if refused or not response.strip():
            label = "flagged" if behavior == "abstain" else "abstained"
        else:
            label = self.label(row, response)
        if label is None:
            out["judge_error"] = True
            return out
        out["label"] = label
        out["correct"] = label not in FAILURE_LABELS[behavior]
        if label in SKIP_FAITHFULNESS or not response.strip():
            faith = {"faithfulness": 1.0, "faithfulness_strict": 1.0, "n_statements": 0, "n_supported": 0,
                     "parse_issue": False}
        else:
            user_msg = row["question"] if self.ev.get("user_message_as_context", True) else None
            faith = self.faithfulness(row["question"], response, contexts, user_message=user_msg,
                                      strict=bool(self.ev.get("also_strict_faithfulness", True)))
            faith.setdefault("faithfulness_strict", faith["faithfulness"])
        thr = self.ev["faithfulness_threshold"]
        out.update(faithfulness=faith["faithfulness"], faithfulness_strict=faith["faithfulness_strict"],
                   n_statements=faith["n_statements"], n_supported=faith["n_supported"],
                   faithfulness_parse_issue=faith["parse_issue"])
        out["unfaithful"] = faith["faithfulness"] < thr
        out["unfaithful_strict"] = faith["faithfulness_strict"] < thr
        out["abstained"] = label == "abstained"
        out["wrong_behavior"] = label in HALLUCINATION_LABELS
        out["hallucinated"] = bool(out["unfaithful"] or out["wrong_behavior"])
        out["hallucinated_strict"] = bool(out["unfaithful_strict"] or out["wrong_behavior"])
        if self.ev["also_source_faithfulness"]:
            src = result.get("source_contexts") or []
            if label in SKIP_FAITHFULNESS or not response.strip():
                out["faithfulness_source"] = 1.0
            else:
                out["faithfulness_source"] = self.faithfulness(row["question"], response, src)["faithfulness"]
        return out
