"""LLM checks that a mutation kept (or changed) the answer as intended.

These checks are what turns the guarantees in README section 13 from "true if the
generator did its job" into "true for every item that is kept". An item that fails
is regenerated, and dropped if it keeps failing. The checks use temperature 0.
"""
from src.mutate.common import passage_text

SYSTEM = "You are a careful, strict verifier. Reply with JSON only, no other text."

PRESERVING_PROMPT = """Passage:
{passage}

Original question: {original}
Reference answer: {answer}

Modified question: {modified}

Compare the modified question with the original. Reply with JSON only, using true or false:
{{"same_information_requested": ..., "reference_answer_still_correct": ..., "added_statements_supported": ..., "reveals_answer": ...}}

- same_information_requested: the modified question asks for exactly the same information as the original and adds no new requirement or condition.
- reference_answer_still_correct: the reference answer is still a correct and complete answer to the modified question, according to the passage.
- added_statements_supported: every factual statement that the modification ADDED is supported by the passage (true if nothing factual was added).
- reveals_answer: the added text states or gives away the reference answer."""

CLAIM_PROMPT = """Passage:
{passage}

Question: {question}
Claim: {claim}

How does the passage relate to the claim?
- "entailed": the passage explicitly states or clearly implies the claim.
- "contradicted": the passage states something that conflicts with the claim.
- "not_mentioned": the passage neither supports nor contradicts the claim.
Also say whether the claim, by itself, directly gives the answer to the question.
Reply with JSON only: {{"label": "entailed" | "contradicted" | "not_mentioned", "gives_answer": true | false}}"""


def _truthy(v):
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def verify_preserving(ctx, item, original_q, modified_q):
    """Did a length mutation keep the question and its answer intact?
    Returns (ok, detail_dict)."""
    obj = ctx.ask_json(
        SYSTEM,
        PRESERVING_PROMPT.format(
            passage=passage_text(item), original=original_q,
            answer=item["answer"], modified=modified_q,
        ),
        temperature=0,
    )
    if not obj:
        return False, {"error": "unparseable_verifier_reply"}
    ok = (
        _truthy(obj.get("same_information_requested"))
        and _truthy(obj.get("reference_answer_still_correct"))
        and _truthy(obj.get("added_statements_supported"))
        and not _truthy(obj.get("reveals_answer"))
    )
    return ok, obj


def verify_claim(ctx, item, question, claim):
    """How does the passage relate to a claim? -> (label, gives_answer, raw)."""
    obj = ctx.ask_json(
        SYSTEM,
        CLAIM_PROMPT.format(passage=passage_text(item), question=question, claim=claim),
        temperature=0,
    )
    if not obj:
        return None, None, {"error": "unparseable_verifier_reply"}
    label = str(obj.get("label", "")).strip().lower()
    if label not in ("entailed", "contradicted", "not_mentioned"):
        return None, None, obj
    return label, _truthy(obj.get("gives_answer")), obj
