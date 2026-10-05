"""MUTATION 1 - LENGTH. Make the prompt longer WITHOUT changing what is asked or the answer.

Four operators; each question gets `length.n_ops` different ones:

  restate     LLM writes a restatement of the question in other words; appended after it.
              Guarantee: K(restatement) = K(question), so adding it changes nothing (verified).
  background  LLM writes ONE sentence of background using only facts of the passage, never
              the answer; added before the question.
              Guarantee: the sentence is entailed by the passage and adds no constraint on the
              asked slot, so every answer supported by the passage stays supported (verified).
  instruction A content-free answer-style instruction chosen from a fixed list (no LLM).
              Guarantee: by construction - the sentence contains no domain content.
  appositive  LLM picks one term copied from the question and describes it using the passage;
              we insert "term (description)" ourselves.
              Guarantee: only an insertion next to an existing term, entailed by the passage.

Verification: if any LLM-written text was added, one LLM check confirms that the modified
question asks the same thing, the reference answer still holds, the added statements are
supported by the passage, and the answer is not revealed. Failures are redone, then dropped.
"""
from src.mutate.common import weighted_sample
from src.mutate.prompt import Prompt
from src.mutate.verify import verify_preserving

GEN_SYSTEM = "You are a careful writing assistant. Reply with JSON only, no other text."

RESTATE_PROMPT = """Question: {question}

Write ONE sentence that asks for exactly the same information as the question, in different words. Do not add, remove, or change any detail or requirement.
Reply with JSON only: {{"paraphrase": "..."}}"""

BACKGROUND_PROMPT = """Passage:
{passage}

Question: {question}

Write ONE sentence of background that a person asking this question might naturally mention before asking it. Use ONLY facts stated in the passage. The sentence must NOT contain, state, or hint at the answer to the question, must not refer to "the passage" or "the text", and must be a complete sentence.
Reply with JSON only: {{"background": "..."}}"""

APPOSITIVE_PROMPT = """Passage:
{passage}

Question: {question}

Choose ONE term (an entity, concept, or technical term) that appears word for word in the question, and write a short description of it (at most 12 words) using ONLY facts from the passage. The description must not contain or hint at the answer to the question and must not contain parentheses.
Reply with JSON only: {{"term": "<exact text copied from the question>", "description": "..."}}"""

CONNECTORS = ["In other words:", "Put differently:", "To say it another way:"]

INSTRUCTIONS = [
    "Answer as precisely as you can.",
    "A brief explanation is fine.",
    "Keep the answer focused on what is asked.",
    "Be specific rather than general.",
    "Give a clear and accurate answer.",
    "Stick to what the question asks.",
]

LLM_OPS = {"restate", "background", "appositive"}


def _passage(item):
    return "\n\n".join(item.get("contexts") or [])


def op_restate(prompt, item, ctx, rng):
    obj = ctx.ask_json(GEN_SYSTEM, RESTATE_PROMPT.format(question=item["question"]))
    text = str((obj or {}).get("paraphrase", "")).strip()
    if len(text.split()) < 3:
        return None, {"error": "no_paraphrase"}
    new = prompt.copy()
    new.post.append(f"{rng.choice(CONNECTORS)} {text}")
    return new, {"paraphrase": text}


def op_background(prompt, item, ctx, rng):
    obj = ctx.ask_json(
        GEN_SYSTEM, BACKGROUND_PROMPT.format(passage=_passage(item), question=item["question"])
    )
    text = str((obj or {}).get("background", "")).strip()
    if len(text.split()) < 4:
        return None, {"error": "no_background"}
    new = prompt.copy()
    new.pre.append(text)
    return new, {"background": text}


def op_instruction(prompt, item, ctx, rng):
    text = rng.choice(INSTRUCTIONS)
    new = prompt.copy()
    new.post.append(text)
    return new, {"instruction": text}


def op_appositive(prompt, item, ctx, rng):
    obj = ctx.ask_json(
        GEN_SYSTEM, APPOSITIVE_PROMPT.format(passage=_passage(item), question=item["question"])
    ) or {}
    term = str(obj.get("term", "")).strip()
    desc = str(obj.get("description", "")).strip().rstrip(".")
    if not term or not desc or term not in prompt.core or "(" in desc or ")" in desc:
        return None, {"error": "bad_term_or_description", "term": term}
    new = prompt.copy()
    new.core = prompt.core.replace(term, f"{term} ({desc})", 1)
    return new, {"term": term, "description": desc}


OPS = {
    "restate": op_restate,
    "background": op_background,
    "instruction": op_instruction,
    "appositive": op_appositive,
}


def mutate_length(item, ctx, mc, rng):
    """Returns {"ok": True, "prompt", "operators", "info"} or {"ok": False, "reason"}."""
    last = "no_attempt"
    for attempt in range(ctx.max_retries + 1):
        ops = weighted_sample(rng, mc["operators"], mc["n_ops"])
        if not ops:
            return {"ok": False, "reason": "no_operators_enabled"}
        prompt, info, failed = Prompt(core=item["question"]), {}, None
        for name in ops:
            new, detail = OPS[name](prompt, item, ctx, rng)
            if new is None:
                failed = f"{name}:{detail.get('error')}"
                break
            prompt, info[name] = new, detail
        if failed:
            last = failed
            continue
        if len(prompt.render().split()) <= len(item["question"].split()):
            last = "not_longer"
            continue
        if mc["verify"] and any(o in LLM_OPS for o in ops):
            ok, detail = verify_preserving(ctx, item, item["question"], prompt.render())
            if not ok:
                last = "verification_failed"
                info["last_verifier_reply"] = detail
                continue
            info["verification"] = detail
        return {"ok": True, "prompt": prompt, "operators": ops, "info": info, "attempts": attempt + 1}
    return {"ok": False, "reason": last}
