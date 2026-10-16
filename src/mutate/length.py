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

LENGTH TARGET (length.target): by default every question gets n_ops operators, which makes prompts much
longer than real ones. With target.mode "real" each question draws a target length from the word counts
of REAL user queries and operators are added one at a time until it is reached (see config.yaml).

Verification: if any LLM-written text was added, one LLM check confirms that the modified
question asks the same thing, the reference answer still holds, the added statements are
supported by the passage, and the answer is not revealed. Failures are redone, then dropped.
"""
import re

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

Choose ONE term (an entity, concept, or technical term) that appears word for word in the question, and write a short description of it (4 to 12 words) using ONLY facts from the passage. The description must be a plain definition-style noun phrase of what the term is (for example "a public building that lends books"), NOT an instruction, NOT a sentence about a figure, diagram, table, colour or caption, and must not contain or hint at the answer to the question and must not contain parentheses.
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


BAD_DESCRIPTION = re.compile(
    r"\b(figure|fig\.?|diagram|table|panel|shown|illustrat\w*|depict\w*|red|blue|green|box(es)?|arrow|"
    r"caption|above|below|left|right|chart|image|photo\w*|fill|see|note|click)\b", re.I)


def op_appositive(prompt, item, ctx, rng):
    obj = ctx.ask_json(
        GEN_SYSTEM, APPOSITIVE_PROMPT.format(passage=_passage(item), question=item["question"])
    ) or {}
    term = str(obj.get("term", "")).strip()
    desc = str(obj.get("description", "")).strip().rstrip(".")
    if not term or not desc or term not in prompt.core or "(" in desc or ")" in desc:
        return None, {"error": "bad_term_or_description", "term": term}
    first = desc.split()[0].lower() if desc.split() else ""
    if not 3 <= len(desc.split()) <= 14 or BAD_DESCRIPTION.search(desc) \
            or first in {"explain", "describe", "show", "give", "list", "tell", "fill", "see"}:
        return None, {"error": "description_not_a_definition", "term": term, "description": desc}
    new = prompt.copy()
    new.core = prompt.core.replace(term, f"{term} ({desc})", 1)
    return new, {"term": term, "description": desc}


OPS = {
    "restate": op_restate,
    "background": op_background,
    "instruction": op_instruction,
    "appositive": op_appositive,
}


def _target_words(item, ctx, tc, rng):
    """Target length (words) for this question, or None for the fixed n_ops behaviour.
    mode real:  drawn from the word counts of REAL user queries of the same domain (only lengths that
                are at least 4 words longer than the question and at most cap_words);
    mode range: uniform in [lo, hi]."""
    mode = (tc or {}).get("mode", "none")
    n0 = len(item["question"].split())
    if mode == "range":
        lo, hi = tc["range"]
        return max(rng.randint(int(lo), int(hi)), n0 + 1)
    if mode == "real":
        domain = "medical" if item.get("rag") == "medrag" else "general"
        pool = [n for n in ctx.real_lengths(domain) if n0 + 4 <= n <= tc.get("cap_words", 100)]
        if not pool:
            if not ctx.real_lengths(domain):
                _warn_once(ctx, domain)
            return None
        return rng.choice(pool)
    return None


def _warn_once(ctx, domain):
    if not getattr(ctx, "_warned_len", False):
        ctx._warned_len = True
        print(f"  note: length.target.mode is 'real' but there are no real queries for '{domain}' "
              f"(run scripts/prepare_real_queries.py {domain}); using n_ops instead.")


def _build_to_target(item, ctx, mc, rng, target):
    """Add operators one at a time (in weighted random order) until the prompt reaches `target` words.
    An operator that would push the prompt beyond target * slack is skipped, so short fixed-size
    operators (instruction, appositive) can finish the job. Returns (prompt, info, ops, error)."""
    slack = mc["target"].get("slack", 1.4)
    order = weighted_sample(rng, mc["operators"], len(mc["operators"]))
    prompt, info, ops, last_err = Prompt(core=item["question"]), {}, [], None
    fallback = None  # first operator that alone overshoots: used only if no other one fits
    for name in order:
        if len(ops) >= mc["target"].get("max_ops", 3) or len(prompt.render().split()) >= target:
            break
        new, detail = OPS[name](prompt, item, ctx, rng)
        if new is None:
            last_err = f"{name}:{detail.get('error')}"
            continue
        if len(new.render().split()) > target * slack:
            if not ops and fallback is None:
                fallback = (name, new, detail)
            continue
        prompt, info[name] = new, detail
        ops.append(name)
    if not ops and fallback:
        name, prompt, detail = fallback
        info[name], ops = detail, [name]
    return prompt, info, ops, (None if ops else (last_err or "no_operator_applied"))


def mutate_length(item, ctx, mc, rng):
    """Returns {"ok": True, "prompt", "operators", "info"} or {"ok": False, "reason"}."""
    last = "no_attempt"
    for attempt in range(ctx.max_retries + 1):
        target = _target_words(item, ctx, mc.get("target"), rng)
        if target:
            prompt, info, ops, failed = _build_to_target(item, ctx, mc, rng, target)
            info["target_words"] = target
        else:
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


def fallback_length(item):
    """Last resort when every attempt failed its check (mutation.fill_all): the question plus ONE content-free
    answer-style instruction from the fixed list. No LLM, no check needed: the sentence has no domain content, so
    what is asked and the answer cannot change. Recorded with info["fallback"] so the report can count them."""
    import random

    text = random.Random(item["id"]).choice(INSTRUCTIONS)
    prompt = Prompt(core=item["question"])
    prompt.post.append(text)
    return {"ok": True, "prompt": prompt, "operators": ["instruction"],
            "info": {"instruction": text, "fallback": "template_only"}}
