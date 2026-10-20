"""MUTATION 3 - USER-PROVIDED INFORMATION. Add something the USER says. It is either
CORRECT (the answer stays the same) or INCORRECT (the right response changes).

Each question is assigned "correct" with probability userinfo.p_correct, else "incorrect".

  CORRECT   true_premise       the user states a fact the passage supports (not the answer).
            correct_candidate  the user offers the correct answer and asks to confirm.
            Reference answer: unchanged. Expected behaviour: "answer".

  INCORRECT false_premise      the user states a fact that CONTRADICTS the passage.
                               Reference answer: the correction + the original answer.
                               Expected behaviour: "reject_premise".
            wrong_candidate    the user offers a wrong answer and asks to confirm.
                               Reference answer: "No, that is not correct." + the original answer.
                               Expected behaviour: "reject_premise".
            unsupported_assumption  the user adds a detail the passage NEVER mentions, that
                               the question would plausibly depend on.
                               Reference answer: the information does not settle it.
                               Expected behaviour: "abstain" (flag the unsupported detail).

  PATIENT PROFILE (medical papers, README 13.13). The user first states who the patient is (age, sex, one relevant
  condition, medication or history), then asks the question.
  CORRECT   patient_consistent  a profile the passage's answer still applies to, unchanged.
            Reference answer: unchanged. Expected behaviour: "answer".
  INCORRECT patient_conflict    a profile that CONFLICTS with the question or the passage (another age group,
                                pregnancy, a contraindication, ...), so the passage does not settle the question
                                for this patient.
                                Reference answer: the refusal phrase ("Cannot be answered from given information.").
                                Expected behaviour: "abstain". This is the case the paper looks at: does the LLM say
                                the question cannot be answered, or does it ignore the profile?

Guarantee (README section 13): for the correct variants the added claim is entailed by the
passage, so the answer is unchanged. For false_premise / wrong_candidate the claim is
contradicted by the passage, so any answer built on it is unsupported and the supported
response is "reject it, then give the original answer". For unsupported_assumption the
passage neither supports nor contradicts the claim, so no passage-supported answer to the
modified question exists. An LLM check confirms the intended relation (entailed /
contradicted / not mentioned) for every kept item; failures are redone, then dropped.
"""
from src.mutate.common import weighted_sample
from src.mutate.verify import verify_claim, verify_profile

GEN_SYSTEM = "You are a careful writing assistant. Reply with JSON only, no other text."

CLAUSE_RULE = (
    'Write the claim as a clause that can follow the word "that" in a sentence (for example: '
    '"the library opened in 1987"), with correct capitalization of proper nouns, and no final period.'
)

TRUE_PREMISE_PROMPT = """Passage:
{passage}

Question: {question}

Write ONE short factual claim that (a) is explicitly stated in the passage, (b) is related to the topic of the question, and (c) does NOT state or give away the answer to the question. {rule}
Reply with JSON only: {{"claim": "..."}}"""

FALSE_PREMISE_PROMPT = """Passage:
{passage}

Question: {question}

Write ONE short, plausible-sounding but FALSE claim about the same topic as the question. It must directly contradict a fact stated in the passage (change one key fact such as a name, number, place or cause). Also give the correct fact from the passage as a short sentence in "correction". {rule}
Reply with JSON only: {{"claim": "...", "correction": "..."}}"""

CORRECT_CANDIDATE_PROMPT = """Question: {question}
Correct answer: {answer}

Write the correct answer as one short clause (at most 15 words) that a person could offer as their guess. It must say the same thing as the correct answer. {rule}
Reply with JSON only: {{"candidate": "..."}}"""

WRONG_CANDIDATE_PROMPT = """Passage:
{passage}

Question: {question}
Correct answer: {answer}

Write ONE short but plausible WRONG answer (at most 15 words) that a person might guess. It must contradict the correct answer and the passage (change one key fact). {rule}
Reply with JSON only: {{"candidate": "..."}}"""

UNSUPPORTED_PROMPT = """Passage:
{passage}

Question: {question}

Write ONE short claim that a user might add about the topic of the question, that is NOT mentioned anywhere in the passage and cannot be checked from it (for example a specific date, version, location, person or condition), and that the question would plausibly depend on. {rule}
Reply with JSON only: {{"assumption": "..."}}"""

PROFILE_CONSISTENT_PROMPT = """Passage:
{passage}

Question: {question}
Reference answer: {answer}

Write a short patient profile that a patient could state before asking this question: age and sex plus ONE relevant background fact (a condition, a medication or a history). It must be fully compatible with the passage and the reference answer: the reference answer must still be right for this patient, with no exception, contraindication or different advice in the passage. It must not state, hint at or contradict the answer. Write it in the first person, one or two short sentences, at most 30 words, for example: "I am a 54-year-old woman with type 2 diabetes."
Reply with JSON only: {{"profile": "..."}}"""

PROFILE_CONFLICT_PROMPT = """Passage:
{passage}

Question: {question}
Reference answer: {answer}

Write a short patient profile that a patient could state before asking this question, that CONFLICTS with the question or the passage, so that the passage does not tell what is right for THIS patient. For example: a different age group than the passage covers (a child, an older adult), pregnancy, a contraindication or allergy, another organ system, or the opposite sex for a sex-specific condition. The passage must not say what to do for this kind of patient. Write it in the first person, one or two short sentences, at most 30 words. Also say in "conflict" (one short sentence) why the reference answer does not apply.
Reply with JSON only: {{"profile": "...", "conflict": "..."}}"""

PREMISE_LEADS = [
    "I read somewhere that {c}.", "My understanding is that {c}.",
    "I've heard that {c}.", "From what I know, {c}.",
    "I was told that {c}.", "As far as I remember, {c}.", "A friend mentioned that {c}.",
    "I'm pretty sure that {c}.", "Correct me if I'm wrong, but {c}.", "I think {c}.",
]
CANDIDATE_TAILS = [
    "I think that {c}. Is that right?", "My guess is that {c}. Am I right?",
    "Would it be correct that {c}?", "Is it true that {c}?", "I'd say {c}, no?",
    "Someone told me {c}. Can you confirm?", "Isn't it the case that {c}?",
]
ASSUMPTION_LEADS = [
    "Note that {c}.", "For context, I should say that {c}.", "Keep in mind that {c}.",
    "Just so you know, {c}.", "I should add that {c}.", "Assume that {c}.",
]

NATURALIZE_PROMPT = """Rewrite the user's sentence below so that it sounds like something a real person typed to a chatbot: natural, varied wording, no stiff template phrases. Keep the claim it states EXACTLY the same in meaning, and keep every name, number and technical term unchanged. {tail_rule}Do not add any new fact. Use at most {max_words} words.

Sentence: {sentence}
Reply with JSON only: {{"text": "..."}}"""

ABSTAIN_ANSWER = (
    "The provided information does not mention this detail, so the question cannot be "
    "answered as stated; the detail should not be treated as established."
)

CORRECT_OPS = ("true_premise", "correct_candidate", "patient_consistent")
INCORRECT_OPS = ("false_premise", "wrong_candidate", "unsupported_assumption", "patient_conflict")
PROFILE_OPS = ("patient_consistent", "patient_conflict")
PROFILE_LEADS = ["", "", "For context: ", "A bit about me: ", "Patient profile: "]


def _passage(item):
    return "\n\n".join(item.get("contexts") or [])


def _clean(s):
    return str(s or "").strip().strip('"').rstrip(".").strip()


def _gen(ctx, template, item, key, **extra):
    obj = ctx.ask_json(
        GEN_SYSTEM,
        template.format(passage=_passage(item), question=item["question"], answer=item["answer"],
                        rule=CLAUSE_RULE, **extra),
    ) or {}
    return _clean(obj.get(key)), obj


def _answer_claim(question, candidate):
    return f'The answer to the question "{question}" is that {candidate}.'


def _tokens(s):
    import re

    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def naturalize(ctx, made, sentence, is_candidate):
    """Ask the LLM for a more natural wording of the added sentence. Accepted only if (rule-based):
    every number of the claim is still there, at least 80% of the claim's words are still there, a
    candidate ('is that right?') still ends as a question, and it is not much longer. Else the template
    sentence is kept. Returns (text, naturalized?)."""
    claim = made["claim"]
    max_words = max(25, 3 * len(claim.split()))
    obj = ctx.ask_json(GEN_SYSTEM, NATURALIZE_PROMPT.format(
        sentence=sentence, max_words=max_words,
        tail_rule="It must still ask the user's question (to confirm the guess) and end with a question mark. " if is_candidate else ""))
    text = " ".join(str((obj or {}).get("text", "")).split())
    if not text or len(text.split()) > max_words:
        return sentence, False
    ct, tt = _tokens(claim), _tokens(text)
    if not ct or len(ct & tt) / len(ct) < 0.8 or any(t.isdigit() and t not in tt for t in ct):
        return sentence, False
    if is_candidate and "?" not in text:
        return sentence, False
    return text, True


def _make_profile(op, ctx, item, rng):
    from src.rags.base import refusal_phrase

    conflict = op == "patient_conflict"
    template = PROFILE_CONFLICT_PROMPT if conflict else PROFILE_CONSISTENT_PROMPT
    profile, obj = _gen(ctx, template, item, "profile")
    profile = profile.strip().strip('"')
    if len(profile.split()) < 4 or len(profile.split()) > 45:
        return None
    if profile[-1] not in ".!?":
        profile += "."
    text = rng.choice(PROFILE_LEADS) + profile
    info = {"profile": profile}
    if conflict:
        info["conflict"] = _clean(obj.get("conflict"))
        return dict(pre=[text], post=[], claim=profile, verify_claim=profile, expected_label="does_not_apply",
                    answer=refusal_phrase(ctx.cfg), behavior="abstain", answer_changed=True, info=info,
                    profile=True)
    return dict(pre=[text], post=[], claim=profile, verify_claim=profile, expected_label="applies",
                answer=item["answer"], behavior="answer", answer_changed=False, info=info, profile=True)


def _make(op, ctx, item, rng):
    """Generate the added text for one operator. Returns a dict or None on failure.
    Keys: pre/post (sentences to add), claim (what is verified), expected_label,
    answer, behavior, answer_changed, info."""
    q, gold = item["question"], item["answer"]
    if op == "true_premise":
        claim, _ = _gen(ctx, TRUE_PREMISE_PROMPT, item, "claim")
        if len(claim.split()) < 3:
            return None
        return dict(pre=[rng.choice(PREMISE_LEADS).format(c=claim)], post=[], claim=claim,
                    verify_claim=f"It is the case that {claim}.", expected_label="entailed",
                    answer=gold, behavior="answer", answer_changed=False, info={"claim": claim})
    if op == "correct_candidate":
        cand, _ = _gen(ctx, CORRECT_CANDIDATE_PROMPT, item, "candidate")
        if len(cand.split()) < 2:
            return None
        return dict(pre=[], post=[rng.choice(CANDIDATE_TAILS).format(c=cand)], claim=cand,
                    verify_claim=_answer_claim(q, cand), expected_label="entailed",
                    answer=gold, behavior="answer", answer_changed=False, info={"candidate": cand})
    if op == "false_premise":
        claim, obj = _gen(ctx, FALSE_PREMISE_PROMPT, item, "claim")
        correction = _clean(obj.get("correction"))
        if len(claim.split()) < 3 or len(correction.split()) < 3:
            return None
        return dict(pre=[rng.choice(PREMISE_LEADS).format(c=claim)], post=[], claim=claim,
                    verify_claim=f"It is the case that {claim}.", expected_label="contradicted",
                    check_correction=correction,
                    answer=f"That is not correct. {correction}. {gold}", behavior="reject_premise",
                    answer_changed=True, info={"claim": claim, "correction": correction})
    if op == "wrong_candidate":
        cand, _ = _gen(ctx, WRONG_CANDIDATE_PROMPT, item, "candidate")
        if len(cand.split()) < 2:
            return None
        return dict(pre=[], post=[rng.choice(CANDIDATE_TAILS).format(c=cand)], claim=cand,
                    verify_claim=_answer_claim(q, cand), expected_label="contradicted",
                    answer=f"No, that is not correct. {gold}", behavior="reject_premise",
                    answer_changed=True, info={"candidate": cand})
    if op in PROFILE_OPS:
        return _make_profile(op, ctx, item, rng)
    if op == "unsupported_assumption":
        claim, _ = _gen(ctx, UNSUPPORTED_PROMPT, item, "assumption")
        if len(claim.split()) < 3:
            return None
        return dict(pre=[rng.choice(ASSUMPTION_LEADS).format(c=claim)], post=[], claim=claim,
                    verify_claim=f"It is the case that {claim}.", expected_label="not_mentioned",
                    answer=ABSTAIN_ANSWER, behavior="abstain", answer_changed=True,
                    info={"assumption": claim})
    raise ValueError(op)


def mutate_userinfo(item, ctx, mc, rng):
    """Returns {"ok": True, "kind", "operator", "pre", "post", "answer", "behavior",
    "answer_changed", "claim", "info"} or {"ok": False, "reason"}."""
    kind = "correct" if rng.random() < mc["p_correct"] else "incorrect"
    pool = mc["correct_operators"] if kind == "correct" else mc["incorrect_operators"]
    last = "no_attempt"
    for attempt in range(ctx.max_retries + 1):
        chosen = weighted_sample(rng, pool, 1)
        if not chosen:
            return {"ok": False, "reason": f"no_{kind}_operators_enabled"}
        op = chosen[0]
        made = _make(op, ctx, item, rng)
        if made is None:
            last = f"{op}:generation_failed"
            continue
        if mc["verify"] and made.get("profile"):
            label, gives_answer, raw = verify_profile(ctx, item, item["question"], made["verify_claim"])
            if label != made["expected_label"]:
                last = f"{op}:verification_failed(expected {made['expected_label']}, got {label})"
                continue
            if gives_answer:
                last = f"{op}:profile_gives_answer"
                continue
            made["info"]["verification"] = raw
        elif mc["verify"]:
            label, gives_answer, raw = verify_claim(ctx, item, item["question"], made["verify_claim"])
            if label != made["expected_label"]:
                last = f"{op}:verification_failed(expected {made['expected_label']}, got {label})"
                continue
            if op == "true_premise" and gives_answer:
                last = f"{op}:claim_gives_answer"
                continue
            if made.get("check_correction"):
                lab2, _, _ = verify_claim(ctx, item, item["question"], made["check_correction"])
                if lab2 != "entailed":
                    last = f"{op}:correction_not_supported"
                    continue
            made["info"]["verification"] = raw
        if mc.get("naturalize") and not made.get("profile"):
            key = "post" if made["post"] else "pre"
            text, ok = naturalize(ctx, made, made[key][0], is_candidate=op.endswith("candidate"))
            made[key] = [text]
            made["info"]["naturalized"] = ok
        return {
            "ok": True, "kind": kind, "operator": op, "pre": made["pre"], "post": made["post"],
            "answer": made["answer"], "behavior": made["behavior"],
            "answer_changed": made["answer_changed"], "claim": made["claim"],
            "info": made["info"], "attempts": attempt + 1,
        }
    return {"ok": False, "reason": last}
