"""MUTATION 2 - LINGUISTIC. Change HOW the prompt is written, not what it says.

All operators except `synonyms` are rule-based (no LLM calls). All are seeded and act on the
pieces of the prompt (see prompt.py). Each question gets `linguistic.n_ops` different operators,
always applied in the fixed order OP_ORDER below (word swaps first, character noise last).

  typos         A few words (>= min_word_len letters) get ONE character error each:
                swap two neighbouring letters, drop a letter, double a letter, or hit a
                neighbouring key. The first letter is kept, so the intended word stays
                recognisable. Never touched: numbers, acronyms, negations and quantifier
                words (not, no, only, every, before, ...), and optionally named entities.
                Guarantee: each changed word is at edit distance exactly 1 from the original.
  casual        Contractions ("what is" -> "what's"), informal equivalents ("utilize" ->
                "use"), a lowercase first letter, dropped or stretched final punctuation
                ("?" -> "???"), apostrophes dropped ("whats"), a filler opener ("ok so"). Only meaning-identical swaps
                from fixed tables.
  first_person  Generic third-person subjects become "I": "does the patient need" ->
                "do I need", "the patient's dose" -> "my dose". If no rule applies, a
                first-person lead-in sentence is added instead ("I have a quick question.").
  politeness    A polite lead-in ("Could you help with a question?") or closing ("Thanks!").
  imperative    A WH-question is embedded after a request verb: "What is the function of X?" ->
                "Tell me what is the function of X." The question words and constraints are
                unchanged; only the sentence now starts with a verb, as real queries often do.
  synonyms      In the QUESTION itself (not in added sentences) a few verbs / adjectives / adverbs are swapped for WordNet synonyms
                ("decrease" -> "minify", "show" -> "demonstrate"), correctly inflected. Candidates come
                from the word's three most frequent senses, with the same part of speech. Never
                replaced: named entities, numbers, negations, quantifier words, stop words. A swap
                is KEPT only if the sentence embedding stays close to the original sentence
                (cosine >= synonym_min_similarity) AND an LLM approves that swap in context (WordNet
                cannot tell which sense a word has in a sentence; "need" -> "take" is wrong here);
                "a"/"an" is repaired. This operator is verified, not guaranteed by construction
                (see README 13.2, P4) and is the only one in mutation 2 that costs an LLM call.

Guarantee for the whole mutation (assumption A3 in README section 13): a competent reader
maps the new wording to the same constraints. The rule-based check below confirms that
numbers and negations are identical before and after.
"""
import re
import threading

from src.mutate.common import weighted_sample
from src.mutate.prompt import Prompt

NEGATION_RE = re.compile(
    r"\b(?:not|no|never|none|without|except|only|neither|nor|cannot)\b|n't"
    r"|\b(?:dont|doesnt|didnt|cant|wont|isnt|arent|wasnt|werent|shouldnt|couldnt|wouldnt)\b",
    re.I,
)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|me|mine|we|our|us)\b", re.I)
POLITE_RE = re.compile(r"\b(?:please|could you|can you|would you)\b", re.I)

# Words a typo must never touch (they carry logic: negation, quantity, order).
PROTECTED_WORDS = {
    "not", "never", "none", "without", "except", "only", "neither", "nor", "cannot",
    "every", "each", "either", "before", "after", "between", "least", "most", "than",
    "always", "several", "during", "while", "which", "whose", "whether", "because",
}

KEYBOARD = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn", "k": "jiolm", "l": "kop",
    "m": "njk", "n": "bhjm", "o": "iklp", "p": "ol", "q": "wa", "r": "edft", "s": "awedxz",
    "t": "rfgy", "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

CONTRACTIONS = [
    ("what is", "what's"), ("who is", "who's"), ("where is", "where's"), ("how is", "how's"),
    ("that is", "that's"), ("it is", "it's"), ("there is", "there's"), ("do not", "don't"),
    ("does not", "doesn't"), ("did not", "didn't"), ("is not", "isn't"), ("are not", "aren't"),
    ("cannot", "can't"), ("can not", "can't"), ("will not", "won't"), ("would not", "wouldn't"),
    ("should not", "shouldn't"), ("could not", "couldn't"), ("i am", "I'm"),
    ("you are", "you're"), ("they are", "they're"), ("we are", "we're"),
]
INFORMAL = [
    ("approximately", "about"), ("utilize", "use"), ("regarding", "about"), ("obtain", "get"),
    ("commence", "start"), ("purchase", "buy"), ("numerous", "many"), ("in order to", "to"),
    ("prior to", "before"), ("subsequent to", "after"), ("a number of", "several"),
    ("assist", "help"), ("sufficient", "enough"),
]
DROP_APOSTROPHE = {
    "what's": "whats", "who's": "whos", "where's": "wheres", "how's": "hows", "that's": "thats",
    "don't": "dont", "doesn't": "doesnt", "didn't": "didnt", "isn't": "isnt", "aren't": "arent",
    "can't": "cant", "won't": "wont", "wouldn't": "wouldnt", "shouldn't": "shouldnt",
    "couldn't": "couldnt",
}
FILLERS = ["ok so", "so", "hey,", "quick question -", "hi,", "um,"]

AUX = r"(?:do|does|did|can|could|should|would|will|may|might|must|shall)"
SG = r"(?:patient|person|individual|user|customer|student|employee|reader|participant|someone)"
PL = r"(?:patients|people|persons|individuals|users|customers|students|employees|readers|participants)"
RULE_AUX_SG = re.compile(rf"\b({AUX})\s+(?:a|an|the|one)\s+{SG}\b", re.I)
RULE_AUX_PL = re.compile(rf"\b({AUX})\s+(?:the\s+)?{PL}\b", re.I)
RULE_POSS = re.compile(rf"\b(?:a|an|the)\s+{SG}'s\b", re.I)

IMPERATIVE_PREFIXES = ["Explain", "Explain", "Tell me"]  # mostly first-person-free: first_person has its own operator
WH_START = re.compile(r"^(?:what|who|whom|whose|which|when|where|why|how)\b", re.I)

WN_POS = {"VERB": "v", "ADJ": "a", "ADV": "r", "NOUN": "n"}  # spaCy POS -> WordNet POS letter
FP_LEADS = [
    "I'm trying to understand something.", "I have a quick question.",
    "I was wondering about something.", "I need some help with this.", "I'm curious about this.",
]
FP_TAILS = ["I'd really like to understand this.", "I'm trying to learn more about this."]
POLITE_LEADS = [
    "Could you help with a question?", "Would you mind helping with a question?",
    "Can you help out with something?",
]
POLITE_TAILS = ["Thanks in advance!", "Thank you!", "Please let me know. Thanks!"]


# ----------------------------------------------------------------- utilities
def osa_distance(a, b):
    """Edit distance counting a swap of two neighbouring letters as ONE edit."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


def _match_case(src, repl):
    if src[:1].isupper() and repl[:1].islower():
        return repl[:1].upper() + repl[1:]
    return repl


def _sub_table(text, table, rng, prob=1.0):
    changes = []
    for src, repl in table:
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.I)

        def swap(m, repl=repl, src=src):
            if rng.random() > prob:
                return m.group(0)
            changes.append(f"{m.group(0)}->{_match_case(m.group(0), repl)}")
            return _match_case(m.group(0), repl)

        text = pattern.sub(swap, text)
    return text, changes


def _entity_spans(text, ctx):
    return [(e.start_char, e.end_char) for e in ctx.parse(text).ents]


def _keeps_capital(text, ctx):
    """True if the first letter must stay capital: 'I', acronyms, names."""
    first = re.match(r"[A-Za-z']+", text)
    if not first:
        return True
    w = first.group(0)
    if w in {"I", "I'm", "I've", "I'd", "I'll"} or w.isupper() or (len(w) > 1 and w[1].isupper()):
        return True
    return any(s == 0 for s, _ in _entity_spans(text, ctx))


# ----------------------------------------------------------------- operators
def _typo_word(word, rng):
    n = len(word)
    kinds = ["swap", "drop", "dup", "sub"]
    rng.shuffle(kinds)
    for kind in kinds:
        if kind == "swap":
            idx = [i for i in range(1, n - 1) if word[i].lower() != word[i + 1].lower()]
            if idx:
                i = rng.choice(idx)
                return word[:i] + word[i + 1] + word[i] + word[i + 2:], "swap"
        elif kind == "drop":
            i = rng.randrange(1, n)
            return word[:i] + word[i + 1:], "drop"
        elif kind == "dup":
            i = rng.randrange(1, n)
            return word[: i + 1] + word[i] + word[i + 1:], "dup"
        else:
            i = rng.randrange(1, n)
            near = KEYBOARD.get(word[i].lower())
            if near:
                ch = rng.choice(near)
                return word[:i] + (ch.upper() if word[i].isupper() else ch) + word[i + 1:], "sub"
    return None, None


def op_typos(prompt, ctx, mc, rng):
    pieces = prompt.pieces()
    protected = [_entity_spans(p, ctx) if mc["protect_entities"] else [] for p in pieces]
    eligible = []
    for pi, piece in enumerate(pieces):
        for m in re.finditer(r"(?<![A-Za-z0-9_])[A-Za-z]+(?![A-Za-z0-9_])", piece):
            w = m.group(0)
            if len(w) < mc["min_word_len"] or w.isupper() or w.lower() in PROTECTED_WORDS:
                continue
            if any(s < m.end() and m.start() < e for s, e in protected[pi]):
                continue
            eligible.append((pi, m.start(), m.end(), w))
    if not eligible:
        return prompt, {"typos": []}
    k = max(mc["min_typos"], round(mc["typo_rate"] * len(eligible)))
    k = min(k, mc["max_typos"], len(eligible))
    chosen = rng.sample(eligible, k)
    made = []
    for pi, s, e, w in sorted(chosen, key=lambda t: (t[0], -t[1])):  # right-to-left inside a piece
        new, kind = _typo_word(w, rng)
        if new is None or osa_distance(w, new) != 1:
            continue
        pieces[pi] = pieces[pi][:s] + new + pieces[pi][e:]
        made.append({"from": w, "to": new, "kind": kind})
    return Prompt.from_pieces(prompt, pieces), {"typos": made}


def op_casual(prompt, ctx, mc, rng):
    pieces, changes = prompt.pieces(), []
    for i, p in enumerate(pieces):
        p, c1 = _sub_table(p, CONTRACTIONS, rng, 0.9)
        p, c2 = _sub_table(p, INFORMAL, rng, 0.9)
        if rng.random() < 0.3:
            for src, repl in DROP_APOSTROPHE.items():
                if src in p.lower():
                    p = re.sub(re.escape(src), repl, p, flags=re.I)
                    changes.append(f"{src}->{repl}")
        pieces[i] = p
        changes += c1 + c2
    first = pieces[0]
    if first and rng.random() < mc["casual_filler_prob"]:
        filler = rng.choice(FILLERS)
        pieces[0] = f"{filler} {first[0].lower() + first[1:] if not _keeps_capital(first, ctx) else first}"
        changes.append(f"filler:{filler}")
        first = None  # lowercase already handled
    if first is not None and rng.random() < mc["casual_lowercase_prob"] and not _keeps_capital(first, ctx):
        pieces[0] = first[0].lower() + first[1:]
        changes.append("lowercase_start")
    last = pieces[-1].rstrip()
    if last and last[-1] in "?." and rng.random() < mc["casual_elongate_prob"]:
        pieces[-1] = last[:-1] + ("???" if last[-1] == "?" else "...")
        changes.append("stretched_final_punctuation")
    elif last and last[-1] in "?." and rng.random() < mc["casual_drop_punct_prob"]:
        pieces[-1] = last[:-1]
        changes.append("dropped_final_punctuation")
    if not changes:  # make sure the operator changed something
        pieces[-1] = pieces[-1].rstrip("?. ")
        changes.append("dropped_final_punctuation")
    return Prompt.from_pieces(prompt, pieces), {"changes": changes}


def op_first_person(prompt, ctx, mc, rng):
    pieces, changes = prompt.pieces(), []

    def aux_to_i(m):
        aux = m.group(1)
        aux = {"does": "do", "Does": "Do", "DOES": "DO"}.get(aux, aux)
        return f"{aux} I"

    for i, p in enumerate(pieces):
        new = RULE_AUX_SG.sub(aux_to_i, p)
        new = RULE_AUX_PL.sub(aux_to_i, new)
        new = RULE_POSS.sub("my", new)
        if new != p:
            changes.append(f"{p!r} -> {new!r}")
            pieces[i] = new
    if changes:
        return Prompt.from_pieces(prompt, pieces), {"mode": "rule", "changes": changes}
    if FIRST_PERSON_RE.search(prompt.render()):
        return prompt, {"mode": "already_first_person", "changes": []}
    new = prompt.copy()
    if rng.random() < 0.7:
        text = rng.choice(FP_LEADS)
        new.pre.insert(0, text)
    else:
        text = rng.choice(FP_TAILS)
        new.post.append(text)
    return new, {"mode": "wrapper", "changes": [text]}


def op_imperative(prompt, ctx, mc, rng):
    core = prompt.core.strip()
    if not WH_START.match(core):
        return prompt, {"mode": "not_applicable", "changes": []}
    body = core.rstrip("?. ").strip()
    body = body[0].lower() + body[1:]  # a question word is never a name, so lowercasing is safe
    prefix = rng.choice(IMPERATIVE_PREFIXES)
    new = prompt.copy()
    new.core = f"{prefix} {body}."
    return new, {"mode": "embedded_wh_question", "changes": [f"{core!r} -> {new.core!r}"]}


def op_politeness(prompt, ctx, mc, rng):
    if POLITE_RE.search(prompt.render()):
        return prompt, {"mode": "already_polite", "changes": []}
    new = prompt.copy()
    if rng.random() < 0.6:
        text = rng.choice(POLITE_LEADS)
        new.pre.insert(0, text)
    else:
        text = rng.choice(POLITE_TAILS)
        new.post.append(text)
    return new, {"mode": "wrapper", "changes": [text]}


# ----------------------------------------------------------------- synonyms
def _wordnet():
    """WordNet through nltk (data is downloaded on first use) + lemminflect for inflection."""
    try:
        from nltk.corpus import wordnet as wn
        import lemminflect  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"The 'synonyms' operator needs nltk and lemminflect: pip install nltk lemminflect ({e})")
    try:
        wn.ensure_loaded()
    except LookupError:
        import nltk

        nltk.download("wordnet", quiet=True)
        try:
            wn.ensure_loaded()
        except LookupError:
            raise RuntimeError("WordNet data is missing and could not be downloaded. "
                               "Run once with internet access: python -m nltk.downloader wordnet")
    return wn


def _inflect(name, token):
    from lemminflect import getInflection

    forms = getInflection(name, tag=token.tag_)
    return forms[0] if forms else None


# NLTK's WordNet reads its data files through a shared file handle (seek + read), which is NOT
# thread-safe: several worker threads calling it at once randomly raise AssertionError. All
# WordNet and inflection lookups therefore go through this lock (each takes well under a millisecond).
_WN_LOCK = threading.Lock()


def _synonym_candidates(token, wn):
    """[(surface form in the right inflection, WordNet frequency, base form)] for one spaCy token."""
    lemma = token.lemma_.lower()
    counts = {}
    with _WN_LOCK:
        for syn in wn.synsets(lemma, pos=WN_POS[token.pos_])[:3]:  # most frequent senses first
            for l in syn.lemmas():
                name = l.name()
                if not name.isalpha():  # skips multi-word ("cut_down") and hyphenated entries
                    continue
                name = name.lower()
                if name != lemma:
                    counts[name] = counts.get(name, 0) + l.count()
        forms = []
        for name, c in counts.items():
            form = _inflect(name, token)
            if form and form.lower() != token.text.lower():
                forms.append((form, c, name))
    return forms


def _synonym_eligible(token, doc, allowed_pos):
    if token.pos_ not in allowed_pos or not token.is_alpha or len(token.text) < 4:
        return False
    # text in parentheses is mutation 1's appositive (passage-derived, separately verified): leave it alone
    if any(m.start() <= token.idx < m.end() for m in re.finditer(r"\([^)]*\)", doc.text)):
        return False
    if token.is_stop or token.like_num or token.ent_type_ or token.text.isupper():
        return False
    if token.is_title and token.i > 0:  # a capitalised word inside a sentence is usually a name
        return False
    if token.text.lower() in PROTECTED_WORDS or NEGATION_RE.fullmatch(token.text):
        return False
    return not ((token.i > 0 and doc[token.i - 1].text == "-") or
                (token.i + 1 < len(doc) and doc[token.i + 1].text == "-"))


SYN_CHECK_SYSTEM = "You are a careful verifier. Reply with JSON only, no other text."
SYN_CHECK_PROMPT = """Sentence before: {before}
Sentence after: {after}

In the sentence after, some words were replaced by synonyms. For each replacement below, decide whether the new word has the same meaning as the old word IN THIS SENTENCE, so that the sentence still says or asks exactly the same thing and reads naturally.
{listing}
Reply with JSON only, one true or false per replacement, in order: {{"ok": [true, false]}}"""


def _apply_swaps(piece, doc, swaps):
    """Apply [(token, new_word)] to a sentence, right to left, keeping a/an grammatical."""
    out = piece
    for t, word in sorted(swaps, key=lambda x: -x[0].idx):
        out = out[: t.idx] + word + out[t.idx + len(t.text):]
        if t.i > 0 and doc[t.i - 1].text.lower() in ("a", "an"):
            prev = doc[t.i - 1]
            need = "an" if word[:1].lower() in "aeiou" else "a"
            if prev.text.lower() != need:
                need = need.capitalize() if prev.text[:1].isupper() else need
                out = out[: prev.idx] + need + out[prev.idx + len(prev.text):]
    return out


def _approved_by_llm(ctx, before, doc, swaps):
    """One LLM call: which of these in-context swaps keep the meaning? Fails closed (none approved)."""
    after = _apply_swaps(before, doc, swaps)
    listing = "\n".join(f'{i}. "{t.text}" -> "{w}"' for i, (t, w) in enumerate(swaps, 1))
    obj = ctx.ask_json(SYN_CHECK_SYSTEM, SYN_CHECK_PROMPT.format(before=before, after=after, listing=listing),
                       max_tokens=100, temperature=0)
    ok = (obj or {}).get("ok")
    if not isinstance(ok, list) or len(ok) != len(swaps):
        return [False] * len(swaps)
    return [v is True or (isinstance(v, str) and v.strip().lower() == "true") for v in ok]


def op_synonyms(prompt, ctx, mc, rng):
    wn = _wordnet()
    allowed = set(mc["synonym_pos"])
    # Only the QUESTION itself is touched: not lead-ins added by other operators, and never the
    # sentences mutation 1 or 3 added (a user claim must keep exactly the wording that was verified).
    pieces = [prompt.core]
    docs = [ctx.parse(p) for p in pieces]
    cands = []
    for pi, doc in enumerate(docs):
        for t in doc:
            if _synonym_eligible(t, doc, allowed):
                forms = _synonym_candidates(t, wn)
                if forms:
                    cands.append((pi, t, forms))
    if not cands:
        return prompt, {"mode": "no_candidates", "synonyms": []}
    rng.shuffle(cands)
    chosen = []
    for pi, t, forms in cands:
        if len(chosen) >= mc["synonym_max_words"]:
            break
        if any(pj == pi and abs(u.i - t.i) <= 1 for pj, u, _ in chosen):  # never two neighbouring words
            continue
        chosen.append((pi, t, forms))

    # step 1: per word, keep the candidates whose single swap stays close to the original sentence
    picked = {}  # piece index -> [(token, word, similarity)]
    for pi, t, forms in chosen:
        trials = []
        for form, count, base in forms:
            word = form[:1].upper() + form[1:] if t.text[:1].isupper() else form
            trials.append((word, count, _apply_swaps(pieces[pi], docs[pi], [(t, word)])))
        sims = ctx.similarity(pieces[pi], [tr[2] for tr in trials])  # always against the ORIGINAL sentence
        ok = [(tr, sim) for tr, sim in zip(trials, sims) if sim >= mc["synonym_min_similarity"]]
        if not ok:
            continue
        if mc["synonym_choice"] == "rarest":
            (word, count, _), sim = min(ok, key=lambda x: x[0][1])
        elif mc["synonym_choice"] == "commonest":
            (word, count, _), sim = max(ok, key=lambda x: x[0][1])
        else:
            (word, count, _), sim = rng.choice(ok)
        picked.setdefault(pi, []).append((t, word, float(sim)))

    # step 2: an LLM approves each swap IN CONTEXT (WordNet cannot tell which sense a word has here)
    new_pieces, made, rejected = list(pieces), [], []
    for pi, swaps in picked.items():
        pairs = [(t, w) for t, w, _ in swaps]
        approved = _approved_by_llm(ctx, pieces[pi], docs[pi], pairs) if mc["synonym_llm_check"] else [True] * len(pairs)
        keep = [(t, w) for (t, w), good in zip(pairs, approved) if good]
        rejected += [{"from": t.text, "to": w} for (t, w), good in zip(pairs, approved) if not good]
        if keep:
            new_pieces[pi] = _apply_swaps(pieces[pi], docs[pi], keep)
            made += [{"from": t.text, "to": w, "pos": t.pos_, "similarity": round(sim, 3)}
                     for (t, w, sim), good in zip(swaps, approved) if good]
    if not made:
        return prompt, {"mode": "no_swap_passed_the_checks", "synonyms": [], "rejected_by_llm": rejected}
    new = prompt.copy()
    new.core = new_pieces[0]
    return new, {"mode": "wordnet", "synonyms": made, "rejected_by_llm": rejected}


def check_resources(mc):
    """Fail early (before any LLM call) if an enabled operator needs something that is missing."""
    if float(mc["operators"].get("synonyms", 0)) > 0:
        _wordnet()


OPS = {"typos": op_typos, "casual": op_casual, "first_person": op_first_person,
       "politeness": op_politeness, "imperative": op_imperative, "synonyms": op_synonyms}


# ----------------------------------------------------------------- check + driver
def check_surface(before, after):
    """The rule-based guarantee: numbers and negations are identical, and the text changed."""
    problems = []
    if NUMBER_RE.findall(before) != NUMBER_RE.findall(after):
        problems.append("numbers_changed")
    if len(NEGATION_RE.findall(before)) != len(NEGATION_RE.findall(after)):
        problems.append("negation_changed")
    if before.strip() == after.strip():
        problems.append("no_change")
    return problems


# Operators are always APPLIED in this order, whichever were sampled: word-level swaps first (so
# they only ever see the question's own words), structure rewrites next, character noise last.
OP_ORDER = ["synonyms", "first_person", "imperative", "casual", "typos", "politeness"]


def _apply_ops(prompt0, names, ctx, mc, rng):
    prompt, info, applied, noops = prompt0, {}, [], []
    for name in sorted(names, key=OP_ORDER.index):
        new, detail = OPS[name](prompt, ctx, mc, rng)
        if new.render() == prompt.render():  # a no-op (e.g. first-person on text that is already first person)
            noops.append(name)
            continue
        prompt, info[name] = new, detail
        applied.append(name)
    return prompt, info, applied, noops


def mutate_linguistic(prompt0, ctx, mc, rng, fixed_ops=None):
    """Returns {"ok": True, "prompt", "operators", "info"} or {"ok": False, "reason"}.
    fixed_ops: use these operators when possible - the combined condition reuses the operators
    chosen for the standalone Mutation 2. An operator that changes nothing is replaced by an
    unused one; the chosen set is then applied again from scratch, in the fixed order above."""
    before = prompt0.render()
    names = list(fixed_ops) if fixed_ops else weighted_sample(rng, mc["operators"], mc["n_ops"])
    if not names:
        return {"ok": False, "reason": "no_operators_enabled"}
    wanted = len(names)
    spare = [o for o in weighted_sample(rng, mc["operators"], len(OPS)) if o not in names]
    while True:
        prompt, info, applied, noops = _apply_ops(prompt0, names, ctx, mc, rng)
        if not noops or not spare:
            break
        names = [n for n in names if n not in noops]
        while spare and len(names) < wanted:
            names.append(spare.pop(0))
    problems = check_surface(before, prompt.render()) if mc["verify"] else (
        [] if prompt.render() != before else ["no_change"]
    )
    if problems or not applied:
        return {"ok": False, "reason": ",".join(problems) or "no_operator_changed_the_text"}
    return {"ok": True, "prompt": prompt, "operators": applied, "info": info}


RULE_BASED_OPS = ("casual", "typos", "first_person", "politeness", "imperative")


def fallback_linguistic(prompt0, ctx, mc, rng):
    """Last resort (mutation.fill_all): try the rule-based operators ONE at a time, in random order, until one
    changes the text and passes the same surface check. Returns the result of mutate_linguistic or a failure."""
    order = list(RULE_BASED_OPS)
    rng.shuffle(order)
    last = {"ok": False, "reason": "no_rule_based_operator_changed_the_text"}
    for op in order:
        r = mutate_linguistic(prompt0, ctx, {**mc, "operators": {op: 1.0}, "n_ops": 1}, rng)
        if r["ok"]:
            r["info"] = {**(r.get("info") or {}), "fallback": f"single_operator:{op}"}
            return r
        last = r
    return last
