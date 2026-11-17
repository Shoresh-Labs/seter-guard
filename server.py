"""Seter Guard — self-hosted PII masking for LLM traffic.

Implements LiteLLM's Generic Guardrail API (POST /beta/litellm_basic_guardrail_api)
plus a plain /v1/redact endpoint. Runs fully on your own machine: no text leaves it.
"""
import bisect, logging, os, re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline

DEFAULT_MODEL = "shoresh-labs/seter-pii-he-ar-fast@v1.1"
# The commit that tag pointed to when this line was written. A tag can be moved on the Hub; if v1.1 ever
# resolves to anything else the server refuses to start instead of running weights nobody chose.
DEFAULT_COMMIT = "2744c393718de9a911d9db3478341984a99a7cdb"
_REPO_ID = re.compile(r"^[\w.-]+(/[\w.-]+)?$")


def _flag(name, strict=True):
    """An on/off environment variable. An unrecognised value stops the server, or counts as off when not strict."""
    value = os.environ.get(name, "").strip().lower()
    if value in ("1", "true", "on", "yes"):
        return True
    if strict and value not in ("", "0", "false", "off", "no"):
        raise ValueError(f"{name}={value!r}: use 1 or 0")
    return False


def parse_model(value: str):
    """SETER_MODEL -> (model, revision, is_local).

    An existing folder is used as it is, whatever characters its path contains. Otherwise the value is a Hub
    repo, `org/name@version` (a tag, branch or commit) or `org/name` alone, which means the newest weights
    and changes when a new version is published."""
    value = value.strip()
    if not value:
        raise ValueError("SETER_MODEL is empty")
    if os.path.isdir(value):
        return value, None, True
    repo, sep, revision = value.rpartition("@") if "@" in value else (value, "", "")
    if not _REPO_ID.match(repo):
        raise ValueError(f"SETER_MODEL={value!r} is neither an existing folder nor a Hub repo (org/name or org/name@version)")
    if sep and not revision:
        raise ValueError(f"SETER_MODEL={value!r}: nothing after '@' (for example {DEFAULT_MODEL}); "
                         "an empty version would silently load the newest weights")
    if os.path.isdir(repo):  # transformers prefers a local folder of that name and would ignore the version
        raise ValueError(f"a folder named {repo!r} exists in the working directory and would be loaded in place of the "
                         f"Hub repo, ignoring the version. Use its path (./{repo}) to load it, or start from another directory")
    return repo, revision or None, False


def resolve_commit(model, revision, is_local):
    """The Hub commit behind `revision`, looked up once. Everything is then loaded by that commit, so the
    pipeline, the window tokenizer and /health cannot disagree, and a tag that moves later changes nothing."""
    if is_local:
        return None
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(model, "config.json", revision=revision)
    except Exception as e:  # noqa: BLE001 - re-raised with what the operator needs to know
        hint = ""
        if _flag("HF_HUB_OFFLINE", strict=False):
            baked = os.environ.get("SETER_BAKED_MODEL")
            hint = (" This process is offline (HF_HUB_OFFLINE=1) and can only load what is already downloaded"
                    + (f": this image was built with SETER_MODEL={baked}" if baked else "")
                    + ". Use that exact value (the version after '@' included), rebuild the image with "
                      "--build-arg SETER_MODEL=..., or set HF_HUB_OFFLINE=0 to allow a download.")
        raise RuntimeError(f"cannot load {model} at version {revision or 'main (not pinned)'}: {type(e).__name__}.{hint}") from e
    m = re.search(r"snapshots[/\\]([0-9a-f]{40})[/\\]", path)
    if not m:
        raise RuntimeError(f"cannot tell which commit of {model} was resolved (cache path {path!r})")
    return m.group(1)


MODEL, REVISION, IS_LOCAL = parse_model(os.environ.get("SETER_MODEL", DEFAULT_MODEL))
COMMIT = resolve_commit(MODEL, REVISION, IS_LOCAL)
if (MODEL, REVISION) == parse_model(DEFAULT_MODEL)[:2] and COMMIT != DEFAULT_COMMIT:
    raise RuntimeError(f"{DEFAULT_MODEL} now resolves to commit {COMMIT}, not {DEFAULT_COMMIT}: the tag was moved. "
                       f"Pin SETER_MODEL to a commit you trust (repo@<commit>) to start anyway")
TAGS = {"private_person": "PERSON", "private_address": "ADDRESS", "private_phone": "PHONE",
        "private_email": "EMAIL", "account_number": "ID"}
MIN_SCORE = float(os.environ.get("SETER_MIN_SCORE", "0.5"))
ACTION = os.environ.get("SETER_ACTION", "mask")  # "mask" or "block"


# ---- Israeli identity numbers ------------------------------------------------------------------------
# An Israeli identity number (Teudat Zehut) is nine digits whose last digit is a check digit. The model
# cannot do that arithmetic, and its account_number class ("ID" here) covers identity, passport, licence
# and bank account numbers alike, so a nine-digit enforcement or proceeding number is masked as one of them.
#
# With SETER_IL_ID_CHECK=1 such a number is left in the text, but only when all of this holds:
#   - the model labelled it ID, and no window called it anything else, at any confidence, trusted or not;
#   - it is exactly nine ASCII digits standing alone, not a piece of a longer number or code;
#   - the check digit is wrong, so it is not a correctly written Israeli identity number;
#   - the words directly before it are one of three Hebrew phrases (REFERENCE_BEFORE): "תיק הוצאה לפועל",
#     "הליך" or "אסמכתא", optionally followed by "מספר", with nothing else between them and the number;
#   - nothing on the line before it says whose record this is (PERSON_RECORD: a patient, an employee, a
#     customer), and nothing names an identity, passport, licence or account number (IDENTITY_CUE) between
#     the previous number on the line and this one;
#   - and neither kind of word follows it: to the end of the line, or, when another number follows, to the
#     end of the clause (a label after that belongs to the other number).
# The narrowness is deliberate. A wrong check digit does not make a number harmless: the first eight digits
# of an identity number determine the ninth, so one with a mistyped or misread last digit still discloses
# the real one, and US social security numbers, many passport numbers and bank account numbers are nine
# digits that fail the check nine times in ten. Wider rules were tried and each leaked: a bare "תיק" is also
# a medical, tax or population-registry file whose number is the person's identity number, and English
# "case", "file" and "reference" are ordinary words ("the number we have on file: ...", "tax file number").
#
# Off by default: these are word lists, not understanding, and a guardrail that is unsure masks.
IL_ID_CHECK = _flag("SETER_IL_ID_CHECK")
_HE = "א-ת"
_HE_PREFIX = "(?:ו?[בלהמשכ]{1,2})?"   # attached ב ל ה מ ש כ, also after ו; a bare ו is left out ("ותיק" is a veteran)
_QUOTE = "[\"״”'׳.]"                  # the mark inside a Hebrew abbreviation: ת"ז, ת.ז, ת׳ז
# Bounded repeats only, and matched against the last REFERENCE_TAIL characters before the number: an open
# "\s*" here backtracked for seconds on a long run of spaces.
REFERENCE_TAIL = 80
REFERENCE_BEFORE = re.compile(
    rf"(?<![{_HE}]){_HE_PREFIX}(?:תיק\s{{1,2}}ה?(?:הוצאה\s{{1,2}}לפועל|הוצל[\"״”]פ)|הליך|אסמכתא)"
    rf"(?:\s{{1,2}}(?:מספר|מס['׳.]?))?\s{{0,2}}:?\s{{0,2}}$")
IDENTITY_CUE = re.compile(
    rf"ת{_QUOTE}\s*ז|מ{_QUOTE}\s*[זא]|ח{_QUOTE}\s*[ןנ]|(?<![{_HE}])תז(?![{_HE}])|זהות|זיהוי|מזהה|דרכו[ןנ]|רי?שיו[ןנ]"
    rf"|חשבו[ןנ](?!י(?:ת|ות)(?![{_HE}]))|ח-ן|עוסק\s+(?:מורשה|פטור)|ביטוח\s+לאומי|מספר\s+אישי|מס\s+הכנסה|פקיד\s+שומה|משרד\s+הפנים"
    r"|קופת\s+חולים|רפואי|هوي[ةه]|جواز|رخصة|حساب"
    r"|(?<![a-z])(?:ids?|i\.d|ssn|ss#|social|passport|licen[cs]e|account|acct|iban|identity|national|mrn|tax|medical|insurance"
    r"|immigration|personnel|personal)(?![a-z])", re.IGNORECASE)
# Stems, so that every inflected form counts (עובדיו, לקוחותיה, חשבונך, מטופלת). Only the look-alikes known to be
# harmless are let through by name: חשבונית / חשבוניות (an invoice) above, עובדה / עובדות / עובדתי (a fact) here.
PERSON_RECORD = re.compile(
    rf"מטופל|חול(?:ה|ים|ת)|עובד(?!(?:ה|ות|תי(?:ת|ים|ות)?)(?![{_HE}]))|לקוח|תלמיד|סטודנט|مريض|موظف"
    r"|(?<![a-z])(?:patient|employee|customer|passenger|student|taxpayer|member|user)(?![a-z])", re.IGNORECASE)
_LONG_NUMBER = re.compile(r"\d[\d-]{3,}\d")           # five characters or more: another number on the line
_CLAUSE_END = re.compile(r"[,;]|[.!?](?=\s|$)")
log = logging.getLogger("seter_guard")
if not log.handlers:  # uvicorn configures only its own loggers; without this the audit lines would go nowhere
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)
    log.propagate = False


def il_id_check(number: str):
    """True or False for nine ASCII digits (does the check digit match?), None for anything else."""
    if len(number) != 9 or not (number.isascii() and number.isdigit()):
        return None
    total = 0
    for i, ch in enumerate(number):
        d = int(ch) * (1 + i % 2)
        total += d if d < 10 else d - 9
    return total % 10 == 0


def _stands_alone(text: str, a: int, b: int):
    """The digits at text[a:b] are a whole number, not a piece of a longer number or code.

    A letter or digit touching it makes it a piece (A034851272). So does a space with a digit beyond it
    (034851272 21, a number written in groups), and any other character with a letter or digit beyond it
    (12-034851272, 034851272/21, 034851272\\21, 034851272-A)."""
    for near, far in ((text[a - 1:a], text[a - 2:a - 1]), (text[b:b + 1], text[b + 1:b + 2])):
        if not near:
            continue
        if near.isalnum() or (far.isdigit() if near.isspace() else far.isalnum()):
            return False
    return True


def _called_a_reference(text: str, a: int, b: int):
    """The words directly before text[a:b] name an enforcement or proceeding number, and nothing around it
    says whose record it is or names an identity, passport, licence or account number."""
    line_start = text.rfind("\n", 0, a) + 1
    line_end = text.find("\n", b)
    before, after = text[line_start:a], text[b:len(text) if line_end < 0 else line_end]
    if not REFERENCE_BEFORE.search(before[-REFERENCE_TAIL:]):
        return False
    # Whose record it is holds for the whole line. A label such as ת.ז. describes the number that follows it,
    # so a label before an earlier number on the line is about that number, not this one.
    earlier = [m.end() for m in _LONG_NUMBER.finditer(before)]
    if PERSON_RECORD.search(before) or IDENTITY_CUE.search(before[earlier[-1] if earlier else 0:]):
        return False
    # A label can follow the number: "מספר הליך 034851272 (ת.ז. של החייב)", "... 034851272, שהוא מספר הזהות".
    # Read to the end of the line. Only when another number follows is the clause the limit: in
    # "מספר הליך 034851272, ת.ז. של החייב 034851279" the label after the comma belongs to the second number.
    nxt = _LONG_NUMBER.search(after)
    if nxt:
        clause = _CLAUSE_END.search(after, 0, nxt.start())  # the full stops inside "ת.ז." are not followed by a space
        after = after[:clause.start() if clause else nxt.start()]
    return not (IDENTITY_CUE.search(after) or PERSON_RECORD.search(after))


def releasable(text: str, a: int, b: int, t: str, other_type_seen: bool):
    """A detected span to leave in the text (see the conditions above). False whenever in doubt."""
    return (IL_ID_CHECK and t == "ID" and not other_type_seen and il_id_check(text[a:b]) is False
            and _stands_alone(text, a, b) and _called_a_reference(text, a, b))


ner = pipeline("token-classification", model=MODEL, revision=COMMIT, aggregation_strategy="simple")
app = FastAPI(title="Seter Guard")

# ---- long texts -------------------------------------------------------------------------------------
# The pipeline reads one window of text per call and silently drops whatever is past the model's limit
# (512 tokens for the large model, about 8,000 for the fast one). A guardrail must read all of it, so
# long texts are cut into overlapping windows, each window is scanned, and the results are joined.
#
# The window tokenizer is a separate object from the pipeline's. The pipeline switches truncation on for
# every call; using its tokenizer here without truncation would flip that setting back and forth between
# requests, and the tokenizer raises "Already borrowed" when two threads do that at once.
wtok = AutoTokenizer.from_pretrained(MODEL, revision=COMMIT)


def _model_limit():
    """Tokens of text the model can read in one call (special tokens excluded)."""
    limit = min(int(ner.tokenizer.model_max_length), 1_000_000)
    if limit >= 1_000_000:  # the tokenizer does not say; fall back to the position table (XLM-R keeps 2 spare rows)
        limit = int(getattr(ner.model.config, "max_position_embeddings", 514)) - 2
    return limit - 2  # room for the two special tokens the pipeline adds to each window


MIN_OVERLAP = 64


def _settings():
    limit = _model_limit()
    # Default 510: the Seter models are trained and scored on windows of up to 512 tokens. The fast model can
    # take more in one call (set SETER_WINDOW_TOKENS up to its limit), but that is not what the scores measure.
    window = int(os.environ.get("SETER_WINDOW_TOKENS", min(510, limit)))
    overlap = int(os.environ.get("SETER_OVERLAP_TOKENS", min(96, window // 2)))
    max_tokens = int(os.environ.get("SETER_MAX_TOKENS", "16000"))
    batch = int(os.environ.get("SETER_BATCH_SIZE", "4"))
    if not 2 * MIN_OVERLAP <= window <= limit:
        raise ValueError(f"SETER_WINDOW_TOKENS={window} must be between {2 * MIN_OVERLAP} and {limit}, the most {MODEL} reads in one call; "
                         "a larger window would be cut short by the model and text would go through unread")
    # An entity that starts right after a cut is taken from the previous window, which must therefore reach
    # well past the cut: with fewer than 64 shared tokens a long address could be seen whole by neither window.
    if not MIN_OVERLAP <= overlap <= window // 2:
        raise ValueError(f"SETER_OVERLAP_TOKENS={overlap} must be between {MIN_OVERLAP} and {window // 2} (half the window)")
    if max_tokens < window or batch < 1:
        raise ValueError("SETER_MAX_TOKENS must be at least one window and SETER_BATCH_SIZE at least 1")
    return window, overlap, max_tokens, batch


WINDOW, OVERLAP, MAX_TOKENS, BATCH = _settings()
# One call on each tokenizer before any request thread exists. A tokenizer.json can carry a truncation setting
# saved at training time; the first call replaces it, and two threads doing that at once raise "Already borrowed".
wtok("warm up", add_special_tokens=False, return_offsets_mapping=True, verbose=False)
ner("warm up")


class TextTooLong(Exception):
    pass


def windows(text: str):
    """[(start, end, trust_from, trust_to, own_from, own_to)] in characters. The windows cover `text`, each is
    at most WINDOW tokens, neighbours share about OVERLAP tokens, and every window starts at the beginning of
    a word, so the piece handed to the model never begins in the middle of a word or a number.

    A window is not trusted near a cut: what it finds in its first and last OVERLAP // 4 tokens was seen with
    little context (an id whose label was cut away looks like a phone number), and the neighbour saw the same
    text well inside its own window. Each character is also "owned" by one window, the one that sees it
    furthest from a cut; when two windows disagree about an entity's type, the owner's answer is used.
    Short texts are one window that trusts and owns everything."""
    whole = [(0, len(text), 0, len(text), 0, len(text))]
    # Always count tokens: one character can become several tokens (ligatures, unit symbols), so a text
    # shorter than WINDOW characters can still be longer than WINDOW tokens.
    offs = wtok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
    n = len(offs)
    if n <= WINDOW:
        return whole
    if n > MAX_TOKENS:
        raise TextTooLong(f"text is {n} tokens; the limit is SETER_MAX_TOKENS={MAX_TOKENS}")
    # token indexes where a word starts: the token itself begins with whitespace, or whitespace precedes it
    word = [i for i, (a, b) in enumerate(offs) if i == 0 or text[a:a + 1].isspace() or text[a - 1:a].isspace()]

    def word_start_at_or_before(i, floor):
        k = bisect.bisect_right(word, i) - 1
        return word[k] if k >= 0 and word[k] > floor else i  # one "word" longer than the search range: cut inside it

    cuts, i = [], 0  # (first token, one past the last token) per window
    while True:
        j = min(i + WINDOW, n)
        if j < n:
            j = word_start_at_or_before(j, i + WINDOW // 2)
        cuts.append((i, j))
        if j >= n:
            break
        nxt = word_start_at_or_before(j - OVERLAP, i)
        assert nxt > i, "windows must advance"  # holds because j > i + WINDOW // 2 and OVERLAP <= WINDOW // 2
        i = nxt
    out, margin = [], OVERLAP // 4
    for k, (i, j) in enumerate(cuts):
        first, last = k == 0, j >= n
        a = 0 if first else offs[i][0]
        b = len(text) if last else offs[j - 1][1]
        shared_before = cuts[k - 1][1] - i if not first else 0   # tokens shared with the previous window
        shared_after = j - cuts[k + 1][0] if not last else 0     # tokens shared with the next one
        trust_from = 0 if first else offs[i + min(margin, shared_before // 2)][0]
        trust_to = len(text) if last else offs[j - min(margin, shared_after // 2) - 1][1]
        own_from = 0 if first else offs[i + shared_before // 2][0]
        own_to = len(text) if last else offs[j - shared_after + shared_after // 2][0]
        out.append((a, b, trust_from, trust_to, own_from, own_to))
    return out


def count_tokens(text: str):
    return len(wtok(text, add_special_tokens=False, verbose=False)["input_ids"])


def entities(text: str):
    """Model output for the whole text, with character offsets into `text` and an "owner" flag (see windows)."""
    wins = windows(text)
    if len(wins) == 1:
        return [{**e, "owner": True} for e in ner(text)]
    out = []
    results = ner([text[a:b] for a, b, *_ in wins], batch_size=BATCH)
    for (a, _b, trust_from, trust_to, own_from, own_to), found in zip(wins, results):
        for e in found:
            s, t = e["start"] + a, e["end"] + a
            while s < t and text[s].isspace():  # the pipeline may include the space before a word
                s += 1
            # Dropped: what lies wholly in the untrusted strip after a cut (seen without its left context; the
            # previous window has it), and what starts in the strip before the next cut. Kept: anything that
            # reaches into the trusted zone, so a long entity that begins just after a cut is never lost.
            if s < t and t > trust_from and s < trust_to:
                out.append({**e, "start": s, "end": t, "owner": own_from <= s < own_to})
            elif s < t:  # not used for masking, but kept as a second opinion (see scan)
                out.append({**e, "start": s, "end": t, "owner": False, "untrusted": True})
    return out


def scan(text: str):
    """(spans to mask, spans detected but left in the text), each a list of (start, end, type)."""
    out = []  # [start, end, type, type_is_from_owner, another_type_was_reported]
    # (start, end, type) of what a window reported below MIN_SCORE or right next to a cut: never masked on that
    # evidence, but enough to stop a later check from releasing the span
    doubts = []
    for e in sorted(entities(text), key=lambda e: (e["start"], not e["owner"], -e["end"])):
        if e["score"] < MIN_SCORE or e.get("untrusted"):
            doubts.append((e["start"], e["end"], TAGS.get(e["entity_group"], "PII")))
            continue
        a, b, t = e["start"], e["end"], TAGS.get(e["entity_group"], "PII")
        while a < b and text[a].isspace():
            a += 1
        if a >= b:
            continue
        last = out[-1] if out else None
        if last and a < last[1]:
            # the same stretch reported by two windows: mask the union (when in doubt, mask); the type comes from
            # the window that owns the text, whichever of the two spans starts first
            last[1] = max(last[1], b)
            if t != last[2]:
                last[4] = True  # the windows disagree: such a span is never released by a later check
            if e["owner"] and not last[3]:
                last[2], last[3] = t, True
        # merge neighbours of the same type separated only by whitespace (e.g. أبو خالد)
        elif last and last[2] == t and text[last[1]:a].strip() == "":
            last[1] = b
        else:
            out.append([a, b, t, e["owner"], False])
    # Judged after merging, so that two numbers the model joined into one span are seen as what they are: not
    # a single nine-digit number, and therefore masked.
    masked, kept = [], []
    for a, b, t, _owner, mixed in out:
        # a weak second opinion counts too: if any window called part of this span something else, it stays masked
        mixed = mixed or any(x < b and a < y and u != t for x, y, u in doubts)
        (kept if releasable(text, a, b, t, mixed) else masked).append((a, b, t))
    for a, b, t in kept:  # both endpoints pass through here; offsets only, never the digits
        log.info("left unmasked by il_id_check: %s at characters %d-%d of a %d-character text", t, a, b, len(text))
    return masked, kept


def spans(text: str):
    return scan(text)[0]


def _render(text: str, to_mask):
    res, pos, found = [], 0, []
    for a, b, t in to_mask:
        res += [text[pos:a], f"[{t}]"]
        found.append({"type": t, "start": a, "end": b})
        pos = b
    res.append(text[pos:])
    return "".join(res), found


def mask(text: str):
    return _render(text, spans(text))


class RedactIn(BaseModel):
    text: str


@app.post("/v1/redact")
def redact(body: RedactIn):
    try:
        to_mask, kept = scan(body.text)
    except TextTooLong as e:  # never return text that was not read
        raise HTTPException(status_code=413, detail=str(e))
    masked, found = _render(body.text, to_mask)
    # kept: detected and deliberately left in the text, so a log can tell that from "nothing found"
    return {"text": masked, "entities": found,
            "kept": [{"type": t, "start": a, "end": b, "reason": "il_id_check"} for a, b, t in kept]}


class GuardIn(BaseModel):
    texts: list[str] = []
    input_type: str | None = None
    structured_messages: list[dict] | None = None
    additional_provider_specific_params: dict | None = None


@app.post("/beta/litellm_basic_guardrail_api")
def litellm_guardrail(body: GuardIn):
    # Count first: the limit is for the request as a whole, and an over-long request is refused before any
    # model time is spent. Blocked, not passed through.
    total = sum(count_tokens(t) for t in body.texts)
    if total > MAX_TOKENS:
        return {"action": "BLOCKED", "blocked_reason": "Text too long to scan for personal data (Seter): "
                                                      f"{total} tokens in this request; the limit is SETER_MAX_TOKENS={MAX_TOKENS}"}
    masked, hit = [], False
    for t in body.texts:
        m, found = mask(t)
        masked.append(m)
        hit = hit or bool(found)
    if not hit:
        return {"action": "NONE"}
    if ACTION == "block":
        return {"action": "BLOCKED", "blocked_reason": "Personal data detected (Seter)"}
    return {"action": "GUARDRAIL_INTERVENED", "texts": masked}


@app.get("/health")
def health():
    # revision: the version asked for (null when none was); commit: what is actually loaded (null for a folder)
    return {"ok": True, "model": MODEL, "source": "local" if IS_LOCAL else "hub", "pinned": bool(REVISION),
            "revision": REVISION, "commit": COMMIT, "il_id_check": IL_ID_CHECK}
