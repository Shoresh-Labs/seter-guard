"""Seter Guard — self-hosted PII masking for LLM traffic.

Implements LiteLLM's Generic Guardrail API (POST /beta/litellm_basic_guardrail_api)
plus a plain /v1/redact endpoint. Runs fully on your own machine: no text leaves it.
"""
import bisect, logging, os, re, unicodedata
from collections import Counter
from functools import lru_cache
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
#   - the words directly before it, on its line, are one of three Hebrew phrases (REFERENCE_BEFORE):
#     "תיק הוצאה לפועל", "הליך" or "אסמכתא", optionally followed by "מספר";
#   - its whole line, and the text around it (CONTEXT_BEFORE characters back and CONTEXT_AFTER on), has no
#     word that says whose record this is (PERSON_RECORD) and, within its paragraph, no word that names an
#     identity, passport, licence or account number (IDENTITY_CUE); the words are looked for in the text
#     with invisible characters and vowel points removed and compatibility forms folded, so that a word
#     cannot be hidden from the lists while reading the same;
#   - and every other place where the same digits appear in the request is released too: one value gets
#     one answer, so a number masked as "ת.ז." in one field is not printed in the next.
# The narrowness is deliberate. A wrong check digit does not make a number harmless: the first eight digits
# of an identity number determine the ninth, so one with a mistyped or misread last digit still discloses
# the real one, and US social security numbers, many passport numbers and bank account numbers are nine
# digits that fail the check nine times in ten. Wider and cleverer rules were tried and each leaked: a bare
# "תיק" is also a medical, tax or registry file numbered by the person's identity number; English "case",
# "file" and "reference" are ordinary words; and deciding which of two numbers a nearby "ת.ז." describes
# went wrong as soon as a date or a phone number stood between them. So an identity word anywhere nearby
# keeps the number masked, even when it describes another number.
#
# Off by default: these are word lists, not understanding, and a guardrail that is unsure masks.
IL_ID_CHECK = _flag("SETER_IL_ID_CHECK")
_HE = "א-ת"
_HE_PREFIX = "(?:ו?[בלהמשכ]{1,2})?"   # attached ב ל ה מ ש כ, also after ו; a bare ו is left out ("ותיק" is a veteran)
# An abbreviation starts a word (so "סגורות. זאת" is not ת.ז) and has one or two marks inside: ת"ז, ת''ז, ת.ז, ת“ז
_ABBR = rf"(?<![{_HE}])[ובלהמשכ]{{0,2}}"
_MARK = "[\"״”“„'׳.]{1,2}"
# Bounded repeats only, and matched against the last REFERENCE_TAIL characters before the number: an open
# "\s*" here backtracked for seconds on a long run of spaces.
REFERENCE_TAIL = 80
CONTEXT_BEFORE, CONTEXT_AFTER = 300, 200
REFERENCE_BEFORE = re.compile(
    rf"(?<![{_HE}]){_HE_PREFIX}(?:תיק\s{{1,2}}ה?(?:הוצאה\s{{1,2}}לפועל|הוצל[\"״”]פ)|הליך|אסמכתא)"
    rf"(?:\s{{1,2}}(?:מספר|מס['׳.]?))?\s{{0,2}}:?\s{{0,2}}$")
# Stems, so that every inflected form counts (דרכונו, חשבונך, עובדיו, לקוחותיה). Only look-alikes known to be
# harmless are let through by name: חשבונית / חשבוניות (an invoice), עוסקת / עוסקים (deals with), and
# עובדה / עובדות / עובדתי (a fact).
IDENTITY_CUE = re.compile(
    rf"{_ABBR}ת{_MARK}\s*ז|{_ABBR}מ{_MARK}\s*[זא]|{_ABBR}ח{_MARK}\s*[ןנ]|{_ABBR}ע{_MARK}\s*מ|(?<![{_HE}])תז(?![{_HE}])"
    rf"|זהות|זיהוי|מזהה|תעודת|דרכו[ןנ]|פספורט|רי?שיו[ןנ]|חשבו[ןנ](?!י(?:ת|ות)(?![{_HE}]))|ח-ן|בנק|עוסק(?!(?:ת|ים|ות)(?![{_HE}]))"
    r"|ביטוח\s+לאומי|מספר\s+אישי|מס\s+הכנסה|פקיד\s+שומה|משרד\s+הפנים|קופת\s+חולים|רפואי|هوي[ةه]|جواز|رخصة|حساب"
    r"|(?<![a-z])(?:ids?|i\.d|t\.z|ssn|ss#|social|passports?|licen[cs]es?|accounts?|acct|iban|identity|national|mrn|tax|medical"
    r"|insurance|immigration|personnel|personal|bank)(?![a-z])", re.IGNORECASE)
PERSON_RECORD = re.compile(
    rf"מטופל|חול(?:ה|ים|ת|י)|עובד(?!(?:ה|ות|תי(?:ת|ים|ות)?)(?![{_HE}]))|לקוח|תלמיד|סטודנט|נוסע|מבוטח|משתמש|נישום|مريض|موظف"
    r"|(?<![a-z])(?:patient|employee|customer|client|passenger|student|taxpayer|member|user)s?(?![a-z])", re.IGNORECASE)
_BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
_NIQQUD = re.compile("[֑-ׇ]")
_INVISIBLE = "\u200e\u200f\u061c"   # direction marks that may sit beside digits


class _WhenNothingElseListens(logging.StreamHandler):
    """Writes only while the root logger has no handler: uvicorn configures just its own loggers, and without
    this the audit lines would go nowhere. A deployment that configures logging gets them through its own
    handlers, once."""
    def emit(self, record):
        if not logging.getLogger().handlers:
            super().emit(record)


log = logging.getLogger("seter_guard")
log.setLevel(logging.INFO)
if not any(isinstance(h, _WhenNothingElseListens) for h in log.handlers):
    _h = _WhenNothingElseListens()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(_h)


def il_id_check(number: str):
    """True or False for nine ASCII digits (does the check digit match?), None for anything else."""
    if len(number) != 9 or not (number.isascii() and number.isdigit()):
        return None
    total = 0
    for i, ch in enumerate(number):
        d = int(ch) * (1 + i % 2)
        total += d if d < 10 else d - 9
    return total % 10 == 0


def _beyond(text: str, i: int, step: int):
    """Walking from text[i] away from a number: (spaces before a separator, the separator, spaces after it,
    the first other character). Spaces include direction marks; the separator is one character that is not
    a letter or digit, "" when there is none."""
    def skip(i):
        n = 0
        while 0 <= i < len(text) and (text[i].isspace() or text[i] in _INVISIBLE):
            i, n = i + step, n + 1
        return i, n
    i, gap = skip(i)
    ch = text[i] if 0 <= i < len(text) else ""
    if not ch or ch.isalnum():
        return gap, "", 0, ch
    i, gap_after = skip(i + step)
    return gap, ch, gap_after, text[i] if 0 <= i < len(text) else ""


def _stands_alone(text: str, a: int, b: int):
    """The digits at text[a:b] are a whole number, not a piece of a longer number or code.

    A piece: a letter or digit touching it (A034851272); a digit beyond spaces (034851272 21, a number written
    in groups); a digit beyond one separator, with or without spaces (12-034851272, 034851272/21,
    034851272 - 21, 034851272. 5); a letter joined by a separator with no space (034851272-A). The one
    exception is a label's colon before the number (אסמכתא:034851272)."""
    for start, step in ((a - 1, -1), (b, 1)):
        gap, sep, gap_after, ch = _beyond(text, start, step)
        if not ch.isalnum():
            continue
        if ch.isdigit():
            return False
        if not sep and not gap:                                   # a letter touching the number
            return False
        if sep and not gap and not gap_after and not (step < 0 and sep == ":"):
            return False
    return True


_BIDI = set("‎‏؜")   # left-to-right, right-to-left and Arabic letter marks: ordinary beside digits
# Letters by category that show nothing (Hangul and other fillers); everything else odd is caught by category.
_FILLERS = set("͏ᅟᅠ឴឵᠋᠌᠍᠎᠏⠀ㅤﾠ")


def _is_odd(ch: str):
    """A character with no ordinary use in Hebrew, Arabic or English prose. Allowed: letters, digits, marks,
    punctuation, spaces, tab and line breaks, the three direction marks, and symbols except pictographic ones
    (U+2000 and up, where blank glyphs such as U+2800 live) and ones that fold into a space or a mark. Everything else is odd: other format characters (zero-width
    characters, soft hyphens, direction overrides, embeddings and isolates, which can make text read
    differently from how it is stored), control characters, fillers, private-use and unassigned code points.
    Nothing is released near an odd character."""
    if ch in _FILLERS:
        return True
    if ch in _BIDI or ch in "\t\n\r\v\f":
        return False
    category = unicodedata.category(ch)
    if category == "Mc":         # a spacing mark takes up room like a letter, and none belongs to Hebrew, Arabic or Latin
        return True
    if category[0] in "LNMPZ":   # Z: spaces and line separators; other control characters are not whitespace here
        return False
    if not (category in ("Sm", "Sc", "Sk") or (category == "So" and ord(ch) < 0x2000)):
        return True
    # a symbol that folds into a space or a combining mark (´ becomes space + accent) would put a gap inside a word
    return any(c.isspace() or unicodedata.category(c)[0] == "M" for c in unicodedata.normalize("NFKC", ch))


@lru_cache(maxsize=8)
def _cue_positions(text: str):
    """(identity words, person words, odd characters, reach) in `text`, found once per text: the first two are
    sorted lists of (start, end) in the original text, the third a sorted list of positions, the last the
    length of the longest word found.

    The words are searched in cleaned copies of the text: compatibility forms folded (NFKC, so full-width
    letters and presentation forms read as plain ones), marks removed (Hebrew points, combining characters,
    variation selectors), and format characters and fillers either removed or turned into a space. Both
    readings are searched and a word found in either counts, so cleaning can only add words, never hide
    one: removing joins "דר|כון", a space keeps "של|ת.ז" apart."""
    joined, spaced, origin, odd = [], [], [], []
    for i, ch in enumerate(text):
        if _is_odd(ch):
            odd.append(i)
        for c in unicodedata.normalize("NFKC", ch):
            category = unicodedata.category(c)
            if category in ("Mn", "Me"):   # marks that sit on a letter: gone in both readings
                continue
            # hidden: shows nothing, is a spacing mark, or is a space that folding made out of something that was
            # not one. Removed in one reading and a space in the other, so it can neither split nor join a word.
            hidden = category in ("Cf", "Mc") or c in _FILLERS or (c.isspace() and not ch.isspace())
            origin.append(i)
            joined.append("" if hidden else c)
            spaced.append(" " if hidden else c)
    # `spaced` has one character per entry of origin; `joined` needs its own map because entries vanish
    joined_origin = [o for o, c in zip(origin, joined) if c]
    found = []
    for rx in (IDENTITY_CUE, PERSON_RECORD):
        spans = {(origin[m.start()], origin[m.end() - 1] + 1) for m in rx.finditer("".join(spaced))}
        spans |= {(joined_origin[m.start()], joined_origin[m.end() - 1] + 1) for m in rx.finditer("".join(joined))}
        found.append(sorted(spans))
    reach = max((e - s for spans in found for s, e in spans), default=0)
    return found[0], found[1], odd, reach


def _any_between(spans, lo, hi, reach):
    """Some (start, end) in the sorted list overlaps [lo, hi): a word that begins before lo and reaches into
    the range counts. `reach` is the longest word in the text, so no such word begins before lo - reach."""
    k = bisect.bisect_left(spans, (lo - reach, 0))
    while k < len(spans) and spans[k][0] < hi:
        if spans[k][1] > lo:
            return True
        k += 1
    return False


def _called_a_reference(text: str, a: int, b: int):
    """The words directly before text[a:b] name an enforcement or proceeding number, and the text around it
    has no word about an identity, passport, licence or account number or about whose record it is."""
    line_start = text.rfind("\n", 0, a) + 1
    tail = "".join(c for c in text[max(line_start, a - REFERENCE_TAIL):a] if c not in _BIDI)  # marks beside the digits
    if not REFERENCE_BEFORE.search(tail):
        return False
    line_end = text.find("\n", b)
    line_end = len(text) if line_end < 0 else line_end
    near_start, near_end = max(0, a - CONTEXT_BEFORE), min(len(text), b + CONTEXT_AFTER)
    identity, person, odd, reach = _cue_positions(text)
    lo, hi = min(line_start, near_start), max(line_end, near_end)
    # a character with no business in prose (see _is_odd): something may be hidden, so nothing is released
    k = bisect.bisect_left(odd, lo)
    if k < len(odd) and odd[k] < hi:
        return False
    # whose record it is: the whole line and everything near, blank lines included
    if _any_between(person, lo, hi, reach):
        return False
    # a label for a number: the whole line and everything near within the paragraph
    blank = [m.end() for m in _BLANK_LINE.finditer(text, near_start, a)]
    if blank:
        near_start = blank[-1]
    blank = _BLANK_LINE.search(text, b, near_end)
    if blank:
        near_end = blank.start()
    return not _any_between(identity, min(line_start, near_start), max(line_end, near_end), reach)


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


def _detect(text: str):
    """(spans to mask, spans that may be left in the text), each a list of (start, end, type), for one text."""
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
    return masked, kept


def scan_many(texts):
    """[(spans to mask, spans detected but left in the text)] for the texts of one request.

    One value gets one answer across the request. A number is left in a text only if every place where its
    digits appear, in any of the texts, is being left too; otherwise the copy that was masked (or that the
    model did not report) would be given away by the copy that was not."""
    found = [_detect(t) for t in texts]
    releasing = Counter(t[a:b] for t, (_m, kept) in zip(texts, found) for a, b, _ in kept)
    appears = {v: sum(t.count(v) for t in texts) for v in releasing}
    out = []
    for text, (masked, kept) in zip(texts, found):
        stay = [s for s in kept if appears[text[s[0]:s[1]]] == releasing[text[s[0]:s[1]]]]
        out.append((sorted(masked + [s for s in kept if s not in stay]), stay))
        for a, b, t in stay:  # both endpoints pass through here; position only, never the digits
            log.info("left unmasked by il_id_check: %s at characters %d-%d of a %d-character text", t, a, b, len(text))
    return out


def scan(text: str):
    return scan_many([text])[0]


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
    for t, (to_mask, _kept) in zip(body.texts, scan_many(body.texts)):  # together: one value, one answer
        m, found = _render(t, to_mask)
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
