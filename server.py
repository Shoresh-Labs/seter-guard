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
from entities import calibrate, select_spans

# v1.1.1 = the v1.1 weights with a config.json that transformers 4.x reads correctly. With v1.1 and
# transformers 4.x the local-attention layers silently used the wrong rotary-position setting and the
# model missed more (held-out recall 0.967 instead of 0.980).
DEFAULT_MODEL = "shoresh-labs/seter-pii-he-ar-fast@v1.1.1"
# The commit that tag pointed to when this line was written. A tag can be moved on the Hub; if v1.1.1 ever
# resolves to anything else the server refuses to start instead of running weights nobody chose.
DEFAULT_COMMIT = "170f528e2aeb93e25b71300bb535a583d640e9f7"
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


def _entity_preset():
    """SETER_ENTITY_PRESET: `default` (or unset) changes nothing; `strict-ids` leaves non-identity numbers in the text."""
    value = os.environ.get("SETER_ENTITY_PRESET", "").strip().lower() or "default"
    if value not in ("default", "strict-ids"):
        raise ValueError(f"SETER_ENTITY_PRESET={value!r}: use one of default, strict-ids")
    return value


ENTITY_PRESET = _entity_preset()


def _decoder():
    """SETER_DECODER: `none` (or unset) is strict masking, the behaviour before the decoder was added; `default` (alias
    `cleanup`) applies the published decoder's account-number calibration (fast model) and clean-up. Reported as
    `default` or `none`."""
    value = os.environ.get("SETER_DECODER", "").strip().lower() or "none"
    if value == "cleanup":
        value = "default"
    if value not in ("default", "none"):
        raise ValueError(f"SETER_DECODER={value!r}: use one of none, default, cleanup")
    return value


DECODER = _decoder()


class _StrictIds(tuple):
    """A (start, end, type) span that the strict-ids preset released. Equal to the plain tuple; it only carries the reason."""


class _AccountCleanup(tuple):
    """A (start, end, type) span that the decoder's account clean-up released (a date, a time, an order-style code)."""


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
# An abbreviation starts a word (so "סגורות. זאת" is not ת.ז) and has one or two marks inside: ת"ז, ת''ז, ת.ז, ת“ז, ת’ז.
# The identity abbreviation is also written with no mark, with a hyphen, a slash or a space (התז, ת-ז, ת/ז, ת ז);
# those count when the ז ends the word, so "בת זוג" does not.
_ABBR = rf"(?<![{_HE}])[ובלהמשכ]{{0,2}}"
_MARK = "[\"״”“„'׳.’‘‚`]{1,2}"
# Bounded repeats only, and matched against the last REFERENCE_TAIL characters before the number: an open
# "\s*" here backtracked for seconds on a long run of spaces.
REFERENCE_TAIL = 80
CONTEXT_BEFORE, CONTEXT_AFTER = 300, 200
REFERENCE_BEFORE = re.compile(
    rf"(?<![{_HE}]){_HE_PREFIX}(?:תיק\s{{1,2}}ה?(?:הוצאה\s{{1,2}}לפועל|הוצל[\"״”]פ)|הליך|אסמכתא)"
    rf"(?:\s{{1,2}}(?:מספר|מס['׳.]?))?\s{{0,2}}:?\s{{0,2}}$")
# Stems, so that every inflected form counts (דרכונו, חשבונך, עובדיו, לקוחותיה). Only look-alikes known to be
# harmless are let through by name: חשבונית / חשבוניות (an invoice), עוסקת / עוסקים (deals with),
# עובדה / עובדות / עובדתי (a fact), and חברה / חברת / חברות (a company).
IDENTITY_CUE = re.compile(
    rf"{_ABBR}ת(?:{_MARK}\s*ז|[-–־/ ]?ז(?![{_HE}]))|{_ABBR}מ{_MARK}\s*[זא]|{_ABBR}ח{_MARK}\s*[ןנ]|{_ABBR}ע{_MARK}\s*מ"
    rf"|זהות|זיהוי|מזהה|תעודת|דרכו[ןנ]|פספורט|רי?שיו[ןנ]|חשבו[ןנ](?!י(?:ת|ות)(?![{_HE}]))|ח-ן|בנק|עוסק(?!(?:ת|ים|ות)(?![{_HE}]))"
    r"|ביטוח\s+לאומי|מספר\s+אישי|מס\s+הכנסה|פקיד\s+שומה|משרד\s+הפנים|קופת\s+חולים|רפואי|אשפוז|هوي[ةه]|جواز|رخص[ةه]|حساب|بطاق[ةه]|ضريب|تأمين"
    # English: short words need a word boundary (camelCase counts as one, see _cue_positions); longer ones count
    # wherever they appear, so that "bankaccount" and "taxid" do
    r"|(?<![a-z])(?:ids?|i\.d|t\.z|ssn|ss#|mrn|a/c|acc)(?![a-z])"
    r"|social|passport|licen[cs]e|account|acct|iban|identity|national|tax|medical|insurance|immigration|personnel|personal|bank",
    re.IGNORECASE)
PERSON_RECORD = re.compile(
    rf"מטופל|חול(?:ה|ים|ת|י)|עובד(?!(?:ה|ות|תי(?:ת|ים|ות)?)(?![{_HE}]))|לקוח|תלמיד|סטודנט|נוסע|מבוטח|משתמש|נישום"
    rf"|מנוי|תושב|אזרח|חבר(?!(?:ה|ת|ות)(?![{_HE}]))"
    r"|مريض|مرضى|موظف|عميل|عملاء|زبون|زبائن|طالب|طلاب|مسافر|مستخدم|عضو|أعضاء|مواطن|مؤمن"
    # English: a word may run on (patientName, customers) but must start one, so "December" is not a member
    r"|(?<![a-z])(?:patient|employee|customer|client|passenger|student|taxpayer|member|user)", re.IGNORECASE)
_BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
_BIDI = frozenset("\u200e\u200f\u061c")   # left-to-right, right-to-left and Arabic letter marks: ordinary beside digits
_TATWEEL = "\u0640"                         # stretches an Arabic word without changing it


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


STANDS_ALONE_REACH = 64


def _stands_alone(text: str, a: int, b: int):
    """The digits at text[a:b] are a whole number, not a piece of a longer number or code.

    Looking outward from each end, past spaces, line breaks, direction marks and punctuation, to the first
    letter or digit. A digit there makes this a piece (034851272 21, 12-034851272, 034851272/21,
    034851272 - 21, 034851272--21, 034851272, 2024): a number written in groups cannot be told from a number
    followed by another one, so a date or a clause number right after it also keeps it masked. A letter makes it
    a piece when nothing but punctuation, or nothing at all, separates them (A034851272, 034851272-A); the one
    exception is a label's colon before the number (אסמכתא:034851272). More than STANDS_ALONE_REACH characters
    of spaces and punctuation: not judged, so not alone."""
    for i, step in ((a - 1, -1), (b, 1)):
        between, spaced = "", False
        while 0 <= i < len(text) and not text[i].isalnum():
            if text[i].isspace() or text[i] in _BIDI:
                spaced = True
            else:
                between += text[i]
            if len(between) > STANDS_ALONE_REACH or abs(i - (a if step < 0 else b)) > 4 * STANDS_ALONE_REACH:
                return False
            i += step
        if not 0 <= i < len(text):
            continue
        if text[i].isdigit():
            return False
        if not spaced and not (step < 0 and between == ":"):       # a letter touching the number or joined to it
            return False
    return True


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
            hidden = category in ("Cf", "Mc") or c in _FILLERS or c == _TATWEEL or (c.isspace() and not ch.isspace())
            # customerId, IDNumber: a capital after a small letter, or before one, starts a word in the spaced reading
            if c.isupper() and spaced and spaced[-1].isalpha() and (spaced[-1].islower() or text[i + 1:i + 2].islower()):
                origin.append(i)
                joined.append("")
                spaced.append(" ")
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


def _cue_ranges(text: str, a: int, b: int):
    """Where to look for words around text[a:b]: ((from, to) for whose record it is and for odd characters,
    (from, to) for a label of a number). Both are the whole line and CONTEXT_BEFORE characters back and
    CONTEXT_AFTER on; the second stops at a blank line, the first does not."""
    line_start = text.rfind("\n", 0, a) + 1
    line_end = text.find("\n", b)
    line_end = len(text) if line_end < 0 else line_end
    near_start, near_end = max(0, a - CONTEXT_BEFORE), min(len(text), b + CONTEXT_AFTER)
    wide = (min(line_start, near_start), max(line_end, near_end))
    blank = [m.end() for m in _BLANK_LINE.finditer(text, near_start, a)]
    if blank:
        near_start = blank[-1]
    blank = _BLANK_LINE.search(text, b, near_end)
    if blank:
        near_end = blank.start()
    return wide, (min(line_start, near_start), max(line_end, near_end))


def _called_a_reference(text: str, a: int, b: int):
    """The words directly before text[a:b] name an enforcement or proceeding number, and the text around it
    has no word about an identity, passport, licence or account number or about whose record it is."""
    line_start = text.rfind("\n", 0, a) + 1
    # direction marks beside the digits and marks on a letter are taken out first: with a vowel point after its
    # ת, "בתְהליך" (in the process) would otherwise read as ב + ת + "הליך"
    tail = "".join(c for c in text[max(line_start, a - REFERENCE_TAIL):a]
                   if c not in _BIDI and unicodedata.category(c) not in ("Mn", "Me"))
    if not REFERENCE_BEFORE.search(tail):
        return False
    identity, person, odd, reach = _cue_positions(text)
    (lo, hi), (label_lo, label_hi) = _cue_ranges(text, a, b)
    # a character with no business in prose (see _is_odd): something may be hidden, so nothing is released
    k = bisect.bisect_left(odd, lo)
    if k < len(odd) and odd[k] < hi:
        return False
    # whose record it is: the whole line and everything near, blank lines included
    if _any_between(person, lo, hi, reach):
        return False
    # a label for a number: the whole line and everything near within the paragraph
    return not _any_between(identity, label_lo, label_hi, reach)


def releasable(text: str, a: int, b: int, t: str, other_type_seen: bool):
    """A detected span to leave in the text (see the conditions above). False whenever in doubt."""
    return (IL_ID_CHECK and t == "ID" and not other_type_seen and il_id_check(text[a:b]) is False
            and _stands_alone(text, a, b) and _called_a_reference(text, a, b))


# ---- Israeli court case numbers ----------------------------------------------------------------------
# A court case number is not personal data: it names a proceeding, and judgments print it. The model's
# account_number class ("ID" here) takes the two ways such a number is written for an account number, so
# 'בג"ץ 5658/23' comes out as 'בג"ץ [ID]'. Training the model out of it was tried (data recipe v10i) and
# cost recall on identity and tax numbers, so it is done here, after the model.
#
# With SETER_IL_COURT_CASE=1 such a number is left in the text, but only when all of this holds:
#   - the model labelled it ID, and no window called any part of it anything else;
#   - it is written the way the courts write it (COURT_CASE_NUMBER): serial-month-year, as in 45127-09-23,
#     with a serial of three digits or more (12-05-85 is a date), or serial/year, as in 8954/11, in ASCII
#     digits, and it is not a piece of a longer number;
#   - the word directly before it, on its line, is the abbreviation of a kind of proceeding (COURT_CASE_TYPES:
#     ע"א, בג"ץ, תמ"ש ...), alone or with an attached ו ב ל ה מ ש כ, optionally followed by the court in
#     brackets, as in ע"א (מחוזי חיפה), and by "מספר" or "מס'";
#   - its parties follow it ("כהן נ' לוי", VERSUS_AFTER), or the brackets after the abbreviation name a kind of
#     court and its seat (COURT_IN_BRACKETS); this is asked of every abbreviation (see _court_evidence);
#   - around it there is no identity, passport, licence, bank, account or medical word (IDENTITY_CUE), no
#     branch, no bank's name and no policy, loan, card or the like (COURT_OTHER_CUE), no word that says whose
#     record it is (PERSON_RECORD) and no character that is out of place in prose (see _is_odd). "Around" is
#     its whole line and CONTEXT_BEFORE characters back and CONTEXT_AFTER on, blank lines included (the
#     identity-number check reads the labels of a number within the paragraph only; here "מספר זהות", a
#     blank line and the number was let through). Where that reach passes the start or the end of the text, it goes on into the
#     neighbouring texts of the request (see _neighbour_has_cue): the label may be the question in the
#     message before;
#   - and every other place where the same digits appear in the request is released too (see scan_many).
# A number the model reported only in part ("23456-11-1" of 23456-11-19) is judged, and released, whole.
#
# The price of reading that widely: a case number stays masked when the party's identity number, or a word
# such as "העובד" or "הלקוח", is within that reach, which is common in a judgment. The narrower reach this
# rule first had (60 characters back, 20 on, its line only) let through a bank account under a table header
# on the line above, a number followed by "... של המבוטח", and an answer to "מספר הזהות או הדרכון?".
#
# Not covered, so still masked: a number after words and no abbreviation ("תיק אזרחי 45127-09-23", "תיק מספר ..."),
# a four-digit year, a dash other than "-", and kinds of proceeding that are not in the list. Left out of the
# list on purpose:
#   - they also label a person's or a company's number or an account: ב"ל (National Insurance), ע"מ (a licensed
#     dealer), בע"מ (Ltd.), מ"א (a soldier's personal number), ע"ח (also "on account of"), ע"ש (also "in the
#     name of");
#   - ת"א (a civil case, the commonest kind): it is also how Tel Aviv is written, and in four reviews most of
#     what leaked was a number given out, or an account held, in Tel Aviv ("משכנתא שנלקחה בת"א 30412-07-19");
#   - they are also everyday words: ע"פ (a criminal appeal, and "according to": "שולם ע"פ 4411/23"), י"ס
#     (with ב it is בי"ס, a school), ת"ת (also a Talmud Torah), ק"ג (also kilograms), ס"ק (also a sub-clause).
# And two words that an attached letter makes out of a listed abbreviation are not one (NOT_A_PROCEEDING).
#
# Known limit: the rule reads words, not meaning. A number that is not a case, written directly after a listed
# abbreviation and followed by a few words and an initial נ' ("... ד"מ 77120/25 עבור נ' לוי"), is released.
#
# Off by default, like the identity-number check: this is a word list, and a guardrail that is unsure masks.
IL_COURT_CASE = _flag("SETER_IL_COURT_CASE")
COURT_CASE_TYPES = (
    # civil, family, insolvency
    'תא"מ', 'תא"ק', 'ת"ק', 'ת"צ', 'ת"ט', 'ה"פ', 'ה"ט', 'ע"א', 'רע"א', 'בר"ע', 'רת"ק', 'בש"א', 'דנ"א', 'ע"ר',
    'פר"ק', 'פש"ר', 'חדל"ת', 'חדל"פ', 'ת"ע', 'תמ"ש', 'תלה"מ', 'עמ"ש', 'רמ"ש', 'א"פ',
    # criminal, traffic
    'ת"פ', 'תפ"ח', 'תפ"ע', 'עפ"ג', 'עפ"ת', 'רע"פ', 'בש"פ', 'דנ"פ', 'מ"ת', 'מ"י', 'עמ"ת', 'עמ"י',
    'תת"ע', 'גמ"ר', 'עת"פ', 'רע"ב',
    # High Court of Justice, administrative
    'בג"ץ', 'בג"צ', 'דנג"ץ', 'עת"מ', 'עע"מ', 'בר"מ', 'עמ"נ', 'עש"א', 'ו"ע',
    # labour
    'סע"ש', 'ס"ע', 'ע"ע', 'ד"מ', 'ד"ט', 'עב"ל', 'פ"ה', 'עס"ק', 'בר"ש',
)
# Words that an attached letter makes out of a listed abbreviation: cooperation (ש + ת"פ), for the time being
# (ל + ע"ע), the Israel Land Administration (מ + מ"י), a coordination and liaison office (מ + ת"ק), the
# institute for technological training (מ + ה"ט).
_QUOTE = "(?:[\"״”“]|'')"   # the mark inside an abbreviation: ", ״, typographic quotes, or two apostrophes
NOT_A_PROCEEDING = frozenset(('שת"פ', 'לע"ע', 'ממ"י', 'מת"ק', 'מה"ט'))
# Evidence that this number is a court's, asked of every abbreviation, and tied to the number itself:
#   - the parties follow it, "<number> כהן נ' לוי": one to five words and then the נ' that stands for "versus"
#     (VERSUS_AFTER); or
#   - the brackets between the abbreviation and the number name a kind of court and where it sits, as in
#     ע"א (מחוזי חיפה) (COURT_IN_BRACKETS).
# A word of the courts elsewhere on the line was tried and let through whatever else the line spoke of
# ("אני גר מול בית המשפט, ...", "מזכירות בית המשפט, שלוחה ..."); so was a נ' anywhere on the line (an initial,
# "מין: נ'" in a form, a fixture: "הפועל חיפה נ' מכבי ..."), a kind of court alone in the brackets, and a seat
# with an attached ב ("עבודה בחיפה" is work, in Haifa). "נוער" is left out: "נוער מרכז" is a youth centre.
# What this cannot tell: an initial that follows the number, "... 77120/25 עבור נ' לוי".
VERSUS_AFTER = re.compile(r"^[,;:]?[ \t]{1,2}(?:\S{1,25}[ \t]{1,2}){1,5}נ['׳][ \t]{1,2}\S")
VERSUS_REACH = 160
_COURT_SEATS = (
    "ירושלים", "י-ם", "תל אביב", "תל-אביב", f"ת{_QUOTE}א", "חיפה", "באר שבע", "באר-שבע", f"ב{_QUOTE}ש", "נצרת", "לוד",
    "פתח תקוו?ה", f"פ{_QUOTE}ת", "ראשון לציון", f"ראשל{_QUOTE}צ", "רמלה", "רחובות", "נתניה", "חדרה", "כפר סבא", f"כ{_QUOTE}ס",
    "הרצליה", "אשדוד", "אשקלון", "קרי?ית גת", "אילת", "עכו", "קריות", "טבריה", "צפת", "עפולה", "קרי?ית שמונה", "נהריה",
    "בית שאן", "בית שמש", "דימונה", "מסעדה", "קצרין", "בת ים", "רמת גן")
COURT_IN_BRACKETS = re.compile(
    rf"^(?:מחוזי\s{{1,2}}מרכז(?:[ -]לוד)?"
    rf"|(?:שלום|מחוזי|משפחה|(?:אזורי\s{{1,2}})?ל?עבודה|תעבורה)\s{{1,2}}(?:{'|'.join(_COURT_SEATS)}))$")


def _court_evidence(text: str, e: int, court):
    """The number that ends at text[e] is followed by its parties, or `court`, the words in the brackets before
    it, are a kind of court and its seat."""
    line_end = text.find("\n", e)
    after = "".join(c for c in text[e:len(text) if line_end < 0 else line_end][:VERSUS_REACH]
                    if c not in _BIDI and unicodedata.category(c) not in ("Mn", "Me"))
    return bool(VERSUS_AFTER.match(after) or (court and COURT_IN_BRACKETS.match(_plain(court).strip())))


COURT_CASE_NUMBER = re.compile(r"[0-9]{3,6}-(?:0[1-9]|1[0-2])-[0-9]{2}|[0-9]{1,5}/[0-9]{2}")
COURT_CASE_LENGTH = 12      # the longest text COURT_CASE_NUMBER matches: 123456-12-23
# A branch, and the banks by name: "לאומי ת"א 12345/67" and "Leumi" in the next column say bank without the word.
# Several are ordinary words as well (national, workers, eastern, a discount); near a number in this shape
# that is reason enough to mask. And the things a person holds that carry a number and are in neither of the
# other two lists: a policy, a loan, a mortgage, a card, a visa, a payslip, a pension, a soldier's number.
# "שכר" alone is left out: "שכר טרחת עו"ד" is in most judgments.
COURT_OTHER_CUE = re.compile(
    rf"סניף|סניפ|branch|فرع"
    rf"|פוליס|ביטוח|הלווא|משכנת|כרטיס|אשראי|ויזה|תלוש|משכורת|פנסי|חייל"
    r"|polic(?:y|ies)|loan|mortgage|(?<![a-z])cards?(?![a-z])|credit|(?<![a-z])visa(?![a-z])|payslip|salary|pension|soldier"
    r"|بوليص|قرض|ائتمان|راتب|تقاعد|جندي"
    rf"|לאומי|פועלים|דיסקונט|מזרחי|טפחות|מרכנתיל|(?<![{_HE}])ה?יהב(?![{_HE}])|אוצר\s+החייל"
    r"|leumi|poalim|discount|mizrahi|tefahot|mercantile|(?<![a-z])fibi(?![a-z])|yahav"
    r"|لئومي|ليئومي|هبوعليم|ديسكونت|مزراحي|مركنتيل", re.IGNORECASE)
# Matched against the last REFERENCE_TAIL characters before the number, with bounded repeats only (see REFERENCE_BEFORE).
COURT_CASE_BEFORE = re.compile(
    rf"(?<!\w)(?P<word>ו?[בלהמשכ]{{0,2}}(?P<type>{'|'.join(t.replace(chr(34), _QUOTE) for t in sorted(COURT_CASE_TYPES, key=len, reverse=True))}))"
    rf"(?:\s{{1,2}}\((?P<court>[{_HE}\"״”“'׳. -]{{1,30}})\))?"
    rf"(?:\s{{1,2}}(?:מספר|מס['׳.]?))?\s{{1,2}}$")
_ANY_QUOTE = re.compile(_QUOTE)


def _plain(text: str):
    """`text` with compatibility forms folded and the marks on a letter and the tatweel that stretches an Arabic
    word removed, for a word search."""
    return "".join(c for c in unicodedata.normalize("NFKC", text)
                   if unicodedata.category(c) not in ("Mn", "Me") and c != _TATWEEL)


def court_case_number(text: str, a: int, b: int):
    """(start, end) of the court case number that text[a:b] is, or is a piece of; None when it is not one,
    or when anything around it leaves room for doubt (see the conditions above). The neighbouring texts of
    the request are looked at separately (_neighbour_has_cue)."""
    s, e = a, b
    while s > 0 and e - s <= COURT_CASE_LENGTH and text[s - 1] in "0123456789-/":
        s -= 1
    while e < len(text) and e - s <= COURT_CASE_LENGTH and text[e] in "0123456789-/":
        e += 1
    if not COURT_CASE_NUMBER.fullmatch(text, s, e) or not _stands_alone(text, s, e):
        return None
    line_start = text.rfind("\n", 0, s) + 1
    tail = "".join(c for c in text[max(line_start, s - REFERENCE_TAIL):s]
                   if c not in _BIDI and unicodedata.category(c) not in ("Mn", "Me"))
    m = COURT_CASE_BEFORE.search(tail)
    if not m or (m.group("court") and IDENTITY_CUE.search(m.group("court"))):
        return None
    if _ANY_QUOTE.sub('"', m.group("word")) in NOT_A_PROCEEDING:
        return None
    if not _court_evidence(text, e, m.group("court")):
        return None
    identity, person, odd, reach = _cue_positions(text)
    # every list is read over the whole reach, blank lines included: a form's label is often a paragraph of
    # its own ("מספר זהות", a blank line, the number)
    (lo, hi), _within_paragraph = _cue_ranges(text, s, e)
    k = bisect.bisect_left(odd, lo)
    if k < len(odd) and odd[k] < hi:
        return None
    if _any_between(person, lo, hi, reach) or _any_between(identity, lo, hi, reach):
        return None
    # no odd character is near (checked above), so folding and removing marks is all the cleaning this needs
    if COURT_OTHER_CUE.search(_plain(text[lo:hi])):
        return None
    return s, e


NEIGHBOUR_MARGIN = 24   # further than the reach, so that a word cut by it is still read whole


def _has_cue(text: str):
    """Some word in `text` that keeps a court case number masked, or a character out of place in prose."""
    identity, person, odd, _reach = _cue_positions(text)
    return bool(identity or person or odd or COURT_OTHER_CUE.search(_plain(text)))


def _neighbour_has_cue(texts, i: int, a: int, b: int):
    """The reach around texts[i][a:b] (CONTEXT_BEFORE characters back, CONTEXT_AFTER on) passes the start or
    the end of its text, and the part of the neighbouring texts it then covers has a word that keeps a court
    case number masked. The texts of a request are read as one text here: a form's label may be one message
    and the number the next. Paragraphs are not told apart across texts."""
    need, j = CONTEXT_BEFORE - a, i - 1
    while need > 0 and j >= 0:
        if _has_cue(texts[j][-(need + NEIGHBOUR_MARGIN):]):
            return True
        need -= len(texts[j])
        j -= 1
    need, j = CONTEXT_AFTER - (len(texts[i]) - b), i + 1
    while need > 0 and j < len(texts):
        if _has_cue(texts[j][:need + NEIGHBOUR_MARGIN]):
            return True
        need -= len(texts[j])
        j += 1
    return False


def release_reason(text: str, a: int, b: int, span=None):
    """Which check left text[a:b] in the text: the first two never apply to the same span (nine bare digits, or not)."""
    if isinstance(span, _StrictIds):
        return "strict_ids"
    if isinstance(span, _AccountCleanup):
        return "account_cleanup"
    return "il_id_check" if il_id_check(text[a:b]) is not None else "il_court_case"


ner = pipeline("token-classification", model=MODEL, revision=COMMIT, aggregation_strategy="simple")
if DECODER == "default":
    calibrate(ner.model)  # fast model: account-number bias -1.0, once; the large model is left as it is
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


def _trim(text: str, s: int, t: int):
    """(s, t) narrowed so that text[s:t] neither starts nor ends with whitespace or punctuation. Some pipeline
    versions start a span one character early (the space before the word); a span can also swallow a trailing
    space or a wrapping quote or bracket. Letters, digits and marks are never trimmed, and neither are symbols
    such as the + of a phone number; punctuation inside the span (the hyphen of a phone number, the @ of an
    email) is not touched. An empty result (s >= t) means the span was only whitespace and punctuation."""
    def edge(c):
        return c.isspace() or unicodedata.category(c).startswith("P")
    while s < t and edge(text[s]):
        s += 1
    while t > s and edge(text[t - 1]):
        t -= 1
    return s, t


def entities(text: str):
    """Model output for the whole text, with character offsets into `text` and an "owner" flag (see windows)."""
    wins = windows(text)
    if len(wins) == 1:
        out = []
        for e in ner(text):
            s, t = _trim(text, e["start"], e["end"])
            if s < t:
                out.append({**e, "start": s, "end": t, "owner": True})
        return out
    out = []
    results = ner([text[a:b] for a, b, *_ in wins], batch_size=BATCH)
    for (a, _b, trust_from, trust_to, own_from, own_to), found in zip(wins, results):
        for e in found:
            s, t = e["start"] + a, e["end"] + a
            s, t = _trim(text, s, t)  # the pipeline may include the space before a word (see _trim)
            # Dropped: what lies wholly in the untrusted strip after a cut (seen without its left context; the
            # previous window has it), and what starts in the strip before the next cut. Kept: anything that
            # reaches into the trusted zone, so a long entity that begins just after a cut is never lost.
            if s < t and t > trust_from and s < trust_to:
                out.append({**e, "start": s, "end": t, "owner": own_from <= s < own_to})
            elif s < t:  # not used for masking, but kept as a second opinion (see scan)
                out.append({**e, "start": s, "end": t, "owner": False, "untrusted": True})
    return out


def _cleanup_drops(text: str, a: int, b: int):
    """The decoder's account clean-up leaves text[a:b] (an [ID] span) in the text: a date or time, or an order-style code."""
    return not select_spans(text, [(a, b, "account_number")], "default")


def _strict_ids_drops(text: str, a: int, b: int):
    """The preset leaves text[a:b] (an [ID] span) in the text."""
    return not select_spans(text, [(a, b, "account_number")], ENTITY_PRESET)


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
        a, b = _trim(text, a, b)
        if a >= b:
            continue
        # The model sometimes reports only some digits of a number ("[ID]3485[ID]2"). A number is masked whole.
        while a > 0 and text[a].isdigit() and text[a - 1].isdigit():
            a -= 1
        while b < len(text) and text[b - 1].isdigit() and text[b].isdigit():
            b += 1
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
        if releasable(text, a, b, t, mixed):
            kept.append((a, b, t))
            continue
        # A court case number is judged whole, also when the model reported a piece of it or two pieces apart:
        # nothing that any window called something else may lie anywhere in it.
        case = court_case_number(text, a, b) if IL_COURT_CASE and t == "ID" and not mixed else None
        if case and not any(x < case[1] and case[0] < y and (u != t or m) for x, y, u, _o, m in out) \
                and not any(x < case[1] and case[0] < y and u != t for x, y, u in doubts):
            if (*case, t) not in kept:
                kept.append((*case, t))
        elif DECODER == "default" and t == "ID" and not mixed and _cleanup_drops(text, a, b):
            kept.append(_AccountCleanup((a, b, t)))
        elif ENTITY_PRESET != "default" and t == "ID" and not mixed and _strict_ids_drops(text, a, b):
            # not an identity or financial number: left in the text, but on the kept list, so that the copy rule
            # in scan_many and the log still see it
            kept.append(_StrictIds((a, b, t)))
        else:
            masked.append((a, b, t))
    return masked, kept


def _digits(text: str):
    """Only the digits of `text`, as ASCII: what is left of a number however it is written (03485127-2,
    0348 51272, with a direction mark inside, in full-width or Arabic-Indic digits)."""
    return "".join(str(unicodedata.digit(c)) for c in text if c.isdigit() and unicodedata.digit(c, None) is not None)


def scan_many(texts):
    """[(spans to mask, spans detected but left in the text)] for the texts of one request.

    One value gets one answer across the request. A number is left in a text only if every place where its
    digits appear, in any of the texts, is being left too; otherwise the copy that was masked (or that the
    model did not report) would be given away by the copy that was not. Copies are compared by their digits
    alone and without leading zeros, so a hyphen, a space or a dropped zero does not make them different
    numbers. Digits that only happen to line up across two other numbers count as a copy too: that masks."""
    found = [_detect(t) for t in texts]
    for i, (text, (masked, kept)) in enumerate(zip(texts, found)):   # a court case number next to a label in a neighbouring text
        back = [s for s in kept if release_reason(text, s[0], s[1], s) == "il_court_case" and _neighbour_has_cue(texts, i, s[0], s[1])]
        if back:
            found[i] = (masked + back, [s for s in kept if s not in back])
    key = lambda text, s: _digits(text[s[0]:s[1]]).lstrip("0") or "0"
    releasing = Counter(key(t, s) for t, (_m, kept) in zip(texts, found) for s in kept)
    digits = [_digits(t) for t in texts] if releasing else []
    appears = {v: sum(d.count(v) for d in digits) for v in releasing}
    out = []
    for text, (masked, kept) in zip(texts, found):
        stay = [s for s in kept if appears[key(text, s)] == releasing[key(text, s)]]
        out.append((sorted(masked + [s for s in kept if s not in stay]), stay))
        for span in stay:  # both endpoints pass through here; position only, never the digits
            a, b, t = span
            log.info("left unmasked by %s: %s at characters %d-%d of a %d-character text",
                     release_reason(text, a, b, span), t, a, b, len(text))
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
            "kept": [{"type": s[2], "start": s[0], "end": s[1], "reason": release_reason(body.text, s[0], s[1], s)} for s in kept]}


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
            "revision": REVISION, "commit": COMMIT, "il_id_check": IL_ID_CHECK, "il_court_case": IL_COURT_CASE,
            "entity_preset": ENTITY_PRESET, "decoder": DECODER}
