import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
import server

c = TestClient(server.app)


def test_hebrew_name_and_phone_masked():
    r = c.post("/v1/redact", json={"text": "שלום, קוראים לי יוסי כהן, הטלפון שלי 054-7812345"}).json()
    assert "יוסי" not in r["text"] and "054-7812345" not in r["text"]
    assert "[PERSON]" in r["text"] and "[PHONE]" in r["text"]


def test_arabic_kunya_masked():
    r = c.post("/v1/redact", json={"text": "يا أبو خالد، ابنك سامر اتصل من 0559023417"}).json()
    assert "سامر" not in r["text"] and "0559023417" not in r["text"]


def test_dates_and_order_numbers_kept():
    t = "Order 22091 shipped on 2026-03-15"
    assert c.post("/v1/redact", json={"text": t}).json()["text"] == t


def test_litellm_guardrail_contract():
    r = c.post("/beta/litellm_basic_guardrail_api",
               json={"texts": ["תעדכנו את רון אברהם", "no pii"], "input_type": "request"}).json()
    assert r["action"] == "GUARDRAIL_INTERVENED"
    assert len(r["texts"]) == 2 and r["texts"][1] == "no pii"
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": ["nothing here"]}).json() == {"action": "NONE"}


# ---- model choice ----

def test_default_is_pinned_and_health_reports_what_was_loaded():
    model, revision, is_local = server.parse_model(server.DEFAULT_MODEL)
    assert revision and not is_local and len(server.DEFAULT_COMMIT) == 40
    h = c.get("/health").json()
    assert h["ok"] and h["model"] == server.MODEL and h["revision"] == server.REVISION and h["pinned"] == bool(server.REVISION)
    assert h["source"] == ("local" if server.IS_LOCAL else "hub")
    if server.IS_LOCAL:
        assert h["commit"] is None
    else:
        assert len(h["commit"]) == 40                           # the commit actually loaded, not the name asked for
    if "SETER_MODEL" not in os.environ:                         # CI runs the default: the pinned fast model
        assert (h["model"], h["revision"], h["commit"]) == (model, revision, server.DEFAULT_COMMIT)


def test_fast_model_local_attention_setting():
    """transformers 4.x reads the rotary-position setting of the local-attention layers from `local_rope_theta`
    and silently falls back to 10000 when the key is missing (fast v1.1); the model was trained with 160000."""
    cfg = server.ner.model.config
    if cfg.model_type != "modernbert":
        return
    theta = getattr(cfg, "local_rope_theta", None)
    if theta is None:
        theta = cfg.rope_parameters["sliding_attention"]["rope_theta"]
    assert theta == 160000


def test_parse_model(tmp_path, monkeypatch):
    assert server.parse_model("org/repo@v1.2") == ("org/repo", "v1.2", False)
    assert server.parse_model("org/repo") == ("org/repo", None, False)
    assert server.parse_model(" org/repo@abc123 ") == ("org/repo", "abc123", False)
    local = tmp_path / "GoogleDrive-me@example.com" / "model"    # a folder whose path contains "@"
    local.mkdir(parents=True)
    assert server.parse_model(str(local)) == (str(local), None, True)
    for bad in ("org/repo@", "@v1.1", "", "   ",                 # an empty version must not silently mean "newest"
                str(tmp_path / "missing@example.com" / "model"),  # a folder that is not there is not a Hub repo
                "~/models/seter", "a/b/c"):
        with pytest.raises(ValueError):
            server.parse_model(bad)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "org" / "repo").mkdir(parents=True)              # a folder named like the repo would override the pin
    with pytest.raises(ValueError, match="in place of the Hub repo"):
        server.parse_model("org/repo@v1.2")
    assert server.parse_model("./org/repo") == ("./org/repo", None, True)


def test_a_moved_tag_or_an_unavailable_model_stops_the_server_with_a_useful_message(monkeypatch):
    assert server.resolve_commit("anything", None, True) is None
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("SETER_BAKED_MODEL", "org/baked@v9")
    import huggingface_hub
    def boom(*a, **k):
        raise OSError("not in the cache")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    with pytest.raises(RuntimeError) as e:
        server.resolve_commit("org/other", None, False)
    msg = str(e.value)
    assert "offline" in msg and "org/baked@v9" in msg and "--build-arg" in msg and "main (not pinned)" in msg


# ---- long texts ----

FILLER = "The quarterly report covers logistics, storage and delivery schedules for the northern region. "


def ntok(t):
    return len(server.wtok(t, add_special_tokens=False)["input_ids"])


def test_personal_data_at_the_end_of_a_long_text_is_masked():
    # longer than any model window (about 8,000 tokens for the fast model, 512 for the large one)
    t = "Contact Dana Levi at 054-7812345. " + FILLER * 700 + "Send the contract to Yossi Cohen, phone 052-3318874."
    assert ntok(t) > 9000
    r = c.post("/v1/redact", json={"text": t}).json()
    assert "054-7812345" not in r["text"] and "Dana Levi" not in r["text"]      # start
    assert "052-3318874" not in r["text"] and "Yossi Cohen" not in r["text"]    # end, past the first window
    assert r["text"].count("[PHONE]") == 2
    assert r["text"].count(FILLER.strip()) == 700                               # nothing else was touched
    for e in r["entities"]:                                                     # offsets point into the original text
        assert 0 <= e["start"] < e["end"] <= len(t)
    assert t[r["entities"][-1]["start"]:r["entities"][-1]["end"]] == "052-3318874"
    assert all(x["end"] <= y["start"] for x, y in zip(r["entities"], r["entities"][1:]))


def test_windows_cover_the_text_start_at_words_and_fit_the_model():
    t = FILLER * 80
    w = server.windows(t)
    assert len(w) > 2 and w[0][0] == 0 and w[-1][1] == len(t)
    for (a, b, *_), (a2, b2, *_) in zip(w, w[1:]):
        assert a < a2 < b < b2                                                  # neighbours overlap, no gap
    for a, b, trust_from, trust_to, own_from, own_to in w:
        assert ntok(t[a:b]) <= server.WINDOW                                    # the model reads all of it
        assert a == 0 or t[a].isspace() or t[a - 1].isspace()                   # never starts inside a word
        assert a <= trust_from <= own_from < own_to <= trust_to <= b
    assert all(y[2] <= x[3] for x, y in zip(w, w[1:]))                          # trusted zones meet
    assert w[0][4] == 0 and w[-1][5] == len(t) and all(x[5] == y[4] for x, y in zip(w, w[1:]))  # one owner per character
    assert server.windows("short text") == [(0, 10, 0, 10, 0, 10)]


def placements(probe):
    """Texts that put `probe` where cut handling can go wrong, located with server.windows() itself, so the
    test cannot drift away from the cuts when a tokenizer or a default changes.
      starts_next_window  the second window begins inside the probe
      just_after_cut      the probe lies wholly in the second window's untrusted first tokens
      across_trust_edge   the probe straddles the point where the second window starts to be trusted
      ends_first_window   the first window ends inside the probe"""
    tail, found = FILLER * 30, {}
    for pad in range(0, 200):
        head = FILLER * 24 + "Filler " * pad
        t = head + probe + tail
        w = server.windows(t)
        if len(w) < 2:
            continue
        ps, pe = len(head), len(head) + len(probe.rstrip())
        start1, trust1, end0 = w[1][0], w[1][2], w[0][1]
        kind = ("starts_next_window" if ps < start1 < pe else
                "just_after_cut" if start1 <= ps and pe <= trust1 else
                "across_trust_edge" if ps < trust1 < pe else
                "ends_first_window" if ps < end0 < pe else None)
        if kind and kind not in found:
            found[kind] = (head, t)
    return found, tail


@pytest.mark.parametrize("probe,hidden,tags,kept", [
    ("Write to dana.levi@example.co.il about it. ", ["dana.levi@example.co.il"], ["[EMAIL]"], ["Write to", "about it."]),
    ("Order 22091 shipped on 2026-03-15 to the warehouse. ", [], [], ["Order 22091 shipped on 2026-03-15 to the warehouse."]),
    ("Please call Dana Levi on 054-7812345 before noon. ", ["Dana Levi", "054-7812345"], ["[PERSON]", "[PHONE]"], ["before noon."]),
    ("מספר תעודת זהות 318845127 נמסר לפקיד. ", ["318845127"], ["[ID]"], ["נמסר לפקיד"]),
    ("يا أبو خالد، ابنك سامر اتصل من 0559023417. ", ["خالد", "سامر", "0559023417"], ["[PERSON]", "[PERSON]", "[PHONE]"], ["اتصل من"]),
    ("הנתבע, מר עזרא בן שושן, מתגורר ברחוב הלל יפה 7, חיפה. ", ["עזרא בן שושן", "הלל יפה 7"], ["[PERSON]", "[ADDRESS]"], ["הנתבע"]),
])
def test_entities_at_a_cut_are_masked_with_the_right_type(probe, hidden, tags, kept):
    # An email cut in half used to come out as a name, an id whose label was cut away as a phone number, and a
    # date as "2026-[PHONE]3-15". The comparison with one pass over the whole text is deliberately not exact:
    # on a 1,300-token context the model itself sometimes masks one more word than on 510 tokens.
    found, tail = placements(probe)
    assert {"starts_next_window", "across_trust_edge", "ends_first_window"} <= set(found), sorted(found)
    for kind, (head, t) in found.items():
        got = server.mask(t)[0]
        assert got.startswith(head) and got.endswith(tail), kind         # nothing around the probe was touched
        mid = got[len(head):len(got) - len(tail)]
        for h in hidden:
            assert h not in mid, (kind, mid)
        for tag in set(tags):
            assert mid.count(tag) == tags.count(tag), (kind, mid)          # each entity masked once, with the right type
        for k in kept:
            assert k in mid, (kind, mid)
        if "[ID]" in tags:
            assert "[PHONE]" not in mid, (kind, mid)


def test_the_window_that_owns_the_text_decides_the_type(monkeypatch):
    # The next window saw the number without its label and called it a phone, starting one character earlier.
    text = "ID (318845127) on file"
    def fake(_text):
        return [{"entity_group": "private_phone", "score": 0.9, "start": 3, "end": 13, "owner": False},
                {"entity_group": "account_number", "score": 0.9, "start": 4, "end": 13, "owner": True}]
    monkeypatch.setattr(server, "entities", fake)
    assert server.spans(text) == [(4, 13, "ID")]   # the bracket the union started with is trimmed
    assert server.mask(text)[0] == "ID ([ID]) on file"


def test_tokens_not_characters_decide_whether_a_text_needs_windows(monkeypatch):
    # Some characters become several tokens, so a text can be short in characters and still too long for one call.
    monkeypatch.setattr(server, "WINDOW", 128)
    monkeypatch.setattr(server, "OVERLAP", 64)
    for unit in ("\ufdfa ", "\u33a1 ", "\u2167 "):
        t = unit * 60
        if len(t) <= 128 < ntok(t):
            w = server.windows(t)
            assert len(w) > 1 and all(ntok(t[a:b]) <= 128 for a, b, *_ in w)
            return
    t = "x " * 127 + "y"                                                 # this tokenizer does not expand those characters
    assert (len(server.windows(t)) > 1) == (ntok(t) > 128)


def test_concurrent_requests():
    # FastAPI runs these endpoints in a thread pool. A tokenizer shared between the window code and the
    # pipeline raised "Already borrowed" under exactly this load.
    long_t = "Contact Dana Levi at 054-7812345. " + FILLER * 60 + "Send the contract to Yossi Cohen, phone 052-3318874."
    short_t = "Please call Dana Levi on 054-7812345 before noon."
    want = {t: server.mask(t) for t in (long_t, short_t)}
    # the truncation setting each tokenizer was left with at import must not change again under load
    state = (server.wtok._tokenizer.truncation, server.wtok._tokenizer.padding)

    def one(i):
        t = long_t if i % 2 else short_t
        return server.mask(t) == want[t]

    with ThreadPoolExecutor(4) as ex:
        assert all(ex.map(one, range(24)))
    assert state == (server.wtok._tokenizer.truncation, server.wtok._tokenizer.padding) == (None, None)


def test_a_text_too_long_to_scan_is_refused_not_passed_through(monkeypatch):
    monkeypatch.setattr(server, "MAX_TOKENS", 600)
    t = FILLER * 80 + "Call Dana Levi on 054-7812345."
    r = c.post("/v1/redact", json={"text": t})
    assert r.status_code == 413 and "054-7812345" not in r.text
    g = c.post("/beta/litellm_basic_guardrail_api", json={"texts": ["fine", t]}).json()
    assert g["action"] == "BLOCKED" and "too long" in g["blocked_reason"].lower()
    # the limit is for the whole request, and it is checked before any text is scanned
    half = FILLER * 20
    assert ntok(half) <= 600 < 2 * ntok(half)
    def must_not_run(_t):
        raise AssertionError("a text was scanned although the request is over the limit")
    monkeypatch.setattr(server, "mask", must_not_run)
    g = c.post("/beta/litellm_basic_guardrail_api", json={"texts": [half, half]}).json()
    assert g["action"] == "BLOCKED" and "too long" in g["blocked_reason"].lower()


def test_settings_that_would_skip_text_are_rejected(monkeypatch):
    limit = server._model_limit()
    for env in ({"SETER_WINDOW_TOKENS": str(limit + 1)}, {"SETER_WINDOW_TOKENS": "100"},
                {"SETER_OVERLAP_TOKENS": "0"}, {"SETER_OVERLAP_TOKENS": "8"},   # 8 let a long address fall between two windows
                {"SETER_WINDOW_TOKENS": "200", "SETER_OVERLAP_TOKENS": "150"}):
        for k in ("SETER_WINDOW_TOKENS", "SETER_OVERLAP_TOKENS"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        with pytest.raises(ValueError):
            server._settings()
    for k in ("SETER_WINDOW_TOKENS", "SETER_OVERLAP_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    window, overlap, _, _ = server._settings()
    assert window <= limit and server.MIN_OVERLAP <= overlap <= window // 2


# ---- Israeli identity numbers ----

VALID_ID, FILE_NO = "034851279", "034851272"   # the same eight digits with the right and a wrong check digit
SSN = "123456789"
FILE_TEXT = f"תיק הוצאה לפועל מספר {FILE_NO} נפתח בלשכת חיפה."
ID_TEXT = f"מספר תעודת הזהות שלו הוא {VALID_ID}."


def redact(text):
    return c.post("/v1/redact", json={"text": text}).json()


def _rule(text, number=None):
    """What the words and characters around the number allow, the model aside."""
    number = number or (SSN if SSN in text else FILE_NO)
    a = text.index(number)
    return server._stands_alone(text, a, a + 9) and server._called_a_reference(text, a, a + 9)


def _shows(out, number):
    """Some four digits in a row of `number` are in `out`: a mask over one or two of its digits does not hide it."""
    return any(number[i:i + 4] in out for i in range(len(number) - 3))


def test_il_id_check_digit():
    assert server.il_id_check(VALID_ID) is True and server.il_id_check(FILE_NO) is False
    assert server.il_id_check("000000018") is True and server.il_id_check("123456782") is True
    wrong = [VALID_ID[:8] + str(d) for d in range(10) if str(d) != VALID_ID[8]]
    assert all(server.il_id_check(n) is False for n in wrong)               # exactly one check digit fits
    for other in ("03485127", "0348512790", "03485127-9", "٠٣٤٨٥١٢٧٩", "03485127a", ""):
        assert server.il_id_check(other) is None                           # not nine plain digits: no opinion


def test_stands_alone():
    for text in (f"מספר {FILE_NO} נפתח", FILE_NO, f"מספר {FILE_NO}.", f"({FILE_NO})", f"מספר: {FILE_NO}, ועוד", f"מספר {FILE_NO}. המשך",
                 f"אסמכתא:{FILE_NO}", f"מספר {FILE_NO}‏ נפתח", f"מספר {FILE_NO} - נפתח"):
        a = text.index(FILE_NO)
        assert server._stands_alone(text, a, a + 9), text
    for joined in (f"12-{FILE_NO}", f"{FILE_NO}-55", f"A{FILE_NO}", f"{FILE_NO}/21", f"1{FILE_NO}", f"{FILE_NO}.5", f"{FILE_NO}\\21",
                   f"{FILE_NO}–21", f"{FILE_NO}־21", f"{FILE_NO}:21", f"+{FILE_NO}1", f"21,{FILE_NO}", f"{FILE_NO}_21", f"ב{FILE_NO}",
                   f"{FILE_NO} 21", f"{FILE_NO} 0123", f"21 {FILE_NO}", f"{FILE_NO}-A", f"A-{FILE_NO}",
                   f"{FILE_NO} - 21", f"{FILE_NO}  21", f"{FILE_NO} /21", f"{FILE_NO}‏-21", f"{FILE_NO}. 5", f"21 - {FILE_NO}",
                   # any amount of punctuation between: a digit beyond it, or a letter joined by it
                   f"{FILE_NO}--21", f"{FILE_NO} - - 21", f"{FILE_NO}--A", f"{FILE_NO}, 2024", f"{FILE_NO}\n\n1. הצדדים", f"{FILE_NO} (21)",
                   f"{FILE_NO}" + " ." * 200 + " 21"):
        text = f"מספר {joined} נפתח"
        a = text.index(FILE_NO)
        assert not server._stands_alone(text, a, a + 9), text


@pytest.mark.parametrize("text", [
    FILE_TEXT, f"בתיק הוצאה לפועל מס' {FILE_NO} נקבע", f"תיק הוצל\"פ {FILE_NO}", f"אסמכתא: {FILE_NO}", f"אסמכתא:{FILE_NO}",
    f"בהליך {FILE_NO} נדחתה הבקשה", f"מספר הליך {FILE_NO}, ועוד",
    # look-alikes of the identity and person words: a fact, an invoice, "deals with", and a sentence end that
    # happens to put ת. before ז or ח. before נ
    f"לאור העובדות שבתיק ההוצאה לפועל {FILE_NO}", f"עובדה היא שבהליך {FILE_NO}", f"הבסיס העובדתי, הליך {FILE_NO}",
    f"חשבונית מס, אסמכתא {FILE_NO}", f"שתי חשבוניות, אסמכתא {FILE_NO}", f"הבקשה עוסקת בהליך {FILE_NO}",
    f"ההחלטה ניתנה בדלתיים סגורות. זאת ועוד, בהליך {FILE_NO} נקבע דיון", f"המכתב נשלח. נפתח הליך {FILE_NO}",
    # direction marks are ordinary beside digits in Hebrew text
    f"מספר הליך \u200f{FILE_NO}\u200f נדחה", f"מספר הליך \u200e{FILE_NO} נדחה ב-5% (₪ 100, 20°)", f"מספר\u00a0הליך {FILE_NO}\tנדחה",
    # more look-alikes: "his partner" is not ת ז, a company is not a member, December is not one either
    f"בת זוגו של המבקש, הליך {FILE_NO}", f"החברה הגישה בקשה בהליך {FILE_NO}", f"חברת הביטוח וחברות הבת, הליך {FILE_NO}",
    f"5 December, הליך {FILE_NO}", f"התזה שלו, הליך {FILE_NO}",
    # an identity word in another paragraph is out of reach
    f"ת.ז. של החייב {VALID_ID}\n\nמספר הליך {FILE_NO}",
])
def test_a_number_named_as_an_enforcement_or_proceeding_number_can_be_released(text):
    assert _rule(text)


@pytest.mark.parametrize("text", [
    # no description, or the phrase is not directly before the number
    f"מספר {FILE_NO}", f"call {FILE_NO} now", f"הליך פתוח\nמספר {FILE_NO}", f"ההליך נסגר. מספר הזיהוי של דני לוי {FILE_NO}",
    f"הליך של דני לוי {FILE_NO}", f"אסמכתא לתשלום של {FILE_NO}",
    # a bare תיק is also a medical, tax or registry file; English reference words are ordinary words
    f"בתיק {FILE_NO}", f"תיק מספר {FILE_NO}", f"שבתיק מס' {FILE_NO} נקבע", f"Case no. {FILE_NO}", f"file #{FILE_NO}", f"ref: {FILE_NO}",
    f"docket number: {FILE_NO}", f"In case you need it, my social is {SSN}", f"Please file the claim for John Smith {SSN}",
    f"Per our file, the customer's social {SSN} is verified", f"Case worker John Smith, SS# {SSN}",
    f"The number we have on file: {SSN}", f"Dana Levi's number on file: {SSN}", f"Here is my number just in case: {SSN}", f"My number for reference: {SSN}",
    f"Tax file number {SSN}", f"medical file no. {SSN}", f"Medical record: file no. {SSN}", f"personnel file no. {SSN} for Dana Levi",
    f"Student file no. {SSN}", f"Immigration file no. {SSN}", f"Insurance file no. {SSN}", f"Taxpayer reference number: {SSN}",
    f"His profile shows the personal number he gave us then: {FILE_NO}", f"The passport-case file of this person lists the following: {FILE_NO}",
    f"ref: employee {FILE_NO} (Dana Levi)", f"ticket opened by customer {FILE_NO}", f"ticket for passenger Dana Levi, frequent flyer {FILE_NO}",
    f"patient file no. {FILE_NO}", f"File for patient John Smith, MRN {SSN}", f"ملف المريض أحمد خالد رقم {FILE_NO}", f"ملف المريض، رقم الهويه {FILE_NO}",
    # Hebrew words that only contain a reference word
    f"אזרח ותיק, מספרו {FILE_NO}", f"אזרח ותיק מספר {FILE_NO}", f"לצורך תיקון הפרטים שלי: {FILE_NO}", f"בתהליך האימות נמסר המספר {FILE_NO}",
    f"בתהליך {FILE_NO}", f"הפניה לרופא עבור מטופל {FILE_NO}", f"כרטיס הפתיקים שלו, והמספר האישי שרשום עליו הוא: {FILE_NO}",
    # personal files
    f"תיק רפואי של מטופל מספר {FILE_NO}", f"תיק רפואי מספר {FILE_NO}", f"תיק אישי של העובד דני לוי {FILE_NO}", f"תיק במס הכנסה של דני לוי: {FILE_NO}",
    f"החולה דני לוי, תיק מספר {FILE_NO}", f"בקופת חולים מכבי, תיק {FILE_NO}", f"במשרד הפנים, תיק מספר {FILE_NO}", f"פקיד שומה חיפה, מספר תיק: {FILE_NO}",
    # whose record it is, with other numbers between, in any inflected form, on the previous line
    f"מטופל דני לוי, נולד 01-02-1980, מספר הליך {FILE_NO}", f"מטופל דני לוי, טל' 054-1234567, אסמכתא {FILE_NO}", f"לקוח 0541234567 הליך {FILE_NO}",
    f"החולה דני לוי, מספר הליך {FILE_NO}", f"העובדת רות כהן, טל' 054-1234567, אסמכתא: {FILE_NO}", f"פרטי המטופל דני לוי\nמספר הליך {FILE_NO}",
    f"אחד מעובדיו, מספר הליך {FILE_NO}", f"לקוחותיה של החברה, אסמכתא {FILE_NO}", f"מטופלו של הרופא, מספר הליך {FILE_NO}",
    f"עובדי המשרד, אסמכתא {FILE_NO}", f"החולים במחלקה, מספר הליך {FILE_NO}", f"מספר הליך {FILE_NO} של לקוחנו", f"מספר הליך {FILE_NO} של המטופל",
    f"אחד מחוליו של ד\"ר כהן, מספר הליך {FILE_NO}", f"חולי המחלקה, מספר הליך {FILE_NO}", f"one of our customers, מספר הליך {FILE_NO}",
    f"patients list, הליך {FILE_NO}", f"המבוטח דני לוי, מספר הליך {FILE_NO}", f"הנוסע דני לוי, אסמכתא {FILE_NO}",
    # an identity, passport, licence or account word anywhere near, in any form or spelling
    f"SSN {FILE_NO}", f"ת.ז. {FILE_NO}", f"ת\"ז {FILE_NO}", f"file ID {FILE_NO}", f"מספר דרכון {FILE_NO}", f"case 12, I.D. {FILE_NO}",
    f"מספר תעודת הזהות של החייב שנגדו נפתח ההליך {FILE_NO}", f"מספר תעודת הזהות של החייב שנגדו נפתח בלשכת חיפה תיק הוצאה לפועל מספר {FILE_NO}",
    f"מספר תעודת הזהות של החייב (יליד 12-03-1975) שנגדו נפתח ההליך {FILE_NO}",
    f"מספר תעודת הזהות של החייב דני לוי (טל' 054-1234567) שנגדו נפתח ההליך {FILE_NO}", f"מספר תעודת הזהות של החייב שנגדו נפתח\nההליך {FILE_NO}",
    f"Passport number of the debtor named in the enforcement case opened in Haifa: {FILE_NO}",
    f"צוין כי מספר דרכונו בהליך {FILE_NO}", f"רישיונו נשלל, אסמכתא {FILE_NO}", f"מספר זיהוי, הליך {FILE_NO}", f"מספר מזהה של החייב בהליך {FILE_NO}",
    f"עוסק מורשה, אסמכתא {FILE_NO}", f"ביטוח לאומי, אסמכתא {FILE_NO}", f"מספר אישי, אסמכתא {FILE_NO}", f"חשבונו בבנק, אסמכתא {FILE_NO}",
    f"חשבונך בבנק, אסמכתא {FILE_NO}", f"מחשבוני הפרטי, אסמכתא {FILE_NO}", f"חשבונותיו, אסמכתא {FILE_NO}",
    '{"file": "a.pdf", "user_id": "%s"}' % FILE_NO, f"Reference check for Dana Levi, national number {FILE_NO}",
    f"ת''ז החייב בהליך {FILE_NO}", f"ת“ז החייב בהליך {FILE_NO}", f"מספר הזֶהוּת, הליך {FILE_NO}", f"מספר הפספורט שלו, הליך {FILE_NO}",
    f"מספר עוסק של הספק, אסמכתא {FILE_NO}", f"ע.מ. של הספק, אסמכתא {FILE_NO}", f"תעודת עולה, הליך {FILE_NO}", f"מספר הבנק שלו, אסמכתא {FILE_NO}",
    f"T.Z., הליך {FILE_NO}", f"passports, הליך {FILE_NO}",
    # far back on a long line, and whose record it is even across a blank line
    f"מספר תעודת הזהות של החייב " + "ועוד ועוד " * 200 + f"מספר הליך {FILE_NO}", f"מספר הליך {FILE_NO} " + "ועוד ועוד " * 200 + "(ת.ז. של החייב)",
    f"פרטי המטופל דני לוי\n\nמספר הליך {FILE_NO}", "המטופל דני לוי " + "ועוד ועוד " * 200 + f"מספר הליך {FILE_NO}",
    # a word hidden from the lists by an invisible character, a soft hyphen, vowel points or full-width letters
    f"ת\u200f.ז. של החייב, מספר הליך {FILE_NO}", f"דר\u200dכון, הליך {FILE_NO}", f"pass\u00adport, הליך {FILE_NO}",
    f"ｐａｓｓｐｏｒｔ, הליך {FILE_NO}", f"מְטוּפָּל, הליך {FILE_NO}", f"מספר הליך {FILE_NO} (ת\u2060.ז.)",
    # removing an invisible character must not join two words and so hide the second ("של" + "ת.ז")
    f"של\u200fת.ז. החייב, מספר הליך {FILE_NO}", f"מספר\u200eדרכון, הליך {FILE_NO}",
    # an invisible character with no place in prose: nothing is released near it, whatever it hides
    f"שלום\u200b רב, מספר הליך {FILE_NO}", f"מספר הליך {FILE_NO} נדחה\u2060 אתמול", f"שלום\u3164רב, מספר הליך {FILE_NO}",
    f"שלום\ufeffרב, הליך {FILE_NO}", f"דר\u034fכון, הליך {FILE_NO}", f"דרכו\ufe0fן, הליך {FILE_NO}",
    # direction overrides, embeddings and isolates can make text read differently from how it is stored
    f"\u202bמספר הליך {FILE_NO}\u202c", f"מספר הליך {FILE_NO} \u202eז.ת\u202c", f"\u2067מספר הליך {FILE_NO}\u2069",
    # blank glyphs and other characters with no place in prose, also where a multi-word label is split by one
    f"שלום\u2800רב, מספר הליך {FILE_NO}", f"ביטוח\u2800לאומי, הליך {FILE_NO}", f"מספר\u3164אישי, הליך {FILE_NO}", f"שלום \x07, הליך {FILE_NO}",
    f"שלום \ue000, הליך {FILE_NO}", f"שלום 🙂, הליך {FILE_NO}",
    # control characters that Python counts as whitespace, and symbols that fold into a space plus an accent
    f"שלום\x1cרב, מספר הליך {FILE_NO}", f"שלום\x85רב, הליך {FILE_NO}", f"דר\u00b4כון, הליך {FILE_NO}", f"דר\u00a8כון, הליך {FILE_NO}",
    f"דר\u0903כון, הליך {FILE_NO}", f"של\u0903ת.ז. החייב, מספר הליך {FILE_NO}", f"שלום\u0903רב, מספר הליך {FILE_NO}",
    # a label stretched with spaces beyond any fixed length still counts when it reaches into the searched text
    "ביטוח" + " " * 400 + "לאומי\n" + f"מספר הליך {FILE_NO}", "מספר הליך " + FILE_NO + " " * 150 + "ביטוח" + " " * 300 + "לאומי",
    # a word that starts just outside the nearby text and reaches into it (the line break keeps it off the number's line)
    "מספר הזהות\n" + "א" * 284 + f" מספר הליך {FILE_NO}",
    # ... also when it describes another number: telling which number a label belongs to is where earlier versions leaked
    f"מספר הליך {FILE_NO}, ת.ז. של החייב {VALID_ID}.", f"ת.ז. של החייב {VALID_ID}, מספר הליך {FILE_NO}",
    # the label comes after the number
    f"מספר הליך {FILE_NO} (ת.ז. של החייב)", f"מספר הליך {FILE_NO} (ת.ז. של החייב), טלפון 0541234567", f"מספר הליך {FILE_NO} (his passport)",
    f"מספר הליך {FILE_NO} - מספר הזהות של הנתבע", f"מספר הליך {FILE_NO}, ת.ז. של החייב", f"מספר הליך {FILE_NO}, שהוא מספר הזהות של החייב.",
    f"מספר הליך {FILE_NO}, which is his passport number", f"מספר הליך {FILE_NO} (Mr. Smith's passport)", f"מספר הליך {FILE_NO} (ת. ז. של החייב)",
    f"מספר הליך {FILE_NO} (מ.ז. של החייב)", f"אסמכתא {FILE_NO} (מ.א. של החייל)", f"אסמכתא {FILE_NO} (ח\"ן בבנק)",
    f"מספר הליך {FILE_NO}, שהוא מספר הזהות של החייב, טלפון 054-1234567", f"מספר הליך {FILE_NO}, שהוא מספר הזהות של החייב, בסך 15000 שקלים",
    f"מספר הליך {FILE_NO} (מס. ת.ז. של החייב), טלפון 054-1234567", f"מספר הליך {FILE_NO} (Mr. Smith's passport), טל' 054-1234567",
    f"מספר הליך {FILE_NO}\n(ת.ז. של החייב)", f"מספר הליך {FILE_NO} (ת''ז של החייב)",
    # other spellings of the identity abbreviation: no mark, a typographic apostrophe, a hyphen, a slash, a space
    f"מספר התז של החייב בהליך {FILE_NO}", f"בתז שלו, הליך {FILE_NO}", f"ת’ז החייב בהליך {FILE_NO}", f"ת‘ז החייב בהליך {FILE_NO}",
    f"ת-ז החייב בהליך {FILE_NO}", f"ת/ז החייב בהליך {FILE_NO}", f"ת ז של החייב, הליך {FILE_NO}", f"ת`ז החייב בהליך {FILE_NO}",
    # English words run together or in camelCase
    f"customerId {VALID_ID}, אסמכתא {FILE_NO}", f"userId 7, אסמכתא {FILE_NO}", f"patientName, אסמכתא {FILE_NO}", f"PassportNo, אסמכתא {FILE_NO}",
    f"IDNumber, אסמכתא {FILE_NO}", f"bankaccount, אסמכתא {FILE_NO}", f"taxid, אסמכתא {FILE_NO}", f"theID, אסמכתא {FILE_NO}",
    f"newCustomer, אסמכתא {FILE_NO}", f"userid, אסמכתא {FILE_NO}",
    # a vowel point must not turn "בתהליך" (in the process) into the word "הליך"
    f"בתְהליך {FILE_NO}", f"בתהליִך {FILE_NO}",
    # more words for whose record it is, in Hebrew and Arabic; an Arabic word stretched with a tatweel
    f"מספר חבר במועדון, אסמכתא {FILE_NO}", f"חברי המועדון, אסמכתא {FILE_NO}", f"מספר מנוי, אסמכתא {FILE_NO}", f"תושב חיפה, הליך {FILE_NO}",
    f"رقم الهـويـة، הליך {FILE_NO}", f"رخصه القيادة، הליך {FILE_NO}", f"العميل أحمد، הליך {FILE_NO}", f"المـريـض أحمد، הליך {FILE_NO}",
    # part of a longer number
    f"תיק הוצאה לפועל {FILE_NO} - 21", f"הליך {FILE_NO}  21", f"הליך {FILE_NO} /21", f"הליך {FILE_NO}‏-21", f"הליך {FILE_NO}-A",
])
def test_everything_else_stays_masked(text):
    assert not _rule(text)


def test_long_lines_cost_nothing():
    import time
    t0 = time.time()
    spaces = "הליך" + " " * 15000
    assert _rule(spaces + f"ועוד מספר הליך {FILE_NO}") and not _rule(spaces + FILE_NO)
    one_line = " ".join(f"הליך {FILE_NO}" for _ in range(2000))             # every number rescanned the whole line before
    for k in range(0, len(one_line), 15):
        server._called_a_reference(one_line, k + 5, k + 14)
    assert time.time() - t0 < 3


def test_il_id_check_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SETER_IL_ID_CHECK", raising=False)
    assert server._flag("SETER_IL_ID_CHECK") is False
    assert server.IL_ID_CHECK is False and c.get("/health").json()["il_id_check"] is False   # conftest clears the variable
    for text, number in ((FILE_TEXT, FILE_NO), (ID_TEXT, VALID_ID)):
        r = redact(text)
        assert number not in r["text"] and r["kept"] == []


def test_il_id_check_releases_an_enforcement_number_and_nothing_else(monkeypatch):
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    assert c.get("/health").json()["il_id_check"] is True
    r = redact(FILE_TEXT)
    a = FILE_TEXT.index(FILE_NO)
    assert r["text"] == FILE_TEXT and r["entities"] == []
    assert r["kept"] == [{"type": "ID", "start": a, "end": a + 9, "reason": "il_id_check"}]   # reported, not silent
    r = redact(ID_TEXT)
    assert VALID_ID not in r["text"] and "[ID]" in r["text"] and r["kept"] == []
    # a wrong check digit alone releases nothing: an identity number with a typo, a passport, a bank account, a
    # US social security number, a personal file, a number next to an identity number, a piece of a longer code
    for text in (f"ת.ז. {FILE_NO}", f"מספר תעודת הזהות שלו הוא {FILE_NO}.", f"מספר דרכון {FILE_NO}",
                 f"בנק 12 סניף 345 חשבון {FILE_NO}", f"SSN {SSN} on file for Dana Levi",
                 f"מספר תעודת הזהות שלו {FILE_NO[:8]}-{FILE_NO[8]}", f"תיק הוצאה לפועל מספר {FILE_NO}\\21 נפתח",
                 f"תיק רפואי מספר {FILE_NO}", f"מספר הליך {FILE_NO} (ת.ז. של החייב)", f"בתיק {FILE_NO} נקבע דיון",
                 f"Tax file number {SSN}", f"The number we have on file: {SSN}"):
        got = redact(text)
        assert not _shows(got["text"], SSN if SSN in text else FILE_NO) and got["kept"] == [], got
    # Next to an identity number the check releases nothing. Whether the model itself reports the proceeding
    # number here differs between library versions (its score sits near the threshold), so only the check's
    # part is asserted: nothing under "kept", and the identity number masked.
    got = redact(f"מספר הליך {FILE_NO}, ת.ז. של החייב {VALID_ID}.")
    assert got["kept"] == [] and VALID_ID not in got["text"], got
    # the LiteLLM endpoint follows: nothing left to mask means no intervention
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [FILE_TEXT]}).json() == {"action": "NONE"}
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [ID_TEXT]}).json()["action"] == "GUARDRAIL_INTERVENED"


def _fake_entities(monkeypatch, reports_for):
    """Stand in for the model: reports_for(text) -> [(start, end, entity_group, score, owner, extra)]."""
    monkeypatch.setattr(server, "entities", lambda t: [
        {"start": a, "end": b, "entity_group": k, "score": sc, "owner": o, **extra} for a, b, k, sc, o, extra in reports_for(t)])


def test_a_number_two_windows_disagree_about_is_not_released(monkeypatch):
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    text = f"מספר הליך {FILE_NO} נדחה"
    a = text.index(FILE_NO)
    span, main = (a, a + 9, "ID"), (a, a + 9, "account_number", 0.9, True, {})
    _fake_entities(monkeypatch, lambda _t: [main])
    assert server.scan(text) == ([], [span])                                     # one opinion: released
    second = [(a, a + 9, "private_phone", 0.9, False, {}),                       # another window calls it a phone number
              (a, a + 9, "private_phone", 0.2, False, {}),                       # ... with a score too low to mask on
              (a, a + 9, "private_phone", 0.9, False, {"untrusted": True})]      # ... from the strip right next to a cut
    for other in second:
        for order in ((main, other), (other, main)):
            _fake_entities(monkeypatch, lambda _t, order=order: list(order))
            assert server.scan(text) == ([span], []), order
    _fake_entities(monkeypatch, lambda _t: [second[2]])                          # an untrusted report alone masks nothing, as before
    assert server.scan(text) == ([], [])


def test_one_value_gets_one_answer(monkeypatch):
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    def every_number(t):
        return [(m.start(), m.end(), "account_number", 0.9, True, {}) for m in __import__("re").finditer(FILE_NO, t)]
    _fake_entities(monkeypatch, every_number)
    proceeding, identity = f"מספר הליך {FILE_NO}", f"ת.ז. של החייב {FILE_NO}"
    assert server.scan(proceeding)[1] and not server.scan(identity)[1]           # each on its own: released, masked
    two_fields = f"{identity}\n\n{proceeding}"                                   # a blank line apart, so only the value links them
    masked, kept = server.scan(two_fields)
    assert kept == [] and len(masked) == 2 and FILE_NO not in server.mask(two_fields)[0]
    twice = f"{proceeding}\n\n{proceeding}"                                      # the same value released in both places: fine
    assert len(server.scan(twice)[1]) == 2
    _fake_entities(monkeypatch, lambda t: every_number(t)[:1])                   # the model reports only the first of two copies
    assert server.scan(f"{proceeding}\n\nראו גם {FILE_NO}")[1] == []
    _fake_entities(monkeypatch, every_number)
    for texts in ([identity, proceeding], [proceeding, identity]):               # the separate texts of one LiteLLM request
        r = c.post("/beta/litellm_basic_guardrail_api", json={"texts": texts}).json()
        assert r["action"] == "GUARDRAIL_INTERVENED" and not any(FILE_NO in t for t in r["texts"]), r
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [proceeding, "no pii"]}).json() == {"action": "NONE"}
    # the same number written another way is the same number: a hyphen, a space, a direction mark, no leading zero,
    # other digit shapes. The model is not asked here (the fake reports only the plain spelling).
    for other in (f"{FILE_NO[:8]}-{FILE_NO[8]}", f"{FILE_NO[:4]} {FILE_NO[4:]}", f"{FILE_NO[:3]}\u200f{FILE_NO[3:]}", FILE_NO[1:],
                  "٠٣٤٨٥١٢٧٢", "０３４８５１２７２"):
        for texts in ([f"ת.ז. של החייב {other}", proceeding], [proceeding, f"ראו {other}"], [f"{proceeding}\n\nת.ז. {other}"]):
            assert all(kept == [] for _masked, kept in server.scan_many(texts)), texts


def test_a_number_is_masked_whole(monkeypatch):
    """The model sometimes reports one or two digits of a number; the rest was printed next to the mask."""
    text = f"מספר הליך {FILE_NO} נדחה, טלפון 054-1234567"
    a, p = text.index(FILE_NO), text.index("054")
    for pieces in ([(a, a + 1)], [(a, a + 1), (a + 5, a + 8)], [(a + 3, a + 4)], [(a + 8, a + 9)]):
        _fake_entities(monkeypatch, lambda _t, pieces=pieces: [(x, y, "account_number", 0.9, True, {}) for x, y in pieces])
        assert server.scan(text) == ([(a, a + 9, "ID")], []), pieces
        assert server.mask(text)[0] == text.replace(FILE_NO, "[ID]")
    # only the digit run itself: a phone number in groups is not pulled in, and words are left alone
    _fake_entities(monkeypatch, lambda _t: [(p + 1, p + 2, "private_phone", 0.9, True, {}), (0, 2, "private_person", 0.9, True, {})])
    assert server.scan(text)[0] == [(0, 2, "PERSON"), (p, p + 3, "PHONE")]
    # two pieces called different things: one mask, and never released
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    _fake_entities(monkeypatch, lambda _t: [(a, a + 1, "account_number", 0.9, True, {}), (a + 5, a + 8, "private_phone", 0.9, True, {})])
    assert server.scan(text) == ([(a, a + 9, "ID")], [])
    _fake_entities(monkeypatch, lambda _t: [(a, a + 1, "account_number", 0.9, True, {})])
    assert server.scan(text) == ([], [(a, a + 9, "ID")])                         # the whole number is what gets judged


def test_a_released_number_is_logged_without_its_digits(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    with caplog.at_level(logging.INFO, logger="seter_guard"):                    # reaches the root handlers like any logger
        assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [FILE_TEXT]}).json() == {"action": "NONE"}
        c.post("/beta/litellm_basic_guardrail_api", json={"texts": [ID_TEXT]})
    lines = [r.getMessage() for r in caplog.records if "il_id_check" in r.getMessage()]
    a = FILE_TEXT.index(FILE_NO)
    assert len(lines) == 1 and f"characters {a}-{a + 9}" in lines[0] and FILE_NO not in lines[0]
    fallback = next(h for h in server.log.handlers if isinstance(h, server._WhenNothingElseListens))
    assert "%(asctime)s" in fallback.formatter._fmt and "%(levelname)s" in fallback.formatter._fmt


def test_flag_values(monkeypatch):
    for value, want in (("1", True), ("true", True), ("ON", True), ("0", False), ("", False), ("off", False)):
        monkeypatch.setenv("SETER_IL_ID_CHECK", value)
        assert server._flag("SETER_IL_ID_CHECK") is want
    monkeypatch.setenv("SETER_IL_ID_CHECK", "maybe")
    with pytest.raises(ValueError):
        server._flag("SETER_IL_ID_CHECK")
    assert server._flag("SETER_IL_ID_CHECK", strict=False) is False             # the lenient form used for HF_HUB_OFFLINE


# ---- Israeli court case numbers ----

# The three sentences with a court case number in the hard4 probe of seter-train (eval/hard4.jsonl, records
# 1, 11 and 12), and their numbers. None of them is personal data.
HARD4_CASES = [
    ('בתמ"ש 4412-03-22 פלונית נ\' אלמוני נקבע כי המשמורת תהיה משותפת.', ["4412-03-22"]),
    ('ת"א 45127-09-23 נדון בבית משפט השלום בתל אביב.', ["45127-09-23"]),
    ('ראו ע"א 8954/11 ורע"א 2146/19.', ["8954/11", "2146/19"]),
]
CASE_TEXT, CASE_NO = HARD4_CASES[1][0], "45127-09-23"


def _case(text, number):
    """What the rule says about `number` in `text`, the model aside: the span of the whole case number, or None."""
    a = text.index(number)
    return server.court_case_number(text, a, a + len(number))


def _whole(text, number):
    a = text.index(number)
    return _case(text, number) == (a, a + len(number))


def test_court_case_rule_on_the_hard4_case_numbers():
    text, numbers = HARD4_CASES[0]
    assert all(_whole(text, n) for n in numbers)
    # the second is ת"א, which is also Tel Aviv and not in the list; the third is a bare citation with no parties
    for text, numbers in HARD4_CASES[1:]:
        assert all(_case(text, n) is None for n in numbers), text


def test_il_court_case_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SETER_IL_COURT_CASE", raising=False)
    assert server._flag("SETER_IL_COURT_CASE") is False
    assert server.IL_COURT_CASE is False and c.get("/health").json()["il_court_case"] is False   # conftest clears the variable
    r = redact(CASE_TEXT)
    assert CASE_NO not in r["text"] and "[ID]" in r["text"] and r["kept"] == []


def test_il_court_case_leaves_the_hard4_case_numbers_in_the_text(monkeypatch):
    monkeypatch.setattr(server, "IL_COURT_CASE", True)
    assert c.get("/health").json()["il_court_case"] is True
    for text, numbers in HARD4_CASES[:1]:
        r = redact(text)
        assert all(n in r["text"] for n in numbers), r
        assert not any(e["type"] == "ID" for e in r["entities"]), r
        assert all(k["reason"] == "il_court_case" and text[k["start"]:k["end"]] in numbers for k in r["kept"]), r
    # the model reports this one, so it is listed under "kept" and the request needs no intervention
    text = f'עת"מ (מחוזי חיפה) {CASE_NO} נדון אתמול.'
    r = redact(text)
    a = text.index(CASE_NO)
    assert r["text"] == text and r["kept"] == [{"type": "ID", "start": a, "end": a + len(CASE_NO), "reason": "il_court_case"}], r
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [text]}).json() == {"action": "NONE"}
    # the case number stays, the personal data in the same sentence does not
    text = f'בתיק עת"מ (מחוזי חיפה) {CASE_NO}, התובע יוסי כהן, טלפון 054-7812345.'
    out = redact(text)["text"]
    assert CASE_NO in out and "יוסי כהן" not in out and "054-7812345" not in out, out
    # ... unless an identity number is given in the same paragraph: then the case number is masked with the rest
    text = f'בתיק עת"מ (מחוזי חיפה) {CASE_NO}, התובע יוסי כהן, טלפון 054-7812345, ת"ז {VALID_ID}.'
    got = redact(text)
    assert CASE_NO not in got["text"] and VALID_ID not in got["text"] and "יוסי כהן" not in got["text"] and got["kept"] == [], got
    # a number in the same shape that is not introduced as a case, a bank account, a phone number after "ת"א"
    for text, number in ((f"מספר חשבון {CASE_NO} בבנק לאומי.", CASE_NO), (f"חשבון בנק בסניף ת\"א {CASE_NO}.", CASE_NO),
                         ("חשבון בנק 12-345-678901 על שם דני לוי.", "12-345-678901"), ('ת"א 03-5551234 התקשרו בבקשה.', "03-5551234"),
                         (f'ת"ז {VALID_ID}', VALID_ID), (ID_TEXT, VALID_ID)):
        got = redact(text)
        assert not _shows(got["text"], number.replace("-", "")) and number not in got["text"] and got["kept"] == [], got
    # the identity-number check is a separate switch
    assert FILE_NO not in redact(FILE_TEXT)["text"]


@pytest.mark.parametrize("text,number", [
    (f'ע"א {CASE_NO} כהן נ\' לוי', CASE_NO), (f'בע"א {CASE_NO} כהן נ\' לוי', CASE_NO), (f'ות"פ {CASE_NO} מדינת ישראל נ\' לוי.', CASE_NO), (f'שבעפ"ג {CASE_NO} פלוני נ\' מדינת ישראל נדחה', CASE_NO),
    ('ראו רע"א 2146/19, עע"מ 311/20 עיריית חיפה נ\' לוי וכן בש"א 1234/07.', "311/20"), ('בג"ץ 5658/23 התנועה נ\' הכנסת', "5658/23"),
    ('בג"צ 1/49 בז\'רנו נ\' שר המשטרה', "1/49"), ('עפ"ג (מחוזי ת"א) 71234-01-20 פלוני נ\' מדינת ישראל', "71234-01-20"),
    ('ע"א (מחוזי חיפה) 3345/08 כהן נ\' לוי', "3345/08"), (f'ע"א (שלום חיפה) {CASE_NO} כהן נ\' לוי', CASE_NO), (f'ע"א מס\' {CASE_NO} כהן נ\' לוי', CASE_NO), (f'ע"א מספר {CASE_NO} כהן נ\' לוי', CASE_NO),
    ('עמ"ש 44123-05-21, תלה"מ 5521-10-22 פלונית נ\' אלמוני, רמ"ש 30111-02-23.', "5521-10-22"), ('סע"ש 23456-11-19 רחל לוי נ\' החברה', "23456-11-19"),
    ('פש"ר 1234-05-16 וחדל"ת (מחוזי ת"א) 22331-04-21', "22331-04-21"),
    # the kind of court and its seat in the brackets is evidence enough
    ('ע"א (מחוזי חיפה) 3345/08', "3345/08"), (f'ע"א (שלום ת"א) {CASE_NO}', CASE_NO), ('תמ"ש (משפחה ראשל"צ) 4412-03-22', "4412-03-22"),
    ('סע"ש (אזורי לעבודה תל אביב) 23456-11-19', "23456-11-19"), ('ע"א (מחוזי מרכז-לוד) 3345/08', "3345/08"), ('ע"א (מחוזי מרכז) 3345/08', "3345/08"), ('עת"מ (מחוזי י-ם) 3345/08', "3345/08"),
    # the parties after the number: after a comma, with up to five words before the נ'
    (f'ע"א {CASE_NO}, כהן נ\' לוי', CASE_NO), ('בג"ץ 5658/23 התנועה למען איכות השלטון בישראל נ\' הכנסת', "5658/23"), ('ע"א 3345/08 כהן נ׳ לוי', "3345/08"),
    # other marks inside the abbreviation, and direction marks beside the number
    (f'ע״א {CASE_NO} כהן נ\' לוי', CASE_NO), (f'ע”א {CASE_NO} כהן נ\' לוי', CASE_NO), (f"ע''א {CASE_NO} כהן נ' לוי", CASE_NO), (f'ע"א ‏{CASE_NO}‏ כהן נ\' לוי', CASE_NO),
    # an identity word in another paragraph, or further away than the rule looks
    ("מספר חשבון " + "א" * 300 + f'\nע"א {CASE_NO} כהן נ\' לוי', CASE_NO), (f'ע"א {CASE_NO} כהן נ\' לוי\n' + "א" * 200 + " מספר חשבון", CASE_NO),
])
def test_a_number_written_as_a_court_case_can_be_released(text, number):
    assert _whole(text, number), text


# Found by reviewing the first version of the rule (PR #6) with the real model: in each of these the number was
# masked with the rule off and left in the text with it on. None is a court case.
REVIEW_LEAKS = [
    ('בנק | סניף | חשבון\nלאומי | 810 | ת"א 12345/67', "12345/67"),               # the label is the table header on the line above
    ('ת"א 12345/67 - זה המספר שמופיע בכרטיס של המטופל', "12345/67"),               # whose record it is, further along the line
    ('ת"א 12345/67, כפי שמופיע בפוליסה של המבוטח', "12345/67"),
    ('התשלום בוצע ע"פ 12345/67', "12345/67"),                                       # ע"פ: "according to"
    ('שולם ע"פ 4411/23 לדני לוי', "4411/23"),
    ('הילד רשום בבי"ס 12345/67', "12345/67"),                                       # ב + י"ס: a school
    ('נועה לומדת בבי"ס 4411/23', "4411/23"),
    ('מספר אשפוז בת"א 12345/67', "12345/67"),                                       # a hospital admission in Tel Aviv
    ('a/c ת"א 12345/67', "12345/67"),
    ('דיסקונט ת"א 123456-01-23', "123456-01-23"),                                   # a bank by its name
    ('Leumi\tת"א 12345/67', "12345/67"),
    ('דני לוי נולד בת"א 12-05-85', "12-05-85"),                                     # a date
    ('תאריך לידה: בת"א 12-05-85', "12-05-85"),
]
# Found by reviewing the second version (PR #7) the same way: 26 more, none a court case.
SECOND_REVIEW_LEAKS = [
    # ת"א is Tel Aviv, and the number was given out there
    ('פוליסה שהופקה בת"א 45127-09-23', "45127-09-23"), ('אשרת שהייה שהונפקה בת"א 771204-05-21', "771204-05-21"),
    ('התיק שלי בלשכת הגיוס ת"א 8841-02-97', "8841-02-97"), ('מספר חוגר, בקו"ם ת"א 7412-03-88', "7412-03-88"),
    ('הלוואה 2231 שנלקחה בת"א 55120-10-22', "55120-10-22"), ('כרטיס מגנטי של מכבי ת"א 20455-08-16', "20455-08-16"),
    ('פוליסת ביטוח חיים שהופקה בסוכנות בת"א 45127-09-23', "45127-09-23"), ('מסגרת אשראי שאושרה בת"א 55120-10-22', "55120-10-22"),
    ('משכנתא שנלקחה בת"א 30412-07-19', "30412-07-19"), ('קרן הפנסיה שלי, מנוהלת בת"א 20455-08-16', "20455-08-16"),
    ('אני גר ברחוב הרצל 12, ת"א 6473-09-21', "6473-09-21"), ('התלונה שהגשתי במשטרת ת"א 88120-03-24', "88120-03-24"),
    ('נותחתי באיכילוב ת"א 77120-05-22', "77120-05-22"), ('אסיר בכלא ת"א 4412-06-20', "4412-06-20"),
    # a label that was in neither word list, before another listed abbreviation
    ('כרטיס אשראי שמספרו מסתיים בפ"ה 4512/27', "4512/27"), ('פוליסה א"פ 88412/19', "88412/19"), ('כרטיס ד"מ 77120/25 בתוקף', "77120/25"),
    ('הלוואה ע"א 55120/22 לפירעון', "55120/22"), ('חייל מ"ת 74125/88', "74125/88"), ('תלוש שכר ס"ע 30412/24', "30412/24"),
    ('ויזה ת"פ 4580/27', "4580/27"),
    # the label is a paragraph of its own, one blank line above
    ('מספר דרכון\n\nע"א 12345/67', "12345/67"), ('מספר זהות\n\nע"א 12345/67', "12345/67"),
    # מ + מ"י: the Israel Land Administration
    ('חוזה החכירה שלי, ממ"י 45127/23', "45127/23"), ('תיק ממ"י 30412/07 על שמי', "30412/07"),
    # a branch, in Arabic, stretched with a tatweel
    ('فــرع ת"א 45127-09-23', "45127-09-23"),
]
# Found by reviewing the third version (PR #7 at e9e01ac): 19 more, none a court case.
THIRD_REVIEW_LEAKS = [
    # ת"א is Tel Aviv, and a word that was taken for the courts' is on the line
    ('תביעה לדמי אבטלה שהוגשה בת"א 55120-10-22', "55120-10-22"), ('תביעה לקופת הגמל, מספר עמית בת"א 20455-08-16', "20455-08-16"),
    ('מספר התביעה שלי במוקד ת"א 88120-03-24', "88120-03-24"), ('אני שופט כדורגל, מספר רישום באיגוד ת"א 7412-03-88', "7412-03-88"),
    ('המשרד המחוזי של משרד הרישוי בת"א 8841-02-97', "8841-02-97"), ('לשכת הבריאות המחוזית ת"א 77120-05-22', "77120-05-22"),
    ('פיקדון אצל עורך הדין של התובע, ת"א 45127-09-23', "45127-09-23"), ('ערעור על אחוזי נכות, ועדה בת"א 55120-10-22', "55120-10-22"),
    # ... or a kind of court alone in the brackets
    ('קוד כניסה ת"א (עבודה) 4412-06-20', "4412-06-20"), ('מספר תיק רווחה ת"א (משפחה) 30412-07-19', "30412-07-19"),
    # a label that was in no word list, before another listed abbreviation
    ('קצבת נכות ע"א 55120/22', "55120/22"), ('דו"ח חניה ת"פ 4580/23', "4580/23"), ('מרשם ד"מ 77120/25', "77120/25"),
    ('מספר רכב ע"א 12345/67', "12345/67"), ('קוד סודי ה"פ 30412/24', "30412/24"), ('מספר הזמנה ת"ק 2512/23 על שם דני לוי', "2512/23"),
    ('תיק ניכויים ע"ר 58012/34', "58012/34"), ('מבקש מקלט, מספר תיק ע"ע 12345/21', "12345/21"), ('שובר תשלום ת"צ 30412/24', "30412/24"),
]
# Found by reviewing the fourth version (PR #7 at 0722ee6): 19 more, none a court case.
FOURTH_REVIEW_LEAKS = [
    # a נ' that is not "versus": an initial, "female" or "married" in a form, a fixture
    ('נ\' אברהמי, מרשם ד"מ 77120/25', "77120/25"), ('נ\' אברהמי, מספר רישום באיגוד ת"א 7412-03-88', "7412-03-88"),
    ('מין: נ\' | עיר: ת"א 6473-09-21', "6473-09-21"), ('מין: נ\', מספר תיק רווחה ת"א 30412-07-19', "30412-07-19"),
    ('מצב משפחתי: נ\'. קצבת ילדים ע"א 55120/22', "55120/22"), ('הפועל חיפה נ\' מכבי ת"א 45127-09-23 - מספר מושב', "45127-09-23"),
    ('נ\' אישי, קוד סודי ה"פ 30412/24', "30412/24"),
    # a court is named on the line, and the number is something else
    ('חניון בית המשפט, מספר רישום ת"א 45127-09-23', "45127-09-23"), ('אני מתורגמן בבית המשפט, מספר ספק ת"א 30412-07-19', "30412-07-19"),
    ('קנס שקיבלתי ליד בית המשפט בת"א 55120-10-22', "55120-10-22"), ('אגרת בית משפט שולמה, אסמכתא ת"א 88120-03-24', "88120-03-24"),
    ('אחרי הדיון בבית המשפט לקחתי מרשם ד"מ 77120/25', "77120/25"), ('אני גר מול בית המשפט, רחוב ויצמן 7, ת"א 6473-09-21', "6473-09-21"),
    ('לפי פסק הדין, מזונות ישולמו לפי קוד משלם ת"א 20455-08-16', "20455-08-16"), ('קוד כניסה לבימ"ש ת"א 4412-06-20', "4412-06-20"),
    ('מזכירות בית המשפט, שלוחה ת"א 612-03-45', "612-03-45"),
    # in the brackets: work, in Haifa; a youth centre
    ('קוד כניסה ת"א (עבודה בחיפה) 4412-06-20', "4412-06-20"), ('תיק רווחה ע"א (נוער מרכז) 12345/21', "12345/21"),
    # the court is far back on one long line
    ('הייתי אתמול בבית המשפט ואחר כך חזרתי הביתה. ' + 'דיברנו על הרבה דברים ולא הגענו לשום מסקנה ברורה, אז נמשיך בשבוע הבא. ' * 5
     + 'אגב, קצבת הנכות שלי ע"א 55120/22', "55120/22"),
]
# ... and a request in which the label is the message before the number
REVIEW_LEAK_REQUEST = ["מספר הזהות או הדרכון של הלקוח?", 'ת"א 45127-09-23']


@pytest.mark.parametrize("text,number", REVIEW_LEAKS + SECOND_REVIEW_LEAKS + THIRD_REVIEW_LEAKS + FOURTH_REVIEW_LEAKS)
def test_the_leaks_found_in_review_stay_masked(text, number):
    assert _case(text, number) is None, text


def test_the_leaks_found_in_review_stay_masked_with_the_model(monkeypatch):
    """Whatever the model reports in these sentences, the rule releases nothing: the output is the same with the
    rule on and off."""
    def run(on):
        monkeypatch.setattr(server, "IL_COURT_CASE", on)
        one = [redact(text) for text, _number in REVIEW_LEAKS + SECOND_REVIEW_LEAKS + THIRD_REVIEW_LEAKS + FOURTH_REVIEW_LEAKS]
        many = [(server._render(t, to_mask)[0], kept) for t, (to_mask, kept) in zip(REVIEW_LEAK_REQUEST, server.scan_many(REVIEW_LEAK_REQUEST))]
        return one, many
    (off, off_many), (on, on_many) = run(False), run(True)
    assert on == off and on_many == off_many
    assert all(r["kept"] == [] for r in on) and all(kept == [] for _text, kept in on_many)


def test_a_label_in_a_neighbouring_text_keeps_a_case_number_masked(monkeypatch):
    monkeypatch.setattr(server, "IL_COURT_CASE", True)
    def every_number(t):
        return [(m.start(), m.end(), "account_number", 0.9, True, {}) for m in __import__("re").finditer(CASE_NO, t)]
    _fake_entities(monkeypatch, every_number)
    case = f'ע"א {CASE_NO} כהן נ\' לוי'
    a = case.index(CASE_NO)
    span = (a, a + len(CASE_NO), "ID")
    kept = lambda texts: [k for _masked, k in server.scan_many(texts)]
    assert kept([case]) == [[span]] and kept(["שלום, מה שלומך?", case, "תודה רבה"]) == [[], [span], []]
    # the label is the text before, or the text after, or two texts back with an empty or a short one between
    for texts, i in ((REVIEW_LEAK_REQUEST, 1), (["מה מספר חשבון הבנק שלך?", case], 1), ([case, "זה מספר החשבון שלי"], 0),
                     (["פרטי הלקוח", "", case], 2), (["Account number?", "שלום", case], 2), ([case, "ok", "(לאומי)"], 0),
                     (["העברה בדיסקונט", case], 1), (["ok\u200b", case], 1)):
        got = server.scan_many(texts)
        assert got[i] == ([span], []), texts
        r = c.post("/beta/litellm_basic_guardrail_api", json={"texts": texts}).json()
        assert r["action"] == "GUARDRAIL_INTERVENED" and not any(CASE_NO in t for t in r["texts"]), r
    # only as far as the rule reads inside a text: 300 characters back and 200 on, counted from the number
    far = "א " * 200
    assert kept(["מספר חשבון " + far, case]) == [[], [span]] and kept([case, far + " מספר חשבון"]) == [[span], []]
    long_case = far + case
    assert kept(["מספר חשבון", long_case])[1] == [(len(far) + a, len(far) + a + len(CASE_NO), "ID")]
    # the identity-number check is not changed by this: it reads its own text only
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    _fake_entities(monkeypatch, lambda t: [(m.start(), m.end(), "account_number", 0.9, True, {}) for m in __import__("re").finditer(FILE_NO, t)])
    assert [k for _m, k in server.scan_many(["שלום", FILE_TEXT])][1] != []


def test_a_piece_of_a_case_number_is_judged_whole():
    text = 'סע"ש 23456-11-19 רחל לוי נ\' החברה'                                           # the model reports 23456-11-1 here
    a = text.index("23456")
    for x, y in ((a, a + 5), (a + 6, a + 8), (a, a + 10), (a + 9, a + 11)):
        assert server.court_case_number(text, x, y) == (a, a + 11), (x, y)


@pytest.mark.parametrize("text,number", [
    # no kind of proceeding directly before it, or a word and no abbreviation
    (f"מספר {CASE_NO}", CASE_NO), (f"תיק מספר {CASE_NO} בבית משפט השלום", CASE_NO), (f"תיק אזרחי {CASE_NO}", CASE_NO), (CASE_NO, CASE_NO),
    (f'ע"א\n{CASE_NO}', CASE_NO), (f'ע"א של דני לוי {CASE_NO}', CASE_NO), (f'ע"א{CASE_NO}', CASE_NO), (f"case no. {CASE_NO}", CASE_NO),
    # abbreviations that label a person's or a company's number, or are not a kind of proceeding
    (f'ת"ז {CASE_NO}', CASE_NO), (f'ב"ל {CASE_NO}', CASE_NO), (f'ע"מ {CASE_NO}', CASE_NO), (f'דני לוי בע"מ {CASE_NO}', CASE_NO),
    (f'מ"א {CASE_NO}', CASE_NO), (f'ח"פ {CASE_NO}', CASE_NO), (f'ת"ד {CASE_NO}', CASE_NO), (f'עו"ד {CASE_NO}', CASE_NO), (f'מע"אם {CASE_NO}', CASE_NO),
    (f'xע"א {CASE_NO}', CASE_NO), (f'7ע"א {CASE_NO}', CASE_NO), (f'_ע"א {CASE_NO}', CASE_NO), (f'אבע"א {CASE_NO}', CASE_NO),
    # not the way a court writes it: no month 13, a phone number, an identity number with a hyphen, a bank account,
    # a four-digit year, another kind of dash, other digit shapes
    ('ע"א 45127-13-23', "45127-13-23"), ('ע"א 45127-00-23', "45127-00-23"), ('ע"א 03-5551234', "03-5551234"), ('ע"א 054-7812345', "054-7812345"),
    ('ע"א 03485127-9', "03485127-9"), ('ע"א 12-345-678901', "12-345-678901"), ('ע"א 123456/78', "123456/78"), ('ע"א 8954/2011', "8954/2011"),
    ('ע"א 1234567-09-23', "1234567-09-23"), ('ע"א 45127–09–23', "45127–09–23"), ('ע"א 45127־09־23', "45127־09־23"), ('ע"א ٤٥١٢٧-٠٩-٢٣', "٤٥١٢٧-٠٩-٢٣"),
    ('ע"א 45127 09 23', "45127"), (f'ע"א {VALID_ID}', VALID_ID),
    # a piece of something longer
    (f'ע"א {CASE_NO}-1', CASE_NO), (f'ע"א {CASE_NO}/21', CASE_NO), (f'ע"א 7-{CASE_NO}', CASE_NO), (f'ע"א {CASE_NO} 21', CASE_NO),
    (f'ע"א {CASE_NO}, 2024', CASE_NO), (f'ע"א {CASE_NO}-A', CASE_NO), (f'ע"א {CASE_NO}א', CASE_NO), ('ע"א 8954/11/3', "8954/11"),
    ('ע"א 8954/11-', "8954/11"), (f'ע"א {CASE_NO}' + "-" * 40, CASE_NO),
    # ע"א is also Tel Aviv: a bank, account, identity, passport or licence word just before or after keeps it masked
    (f'חשבון בנק בסניף ע"א {CASE_NO}', CASE_NO), ('חשבון בסניף ע"א 12345/67', "12345/67"), (f'ע"א {CASE_NO} (מספר חשבון)', CASE_NO),
    (f'בנק לאומי ע"א {CASE_NO}', CASE_NO), (f'ת.ז. של התובע בע"א {CASE_NO}', CASE_NO), (f'מספר דרכון, ע"א {CASE_NO}', CASE_NO),
    (f'ע"א (חשבון בנק) {CASE_NO}', CASE_NO), (f'account, ע"א {CASE_NO}', CASE_NO), (f'חֶשבון בסניף ע"א {CASE_NO}', CASE_NO),
    (f'חש‍בון בסניף ע"א {CASE_NO}', CASE_NO),
    # ... and so does a word for whose record it is, or for a branch
    ('מספר מבוטח בסניף ע"א 12345/67', "12345/67"), (f'לקוח ע"א {CASE_NO}', CASE_NO), (f'לאומי, סניף ע"א {CASE_NO}', CASE_NO),
    (f'סְניף ע"א {CASE_NO}', CASE_NO), (f'ע"א {CASE_NO} של הלקוח', CASE_NO), (f'Tel Aviv branch, ע"א {CASE_NO}', CASE_NO),
    # "on account of" and "in the name of" are written like two kinds of proceeding, which are therefore not in the list
    (f'שולם ע"ח {CASE_NO}', CASE_NO), (f'ע"ש {CASE_NO}', CASE_NO),
    # ... wherever it stands in the paragraph: the line above, further along the line, the next line
    (f'ת"פ 12345-06-21 מדינת ישראל נ\' דני לוי, ת.ז. {VALID_ID}.', "12345-06-21"), (f'ת.ז. {VALID_ID}\nע"א {CASE_NO}', CASE_NO),
    (f'מספר חשבון:\nע"א {CASE_NO}', CASE_NO), (f'ע"א {CASE_NO} הוא המספר שאליו יש להעביר, זה חשבון הבנק של דני', CASE_NO),
    (f'ע"א {CASE_NO}\nמספר דרכון', CASE_NO), (f'פרטי הלקוח:\n\nע"א {CASE_NO}', CASE_NO), (f'בג"ץ 5658/23 העותר, ת"ז {VALID_ID}', "5658/23"),
    # a bank by its name, without the word bank; "a/c"; a hospital admission
    (f'לאומי ע"א {CASE_NO}', CASE_NO), (f'פועלים, ע"א {CASE_NO}', CASE_NO), (f'העברה לדיסקונט\nע"א {CASE_NO}', CASE_NO),
    (f'מזרחי טפחות ע"א {CASE_NO}', CASE_NO), (f'Leumi ע"א {CASE_NO}', CASE_NO), (f'Ｌｅｕｍｉ ע"א {CASE_NO}', CASE_NO), (f'לְאוּמי ע"א {CASE_NO}', CASE_NO),
    (f'בנק יהב ע"א {CASE_NO}', CASE_NO), (f'a/c ע"א {CASE_NO}', CASE_NO), (f'acc. no ע"א {CASE_NO}', CASE_NO), (f'מספר אשפוז בע"א {CASE_NO}', CASE_NO),
    # a label of a number is read across a blank line too
    (f'ת.ז. {VALID_ID}\n\nע"א 3345/08', "3345/08"), ('ע"א 3345/08\n\nמספר חשבון: 123', "3345/08"), ('סניף 810\n\nע"א 3345/08', "3345/08"),
    # words an attached letter makes, and things a person holds that carry a number
    ('מת"ק 12345/24', "12345/24"), ('מה"ט 4512/19', "4512/19"), ('ממ"י 45127/23', "45127/23"), ('משכורת, ע"א 3345/08', "3345/08"),
    ('policy no. ע"א 3345/08', "3345/08"), ('credit card ע"א 3345/08', "3345/08"), ('قرض ע"א 3345/08', "3345/08"), ('لئــومي ע"א 3345/08', "3345/08"),
    # ת"א (also Tel Aviv) is not in the list: "ת"א 12345/67" is how a bank account at a Tel Aviv branch is written
    ('ת"א 12345/67', "12345/67"), ('ת"א 8954/11 כהן נ\' לוי', "8954/11"), ('בת"א 3345/08 נקבע', "3345/08"), ('ת"א (שלום חיפה) 3345/08 כהן נ\' לוי', "3345/08"),
    ('ת״א 12345/67', "12345/67"), ("ת''א 12345/67", "12345/67"), ('ת"א מספר 12345/67', "12345/67"),
    # a date is written like a case number with a short serial
    ('נולד בע"א 12-05-85', "12-05-85"), ('בג"ץ 1-01-20', "1-01-20"), ('תמ"ש 31-12-99', "31-12-99"),
    # abbreviations that are also everyday words (according to, a school, a Talmud Torah, kilograms, a sub-clause),
    # and words that an attached letter makes out of a listed abbreviation (cooperation, for the time being)
    (f'ע"פ {CASE_NO}', CASE_NO), ('שולם ע"פ 4411/23', "4411/23"), (f'נמסר בע"פ {CASE_NO}', CASE_NO), ('בי"ס 4411/23', "4411/23"), (f'י"ס {CASE_NO}', CASE_NO),
    (f'ת"ת {CASE_NO}', CASE_NO), ('70 ק"ג 123/45', "123/45"), ('ס"ק 12/34', "12/34"), (f'שת"פ {CASE_NO}', CASE_NO), ('לע"ע 123/45', "123/45"),
    # a character with no place in prose on the line: nothing is released near it
    (f'ע"א {CASE_NO}​', CASE_NO), (f'שלום​ ע"א {CASE_NO}', CASE_NO), (f'‮ע"א {CASE_NO}‬', CASE_NO), (f'ע"א {CASE_NO} 🙂', CASE_NO),
    (f'ע"א 45127-09-23​45', CASE_NO), (f'ת‍"א {CASE_NO}', CASE_NO),
    # the court in brackets is a short Hebrew name and nothing else
    (f'ע"א (John Smith) {CASE_NO}', CASE_NO), (f'ע"א (דירה 5) {CASE_NO}', CASE_NO), (f'ע"א ({"א" * 31}) {CASE_NO}', CASE_NO),
])
def test_everything_else_in_that_shape_stays_masked(text, number, monkeypatch):
    assert _case(text, number) is None, text
    # ... and not for want of parties or a court: each stays masked when that condition is taken as met
    monkeypatch.setattr(server, "_court_evidence", lambda *_: True)
    assert _case(text, number) is None, text


@pytest.mark.parametrize("text,number", [
    # neither its parties after it nor a court and its seat in the brackets: a bare citation, a court named elsewhere on the line
    ('ע"א 3345/08', "3345/08"), ('ראו ע"א 8954/11 ורע"א 2146/19.', "8954/11"), ('בג"ץ 5658/23', "5658/23"), ('ע"א 3345/08 נדון', "3345/08"),
    ('בבית המשפט: ע"א 3345/08', "3345/08"), ('פסק דין בע"א 3345/08', "3345/08"), ('ע"א 3345/08, פס"ד מיום 3.5.24', "3345/08"), ('בימ"ש מחוזי, ע"א 3345/08', "3345/08"),
    ('הערעור בע"א 3345/08 נדחה', "3345/08"), ('השופט דן בע"א 3345/08', "3345/08"), ('תביעה, תמ"ש 4412-03-22', "4412-03-22"),
    # a נ' that does not follow the number as "versus" does
    ('כהן נ\' לוי, ע"א 3345/08', "3345/08"), ('מין: נ\', ע"א 3345/08', "3345/08"), ('ע"א 3345/08\nכהן נ\' לוי', "3345/08"), ('ע"א 3345/08 נ\' לוי', "3345/08"),
    ('ע"א 3345/08 כהן נ\'', "3345/08"), ('ע"א 3345/08 ' + "מילה " * 6 + 'נ\' לוי', "3345/08"), ('ע"א 3345/08 הנ"ל ולנ\' לוי', "3345/08"), ('ע"א 3345/08. כהן נ\' לוי', "3345/08"),
    # ת"א is Tel Aviv, and not in the list, whatever follows
    (f'ת"א {CASE_NO} כהן נ\' לוי', CASE_NO), (f'בת"א {CASE_NO} כהן נ\' לוי', CASE_NO), (f'ת"א (שלום ת"א) {CASE_NO}', CASE_NO), (f'ת״א {CASE_NO} כהן נ\' לוי', CASE_NO),
    ('ת"א 45127-09-23 נדון בבית משפט השלום בתל אביב.', CASE_NO), (f'ת"א (מחוזי חיפה) {CASE_NO} כהן נ\' לוי', CASE_NO),
    # in the brackets: a kind of court alone, a place alone, a seat with an attached ב, something after the seat
    ('ע"א (עבודה) 3345/08', "3345/08"), ('ע"א (משפחה) 3345/08', "3345/08"), ('ע"א (יפו) 3345/08', "3345/08"), ('ע"א (מחוזי) 3345/08', "3345/08"),
    ('ע"א (חיפה) 3345/08', "3345/08"), ('ע"א (עבודה מהבית) 3345/08', "3345/08"), ('ע"א (דרך השלום חיפה) 3345/08', "3345/08"),
    ('ע"א (עבודה בחיפה) 3345/08', "3345/08"), ('ע"א (משפחה בירושלים) 3345/08', "3345/08"), ('ע"א (נוער מרכז) 3345/08', "3345/08"),
    ('ע"א (עבודה מרכז) 3345/08', "3345/08"), ('ע"א (שלום חיפה שלי) 3345/08', "3345/08"),
])
def test_a_number_with_no_court_on_its_line_stays_masked(text, number):
    assert _case(text, number) is None, text


def test_a_case_number_any_window_doubts_is_not_released(monkeypatch):
    monkeypatch.setattr(server, "IL_COURT_CASE", True)
    text = f'ע"א {CASE_NO} כהן נ\' לוי'
    a = text.index(CASE_NO)
    b = a + len(CASE_NO)
    span, main = (a, b, "ID"), (a, b, "account_number", 0.9, True, {})
    _fake_entities(monkeypatch, lambda _t: [main])
    assert server.scan(text) == ([], [span])
    # reported in pieces: released whole, once
    _fake_entities(monkeypatch, lambda _t: [(a, a + 5, "account_number", 0.9, True, {}), (a + 6, b, "account_number", 0.9, True, {})])
    assert server.scan(text) == ([], [span])
    _fake_entities(monkeypatch, lambda _t: [(a + 6, a + 8, "account_number", 0.9, True, {})])
    assert server.scan(text) == ([], [span])
    # the model called it, or a piece of it, something else: masked as before
    _fake_entities(monkeypatch, lambda _t: [(a, b, "private_phone", 0.9, True, {})])
    assert server.scan(text) == ([(a, b, "PHONE")], [])
    for other in ((a, b, "private_phone", 0.9, False, {}), (a, b, "private_phone", 0.2, False, {}),
                  (a, b, "private_phone", 0.9, False, {"untrusted": True}), (a + 6, b, "private_phone", 0.2, False, {})):
        for order in ((main, other), (other, main)):
            _fake_entities(monkeypatch, lambda _t, order=order: list(order))
            assert server.scan(text) == ([span], []), order
    pieces = [(a, a + 5, "account_number", 0.9, True, {}), (a + 6, b, "private_phone", 0.9, True, {})]
    _fake_entities(monkeypatch, lambda _t: pieces)
    assert server.scan(text) == ([(a, a + 5, "ID"), (a + 6, b, "PHONE")], [])
    _fake_entities(monkeypatch, lambda _t: [pieces[0], (a + 6, b, "private_phone", 0.2, True, {})])
    assert server.scan(text) == ([(a, a + 5, "ID")], [])
    # with the switch off nothing changes
    monkeypatch.setattr(server, "IL_COURT_CASE", False)
    _fake_entities(monkeypatch, lambda _t: [main])
    assert server.scan(text) == ([span], [])


def test_one_case_number_gets_one_answer(monkeypatch):
    monkeypatch.setattr(server, "IL_COURT_CASE", True)
    def every_number(t):
        return [(m.start(), m.end(), "account_number", 0.9, True, {}) for m in __import__("re").finditer(CASE_NO, t)]
    _fake_entities(monkeypatch, every_number)
    case, account = f'ע"א {CASE_NO} כהן נ\' לוי', f"מספר חשבון {CASE_NO}"
    assert server.scan(case)[1] and not server.scan(account)[1]
    for texts in ([account, case], [case, account], [f"{account}\n\n{case}"]):
        assert all(kept == [] for _masked, kept in server.scan_many(texts)), texts
        r = c.post("/beta/litellm_basic_guardrail_api", json={"texts": texts}).json()
        assert r["action"] == "GUARDRAIL_INTERVENED" and not any(CASE_NO in t for t in r["texts"]), r
    assert len(server.scan(f"{case}\n\n{case}")[1]) == 2                         # released in both places: fine
    for other in (CASE_NO.replace("-", ""), CASE_NO.replace("-", " "), CASE_NO.replace("-", "/")):   # written another way elsewhere
        assert server.scan_many([case, f"ראו {other}"])[0][1] == [], other
    _fake_entities(monkeypatch, lambda t: every_number(t)[:1])                   # the model reports only the first of two copies
    assert server.scan(f"{case}\n\nראו גם {CASE_NO}")[1] == []


def test_a_released_case_number_is_logged_without_its_digits(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(server, "IL_COURT_CASE", True)
    with caplog.at_level(logging.INFO, logger="seter_guard"):
        text = f'עת"מ (מחוזי חיפה) {CASE_NO} נדון אתמול.'
        assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [text]}).json() == {"action": "NONE"}
    lines = [r.getMessage() for r in caplog.records if "left unmasked" in r.getMessage()]
    a = text.index(CASE_NO)
    assert len(lines) == 1 and "il_court_case" in lines[0] and f"characters {a}-{a + len(CASE_NO)}" in lines[0]
    assert not any(part in lines[0] for part in ("45127", "09-23"))


# ---- span trimming (fast model + transformers 5.x starts spans one character early) ----

def _stub_ner(monkeypatch, text, pieces, group="private_person"):
    """Stand in for the token-classification pipeline: pieces are (start, end) as the pipeline would report them."""
    monkeypatch.setattr(server, "ner", lambda t, **_k: [
        {"entity_group": group, "score": 0.99, "start": a, "end": b, "word": t[a:b]} for a, b in pieces])


@pytest.mark.parametrize("text,name,group", [
    ("שלום, קוראים לי יוסי כהן היום", "יוסי כהן", "private_person"),
    ("يا صديقي، اسمي سامر الخطيب اليوم", "سامر الخطيب", "private_person"),
    ("My name is John Smith today", "John Smith", "private_person"),
])
@pytest.mark.parametrize("shift_start,shift_end", [(-1, 0), (0, 1), (-1, 1)])
def test_a_span_with_a_stray_space_is_trimmed(monkeypatch, text, name, group, shift_start, shift_end):
    a = text.index(name)
    b = a + len(name)
    _stub_ner(monkeypatch, text, [(a + shift_start, b + shift_end)], group)
    assert text[a - 1] == " "
    assert server.spans(text) == [(a, b, "PERSON")]
    out, found = server.mask(text)
    assert out == text[:a] + "[PERSON]" + text[b:]
    assert found == [{"type": "PERSON", "start": a, "end": b}]


def test_wrapping_punctuation_is_trimmed_but_inner_punctuation_is_kept(monkeypatch):
    text = 'Call "054-7812345", or mail (dana.k@example.com).'
    phone = text.index("054")
    mail = text.index("dana")
    _stub_ner(monkeypatch, text, [(phone - 1, phone + 11), (mail - 1, mail + len("dana.k@example.com") + 1)], "private_phone")
    assert server.spans(text) == [(phone, phone + 11, "PHONE"), (mail, mail + len("dana.k@example.com"), "PHONE")]
    assert server.mask(text)[0] == 'Call "[PHONE]", or mail ([PHONE]).'


def test_a_phone_keeps_its_plus_and_a_span_of_only_punctuation_is_dropped(monkeypatch):
    text = "Reach me on +972-54-7812345 ok"
    a = text.index("+")
    _stub_ner(monkeypatch, text, [(a - 1, a + 15), (len(text) - 3, len(text) - 2)], "private_phone")
    assert server.spans(text) == [(a, a + 15, "PHONE")]
    _stub_ner(monkeypatch, "a - b", [(1, 4)])
    assert server.spans("a - b") == []


def test_a_correct_span_is_left_alone(monkeypatch):
    text = "שלום, קוראים לי יוסי כהן, הטלפון 054-7812345"
    a, p = text.index("יוסי"), text.index("054")
    pieces = [(a, a + 8), (p, p + 11)]
    monkeypatch.setattr(server, "ner", lambda t, **_k: [
        {"entity_group": g, "score": 0.99, "start": x, "end": y} for g, (x, y) in zip(["private_person", "private_phone"], pieces)])
    assert server.entities(text) == [
        {"entity_group": "private_person", "score": 0.99, "start": a, "end": a + 8, "owner": True},
        {"entity_group": "private_phone", "score": 0.99, "start": p, "end": p + 11, "owner": True}]
    assert server.mask(text)[0] == "שלום, קוראים לי [PERSON], הטלפון [PHONE]"


def test_trimmed_neighbours_still_merge_and_digits_still_expand(monkeypatch):
    text = "يا أبو خالد اتصل 0559023417"
    a = text.index("أبو")
    k = text.index("خالد")
    _stub_ner(monkeypatch, text, [(a - 1, k - 1), (k - 1, k + 4)])
    assert server.spans(text) == [(a, k + 4, "PERSON")]
    t2 = "tel 0559023417 now"
    d = t2.index("0559")
    _stub_ner(monkeypatch, t2, [(d - 1, d + 5)], "private_phone")
    assert server.spans(t2) == [(d, d + 10, "PHONE")]


def test_trim_also_applies_to_windowed_texts(monkeypatch):
    text = "x " * 3000 + "Dana Levi"
    wins = server.windows(text)
    assert len(wins) > 1
    a = text.index("Dana")

    def fake(chunks, **_k):
        out = []
        for (wa, wb, *_r), chunk in zip(wins, chunks):
            hit = chunk.find("Dana Levi")
            out.append([{"entity_group": "private_person", "score": 0.99, "start": hit - 1, "end": hit + 9, "word": ""}]
                       if hit >= 0 else [])
        return out
    monkeypatch.setattr(server, "ner", fake)
    assert server.spans(text) == [(a, a + 9, "PERSON")]
