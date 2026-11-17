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
    assert server.spans(text) == [(3, 13, "ID")]
    assert server.mask(text)[0] == "ID [ID]) on file"


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
FILE_TEXT = f"תיק הוצאה לפועל מספר {FILE_NO} נפתח בלשכת חיפה."
ID_TEXT = f"מספר תעודת הזהות שלו הוא {VALID_ID}."


def redact(text):
    return c.post("/v1/redact", json={"text": text}).json()


def test_il_id_check_digit():
    assert server.il_id_check(VALID_ID) is True and server.il_id_check(FILE_NO) is False
    assert server.il_id_check("000000018") is True and server.il_id_check("123456782") is True
    wrong = [VALID_ID[:8] + str(d) for d in range(10) if str(d) != VALID_ID[8]]
    assert all(server.il_id_check(n) is False for n in wrong)               # exactly one check digit fits
    for other in ("03485127", "0348512790", "03485127-9", "٠٣٤٨٥١٢٧٩", "03485127a", ""):
        assert server.il_id_check(other) is None                           # not nine plain digits: no opinion


def test_stands_alone():
    for text in (f"מספר {FILE_NO} נפתח", FILE_NO, f"מספר {FILE_NO}.", f"({FILE_NO})", f"מספר: {FILE_NO}, ועוד", f"{FILE_NO}. 5 ימים"):
        a = text.index(FILE_NO)
        assert server._stands_alone(text, a, a + 9), text
    for joined in (f"12-{FILE_NO}", f"{FILE_NO}-55", f"A{FILE_NO}", f"{FILE_NO}/21", f"1{FILE_NO}", f"{FILE_NO}.5", f"{FILE_NO}\\21",
                   f"{FILE_NO}–21", f"{FILE_NO}־21", f"{FILE_NO}:21", f"+{FILE_NO}1", f"21,{FILE_NO}", f"{FILE_NO}_21", f"ב{FILE_NO}",
                   f"{FILE_NO} 21", f"{FILE_NO} 0123", f"21 {FILE_NO}", f"{FILE_NO}-A", f"A-{FILE_NO}"):
        text = f"מספר {joined} נפתח"
        a = text.index(FILE_NO)
        assert not server._stands_alone(text, a, a + 9), text


def _called(text, number=FILE_NO):
    a = text.index(number)
    return server._called_a_reference(text, a, a + 9)


@pytest.mark.parametrize("text", [
    FILE_TEXT, f"מספר הליך {FILE_NO}, ת.ז. של החייב {VALID_ID}.", f"ת.ז. של החייב {VALID_ID}, מספר הליך {FILE_NO}",
    f"בתיק הוצאה לפועל מס' {FILE_NO} נקבע", f"תיק הוצל\"פ {FILE_NO}", f"אסמכתא: {FILE_NO}", f"בהליך {FILE_NO} נדחתה הבקשה",
    f"לאור העובדות שבתיק ההוצאה לפועל {FILE_NO}", f"חשבונית מס, אסמכתא {FILE_NO}",   # עובדות, חשבונית: not identity words
])
def test_a_number_named_as_an_enforcement_or_proceeding_number_can_be_released(text):
    assert _called(text)


SSN = "123456789"


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
    # whose record it is, anywhere before the number on the line, even with other numbers between
    f"מטופל דני לוי, נולד 01-02-1980, מספר הליך {FILE_NO}", f"מטופל דני לוי, טל' 054-1234567, אסמכתא {FILE_NO}", f"לקוח 0541234567 הליך {FILE_NO}",
    f"החולה דני לוי, מספר הליך {FILE_NO}", f"העובדת רות כהן, טל' 054-1234567, אסמכתא: {FILE_NO}",
    # named as an identity, passport, licence or account number, near or far, in any form
    f"SSN {FILE_NO}", f"ת.ז. {FILE_NO}", f"ת\"ז {FILE_NO}", f"file ID {FILE_NO}", f"מספר דרכון {FILE_NO}", f"case 12, I.D. {FILE_NO}",
    f"מספר תעודת הזהות של החייב שנגדו נפתח ההליך {FILE_NO}", f"מספר תעודת הזהות של החייב שנגדו נפתח בלשכת חיפה תיק הוצאה לפועל מספר {FILE_NO}",
    f"Passport number of the debtor named in the enforcement case opened in Haifa: {FILE_NO}",
    f"צוין כי מספר דרכונו בהליך {FILE_NO}", f"רישיונו נשלל, אסמכתא {FILE_NO}", f"מספר זיהוי, הליך {FILE_NO}", f"מספר מזהה של החייב בהליך {FILE_NO}",
    f"עוסק מורשה, אסמכתא {FILE_NO}", f"ביטוח לאומי, אסמכתא {FILE_NO}", f"מספר אישי, אסמכתא {FILE_NO}", f"חשבונו בבנק, אסמכתא {FILE_NO}",
    '{"file": "a.pdf", "user_id": "%s"}' % FILE_NO, f"Reference check for Dana Levi, national number {FILE_NO}",
    # the label comes after the number
    f"מספר הליך {FILE_NO} (ת.ז. של החייב)", f"מספר הליך {FILE_NO} (ת.ז. של החייב), טלפון 0541234567", f"מספר הליך {FILE_NO} (his passport)",
    f"מספר הליך {FILE_NO} - מספר הזהות של הנתבע", f"מספר הליך {FILE_NO}, ת.ז. של החייב", f"מספר הליך {FILE_NO}, שהוא מספר הזהות של החייב.",
    f"מספר הליך {FILE_NO}, which is his passport number", f"מספר הליך {FILE_NO} (Mr. Smith's passport)", f"מספר הליך {FILE_NO} (ת. ז. של החייב)",
    f"מספר הליך {FILE_NO} (מ.ז. של החייב)", f"אסמכתא {FILE_NO} (מ.א. של החייל)", f"אסמכתא {FILE_NO} (ח\"ן בבנק)", f"מספר הליך {FILE_NO} של המטופל",
])
def test_everything_else_stays_masked(text):
    assert not _called(text, SSN if SSN in text else FILE_NO)


def test_a_long_run_of_spaces_before_the_number_costs_nothing():
    import time
    text = "הליך" + " " * 15000 + f"ועוד מספר הליך {FILE_NO}"
    t0 = time.time()
    assert _called(text) and not _called("הליך" + " " * 15000 + FILE_NO)
    assert time.time() - t0 < 0.5


def test_il_id_check_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SETER_IL_ID_CHECK", raising=False)
    assert server._flag("SETER_IL_ID_CHECK") is False
    assert server.IL_ID_CHECK is False and c.get("/health").json()["il_id_check"] is False   # conftest clears the variable
    for text, number in ((FILE_TEXT, FILE_NO), (ID_TEXT, VALID_ID)):
        r = redact(text)
        assert number not in r["text"] and r["kept"] == []


def test_il_id_check_releases_a_file_number_and_nothing_else(monkeypatch):
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    assert c.get("/health").json()["il_id_check"] is True
    r = redact(FILE_TEXT)
    a = FILE_TEXT.index(FILE_NO)
    assert r["text"] == FILE_TEXT and r["entities"] == []
    assert r["kept"] == [{"type": "ID", "start": a, "end": a + 9, "reason": "il_id_check"}]   # reported, not silent
    r = redact(ID_TEXT)
    assert VALID_ID not in r["text"] and "[ID]" in r["text"] and r["kept"] == []
    r = redact(f"מספר הליך {FILE_NO}, ת.ז. של החייב {VALID_ID}.")["text"]
    assert FILE_NO in r and VALID_ID not in r
    # a wrong check digit alone releases nothing: an identity number with a typo, a passport, a bank account,
    # a US social security number, a number with no description, a piece of a longer code
    for text in (f"ת.ז. {FILE_NO}", f"מספר תעודת הזהות שלו הוא {FILE_NO}.", f"מספר דרכון {FILE_NO}",
                 f"בנק 12 סניף 345 חשבון {FILE_NO}", "SSN 123456789 on file for Dana Levi",
                 f"מספר תעודת הזהות שלו {FILE_NO[:8]}-{FILE_NO[8]}", f"תיק הוצאה לפועל מספר {FILE_NO}\\21 נפתח",
                 f"תיק רפואי מספר {FILE_NO}", f"מספר הליך {FILE_NO} (ת.ז. של החייב)", f"בתיק {FILE_NO} נקבע דיון",
                 "Tax file number 123456789", "The number we have on file: 123456789"):
        got = redact(text)
        digits = "123456789" if "123456789" in text else FILE_NO[:8]
        assert digits not in got["text"] and got["kept"] == [], got
    # the LiteLLM endpoint follows: nothing left to mask means no intervention
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [FILE_TEXT]}).json() == {"action": "NONE"}
    assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [ID_TEXT]}).json()["action"] == "GUARDRAIL_INTERVENED"


def test_a_number_two_windows_disagree_about_is_not_released(monkeypatch):
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    text = f"מספר הליך {FILE_NO} נדחה"
    a = text.index(FILE_NO)
    span = (a, a + 9, "ID")
    def fake(*reports):
        return lambda _t: [{"start": a, "end": a + 9, "entity_group": k, "score": sc, "owner": o, **extra} for k, sc, o, extra in reports]
    monkeypatch.setattr(server, "entities", fake(("account_number", 0.9, True, {})))
    assert server.scan(text) == ([], [span])                                     # one opinion: released
    second = [("private_phone", 0.9, False, {}),                                 # another window calls it a phone number
              ("private_phone", 0.2, False, {}),                                 # ... with a score too low to mask on
              ("private_phone", 0.9, False, {"untrusted": True})]                # ... from the strip right next to a cut
    for other in second:
        for order in ((("account_number", 0.9, True, {}), other), (other, ("account_number", 0.9, True, {}))):
            monkeypatch.setattr(server, "entities", fake(*order))
            assert server.scan(text) == ([span], []), order
    # an untrusted report alone masks nothing, as before
    monkeypatch.setattr(server, "entities", fake(("private_phone", 0.9, False, {"untrusted": True})))
    assert server.scan(text) == ([], [])


def test_a_released_number_is_logged_without_its_digits(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(server, "IL_ID_CHECK", True)
    server.log.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="seter_guard"):
            assert c.post("/beta/litellm_basic_guardrail_api", json={"texts": [FILE_TEXT]}).json() == {"action": "NONE"}
            c.post("/beta/litellm_basic_guardrail_api", json={"texts": [ID_TEXT]})
    finally:
        server.log.removeHandler(caplog.handler)
    lines = [r.getMessage() for r in caplog.records if "il_id_check" in r.getMessage()]
    a = FILE_TEXT.index(FILE_NO)
    assert len(lines) == 1 and f"characters {a}-{a + 9}" in lines[0] and FILE_NO not in lines[0]


def test_flag_values(monkeypatch):
    for value, want in (("1", True), ("true", True), ("ON", True), ("0", False), ("", False), ("off", False)):
        monkeypatch.setenv("SETER_IL_ID_CHECK", value)
        assert server._flag("SETER_IL_ID_CHECK") is want
    monkeypatch.setenv("SETER_IL_ID_CHECK", "maybe")
    with pytest.raises(ValueError):
        server._flag("SETER_IL_ID_CHECK")
    assert server._flag("SETER_IL_ID_CHECK", strict=False) is False             # the lenient form used for HF_HUB_OFFLINE
