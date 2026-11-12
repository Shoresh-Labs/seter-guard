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
