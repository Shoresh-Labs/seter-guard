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


@pytest.mark.parametrize("probe", [
    "Write to dana.levi@example.co.il about it. ",
    "Order 22091 shipped on 2026-03-15 to the warehouse. ",
    "Please call Dana Levi on 054-7812345 before noon. ",
])
def test_text_near_a_window_boundary_reads_as_it_does_alone(probe):
    # Slide the probe across the first boundary. Whatever the cut does, an email must not turn into a name,
    # a date must not be half-masked, and an entity must be masked exactly once.
    alone = c.post("/v1/redact", json={"text": probe}).json()["text"]
    tail = FILLER * 30
    for pad in range(58, 82, 3):
        head = FILLER * 27 + "Filler " * pad
        got = c.post("/v1/redact", json={"text": head + probe + tail}).json()["text"]
        assert got == head + alone + tail, (pad, got[len(head) - 20:len(head) + 80])


@pytest.mark.parametrize("probe", [
    "מספר תעודת זהות 318845127 נמסר לפקיד. ",
    "يا أبو خالد، ابنك سامر اتصل من 0559023417. ",
    "הנתבע, מר עזרא בן שושן, מתגורר ברחוב הלל יפה 7, חיפה. ",
])
def test_windowed_scan_equals_one_pass_near_a_boundary(probe, monkeypatch):
    # For a model that can read the whole text in one call, cutting it into windows must not change the
    # answer: same spans, same types. (Skipped for models whose limit is one window, such as the large one.)
    limit = server._model_limit()
    if limit < 2000:
        pytest.skip("this model cannot read the test text in one pass")
    tail = FILLER * 30
    differ = []
    for pad in range(58, 82, 3):
        t = FILLER * 27 + "Filler " * pad + probe + tail
        windowed = server.mask(t)
        assert len(server.windows(t)) > 1
        with monkeypatch.context() as m:
            m.setattr(server, "WINDOW", limit)
            assert len(server.windows(t)) == 1
            one_pass = server.mask(t)
        if windowed != one_pass:
            differ.append((pad, windowed[0][len(t) - len(tail) - len(probe) - 10:len(t) - len(tail) + 10], one_pass[1][-3:]))
    assert not differ, differ


def test_concurrent_requests():
    # FastAPI runs these endpoints in a thread pool. A tokenizer shared between the window code and the
    # pipeline raised "Already borrowed" under exactly this load.
    long_t = "Contact Dana Levi at 054-7812345. " + FILLER * 60 + "Send the contract to Yossi Cohen, phone 052-3318874."
    short_t = "Please call Dana Levi on 054-7812345 before noon."
    want = {t: server.mask(t) for t in (long_t, short_t)}

    def one(i):
        t = long_t if i % 2 else short_t
        return server.mask(t) == want[t]

    with ThreadPoolExecutor(4) as ex:
        assert all(ex.map(one, range(24)))


def test_a_text_too_long_to_scan_is_refused_not_passed_through(monkeypatch):
    monkeypatch.setattr(server, "MAX_TOKENS", 600)
    t = FILLER * 80 + "Call Dana Levi on 054-7812345."
    r = c.post("/v1/redact", json={"text": t})
    assert r.status_code == 413 and "054-7812345" not in r.text
    g = c.post("/beta/litellm_basic_guardrail_api", json={"texts": ["fine", t]}).json()
    assert g["action"] == "BLOCKED" and "too long" in g["blocked_reason"].lower()


def test_settings_that_would_skip_text_are_rejected(monkeypatch):
    limit = server._model_limit()
    for env in ({"SETER_WINDOW_TOKENS": str(limit + 1)}, {"SETER_WINDOW_TOKENS": "8"},
                {"SETER_OVERLAP_TOKENS": "0"}, {"SETER_WINDOW_TOKENS": "200", "SETER_OVERLAP_TOKENS": "150"}):
        for k in ("SETER_WINDOW_TOKENS", "SETER_OVERLAP_TOKENS"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        with pytest.raises(ValueError):
            server._settings()
    for k in ("SETER_WINDOW_TOKENS", "SETER_OVERLAP_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    window, overlap, _, _ = server._settings()
    assert window <= limit and 8 <= overlap <= window // 2
