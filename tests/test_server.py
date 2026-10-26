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
