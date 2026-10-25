"""Seter Guard — self-hosted PII masking for LLM traffic.

Implements LiteLLM's Generic Guardrail API (POST /beta/litellm_basic_guardrail_api)
plus a plain /v1/redact endpoint. Runs fully on your own machine: no text leaves it.
"""
import os, re
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL = os.environ.get("SETER_MODEL", "shoresh-labs/seter-pii-he-ar-fast")
TAGS = {"private_person": "PERSON", "private_address": "ADDRESS", "private_phone": "PHONE",
        "private_email": "EMAIL", "account_number": "ID"}
MIN_SCORE = float(os.environ.get("SETER_MIN_SCORE", "0.5"))
ACTION = os.environ.get("SETER_ACTION", "mask")  # "mask" or "block"

ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")
app = FastAPI(title="Seter Guard")


def spans(text: str):
    out = []
    for e in sorted(ner(text), key=lambda e: e["start"]):
        if e["score"] < MIN_SCORE:
            continue
        a, b, t = e["start"], e["end"], TAGS.get(e["entity_group"], "PII")
        while a < b and text[a].isspace():
            a += 1
        # merge neighbours of the same type separated only by whitespace (e.g. أبو خالد)
        if out and out[-1][2] == t and text[out[-1][1]:a].strip() == "":
            out[-1] = (out[-1][0], b, t)
        else:
            out.append((a, b, t))
    return out


def mask(text: str):
    res, pos, found = [], 0, []
    for a, b, t in spans(text):
        res += [text[pos:a], f"[{t}]"]
        found.append({"type": t, "start": a, "end": b})
        pos = b
    res.append(text[pos:])
    return "".join(res), found


class RedactIn(BaseModel):
    text: str


@app.post("/v1/redact")
def redact(body: RedactIn):
    masked, found = mask(body.text)
    return {"text": masked, "entities": found}


class GuardIn(BaseModel):
    texts: list[str] = []
    input_type: str | None = None
    structured_messages: list[dict] | None = None
    additional_provider_specific_params: dict | None = None


@app.post("/beta/litellm_basic_guardrail_api")
def litellm_guardrail(body: GuardIn):
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
    return {"ok": True, "model": MODEL}
