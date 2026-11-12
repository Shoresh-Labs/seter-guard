"""Seter Guard — self-hosted PII masking for LLM traffic.

Implements LiteLLM's Generic Guardrail API (POST /beta/litellm_basic_guardrail_api)
plus a plain /v1/redact endpoint. Runs fully on your own machine: no text leaves it.
"""
import os, re
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# SETER_MODEL is a Hub repo, optionally pinned to a version: "repo@v1.1". Without "@" the newest
# published weights are used, which can change under you when a new version is released.
MODEL, _, REVISION = os.environ.get("SETER_MODEL", "shoresh-labs/seter-pii-he-ar-fast").partition("@")
REVISION = REVISION or None
TAGS = {"private_person": "PERSON", "private_address": "ADDRESS", "private_phone": "PHONE",
        "private_email": "EMAIL", "account_number": "ID"}
MIN_SCORE = float(os.environ.get("SETER_MIN_SCORE", "0.5"))
ACTION = os.environ.get("SETER_ACTION", "mask")  # "mask" or "block"

ner = pipeline("token-classification", model=MODEL, revision=REVISION, aggregation_strategy="simple")
app = FastAPI(title="Seter Guard")

# The pipeline reads one window of text per call and silently drops whatever is past the model's limit
# (512 tokens for the large model, about 8,000 for the fast one). A guardrail must read all of it, so long
# texts are scanned in overlapping windows and the results joined.
WINDOW = int(os.environ.get("SETER_WINDOW_TOKENS", "480"))    # tokens per window, below every model's limit
OVERLAP = int(os.environ.get("SETER_OVERLAP_TOKENS", "96"))   # shared by neighbouring windows
assert 0 <= OVERLAP < WINDOW, "SETER_OVERLAP_TOKENS must be smaller than SETER_WINDOW_TOKENS"


def windows(text: str):
    """Character ranges that cover `text`, each at most WINDOW tokens long."""
    offs = ner.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offs) <= WINDOW:
        return [(0, len(text))]
    out, i = [], 0
    while True:
        j = min(i + WINDOW, len(offs))
        out.append((0 if i == 0 else offs[i][0], len(text) if j == len(offs) else offs[j - 1][1]))
        if j == len(offs):
            return out
        i = j - OVERLAP


def entities(text: str):
    """Model output for the whole text, with character offsets into `text`."""
    out = []
    for a, b in windows(text):
        for e in ner(text[a:b]):
            out.append({**e, "start": e["start"] + a, "end": e["end"] + a})
    return out


def spans(text: str):
    out = []
    for e in sorted(entities(text), key=lambda e: (e["start"], e["end"])):
        if e["score"] < MIN_SCORE:
            continue
        a, b, t = e["start"], e["end"], TAGS.get(e["entity_group"], "PII")
        while a < b and text[a].isspace():
            a += 1
        if a >= b:
            continue
        if out and a < out[-1][1]:
            # the same text seen by two windows: keep the union (when in doubt, mask); the first type wins
            out[-1] = (out[-1][0], max(out[-1][1], b), out[-1][2])
        # merge neighbours of the same type separated only by whitespace (e.g. أبو خالد)
        elif out and out[-1][2] == t and text[out[-1][1]:a].strip() == "":
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
    return {"ok": True, "model": MODEL, "revision": REVISION or "main"}
