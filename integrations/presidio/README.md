# Seter with Presidio

Registers the open Seter models as a [Presidio](https://github.com/data-privacy-stack/presidio) `HuggingFaceNerRecognizer` for Hebrew, Arabic and English. It uses `NoOpNlpEngine`, so no spaCy model is needed. A custom `EntityRecognizer` is not required.

- Models: [seter-pii-he-ar](https://huggingface.co/shoresh-labs/seter-pii-he-ar) (large), [seter-pii-he-ar-fast](https://huggingface.co/shoresh-labs/seter-pii-he-ar-fast) (fast)
- Code: [seter-guard](https://github.com/Shoresh-Labs/seter-guard) (this repo)
- Presidio discussion: [Sample recognizer: Seter models for Hebrew, Arabic and English PII](https://github.com/data-privacy-stack/presidio/discussions/2499)

## Install

```bash
pip install presidio-analyzer transformers torch
```

The models are public; no `HF_TOKEN` is required.

## Usage

```python
from seter_hf_recognizer import create_analyzer

analyzer = create_analyzer()  # shoresh-labs/seter-pii-he-ar

analyzer.analyze(text="היי, כאן דנה מחיפה, הטלפון שלי 052-1234567", language="he")
analyzer.analyze(text="...", language="ar")
analyzer.analyze(text="...", language="en")
```

`HuggingFaceNerRecognizer` accepts a single `supported_language`, so one recognizer is registered per language (`he`, `ar`, `en`). A thin subclass overrides `load()` so all three share a single pipeline built on the first call (Presidio's constructor always calls `load()`, so assigning the pipeline afterwards would still load the model three times). If `language` does not match a registered recognizer, Presidio raises `ValueError: No matching recognizers were found to serve the request.`

Run the bundled sample: `python seter_hf_recognizer.py`.

## Label mapping

| Seter label | Presidio entity |
|---|---|
| `private_person` | `PERSON` |
| `private_address` | `ADDRESS` (custom type; Seter spans are full addresses, not `LOCATION`) |
| `private_phone` | `PHONE_NUMBER` |
| `private_email` | `EMAIL_ADDRESS` |
| `account_number` | `ACCOUNT_NUMBER` |

## Fast vs large

```python
analyzer = create_analyzer("shoresh-labs/seter-pii-he-ar-fast")  # faster on CPU
```

In production, pin a revision (`v1.2` for large, `v1.1.1` for fast). Presidio's `HuggingFaceNerRecognizer` has no `revision` argument and loads the model by name, so this example cannot pin it and loads the newest revision of the repo. To pin, download the model at a fixed revision (for example with `huggingface_hub.snapshot_download(repo_id, revision="v1.2")`) and pass the local folder as `model_name`. The server in this repo pins with `SETER_MODEL=repo@version`.

## Caveat: fast model span offsets

On transformers 5.24.1 the fast model's pipeline often starts a span one character early (a leading space; 36 of 50 rows in our check). The large model was unaffected. Trim leading and trailing whitespace and punctuation on returned spans before using them. See the server's `_trim()` in this repo for a reference implementation.

This is a redaction aid, not an anonymization guarantee.
