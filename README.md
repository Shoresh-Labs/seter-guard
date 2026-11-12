# Seter Guard

Self-hosted PII masking for LLM traffic — **Hebrew, Arabic and English**, including mixed text.
Wraps the open [Seter models](https://huggingface.co/shoresh-labs/seter-pii-he-ar-fast) in a small API and plugs into **LiteLLM** as a guardrail, so names, IDs, phones, emails and addresses are masked before prompts reach OpenAI/Anthropic/Gemini. Nothing leaves your servers.

Try the model in your browser: [seter-demo](https://huggingface.co/spaces/yinonov/seter-demo)

## Run

```bash
docker build -t seter-guard .
docker run -p 8080:8080 seter-guard
curl -s localhost:8080/v1/redact -H 'content-type: application/json' \
  -d '{"text":"שלום, קוראים לי יוסי כהן, 054-7812345"}'
# {"text":"שלום, קוראים לי [PERSON], [PHONE]", "entities":[...]}
```

### Hugging Face token (optional)

The Seter models are public, so no token is needed. To use a private or gated model, or to avoid anonymous rate limits, set `HF_TOKEN`:

```bash
# build: passed as a secret, never stored in the image
HF_TOKEN=hf_... docker build --secret id=hf_token,env=HF_TOKEN -t seter-guard .
# local run without Docker: transformers picks up HF_TOKEN from the environment
HF_TOKEN=hf_... uvicorn server:app
```

## Use with LiteLLM

LiteLLM's [Generic Guardrail API](https://docs.litellm.ai/docs/adding_provider/generic_guardrail_api) calls `POST /beta/litellm_basic_guardrail_api`, which this server implements. See `litellm_config.yaml`:

```yaml
guardrails:
  - guardrail_name: seter-pii
    litellm_params:
      guardrail: generic_guardrail_api
      mode: pre_call
      api_base: http://localhost:8080
```

`SETER_ACTION=block` blocks requests containing personal data instead of masking them. `SETER_MODEL=shoresh-labs/seter-pii-he-ar` uses the larger, more accurate **v1.1** model. The default image still loads `seter-pii-he-ar-fast`, which is the **v1.0** (25 Oct 2026) fast checkpoint.

## Long texts

A model reads a limited number of tokens per call (512 for the large model, about 8,000 for the fast one) and the plain pipeline silently ignores the rest. Seter Guard therefore cuts long texts into overlapping windows that start at word boundaries, scans each, and joins the results, so personal data at the end of a long prompt is masked like data at the start.

| Variable | Default | Meaning |
|---|---|---|
| `SETER_WINDOW_TOKENS` | 510 | Tokens per window. Must not exceed what the model reads in one call (the server refuses to start otherwise). The models are trained and scored on windows of this size; the fast model accepts up to about 8,000, which is faster per token of text but is not what the published scores measure. |
| `SETER_OVERLAP_TOKENS` | 96 | Tokens shared by neighbouring windows (8 to half the window). What a window finds right next to a cut is taken from its neighbour instead. |
| `SETER_BATCH_SIZE` | 4 | Windows sent to the model together. |
| `SETER_MAX_TOKENS` | 50000 | Longest text accepted. A longer one is refused (`/v1/redact` returns 413; the LiteLLM endpoint returns `BLOCKED`), never passed through unread. |

Texts that fit in one window take a single model call, as before.

## Accuracy (REDACT benchmark, char-level F1)

| | OpenAI privacy-filter | Seter v1.1 large, mean of 3 seeds (released) | Seter Fast (v1.0) |
|---|---|---|---|
| Hebrew | 0.63 | 0.900 (0.904) | 0.84 |
| Arabic | 0.72 | 0.912 (0.917) | 0.88 |
| English | 0.70 | 0.871 (0.878) | 0.83 |

A redaction aid, not an anonymization guarantee. MIT licensed. Numbers and caveats: [seter-pii-he-ar model card](https://huggingface.co/shoresh-labs/seter-pii-he-ar).
