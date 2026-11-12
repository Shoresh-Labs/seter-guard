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

`SETER_ACTION=block` blocks requests containing personal data instead of masking them.

## Choosing and pinning the model

`SETER_MODEL` is a Hugging Face repo followed by `@` and a version, or a local folder:

| `SETER_MODEL` | What you get |
|---|---|
| `shoresh-labs/seter-pii-he-ar-fast@v1.1` | the fast model, pinned. This is the default, with or without Docker. |
| `shoresh-labs/seter-pii-he-ar@v1.2` | the larger, more accurate model, pinned (about 2.3× slower on CPU) |
| a repo without `@version` | whatever is newest in that repo; it changes when a new version is published |
| a local folder | the model in that folder |

`repo@` with nothing after it is refused, so a template with an unset version cannot quietly load the newest weights. `GET /health` reports the model, the version asked for, and the commit that was actually loaded.

With Docker the model is downloaded when the image is built, and the container never contacts the Hub afterwards (`HF_HUB_OFFLINE=1`), so a running image cannot change model. To build with another model:

```bash
docker build --build-arg SETER_MODEL=shoresh-labs/seter-pii-he-ar@v1.2 -t seter-guard .
```

Setting `SETER_MODEL` only at `docker run` to a model that is not in the image fails at start; rebuild, or add `-e HF_HUB_OFFLINE=0` to let the container download it. Without Docker: `SETER_MODEL=shoresh-labs/seter-pii-he-ar@v1.2 uvicorn server:app`.

## Long texts

A model reads a limited number of tokens per call (512 for the large model, about 8,000 for the fast one) and the plain pipeline silently ignores the rest. Seter Guard therefore cuts long texts into overlapping windows that start at word boundaries, scans each, and joins the results, so personal data at the end of a long prompt is masked like data at the start.

| Variable | Default | Meaning |
|---|---|---|
| `SETER_WINDOW_TOKENS` | 510 | Tokens per window. Must not exceed what the model reads in one call (the server refuses to start otherwise). The models are trained and scored on windows of this size; the fast model accepts up to about 8,000, which is faster per token of text but is not what the published scores measure. |
| `SETER_OVERLAP_TOKENS` | 96 | Tokens shared by neighbouring windows (8 to half the window). What a window finds right next to a cut is taken from its neighbour instead. |
| `SETER_BATCH_SIZE` | 4 | Windows sent to the model together. |
| `SETER_MAX_TOKENS` | 50000 | Longest text accepted. A longer one is refused (`/v1/redact` returns 413; the LiteLLM endpoint returns `BLOCKED`), never passed through unread. |

Texts that fit in one window take a single model call, as before.

## Accuracy

Character-level F1 on the Hebrew, Arabic and English records of the REDACT benchmark:

| | OpenAI privacy-filter | Seter large v1.2, mean of 3 training seeds (released weights) | Seter Fast v1.1, mean of 3 training seeds (released weights) |
|---|---|---|---|
| Hebrew | 0.642 | 0.915 (0.914) | 0.905 (0.908) |
| Arabic | 0.735 | 0.922 (0.925) | 0.913 (0.914) |
| English | 0.709 | 0.890 (0.890) | 0.885 (0.886) |

On 150 held-out documents written for evaluation (support tickets, chats, emails, forms), released weights: large v1.2 F1 0.968 (precision 0.962, recall 0.975); Fast v1.1 F1 0.947 (precision 0.916, recall 0.980).

These figures were measured with the models' own post-processing; this server uses the plain pipeline output, which can differ slightly. Earlier versions of this table (0.63 / 0.72 / 0.70 for OpenAI privacy-filter, 0.84 / 0.88 / 0.83 for Fast v1.0) were computed with a scoring bug that understated every system by 0.01-0.02.

A redaction aid, not an anonymization guarantee. MIT licensed. Numbers and caveats: [seter-pii-he-ar](https://huggingface.co/shoresh-labs/seter-pii-he-ar) and [seter-pii-he-ar-fast](https://huggingface.co/shoresh-labs/seter-pii-he-ar-fast) model cards.
