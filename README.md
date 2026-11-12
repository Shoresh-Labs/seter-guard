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

`SETER_MODEL` is a Hugging Face repo, optionally followed by `@` and a version:

| `SETER_MODEL` | What you get |
|---|---|
| `shoresh-labs/seter-pii-he-ar-fast@v1.1` | the fast model, pinned (the Docker image's default) |
| `shoresh-labs/seter-pii-he-ar@v1.2` | the larger, more accurate model, pinned (about 2.3× slower on CPU) |
| a repo without `@version` | whatever is newest in that repo; it changes when a new version is published |

The Docker image downloads the model when it is built. To use another model, change `ENV SETER_MODEL` in the Dockerfile and rebuild, or run without Docker: `SETER_MODEL=shoresh-labs/seter-pii-he-ar@v1.2 uvicorn server:app`. `GET /health` reports the model and version in use.

## Long texts

Texts longer than one model window are scanned in overlapping windows (480 tokens, 96 shared), so personal data at the end of a long prompt is masked like data at the start. `SETER_WINDOW_TOKENS` and `SETER_OVERLAP_TOKENS` change the sizes.

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
