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

`SETER_ACTION=block` blocks requests containing personal data instead of masking them. `SETER_MODEL=shoresh-labs/seter-pii-he-ar` uses the larger, more accurate model.

## Accuracy (REDACT benchmark, char-level F1)

| | OpenAI privacy-filter | Seter | Seter Fast |
|---|---|---|---|
| Hebrew | 0.63 | 0.87 | 0.84 |
| Arabic | 0.72 | 0.89 | 0.88 |
| English | 0.70 | 0.84 | 0.83 |

A redaction aid, not an anonymization guarantee. MIT licensed.
