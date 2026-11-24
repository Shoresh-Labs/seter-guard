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
| `shoresh-labs/seter-pii-he-ar-fast@v1.1.1` | the fast model, pinned. This is the default, with or without Docker. |
| `shoresh-labs/seter-pii-he-ar@v1.2` | the larger, more accurate model, pinned (about 2.3× slower on CPU) |
| a repo without `@version` | whatever is newest in that repo; it changes when a new version is published |
| a local folder | the model in that folder |

`repo@` with nothing after it is refused, so a template with an unset version cannot quietly load the newest weights. The version is resolved to a commit once at start and everything is loaded by that commit. For the default, the server also checks that `v1.1.1` still points to the commit it did when this release was made, and refuses to start if the tag was moved. `GET /health` reports `model`, `source` (`hub` or `local`), `pinned`, the `revision` asked for and the `commit` actually loaded.

With Docker the model is downloaded when the image is built, and the container never contacts the Hub afterwards (`HF_HUB_OFFLINE=1`), so a running image cannot change model. To build with another model:

```bash
docker build --build-arg SETER_MODEL=shoresh-labs/seter-pii-he-ar@v1.2 -t seter-guard .
```

The image can load only the exact `SETER_MODEL` it was built with. Setting another value at `docker run` fails at start with a message saying so; this includes the same repo without its `@version`, which means "newest" and needs the Hub. Rebuild, or add `-e HF_HUB_OFFLINE=0` to let the container download. Without Docker: `SETER_MODEL=shoresh-labs/seter-pii-he-ar@v1.2 uvicorn server:app`.

## Long texts

A model reads a limited number of tokens per call (512 for the large model, about 8,000 for the fast one) and the plain pipeline silently ignores the rest. Seter Guard therefore cuts long texts into overlapping windows that start at word boundaries, scans each, and joins the results, so personal data at the end of a long prompt is masked like data at the start.

| Variable | Default | Meaning |
|---|---|---|
| `SETER_WINDOW_TOKENS` | 510 | Tokens per window. Must not exceed what the model reads in one call (the server refuses to start otherwise). The models are trained and scored on windows of this size; the fast model accepts up to about 8,000, which is faster per token of text but is not what the published scores measure. |
| `SETER_OVERLAP_TOKENS` | 96 | Tokens shared by neighbouring windows (64 to half the window). What a window finds right next to a cut is taken from its neighbour instead, so the shared part must be longer than any single entity. |
| `SETER_BATCH_SIZE` | 4 | Windows sent to the model together. |
| `SETER_MAX_TOKENS` | 16000 | Most tokens accepted in one request (all texts of a LiteLLM call together). More is refused before any scanning (`/v1/redact` returns 413; the LiteLLM endpoint returns `BLOCKED`), never passed through unread. On a CPU a window takes on the order of a second, so the default is about 40 windows. |

Texts that fit in one window take a single model call, as before.

**Use fast `v1.1.1`, not `v1.1`, with transformers 4.x.** The two have the same weights. `v1.1` has a `config.json` that transformers 4.x misreads without any warning, which makes the model miss more personal data (recall 0.967 instead of 0.980 on the held-out documents). `v1.1.1` is read correctly by transformers 4.57.6 and 5; `requirements.txt` asks for 4.57.6 or newer, the oldest version this was checked with.

## Israeli identity numbers (optional check)

An Israeli identity number (Teudat Zehut) is nine digits, and the last one is a check digit. The model cannot verify it, and its id class covers identity, passport, licence and bank account numbers alike, so a nine-digit enforcement or proceeding number is masked as one of them. With `SETER_IL_ID_CHECK=1` such a number is left in the text when all of these hold:

- the model labelled it as an id, and in a long text no window labelled it as anything else;
- it is exactly nine digits standing alone: no letter or digit touches it or is joined to it by punctuation, and no other digits follow or precede it across spaces, punctuation or a line break (so a number followed by a date or by a numbered clause stays masked);
- its check digit is wrong;
- the words directly before it, on its line, are `תיק הוצאה לפועל` (or `תיק הוצל"פ`), `הליך` or `אסמכתא`, alone or with an attached ב, ל, ה, מ, ש, כ (also after ו), optionally followed by `מספר` or `מס'`;
- its whole line and the text around it (about 300 characters before and 200 after) have no word that says whose record it is (a patient, an employee, a customer, a member) and, within the paragraph, no word about an identity, passport, licence or account number;
- the same stretch of text has no character that is out of place in Hebrew, Arabic or English prose: an emoji or other pictograph, a zero-width character, a byte-order mark, a direction override. Such characters can hide a word, so nothing is released near one;
- the same digits are not masked, or left unreported, anywhere else in the request, however they are written there (with a hyphen, a space, or without the leading zero).

| Text | With the check on |
|---|---|
| `תיק הוצאה לפועל מספר 034851272 נפתח בלשכת חיפה.` | left as it is |
| `ת.ז. 034851272`, `מספר דרכון 034851272`, `SSN 123456789`, a bare `034851272` | masked |
| `בתיק 034851272`, `תיק רפואי מספר 034851272`, `Tax file number 123456789` | masked |
| `מספר הליך 034851272 (ת.ז. של החייב)`, `החולה דני לוי, מספר הליך 034851272` | masked |
| `מספר התז של החייב בהליך 034851272`, `מספר חבר במועדון, אסמכתא 034851272` | masked |
| `מספר הליך 034851272, ת.ז. של החייב 034851279.` | the check releases neither number |

The rule is narrow on purpose. A wrong check digit alone is not enough: an identity number with one mistyped digit still gives the real one away, and many passport, social security and bank account numbers are nine digits that fail the check. A bare `תיק` is not enough either, because a medical, tax or population-registry file is numbered by the person's identity number, and English `case`, `file` and `reference` are ordinary words, so no English phrase releases a number. An identity word nearby keeps the number masked even when it describes a different number, as in the last row. Working out which number a label belongs to is where earlier versions of this rule went wrong.

The check only decides about numbers the model reported; it never adds a mask. With the default model the model itself does not report the first number in the last row, nor the number in `Case no. 034851272`, with the check on or off. The table was measured with fast `v1.1.1` on transformers 4.57.6.

It is off by default. The rule is word lists (`REFERENCE_BEFORE`, `IDENTITY_CUE` and `PERSON_RECORD` in `server.py`), not understanding: an enforcement number described in other words stays masked, and a personal number that the text introduces only as a proceeding number, with none of the listed words near it, would be released. `GET /health` reports `il_id_check`. Every number left in the text this way is listed under `kept` by `/v1/redact` (reason `il_id_check`) and written to the log on both endpoints (logger `seter_guard`: type and position, never the digits; it goes through your logging configuration, or to stderr with a timestamp when there is none).

## Israeli court case numbers (optional rule)

A court case number names a proceeding, not a person, and judgments print it. The model's id class takes the two ways such a number is written for an account number, so `בג"ץ 5658/23` comes out as `בג"ץ [ID]`. With `SETER_IL_COURT_CASE=1` the number is left in the text when all of these hold:

- the model labelled it as an id, and no window labelled any part of it as anything else;
- it is written as serial-month-year (`45127-09-23`, with a serial of three digits or more: `12-05-85` is a date) or serial/year (`8954/11`), in plain digits, and is not a piece of a longer number;
- the word directly before it, on its line, is the abbreviation of a kind of proceeding (`ע"א`, `בג"ץ`, `תמ"ש` and the rest of the 58 listed in `COURT_CASE_TYPES` in `server.py`), alone or with an attached ו, ב, ל, ה, מ, ש, כ, optionally followed by the court in brackets, as in `ע"א (מחוזי חיפה)`, and by `מספר` or `מס'`;
- something tied to the number itself says it is a court's: its parties follow it, one to five words and then the `נ'` that stands for "versus" (`ע"א 8954/11 כהן נ' לוי`), or the brackets after the abbreviation name a kind of court together with its seat (`ע"א (מחוזי חיפה) 3345/08`). A court named elsewhere on the line does not count, nor does a `נ'` before the number, a kind of court alone in the brackets, or a seat with an attached ב (`עבודה בחיפה` is work, in Haifa);
- no identity, passport, licence, bank, account, branch or medical word, no bank's name (`לאומי`, `דיסקונט`, `Leumi` …), no policy, insurance, loan, mortgage, card, credit, visa, payslip, pension or soldier word, and no word that says whose record it is (a customer, an insured person, an employee) is near it: on its line, or within 300 characters before it or 200 after it, blank lines included (a form's label is often a paragraph of its own);
- where those 300 and 200 characters pass the start or the end of the text, they continue into the neighbouring texts of the request: the label may be the question in the message before the number;
- no character that is out of place in prose is near it (the same test as for identity numbers);
- the same digits are not masked, or left unreported, anywhere else in the request.

| Text | With the rule on |
|---|---|
| `בתמ"ש 4412-03-22 פלונית נ' אלמוני נקבע כי המשמורת תהיה משותפת.` | the number is left |
| `עפ"ג (מחוזי ת"א) 71234-01-20 פלוני נ' מדינת ישראל` | the number is left; the model's other masks stay |
| `בתיק עת"מ (מחוזי חיפה) 45127-09-23, התובע יוסי כהן, טלפון 054-7812345` | the number is left; the name and the phone number are masked |
| `בתיק עת"מ (מחוזי חיפה) 45127-09-23, התובע יוסי כהן, ת"ז 034851279` | masked: an identity number is given near it |
| `ת"א 45127-09-23 נדון בבית משפט השלום בתל אביב.`, `ת"א 45127-09-23 כהן נ' לוי` | masked: `ת"א` is not in the list |
| `פסק הדין בע"א 8954/11` | masked (`[ID]/11`): no parties after the number, no court and seat in the brackets |
| `מין: נ', ע"א 3345/08`, `אחרי הדיון בבית המשפט לקחתי מרשם ד"מ 77120/25` | masked: the same |
| `ע"א (עבודה בחיפה) 3345/08`, `ע"א (משפחה) 3345/08` | masked: not a court and its seat |
| `תיק מספר 45127-09-23`, `תיק אזרחי 45127-09-23` | masked: words and no abbreviation |
| `מספר חשבון 45127-09-23 בבנק לאומי`, `משכנתא שנלקחה בת"א 30412-07-19`, `לאומי ת"א 12345/67` | masked |
| `מספר דרכון`, a blank line, `ע"א 12345/67 כהן נ' לוי` | masked: the label is one paragraph up |
| `תיק ממ"י 30412/07`, `הלוואה ע"א 55120/22`, `תלוש שכר ס"ע 30412/24` | masked: the Land Administration, a loan, a payslip |
| `ע"א 45127-09-23-1 כהן נ' לוי` | masked: not written the way a court writes it |
| `שולם ע"פ 4411/23`, `בבי"ס 4411/23` | masked: not in the list |

**What the rule gives up.** It was reviewed four times with the real model, and each time about twenty invented sentences came out that were masked with the rule off and printed with it on. Each review narrowed it, and what is left is narrow:

- `ת"א`, a civil case and the commonest kind of all, is not in the list. It is also how Tel Aviv is written, and most of what leaked was a number given out, or an account held, in Tel Aviv (`משכנתא שנלקחה בת"א 30412-07-19`, `אני גר מול בית המשפט, רחוב ויצמן 7, ת"א 6473-09-21`).
- A bare citation is not released: in `פסק הדין בע"א 8954/11` no parties follow the number, so what the model reports of it stays masked.
- A case number stays masked whenever the party's identity number, or a word such as `העובד` or `הלקוח`, is within the 300 and 200 characters, which is common in a judgment.

**Known limit.** The rule reads words, not meaning. A number that is not a court case is still released when all of this happens at once: the model reports it as an id, it has the shape of a case number, it is written directly after a listed abbreviation, none of the words above is near it, and it is followed by one to five words and a `נ'`, or preceded by brackets that read as a court and its seat (`קוד כניסה ה"פ (עבודה חיפה) 30412/24`). The rule takes every such `נ'` for "versus", though it is also an initial (`מרשם ד"מ 77120/25 עבור נ' לוי`), female (`מין נ'`), married (`מצב משפחתי נ' + 2`), born (`רחל אברהמי נ' 12.3.1985`) and the word between two teams in a fixture (`הפועל נ' מכבי`). The number in each of these sentences is released. Nobody writes a court abbreviation after "prescription", but the rule cannot know that. A released number can also belong to a proceeding heard behind closed doors, such as a family (`תמ"ש`), guardianship or arrest (`מ"י`) case: the names beside it are still masked, but the number alone points to one person's case. Leave the rule off where a court case number in the model's hands is not worth these risks.

A number the model reported only in part (it sometimes masks `23456-11-1` of `23456-11-19`) is judged and left whole. Like the identity-number check, the rule only decides about numbers the model reported and never adds a mask; it is off by default, `GET /health` reports `il_court_case`, and every number left in the text is listed under `kept` by `/v1/redact` with the reason `il_court_case` and written to the log without its digits. Also left out of the list on purpose: `ב"ל`, `ע"מ`, `בע"מ`, `מ"א`, `ע"ח` (also "on account of") and `ע"ש` (also "in the name of"), which also label a person's or a company's number or an account; and `ע"פ` (also "according to"), `י"ס` (with ב, a school), `ת"ת` (also a Talmud Torah), `ק"ג` (also kilograms) and `ס"ק` (also a sub-clause), which are also everyday words. So a criminal appeal's number (`ע"פ 1234/20`) stays masked. `שת"פ`, `לע"ע`, `ממ"י`, `מת"ק` and `מה"ט`, which an attached letter makes out of a listed abbreviation, are refused by name. The two switches are independent. The table was measured with fast `v1.1.1` on transformers 4.57.6.

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
