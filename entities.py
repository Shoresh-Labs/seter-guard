"""Seter span decoder: account-number clean-up and entity selection.

Apply this to the output of a Seter PII model. The model card's benchmark figures are measured with it.

Presets
-------
"default"     Account-number clean-up (on by default). Only `account_number` spans are touched: edge
              punctuation is trimmed, full dates and times are dropped, and a number is dropped when the
              nearest word before it names a non-personal code (order, invoice, ticket, reference, case,
              tracking, SKU, version, booking, ... in Hebrew, Arabic and English) and no identity word is
              as close. IBAN, card and passport shapes are always kept.
"strict-ids"  The clean-up, then `account_number` keeps only identity and financial numbers: national /
              identity numbers, passports, driving licences, payment cards (including masked and last-4
              forms), tax ids, bank accounts, IBANs. It also drops customer / client / user / member /
              employee / student / patient ids, medical-file numbers, insurance policy numbers, professional
              licence numbers, reference-style codes and card expiry dates.
"none"        The spans exactly as the model returned them.
Person, address, phone and email spans are never changed by any preset.

Account-number calibration (fast model)
---------------------------------------
The fast model's figures also use an account-number logit offset of 1.0 (its account scores run high).
`calibrate(model)` applies it once to a loaded transformers model; it does nothing for the large model.
For ONNX, subtract ACCOUNT_BIAS from the B-/I-account_number logits before the argmax.

Usage
-----
    from transformers import pipeline
    from seter_entities import calibrate, select_entities
    pii = pipeline("token-classification", model="shoresh-labs/seter-pii-he-ar-fast", aggregation_strategy="simple")
    calibrate(pii.model)
    text = "ת\"ז 039876541, מספר לקוח 448812"
    ents = select_entities(text, pii(text))                         # default: account clean-up, both numbers kept
    ids = select_entities(text, pii(text), preset="strict-ids")     # keeps the ID, drops the customer number

`entities` may be the list of dicts returned by the pipeline (keys start / end / entity_group)
or a list of [start, end, label] spans. The same shape is returned.
MIT License, Shoresh Labs.
"""
import re

__all__ = ["PRESETS", "ACCOUNT_BIAS", "calibrate", "select_entities", "select_spans"]
__version__ = "1.1.0"
PRESETS = ("default", "strict-ids", "none")
# account-number logit offset by model type: fast (ModernBERT / mmBERT) 1.0, large (XLM-R) 0.0
ACCOUNT_BIAS = {"modernbert": 1.0, "xlm-roberta": 0.0}
ACC = "account_number"

# ---- stage 1: account span clean-up (dates, look-alike codes, edge punctuation)
LEAD_P = "=:#([{-.,;'\"/ "
TRAIL_P = ")]}.,;:-'\"/ "
DATE_FULL = re.compile(r"(\d{1,2}[./-]\d{1,2}[./-](\d{2}|(19|20)\d{2})|(19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2}|(0?[1-9]|1[0-2])[./](19|20)\d{2}|(19|20)\d{2}[./-](0?[1-9]|1[0-2])|\d{1,2}:\d{2}(:\d{2})?)")
IBAN = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}")
PASSPORT = re.compile(r"[A-Z]\d{6,9}")
CARDLIKE = re.compile(r"[\d*xX•][\d*xX• -]{11,22}\d")
LOOKALIKE_KW = ["order", "invoice", "ticket", "ref", "reference", "case", "tracking", "sku", "po-", "version", "build", "shipment", "booking", "confirmation", "request", "incident", "flight", "room", "batch", "serial", "model",
                "הזמנה", "חשבונית", "קריאה", "קריאת", "פנייה", "פניה", "אסמכתא", "תיק מס", "תיק בית", "משלוח", "מק\"ט", "מקט", "גרסה", "גרסת", "אצווה", "מספר סידורי", "טיסה", "חדר", "הזמנת", "בקשה מס", "אירוע מס",
                "طلب", "الطلب", "فاتورة", "تذكرة", "التذكرة", "مرجع", "المرجع", "قضية", "القضية", "شحنة", "الشحنة", "رمز المنتج", "الإصدار", "إصدار", "حجز", "رحلة", "غرفة"]
ID_KW = ["id", "identity", "passport", "account", "iban", "card", "licen", "ssn", "social security", "tax", "customer", "client", "employee", "member", "user", "patient", "medical", "file no", "national",
         "ת\"ז", "ת.ז", "תעודת זהות", "מספר זהות", "דרכון", "חשבון", "כרטיס", "רישיון", "רשיון", "עוסק", "ח.פ", "לקוח", "עובד", "מנוי", "חבר", "מטופל", "תיק רפואי", "מ.ר", "ע.מ",
         "الهوية", "هوية", "جواز", "حساب", "الحساب", "آيبان", "الآيبان", "بطاقة", "البطاقة", "رخصة", "ضريبي", "الضريبي", "العميل", "عميل", "الوظيفي", "المدني", "الوطني", "المريض",
         "account ref", "tax ref", "last4", "last 4", "last four", "ending", "ends in", "מסתיים", "מסתיימת", "ספרות אחרונות", "تنتهي", "المنتهية", "ينتهي", "آخر أربع", "policy", "פוליסה", "وثيقة", "mrn", "hospital no", "رقم الملف", "insurance", "ביטוח", "تأمين", "application", "متقدم"]


def _last_end(ctx, kws):
    best = -1
    for k in kws:
        i = ctx.rfind(k)
        if i >= 0:
            best = max(best, i + len(k))
    return best


def _cleanup(text, spans, win=24):
    out = []
    for a, b, ty in spans:
        if ty != ACC:
            out.append([a, b, ty]); continue
        while a < b and text[a] in LEAD_P: a += 1
        while b > a and text[b - 1] in TRAIL_P: b -= 1
        if b <= a: continue
        s = text[a:b]
        if DATE_FULL.fullmatch(s): continue
        protected = IBAN.fullmatch(s.replace(" ", "")) or CARDLIKE.fullmatch(s) or PASSPORT.fullmatch(s)
        if not protected:
            ctx = text[max(0, a - win):a].lower()
            for x, y in (("account reference", "account"), ("tax reference", "tax"), ("account ref", "account"), ("tax ref", "tax")):
                ctx = ctx.replace(x, y)
            n, p = _last_end(ctx, LOOKALIKE_KW), _last_end(ctx, ID_KW)
            if n >= 0 and n > p: continue
        out.append([a, b, ty])
    return out


# ---- stage 2: keep identity and financial numbers only
EXPIRY = re.compile(r"(0?[1-9]|1[0-2])\s?/\s?(\d{2}|20\d{2})")
KEEP_KW = ["national id", "id number", "id no", "identity", "teudat", "passport", "card", "drivinglicence", "driving", "driver", "ssn", "social security", "tax", "bank", "iban", "account", "ending", "ends in", "last 4", "last four", " id ", "(id ", "id:",
           "ת\"ז", "ת.ז", "תעודת זהות", "מספר זהות", "זהות", "דרכון", "כרטיס", "רישיון נהיגה", "רשיון נהיגה", "עוסק", "חשבון", "מסתיים", "ספרות אחרונות",
           "الهوية", "هوية", "الرقم الوطني", "الوطني", "المدني", "جواز", "البطاقة", "بطاقة", "رخصة القيادة", "رخصة", "ضريبي", "الضريبي", "الحساب", "حساب", "آيبان", "الآيبان", "المنتهية", "آخر 4", "آخر أربع"]
DROP_KW = ["customer", "client", "user id", "user no", "member", "employee", "staff", "student", "patient", "medical", "mrn", "file no", "hospital", "policy", "insurance", "licence no", "license no", "reference", "account ref", "ref",
           "expiry", "expires", "exp.", "valid", "application", "droptoken",
           "order", "invoice", "ticket", "case", "claim", "tracking", "txn", "transaction", "request", "session", "batch", "po number", "po-", "sku", "build", "version",
           "error", "line", "qty", "amount", "booking", "shipment", "confirmation", "incident", "serial", "epoch", "timestamp",
           "הזמנה", "חשבונית", "קריאה", "פנייה", "תביעה", "משלוח", "עסקה", "בקשה", "סשן", "אצווה", "רכש", "מק\"ט", "גרסה", "שגיאה", "שורה", "כמות", "סכום", "חותמת",
           "الطلب", "طلب", "الفاتورة", "فاتورة", "التذكرة", "الحالة", "المطالبة", "التتبع", "المعاملة", "الجلسة", "الدفعة", "الشراء", "المنتج", "الإصدار", "الخطأ", "سطر", "الكمية", "المبلغ", "الحجز", "الطابع",
           "לקוח", "עובד", "מנוי", "חבר", "סטודנט", "מטופל", "תיק רפואי", "רפואי", "מ.ר", "פוליסה", "ביטוח", "אסמכתא", "תוקף",
           "العميل", "عميل", "الوظيفي", "الموظف", "العضوية", "عضوية", "الطالب", "المريض", "الملف", "وثيقة", "تأمين", "الترخيص", "مرجع", "المرجع", "الانتهاء", "انتهاء", "صلاحية"]
# context normalisation: "<type>_id" style keys become one non-identity token; driving vs professional licence
NORM = [(re.compile(r"(customer|client|member|membership|employee|staff|student|patient|user|account)[ _\-]?(id|no|num|number|ref|reference|#)"), "droptoken"),
        (re.compile(r"driv\w*[ _\-]?licen[cs]e([ _\-]?(no|num|number))?\.?"), "drivinglicence"),
        (re.compile(r"(?<!driving)licen[cs]e[ _\-]?(no|num|number)"), "licence no"),
        (re.compile(r"_"), " ")]
REFCODE = re.compile(r"[A-Za-z]{3,}[-_]?\d[\dA-Za-z\-_]*")


def _norm(ctx):
    for rx, rep in NORM:
        ctx = rx.sub(rep, ctx)
    return ctx


def _column_header(text, a):
    """In a CSV / TSV / pipe table, the header cell above the span. Used only when the span's row and
    the block's first row have the same number of cells (>= 3) and the header row has no digits."""
    ls = text.rfind("\n", 0, a) + 1
    le = text.find("\n", a)
    le = len(text) if le < 0 else le
    row = text[ls:le]
    for dl in (",", "\t", "|"):
        n = row.count(dl)
        if n < 2: continue
        hs = ls
        while hs > 0:
            prev = text.rfind("\n", 0, hs - 1) + 1
            if text[prev:hs - 1].count(dl) != n: break
            hs = prev
        if hs == ls: return None
        header = text[hs:text.find("\n", hs)]
        if any(ch.isdigit() for ch in header): return None
        col = text[ls:a].count(dl)
        cells = header.split(dl)
        return cells[col].strip().lower() if col < len(cells) else None
    return None


def _ids_only(text, spans, win=32):
    out = []
    for a, b, ty in spans:
        if ty != ACC:
            out.append([a, b, ty]); continue
        s = text[a:b]
        if EXPIRY.fullmatch(s): continue                                   # card expiry is a date
        if IBAN.fullmatch(s.replace(" ", "")) or CARDLIKE.fullmatch(s) or PASSPORT.fullmatch(s):
            out.append([a, b, ty]); continue
        if REFCODE.fullmatch(s): continue                                  # letter-prefixed reference codes
        hdr = _column_header(text, a)
        ctx = _norm(hdr if hdr is not None else text[max(0, a - win):a].lower())
        d, k = _last_end(ctx, DROP_KW), _last_end(ctx, KEEP_KW)
        if d >= 0 and d >= k: continue                                     # nearest cue is a non-identity number type
        out.append([a, b, ty])
    return out


def calibrate(model, account_bias=None):
    """Subtract the account-number logit offset from a loaded transformers token-classification model, once.
    account_bias=None picks it from the model type (ACCOUNT_BIAS). Returns the model."""
    if getattr(model, "_seter_calibrated", False):
        return model
    if account_bias is None:
        account_bias = ACCOUNT_BIAS.get(model.config.model_type, 0.0)
    if account_bias:
        import torch
        ids = [int(i) for i, l in model.config.id2label.items() if str(l).endswith(ACC)]
        with torch.no_grad():   # the token-classification head's final linear layer
            model.classifier.bias[ids] -= float(account_bias)
    model._seter_calibrated = True
    return model


def select_spans(text, spans, preset="default"):
    """spans: iterable of (start, end, label). Returns a new list of [start, end, label]."""
    spans = [[int(a), int(b), ty] for a, b, ty in spans]
    if preset == "none":
        return spans
    if preset in (None, "default"):
        return _cleanup(text, spans)
    if preset == "strict-ids":
        return _ids_only(text, _cleanup(text, spans))
    raise ValueError(f"unknown preset {preset!r}; choose one of {PRESETS}")


def select_entities(text, entities, preset="default"):
    """entities: pipeline output (dicts with start / end / entity_group or entity) or [start, end, label] spans."""
    entities = list(entities)
    if preset == "none":
        return entities
    if not entities or not isinstance(entities[0], dict):
        return select_spans(text, entities, preset)
    def lab(e):
        x = e.get("entity_group") or e.get("entity") or ""
        return x[2:] if x[:2] in ("B-", "I-") else x
    out = []
    for e in entities:
        if lab(e) != ACC:
            out.append(e); continue
        kept = select_spans(text, [[e["start"], e["end"], ACC]], preset)
        if kept:
            a, b, _ = kept[0]
            out.append(dict(e, start=a, end=b, word=text[a:b]))
    return out
