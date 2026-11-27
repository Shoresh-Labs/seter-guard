"""Minimal Seter + Presidio setup.

Uses HuggingFaceNerRecognizer and NoOpNlpEngine. One recognizer is
registered per language (he, ar, en) because HuggingFaceNerRecognizer
accepts a single supported_language; a thin subclass makes them share one
pipeline, so the model is built only once.
"""

import unicodedata

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NoOpNlpEngine
from presidio_analyzer.predefined_recognizers import HuggingFaceNerRecognizer

LABEL_MAPPING = {
    "private_person": "PERSON",
    "private_address": "ADDRESS",  # custom type; Seter addresses are full addresses, not LOCATION
    "private_phone": "PHONE_NUMBER",
    "private_email": "EMAIL_ADDRESS",
    "account_number": "ACCOUNT_NUMBER",
}

MODEL_ID = "shoresh-labs/seter-pii-he-ar"  # or shoresh-labs/seter-pii-he-ar-fast
LANGS = ["he", "ar", "en"]


def _shared_recognizer_class(shared: dict):
    """Build a recognizer class whose load() reuses one pipeline stored in `shared`.

    EntityRecognizer.__init__ always calls self.load(), so assigning
    ner_pipeline after construction would still build a full pipeline per
    recognizer. Overriding load() makes the first call build it and the
    rest reuse it.
    """

    class SharedPipelineRecognizer(HuggingFaceNerRecognizer):
        def load(self):
            if "pipeline" not in shared:
                super().load()
                shared["pipeline"] = self.ner_pipeline
            else:
                # Reusing the shared pipeline skips Presidio's deferred TokenizerBasedTextChunker resolution;
                # that is fine with the default CharacterBasedTextChunker.
                self.ner_pipeline = shared["pipeline"]

    return SharedPipelineRecognizer


def create_analyzer(model_name: str = MODEL_ID) -> AnalyzerEngine:
    recognizer_cls = _shared_recognizer_class({})
    registry = RecognizerRegistry(supported_languages=LANGS)
    for lang in LANGS:
        registry.add_recognizer(
            recognizer_cls(
                model_name=model_name,
                name=f"SeterRecognizer_{lang}",
                supported_language=lang,
                label_mapping=LABEL_MAPPING,
                aggregation_strategy="simple",
            )
        )

    return AnalyzerEngine(
        nlp_engine=NoOpNlpEngine(
            models=[{"lang_code": lang, "model_name": "no_op"} for lang in LANGS]
        ),
        registry=registry,
        supported_languages=LANGS,
    )


def trim_span(text: str, start: int, end: int):
    """Narrow (start, end) so the span neither starts nor ends with whitespace or Unicode punctuation
    (category P), the same rule as _trim() in server.py."""
    def edge(c):
        return c.isspace() or unicodedata.category(c).startswith("P")

    while start < end and edge(text[start]):
        start += 1
    while end > start and edge(text[end - 1]):
        end -= 1
    return start, end


if __name__ == "__main__":
    analyzer = create_analyzer()
    text = "היי, כאן דנה מחיפה, הטלפון שלי 052-1234567"
    for result in analyzer.analyze(text=text, language="he"):
        start, end = trim_span(text, result.start, result.end)
        if start < end:
            print(result.entity_type, result.score, text[start:end])
