"""Microsoft Presidio with the Italian spaCy pipeline (it_core_news_lg).

Presidio ships Italian recognizers for fiscal code, VAT, identity card, passport and driver licence
plus language-agnostic IBAN / e-mail / phone. NER (PERSON, LOCATION, ORGANIZATION, DATE_TIME) comes
from spaCy. We map Presidio's labels onto the benchmark categories and optionally union the regex
layer (hybrid `presidio+regex`), which is the configuration most deployments actually run.
"""
from __future__ import annotations

from ..schema import Span
from .base import Pseudonymizer
from .regex_ps import RegexPseudonymizer, BIRTH_CONTEXT

PRESIDIO_TO_CATEGORY = {
    "PERSON": "PER", "IT_FISCAL_CODE": "CF", "IBAN_CODE": "IBAN", "IT_VAT_CODE": "PIVA", "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE", "LOCATION": "LOC", "ORGANIZATION": "ORG", "DATE_TIME": "DATE",
    "IT_IDENTITY_CARD": "DOC_ID", "IT_PASSPORT": "DOC_ID", "IT_DRIVER_LICENSE": "DOC_ID",
    "NRP": "LOC", "URL": None, "CREDIT_CARD": None, "CRYPTO": None, "IP_ADDRESS": None, "MEDICAL_LICENSE": None,
    "US_SSN": None, "US_PASSPORT": None, "US_DRIVER_LICENSE": None, "US_BANK_NUMBER": None, "US_ITIN": None,
}


class PresidioPseudonymizer(Pseudonymizer):
    def __init__(self, with_regex: bool = False, score_threshold: float = 0.4, model: str = "it_core_news_lg", **kw):
        super().__init__(**kw)
        self.name = "presidio+regex" if with_regex else "presidio"
        self.with_regex = with_regex
        self.threshold = score_threshold
        self._regex = RegexPseudonymizer(**kw) if with_regex else None
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        provider = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy",
                                                        "models": [{"lang_code": "it", "model_name": model}]})
        self.engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["it"])

    def detect(self, text: str) -> list[Span]:
        results = self.engine.analyze(text=text, language="it", score_threshold=self.threshold)
        spans: list[Span] = []
        for r in results:
            cat = PRESIDIO_TO_CATEGORY.get(r.entity_type)
            if cat is None:
                continue
            if cat == "DATE" and BIRTH_CONTEXT.search(text[max(0, r.start - 60):r.start]):
                cat = "DATE_BIRTH"
            spans.append(Span(r.start, r.end, cat, float(r.score), text[r.start:r.end]))
        if self._regex is not None:
            spans.extend(self._regex.detect(text))
        return spans
