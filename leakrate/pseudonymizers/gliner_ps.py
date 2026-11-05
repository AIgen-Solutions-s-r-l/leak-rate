"""GLiNER zero-shot NER (Zaratiana et al., NAACL 2024) prompted with Italian labels.

Requires `pip install gliner torch`. Structured identifiers are better handled by the regex layer,
so the recommended configuration is the hybrid `gliner+regex`.
"""
from __future__ import annotations

import os

from ..schema import Span
from .base import Pseudonymizer
from .regex_ps import RegexPseudonymizer, BIRTH_CONTEXT

LABELS_IT = {
    "nome di persona": "PER", "organizzazione": "ORG", "azienda": "ORG", "città": "LOC", "luogo": "LOC",
    "data": "DATE", "data di nascita": "DATE_BIRTH", "indirizzo": "ADDRESS", "numero di telefono": "PHONE",
    "indirizzo email": "EMAIL", "codice fiscale": "CF", "IBAN": "IBAN", "partita IVA": "PIVA",
    "targa": "PLATE", "numero documento di identità": "DOC_ID", "professione": "PROFESSION",
    "malattia o diagnosi": "HEALTH", "farmaco": "HEALTH", "età": "AGE",
}


# English labels: gliner_multi was trained on English label strings, so English labels are the stronger prompt
LABELS_EN = {
    "person": "PER", "person name": "PER", "organization": "ORG", "company": "ORG", "city": "LOC", "location": "LOC",
    "date": "DATE", "date of birth": "DATE_BIRTH", "street address": "ADDRESS", "phone number": "PHONE",
    "email address": "EMAIL", "italian fiscal code": "CF", "IBAN": "IBAN", "VAT number": "PIVA",
    "license plate": "PLATE", "identity document number": "DOC_ID", "profession": "PROFESSION",
    "disease or diagnosis": "HEALTH", "medication": "HEALTH", "age": "AGE",
}


class GlinerPseudonymizer(Pseudonymizer):
    def __init__(self, with_regex: bool = True, model: str = "urchade/gliner_multi-v2.1",
                 threshold: float = float(os.environ.get("LEAKRATE_GLINER_THRESHOLD", "0.4")),
                 labels: str = os.environ.get("LEAKRATE_GLINER_LABELS", "it"), chunk_chars: int = 1500, **kw):
        super().__init__(**kw)
        self.labels_map = LABELS_IT if labels == "it" else LABELS_EN
        suffix = "" if (labels == "it" and abs(threshold - 0.4) < 1e-9) else f"[{labels},{threshold:g}]"
        self.name = ("gliner+regex" if with_regex else "gliner") + suffix
        self.threshold = threshold
        self.chunk_chars = chunk_chars
        self._regex = RegexPseudonymizer(**kw) if with_regex else None
        try:
            from gliner import GLiNER
        except ImportError as e:  # pragma: no cover
            raise ImportError("GLiNER adapter needs `pip install gliner torch`") from e
        self.model = GLiNER.from_pretrained(model)

    def _chunks(self, text: str):
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_chars)
            if end < len(text):
                cut = text.rfind("\n", start, end)
                if cut > start + 200:
                    end = cut
            yield start, text[start:end]
            start = end

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        labels = list(self.labels_map.keys())
        for offset, chunk in self._chunks(text):
            for e in self.model.predict_entities(chunk, labels, threshold=self.threshold):
                cat = self.labels_map[e["label"]]
                s, t = offset + e["start"], offset + e["end"]
                if cat == "DATE" and BIRTH_CONTEXT.search(text[max(0, s - 60):s]):
                    cat = "DATE_BIRTH"
                spans.append(Span(s, t, cat, float(e["score"]), text[s:t]))
        if self._regex is not None:
            spans.extend(self._regex.detect(text))
        return spans
