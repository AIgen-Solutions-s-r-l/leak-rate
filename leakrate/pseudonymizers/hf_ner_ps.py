"""Supervised multilingual NER from the Hugging Face hub as a local detector (reviewer baseline).

Default model: Babelscape/wikineural-multilingual-ner (mBERT fine-tuned on WikiNEuRal, covers Italian;
labels PER / ORG / LOC / MISC). Runs on CPU; text is split into chunks at line breaks. MISC is ignored.
"""
from __future__ import annotations

import os

from ..schema import Span
from .base import Pseudonymizer
from .regex_ps import RegexPseudonymizer

DEFAULT_MODEL = os.environ.get("LEAKRATE_HF_NER", "Babelscape/wikineural-multilingual-ner")
LABEL_MAP = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}


class HFNERPseudonymizer(Pseudonymizer):
    def __init__(self, with_regex: bool = True, model: str = DEFAULT_MODEL, threshold: float = 0.5, chunk_chars: int = 1200, **kw):
        super().__init__(**kw)
        short = model.split("/")[-1].split("-")[0].lower()
        self.name = f"{short}+regex" if with_regex else short
        self.threshold = threshold
        self.chunk_chars = chunk_chars
        self._regex = RegexPseudonymizer(**kw) if with_regex else None
        from transformers import pipeline
        self.pipe = pipeline("ner", model=model, aggregation_strategy="simple", device=-1)

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
        for off, chunk in self._chunks(text):
            for e in self.pipe(chunk):
                cat = LABEL_MAP.get(e["entity_group"])
                if cat is None or float(e["score"]) < self.threshold:
                    continue
                s, t = off + int(e["start"]), off + int(e["end"])
                while s < t and text[s].isspace():
                    s += 1
                while t > s and text[t - 1].isspace():
                    t -= 1
                if t > s:
                    spans.append(Span(s, t, cat, float(e["score"]), text[s:t]))
        if self._regex is not None:
            spans.extend(self._regex.detect(text))
        return spans
