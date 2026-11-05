"""Pattern-only baseline. Finds structured identifiers; by construction misses PER/ORG/LOC.

Also reused as the structured-identifier layer of the hybrid systems (presidio+regex, gliner+regex).
"""
from __future__ import annotations

import re

from ..schema import Span
from .base import Pseudonymizer
from ..data.italian_ids import validate_cf, validate_iban, validate_piva

MONTHS = "gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre"

PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("CF", re.compile(r"\b[A-Z]{6}\d{2}[A-EHLMPR-T]\d{2}[A-Z]\d{3}[A-Z]\b"), 0.9),
    ("IBAN", re.compile(r"\bIT\s?\d{2}(?:\s?[A-Z0-9]){23}\b"), 0.9),
    ("PIVA", re.compile(r"(?<![\d])(?:IT)?\d{11}(?![\d])"), 0.6),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), 0.95),
    ("PHONE", re.compile(r"(?<![\d])(?:\+39\s?)?(?:3\d{2}[\s.-]?\d{3}[\s.-]?\d{4}|3\d{8,9}|0\d{1,3}[\s/.-]?\d{5,8})(?![\d])"), 0.6),
    ("PLATE", re.compile(r"\b[A-Z]{2}\s?\d{3}\s?[A-Z]{2}\b"), 0.7),
    ("DOC_ID", re.compile(r"\bC[A-Z]\d{5}[A-Z]{2}\b|\bY[A-Z]\d{7}\b"), 0.8),
    ("ADDRESS", re.compile(r"\b(?:Via|Viale|Piazza|Piazzale|Corso|Largo|Vicolo|Strada|Contrada)\s+(?:[A-ZÀ-Ü][\w'.À-ü]*\s?|d[aeio]l?l?[aeio']?\s|dei\s|delle\s|degli\s|[IVX]+\s)+\d{1,4}\b"), 0.7),
    ("DATE", re.compile(rf"\b\d{{1,2}}[/.-]\d{{1,2}}[/.-](?:19|20)\d{{2}}\b|\b\d{{1,2}}\s(?:{MONTHS})\s(?:19|20)\d{{2}}\b", re.I), 0.7),
]
BIRTH_CONTEXT = re.compile(r"(nat[oa]\s+(?:a\s+[^,]{2,40}\s+)?il|nat[oa]\s+il|data\s+di\s+nascita\s*:?)\s*$", re.I)


class RegexPseudonymizer(Pseudonymizer):
    name = "regex"

    def detect(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for cat, pat, score in PATTERNS:
            for m in pat.finditer(text):
                s = m.group(0)
                if cat == "CF" and not validate_cf(s):
                    continue
                if cat == "IBAN" and not validate_iban(s):
                    continue
                if cat == "PIVA":
                    digits = s[2:] if s.startswith("IT") else s
                    if not validate_piva(digits):
                        continue
                if cat == "DATE" and BIRTH_CONTEXT.search(text[max(0, m.start() - 60):m.start()]):
                    cat_out = "DATE_BIRTH"
                else:
                    cat_out = cat
                spans.append(Span(m.start(), m.end(), cat_out, score, s))
        return spans
