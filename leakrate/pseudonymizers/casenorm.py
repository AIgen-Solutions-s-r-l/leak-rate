"""Case normalization in front of a NER detector (reviewer baseline).

Italian institutional prose writes names as ``MARIO ROSSI`` and ``ROSSI Mario``; NER models trained on
news text miss both. Before detection we (i) rewrite ``COGNOME Nome`` as ``Nome Cognome`` and
(ii) title-case all-caps words of three or more letters (acronyms excluded), keeping a character-level
map back to the original text so that detected spans are projected onto the original offsets. The regex
layer always runs on the original text. Only the detector's input changes; the pseudonymized document
is built from the original.
"""
from __future__ import annotations

import os
import re

from ..schema import Span
from .base import Pseudonymizer
from .regex_ps import RegexPseudonymizer

INVERTED = re.compile(r"\b([A-ZÀ-Ý]{3,}(?:['’][A-ZÀ-Ý]{2,})?)\s+([A-ZÀ-Ý][a-zà-ÿ]{2,})\b")
ALLCAPS = re.compile(r"\b[A-ZÀ-Ý]{3,}\b")
ACRONYMS = {"IBAN", "PEC", "CCNL", "INPS", "INAIL", "SRL", "SPA", "SNC", "SAS", "IVA", "RAL", "ASL", "TAC", "ICD", "DPR",
            "DPCM", "GDPR", "CAP", "SEPA", "BIC", "CIG", "CUP", "RSA", "PIVA", "USL", "ATS", "UOC", "UOS", "PS", "CF",
            "TFR", "HIV", "ECG", "RM", "RX", "PET", "BPCO", "DSM", "ART", "LGS", "CPC", "CPP", "REA", "SIAE", "CNR"}


def normalize(text: str) -> tuple[str, list[int]]:
    """Return (normalized text, back) with back[i] = index in the original text of normalized char i."""
    out: list[str] = []
    back: list[int] = []
    pos = 0
    for m in INVERTED.finditer(text):
        a_start, a_end = m.span(1)
        b_start, b_end = m.span(2)
        if m.group(1) in ACRONYMS:
            continue
        for i in range(pos, a_start):
            out.append(text[i]); back.append(i)
        first = text[b_start:b_end]
        last = text[a_start:a_end].capitalize()
        for k, ch in enumerate(first):
            out.append(ch); back.append(b_start + k)
        out.append(" "); back.append(a_end)   # the separating whitespace
        for k, ch in enumerate(last):
            out.append(ch); back.append(a_start + k)
        pos = b_end
    for i in range(pos, len(text)):
        out.append(text[i]); back.append(i)
    norm = "".join(out)
    # title-case remaining all-caps words (same length, map unchanged)
    def _tc(m: re.Match) -> str:
        w = m.group(0)
        return w if w in ACRONYMS else w.capitalize()
    norm = ALLCAPS.sub(_tc, norm)
    assert len(norm) == len(back)
    return norm, back


def project(spans: list[Span], back: list[int], original: str) -> list[Span]:
    """Project spans found on the normalized text onto the original; split spans whose original chars are not contiguous."""
    out: list[Span] = []
    for s in spans:
        idx = back[s.start:s.end]
        if not idx:
            continue
        run_start = idx[0]; prev = idx[0]
        for j in idx[1:] + [None]:
            if j is None or j != prev + 1:
                out.append(Span(run_start, prev + 1, s.category, s.score, original[run_start:prev + 1]))
                if j is not None:
                    run_start = j
            if j is not None:
                prev = j
    return [s for s in out if s.text.strip() and len(s.text.strip()) >= 2]


class CaseNormPseudonymizer(Pseudonymizer):
    def __init__(self, inner: Pseudonymizer, with_regex: bool = True, **kw):
        super().__init__(**kw)
        self.inner = inner
        base = inner.name.replace("+regex", "")
        self.name = f"{base}+casenorm" + ("+regex" if with_regex else "")
        self._regex = RegexPseudonymizer(**kw) if with_regex else None

    def detect(self, text: str) -> list[Span]:
        norm, back = normalize(text)
        spans = project(self.inner.detect(norm), back, text)
        if self._regex is not None:
            spans.extend(self._regex.detect(text))
        return spans
