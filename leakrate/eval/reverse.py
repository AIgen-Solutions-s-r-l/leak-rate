"""Reverse mapping M⁻¹ applied to frontier-model output (METHODS.md §4).

Tolerant to the mutations models actually produce: lower-casing, lost or doubled underscores,
trailing punctuation, possessives ("PER_f_0001's"), markdown emphasis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import Mapping

TYPED_LOOSE = re.compile(r"(?<![A-Za-z0-9])(PER|ORG|LOC|CF|IBAN|PIVA|EMAIL|PHONE|ADDRESS|DOC_?ID|PLATE|DATE_?BIRTH|DATE|AGE|PROFESSION|HEALTH)[ _-]?(?:([mfx])[ _-]?)?(\d{3,4})(?![0-9])", re.I)
OPAQUE = re.compile(r"\[\[\s*([0-9a-f]{8})\s*\]\]", re.I)


@dataclass
class ReverseResult:
    text: str
    replaced: int
    unresolved: list[str]       # pseudonym-looking tokens with no mapping entry
    residual_pseudonyms: int    # after replacement, how many pseudonym-looking tokens remain


def _canon(tok: str) -> str:
    m = TYPED_LOOSE.fullmatch(tok)
    if not m:
        return tok
    cat = m.group(1).upper().replace("DOCID", "DOC_ID").replace("DATEBIRTH", "DATE_BIRTH")
    g = (m.group(2) or "").lower()
    num = m.group(3).zfill(4)
    return f"{cat}_{g}_{num}" if cat == "PER" and g else f"{cat}_{num}"


def reverse_map(text: str, mapping: Mapping) -> ReverseResult:
    table = {e.pseudonym: e.canonical for e in mapping.entries}
    unresolved: list[str] = []
    count = 0

    def sub_typed(m: re.Match) -> str:
        nonlocal count
        key = _canon(m.group(0))
        if key in table:
            count += 1
            return table[key]
        unresolved.append(m.group(0))
        return m.group(0)

    def sub_opaque(m: re.Match) -> str:
        nonlocal count
        key = f"[[{m.group(1).lower()}]]"
        if key in table:
            count += 1
            return table[key]
        unresolved.append(m.group(0))
        return m.group(0)

    if mapping.mode == "typed":
        out = TYPED_LOOSE.sub(sub_typed, text)
    else:
        out = OPAQUE.sub(sub_opaque, text)
    residual = len(TYPED_LOOSE.findall(out)) if mapping.mode == "typed" else len(OPAQUE.findall(out))
    return ReverseResult(out, count, unresolved, residual)
