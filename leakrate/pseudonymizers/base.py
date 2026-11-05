"""Common pseudonymization machinery (METHODS.md §4).

detect() is system-specific. Everything after detection is shared so that systems differ only in
*what they find*, never in how pseudonyms are assigned:

1. resolve overlapping spans (longest wins, then higher score);
2. drop categories outside the replacement policy;
3. link spans into entities (coreference by normalized surface; PER also by surname subset);
4. assign typed pseudonyms PER_f_0001 / ORG_0002 … or opaque [[a3f9c2e1]];
5. rewrite the text and emit the mapping.
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from abc import ABC, abstractmethod

from ..schema import Span, Mapping, MappingEntry, PseudonymizedDocument, DIRECT_CATEGORIES, ALL_CATEGORIES
from ..data.lexicons import gender_of_first_name

POLICIES = {
    "direct_only": set(DIRECT_CATEGORIES),
    "direct+org+loc": set(DIRECT_CATEGORIES) | {"ORG", "LOC"},
    "all": set(ALL_CATEGORIES),
}
DEFAULT_REPLACE_POLICY = "direct+org+loc"

COMPACT_CATEGORIES = {"CF", "IBAN", "PIVA", "PHONE", "PLATE", "DOC_ID"}
TITLE_F = re.compile(r"(sig\.ra|sig\.na|dott\.ssa|prof\.ssa|signora|dottoressa|avvocatessa|la|alla|della|dalla|nella)\s*$", re.I)
TITLE_M = re.compile(r"(sig\.|dott\.|prof\.|signor|dottor|il|al|del|dal|nel)\s*$", re.I)


def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def norm_key(category: str, surface: str) -> str:
    s = strip_accents(surface).lower()
    if category in COMPACT_CATEGORIES:
        s = re.sub(r"[^a-z0-9]", "", s)
        if category == "PHONE":
            s = re.sub(r"\D", "", s)[-9:]
        if category == "PIVA" and s.startswith("it"):
            s = s[2:]
        return s
    s = re.sub(r"[^a-z0-9@.]+", " ", s).strip()
    return s


def _per_tokens(key: str) -> set[str]:
    return {t for t in re.split(r"[ .]+", key) if len(t) >= 2}


class Pseudonymizer(ABC):
    name = "abstract"

    def __init__(self, policy: str = DEFAULT_REPLACE_POLICY, **_):
        self.policy = policy
        self.replace_categories = POLICIES[policy]

    @abstractmethod
    def detect(self, text: str) -> list[Span]:
        ...

    # ------------------------------------------------------------------ pipeline
    def pseudonymize(self, text: str, doc_id: str, mode: str = "typed") -> PseudonymizedDocument:
        t0 = time.perf_counter()
        spans = self.detect(text)
        spans = resolve_overlaps(spans)
        spans = [s for s in spans if s.category in self.replace_categories]
        for s in spans:
            s.text = text[s.start:s.end]
        new_text, mapping, replaced = rewrite(text, spans, doc_id, mode)
        return PseudonymizedDocument(doc_id=doc_id, system=self.name, mode=mode, text=new_text, mapping=mapping,
                                     replaced_spans=replaced, seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------- shared steps

def _priority(s: Span) -> tuple:
    # On overlap: a direct-identifier detection beats a quasi-identifier one (an IBAN mis-read as an
    # ORGANIZATION must still be removed under a direct-only policy), then longer, then higher score.
    return (s.category in DIRECT_CATEGORIES, s.end - s.start, s.score)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    spans = sorted((s for s in spans if s.end > s.start), key=lambda s: (s.start, tuple(-x for x in _priority(s)[1:])))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            if _priority(s) > _priority(out[-1]):
                out[-1] = s
            continue
        out.append(s)
    return out


def infer_gender(text: str, span: Span) -> str | None:
    surface = text[span.start:span.end]
    toks = [t for t in re.split(r"\s+", surface.strip()) if t]
    # "FERRARO Giulia" (inverted): first name is the first non-uppercase token
    cands = [t for t in toks if not (t.isupper() and len(t) > 2)] or toks
    # 1) lexicon match on any token (handles "Giulia Ferraro", "FERRARO Giulia", "Giulia")
    for t in cands:
        g = gender_of_first_name(t.strip(".,"), lexicon_only=True)
        if g:
            return g
    # 2) title / article immediately before the mention ("alla sig.ra Greco", "il dott. Rossi")
    left = text[max(0, span.start - 12):span.start]
    if TITLE_F.search(left):
        return "f"
    if TITLE_M.search(left):
        return "m"
    # 3) ending heuristic, only for multi-token mentions whose first token looks like a first name
    if len(cands) >= 2:
        return gender_of_first_name(cands[0].strip(".,"))
    return None


def link_entities(text: str, spans: list[Span]) -> list[list[Span]]:
    """Group spans referring to the same entity. Returns groups in order of first appearance."""
    groups: list[dict] = []
    for s in spans:
        key = norm_key(s.category, text[s.start:s.end])
        toks = _per_tokens(key) if s.category == "PER" else None
        target = None
        for g in groups:
            if g["category"] != s.category:
                continue
            if key in g["keys"]:
                target = g
                break
            if s.category == "PER" and toks:
                for gt in g["tokensets"]:
                    shared = toks & gt
                    if shared and (toks <= gt or gt <= toks) and max(len(t) for t in shared) >= 3:
                        target = g
                        break
                if target:
                    break
        if target is None:
            target = {"category": s.category, "keys": set(), "tokensets": [], "spans": []}
            groups.append(target)
        target["keys"].add(key)
        if toks:
            target["tokensets"].append(toks)
        target["spans"].append(s)
    return [g["spans"] for g in groups]


def opaque_token(doc_id: str, category: str, canonical: str, idx: int) -> str:
    h = hashlib.sha256(f"{doc_id}|{category}|{canonical}|{idx}".encode()).hexdigest()[:8]
    return f"[[{h}]]"


def rewrite(text: str, spans: list[Span], doc_id: str, mode: str) -> tuple[str, Mapping, list[Span]]:
    groups = link_entities(text, spans)
    counters: dict[str, int] = {}
    entries: list[MappingEntry] = []
    token_for_span: dict[int, str] = {}
    for gi, group in enumerate(groups):
        cat = group[0].category
        surfaces = sorted({text[s.start:s.end] for s in group}, key=len, reverse=True)
        canonical = surfaces[0]
        attrs = {}
        if cat == "PER":
            g = None
            for s in group:
                g = infer_gender(text, s)
                if g:
                    break
            attrs["gender"] = g or "x"
        counters[cat] = counters.get(cat, 0) + 1
        if mode == "typed":
            pseud = f"{cat}_{attrs['gender']}_{counters[cat]:04d}" if cat == "PER" else f"{cat}_{counters[cat]:04d}"
        else:
            pseud = opaque_token(doc_id, cat, canonical, gi)
        entries.append(MappingEntry(pseudonym=pseud, category=cat, canonical=canonical, surfaces=surfaces, attrs=attrs))
        for s in group:
            token_for_span[id(s)] = pseud
    out = []
    last = 0
    replaced = []
    for s in sorted(spans, key=lambda s: s.start):
        out.append(text[last:s.start])
        out.append(token_for_span[id(s)])
        last = s.end
        replaced.append(s)
    out.append(text[last:])
    return "".join(out), Mapping(doc_id=doc_id, mode=mode, entries=entries), replaced


# ---------------------------------------------------------------------- reference systems

class IdentityPseudonymizer(Pseudonymizer):
    """Detects nothing. Leak rate must be 100 %: the sanity floor."""
    name = "identity"

    def detect(self, text: str) -> list[Span]:
        return []


class OraclePseudonymizer(Pseudonymizer):
    """Uses the gold annotations. Leak rate must be 0 % for replaced categories: the sanity ceiling."""
    name = "oracle"

    def __init__(self, gold_by_doc: dict | None = None, **kw):
        super().__init__(**kw)
        self.gold_by_doc = gold_by_doc or {}
        self._current = None

    def pseudonymize(self, text: str, doc_id: str, mode: str = "typed") -> PseudonymizedDocument:
        self._current = self.gold_by_doc.get(doc_id, [])
        return super().pseudonymize(text, doc_id, mode)

    def detect(self, text: str) -> list[Span]:
        return [Span(e.start, e.end, e.category, 1.0) for e in (self._current or [])]
