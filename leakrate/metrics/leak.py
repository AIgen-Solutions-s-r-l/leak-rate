"""Leak rate (METHODS.md §5).

A ground-truth mention leaks (strict) if its normalized surface still occurs in the pseudonymized
text; it leaks partially if a discriminating fragment of its canonical value survives.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..schema import Document, PseudonymizedDocument, Entity, DIRECT_CATEGORIES, QUASI_CATEGORIES, ALL_CATEGORIES
from ..pseudonymizers.base import strip_accents, COMPACT_CATEGORIES
from .stats import cluster_bootstrap_rate, RateCI

MONTHS_IT = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto",
             "settembre", "ottobre", "novembre", "dicembre"]


class NormalizedText:
    """Two views of d': word view for names/addresses, compact view for numeric identifiers."""

    def __init__(self, text: str):
        low = strip_accents(text).lower()
        self.words = " " + re.sub(r"[^a-z0-9@]+", " ", low).strip() + " "
        self.compact = re.sub(r"[^a-z0-9]", "", low)
        self.digits = re.sub(r"\D", "", low)
        self.raw = low

    def has_words(self, phrase: str) -> bool:
        p = re.sub(r"[^a-z0-9@]+", " ", strip_accents(phrase).lower()).strip()
        return bool(p) and (" " + p + " ") in self.words

    def has_compact(self, s: str) -> bool:
        c = re.sub(r"[^a-z0-9]", "", strip_accents(s).lower())
        return bool(c) and c in self.compact

    def has_digits(self, s: str) -> bool:
        d = re.sub(r"\D", "", s)
        return bool(d) and d in self.digits


def date_renderings(iso: str) -> list[str]:
    d = date.fromisoformat(iso)
    return [f"{d.day:02d}/{d.month:02d}/{d.year}", f"{d.day}/{d.month}/{d.year}", f"{d.day} {MONTHS_IT[d.month - 1]} {d.year}",
            f"{d.day:02d}.{d.month:02d}.{d.year}", f"{d.day:02d}-{d.month:02d}-{d.year}", f"{d.year}-{d.month:02d}-{d.day:02d}"]


def strict_leak(e: Entity, nt: NormalizedText) -> bool:
    if e.category in COMPACT_CATEGORIES:
        if e.category == "PHONE":
            return nt.has_digits(e.canonical)
        if e.category == "PIVA":
            return nt.has_compact(e.canonical)
        return nt.has_compact(e.surface)
    if e.category == "DATE_BIRTH" or e.category == "DATE":
        return any(nt.has_words(r) for r in date_renderings(e.canonical))
    return nt.has_words(e.surface)


def partial_leak(e: Entity, nt: NormalizedText) -> bool:
    """Discriminating fragment survives (METHODS.md §5). Superset of strict."""
    c = e.canonical
    cat = e.category
    if cat == "PER":
        toks = [t for t in re.split(r"[\s.]+", strip_accents(c).lower()) if len(t) >= 4]
        return any(nt.has_words(t) for t in toks)
    if cat == "CF":
        return nt.has_compact(c[:6]) or nt.has_compact(c[-5:])
    if cat == "IBAN":
        return nt.has_compact(c[-12:])
    if cat == "PIVA":
        return nt.has_compact(c[-7:])
    if cat == "PHONE":
        return nt.has_digits(re.sub(r"\D", "", c)[-7:])
    if cat == "EMAIL":
        local = strip_accents(c.split("@")[0]).lower()
        return len(local) >= 5 and local in nt.raw      # the local part itself survives (not merely the name)
    if cat == "ADDRESS":
        m = re.match(r"^(?:\S+)\s+(.+?)\s+(\d+)$", c)
        if m:
            return nt.has_words(m.group(1)) and nt.has_words(m.group(2))
        return nt.has_words(c)
    if cat == "DOC_ID":
        return nt.has_compact(c[-5:])
    if cat == "PLATE":
        return nt.has_compact(c[2:5]) and (nt.has_compact(c[:2]) or nt.has_compact(c[5:]))
    if cat in ("DATE_BIRTH", "DATE"):
        d = date.fromisoformat(c)
        return any(nt.has_words(r) for r in (f"{d.day:02d}/{d.month:02d}", f"{d.day}/{d.month}", f"{d.day} {MONTHS_IT[d.month - 1]}", str(d.year)))
    return strict_leak(e, nt)


def per_form(e: Entity) -> str:
    """Surface form of a PER mention (for error analysis)."""
    s, c = e.surface, e.canonical
    if s == c:
        return "full"
    if s.isupper():
        return "upper"
    if "." in s:
        return "initial"
    parts = c.split()
    if s == parts[-1] or (len(parts) > 2 and s == " ".join(parts[-2:])):
        return "surname"
    if s == parts[0] or (len(parts) > 2 and s == " ".join(parts[:2])):
        return "first"
    if s.split()[0].isupper():
        return "inverted"
    return "other"


@dataclass
class DocLeak:
    doc_id: str
    domain: str
    per_cat_total: dict = field(default_factory=lambda: defaultdict(int))
    per_cat_strict: dict = field(default_factory=lambda: defaultdict(int))
    per_cat_partial: dict = field(default_factory=lambda: defaultdict(int))
    form_total: dict = field(default_factory=lambda: defaultdict(int))
    form_strict: dict = field(default_factory=lambda: defaultdict(int))
    ent_total: dict = field(default_factory=lambda: defaultdict(int))
    ent_strict: dict = field(default_factory=lambda: defaultdict(int))
    any_direct_strict: bool = False
    any_direct_partial: bool = False
    over_redacted_spans: int = 0
    tokens: int = 0
    leaked_examples: list = field(default_factory=list)
    surviving_quasi: dict = field(default_factory=dict)   # for k-anonymity


def score_document(doc: Document, pd: PseudonymizedDocument, max_examples: int = 3) -> DocLeak:
    nt = NormalizedText(pd.text)
    r = DocLeak(doc.doc_id, doc.domain)
    r.tokens = len(doc.text.split())
    ent_leaked: dict[str, bool] = {}
    for e in doc.entities:
        s = strict_leak(e, nt)
        p = s or partial_leak(e, nt)
        r.per_cat_total[e.category] += 1
        r.per_cat_strict[e.category] += int(s)
        r.per_cat_partial[e.category] += int(p)
        key = (e.category, e.id)
        ent_leaked[key] = ent_leaked.get(key, False) or s
        if e.category == "PER":
            f = per_form(e)
            r.form_total[f] += 1
            r.form_strict[f] += int(s)
        if e.kind == "direct":
            r.any_direct_strict |= s
            r.any_direct_partial |= p
            if s and len(r.leaked_examples) < max_examples:
                r.leaked_examples.append({"category": e.category, "surface": e.surface})
    for (cat, _), leaked in ent_leaked.items():
        r.ent_total[cat] += 1
        r.ent_strict[cat] += int(leaked)
    # over-redaction: replaced spans that do not overlap any gold span
    gold = [(e.start, e.end) for e in doc.entities]
    for s in pd.replaced_spans:
        if not any(s.start < ge and s.end > gs for gs, ge in gold):
            r.over_redacted_spans += 1
    # surviving quasi-identifiers of the data subject (for k-anonymity)
    q = doc.quasi_profile
    surv = {"gender": True}   # gender survives by design (typed pseudonyms) and through agreement anyway
    surv["age"] = nt.has_words(str(q["age"])) or nt.has_words(str(q["birth_year"])) or any(
        strict_leak(e, nt) for e in doc.entities if e.category == "DATE_BIRTH")
    surv["city"] = nt.has_words(q["city"])
    surv["profession"] = any(strict_leak(e, nt) for e in doc.entities if e.category == "PROFESSION")
    surv["employer"] = bool(q.get("employer")) and nt.has_words(q["employer"])
    surv["nationality"] = nt.has_words(q["nationality"])
    surv["marital_status"] = any(nt.has_words(w) for w in (q["marital_status"], q["marital_status"][:-1] + "a"))
    surv["diagnosis"] = bool(q.get("diagnosis")) and nt.has_words(q["diagnosis"])
    r.surviving_quasi = surv
    return r


@dataclass
class LeakReport:
    system: str
    mode: str
    policy: str
    n_docs: int
    per_category: dict           # cat -> {"strict": RateCI, "partial": RateCI, "entity": RateCI}
    doc_leak_rate: RateCI
    doc_leak_rate_partial: RateCI
    direct_mention_leak: RateCI
    over_redaction_per_1k_tokens: float
    examples: list
    by_domain: dict              # domain -> doc_leak_rate RateCI
    per_form: dict = field(default_factory=dict)   # PER surface form -> strict RateCI

    def to_json(self) -> dict:
        def ser(x):
            return x.to_json() if hasattr(x, "to_json") else x
        return {"system": self.system, "mode": self.mode, "policy": self.policy, "n_docs": self.n_docs,
                "per_category": {c: {k: ser(v) for k, v in d.items()} for c, d in self.per_category.items()},
                "doc_leak_rate": ser(self.doc_leak_rate), "doc_leak_rate_partial": ser(self.doc_leak_rate_partial),
                "direct_mention_leak": ser(self.direct_mention_leak),
                "over_redaction_per_1k_tokens": self.over_redaction_per_1k_tokens, "examples": self.examples,
                "by_domain": {d: ser(v) for d, v in self.by_domain.items()},
                "per_form": {f: ser(v) for f, v in self.per_form.items()}}


def evaluate_leak(docs: list[Document], pdocs: list[PseudonymizedDocument], policy: str = "", B: int = 2000, seed: int = 0) -> tuple[LeakReport, list[DocLeak]]:
    by_id = {p.doc_id: p for p in pdocs}
    rows = [score_document(d, by_id[d.doc_id]) for d in docs if d.doc_id in by_id]
    cats = sorted({c for r in rows for c in r.per_cat_total})
    per_cat = {}
    for c in cats:
        tot = np.array([r.per_cat_total[c] for r in rows])
        per_cat[c] = {
            "strict": cluster_bootstrap_rate(np.array([r.per_cat_strict[c] for r in rows]), tot, B, seed),
            "partial": cluster_bootstrap_rate(np.array([r.per_cat_partial[c] for r in rows]), tot, B, seed),
            "entity": cluster_bootstrap_rate(np.array([r.ent_strict[c] for r in rows]), np.array([r.ent_total[c] for r in rows]), B, seed),
            "kind": "direct" if c in DIRECT_CATEGORIES else "quasi",
        }
    ones = np.ones(len(rows))
    doc_rate = cluster_bootstrap_rate(np.array([int(r.any_direct_strict) for r in rows]), ones, B, seed)
    doc_rate_p = cluster_bootstrap_rate(np.array([int(r.any_direct_partial) for r in rows]), ones, B, seed)
    dnum = np.array([sum(v for c, v in r.per_cat_strict.items() if c in DIRECT_CATEGORIES) for r in rows])
    dden = np.array([sum(v for c, v in r.per_cat_total.items() if c in DIRECT_CATEGORIES) for r in rows])
    direct_rate = cluster_bootstrap_rate(dnum, dden, B, seed)
    tokens = sum(r.tokens for r in rows) or 1
    over = 1000 * sum(r.over_redacted_spans for r in rows) / tokens
    examples = [ex | {"doc_id": r.doc_id} for r in rows for ex in r.leaked_examples][:12]
    by_domain = {}
    for dom in sorted({r.domain for r in rows}):
        sub = [r for r in rows if r.domain == dom]
        by_domain[dom] = cluster_bootstrap_rate(np.array([int(r.any_direct_strict) for r in sub]), np.ones(len(sub)), B, seed)
    forms = {}
    for f in sorted({f for r in rows for f in r.form_total}):
        forms[f] = cluster_bootstrap_rate(np.array([r.form_strict[f] for r in rows]), np.array([r.form_total[f] for r in rows]), B, seed)
    sysname = pdocs[0].system if pdocs else "?"
    mode = pdocs[0].mode if pdocs else "?"
    return LeakReport(sysname, mode, policy, len(rows), per_cat, doc_rate, doc_rate_p, direct_rate, over, examples, by_domain, forms), rows
