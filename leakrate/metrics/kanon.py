"""Re-identification risk from surviving quasi-identifiers (METHODS.md §6).

* corpus k: size of the equivalence class of the surviving quasi-identifier tuple within the
  released corpus (classic k-anonymity, the corpus as population);
* population k̂: estimated number of Italian residents sharing the tuple, from reference marginals
  under independence (optimistic upper bound, illustrative marginals — see data/reference.py);
* l-diversity of the diagnosis within corpus equivalence classes (clinical domain).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

import numpy as np

from ..schema import Document
from ..data import reference as R
from .leak import DocLeak


@dataclass
class KAnonReport:
    n_docs: int
    corpus_k_median: float
    corpus_k_p10: float
    corpus_k_unique_share: float          # share of documents with corpus k == 1
    pop_k_median: float
    pop_k_p10: float
    pop_k_lt5_share: float
    pop_k_lt11_share: float
    pop_k_lt100_share: float
    attr_survival: dict                    # attribute -> share of docs where it survived
    by_domain: dict
    l_diversity_clinical: dict | None

    def to_json(self) -> dict:
        return asdict(self)


def surviving_tuple(doc: Document, row: DocLeak) -> tuple:
    q, s = doc.quasi_profile, row.surviving_quasi
    return (
        q["gender"] if s.get("gender") else None,
        q["age"] if s.get("age") else None,
        q["city"] if s.get("city") else None,
        q["profession"] if s.get("profession") else None,
        q.get("employer") if s.get("employer") else None,
        q["nationality"] if s.get("nationality") else None,
        q["marital_status"] if s.get("marital_status") else None,
        q.get("diagnosis") if s.get("diagnosis") else None,
    )


def population_k(doc: Document, row: DocLeak) -> float:
    q, s = doc.quasi_profile, row.surviving_quasi
    if s.get("employer") and q.get("employer"):
        base = float(R.EMPLOYER_SIZE)
    elif s.get("city"):
        base = float(R.CITY_POPULATION.get(q["city"]) or q.get("city_population") or R.ITALY_POPULATION)
    else:
        base = float(R.ITALY_POPULATION)
    k = base
    if s.get("gender"):
        k *= R.GENDER_SHARE.get(q["gender"], 0.5)
    if s.get("age"):
        k *= R.share_of_age(q.get("age"))
    if s.get("profession") and not s.get("employer"):
        k *= R.PROFESSION_SHARE.get(q["profession"]) or q.get("profession_share") or 0.01
    if s.get("nationality"):
        k *= R.NATIONALITY_SHARE.get(q["nationality"], 0.002)
    if s.get("marital_status"):
        k *= R.MARITAL_SHARE.get(q["marital_status"], 0.1)
    if s.get("diagnosis") and q.get("diagnosis"):
        k *= R.DIAGNOSIS_SHARE.get(q["diagnosis"]) or q.get("diagnosis_share") or 0.01
    return max(k, 1.0)


def evaluate_kanon(docs: list[Document], rows: list[DocLeak]) -> KAnonReport:
    by_id = {r.doc_id: r for r in rows}
    pairs = [(d, by_id[d.doc_id]) for d in docs if d.doc_id in by_id]
    tuples = [surviving_tuple(d, r) for d, r in pairs]
    counts = Counter(tuples)
    corpus_k = np.array([counts[t] for t in tuples], dtype=float)
    pop_k = np.array([population_k(d, r) for d, r in pairs])
    attrs = defaultdict(int)
    for _, r in pairs:
        for a, v in r.surviving_quasi.items():
            attrs[a] += int(bool(v))
    n = len(pairs) or 1
    by_domain = {}
    for dom in sorted({d.domain for d, _ in pairs}):
        idx = [i for i, (d, _) in enumerate(pairs) if d.domain == dom]
        by_domain[dom] = {"pop_k_median": float(np.median(pop_k[idx])), "pop_k_lt11_share": float((pop_k[idx] < 11).mean()),
                          "corpus_k_unique_share": float((corpus_k[idx] == 1).mean()), "n": len(idx)}
    ldiv = None
    clin = [(t, d.quasi_profile.get("diagnosis")) for (d, _), t in zip(pairs, tuples) if d.domain == "clinical"]
    if clin:
        classes = defaultdict(set)
        for t, dx in clin:
            classes[t[:-1]].add(dx)       # equivalence class on the non-sensitive attributes
        ls = np.array([len(classes[t[:-1]]) for t, _ in clin], dtype=float)
        ldiv = {"l_median": float(np.median(ls)), "l1_share": float((ls == 1).mean()), "n": len(clin)}
    return KAnonReport(n_docs=len(pairs), corpus_k_median=float(np.median(corpus_k)), corpus_k_p10=float(np.quantile(corpus_k, 0.1)),
                       corpus_k_unique_share=float((corpus_k == 1).mean()), pop_k_median=float(np.median(pop_k)),
                       pop_k_p10=float(np.quantile(pop_k, 0.1)), pop_k_lt5_share=float((pop_k < 5).mean()),
                       pop_k_lt11_share=float((pop_k < 11).mean()), pop_k_lt100_share=float((pop_k < 100).mean()),
                       attr_survival={a: v / n for a, v in sorted(attrs.items())}, by_domain=by_domain, l_diversity_clinical=ldiv)
