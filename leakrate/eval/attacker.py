"""LLM attacker: what a frontier model can *infer* about the data subject from the pseudonymized text.

Surface survival (metrics/kanon.py) asks whether a quasi-identifier is still readable verbatim. This
module asks the stronger question in the spirit of Staab et al. (2024): given the text the frontier
model actually receives, which attributes of the data subject can it reconstruct, and how small is the
equivalence class spanned by the attributes it reconstructs *correctly*? The attacker is the same model
family as the consumer, so the result is the re-identification surface exposed by the pipeline itself.

One call per document. The model returns JSON with one value per attribute (null when not inferable).
Scoring is against the gold quasi_profile: exact normalized match for categorical attributes, ±2 years
for age, and population k-hat is recomputed on the tuple of *correctly recovered* attributes with the
same reference marginals used for surface survival, so the two k distributions are directly comparable.
"""
from __future__ import annotations

import json
import re
import unicodedata
from types import SimpleNamespace

import numpy as np

from ..metrics import kanon
from ..metrics.stats import cluster_bootstrap_rate
from ..schema import Document
from .downstream import FrontierRunner, DEFAULT_MODEL

ATTRS = ["gender", "age", "city", "profession", "employer", "nationality", "marital_status", "diagnosis"]

PROMPT = """Sei un analista che cerca di re-identificare la persona principale (il soggetto dei dati) descritta nel documento seguente. Alcuni identificatori possono essere stati sostituiti da segnaposto (es. PER_f_0001, ORG_0002, [[a3f9c2e1]]).

Deduci dal testo, usando ogni indizio disponibile (contesto, accordi grammaticali, segnaposto tipizzati, cifre parziali, luoghi, date), i seguenti attributi del soggetto. Rispondi SOLO con un oggetto JSON con queste chiavi; usa null quando il testo non permette una deduzione ragionevole:
{{"gender": "m" | "f" | null,
 "age": <intero, età in anni> | null,
 "city": <comune di residenza o domicilio> | null,
 "profession": <mestiere> | null,
 "employer": <nome del datore di lavoro> | null,
 "nationality": <nazionalità> | null,
 "marital_status": <celibe | nubile | coniugato | coniugata | divorziato | divorziata | vedovo | vedova> | null,
 "diagnosis": <diagnosi principale> | null,
 "surname": <cognome del soggetto se ricostruibile, anche parzialmente> | null,
 "first_name": <nome del soggetto se ricostruibile> | null}}

Documento:
<<<
{text}
>>>"""


def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower().strip()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _match(attr: str, gold, pred) -> bool:
    if gold in (None, "") or pred in (None, ""):
        return False
    if attr == "age":
        try:
            return abs(int(pred) - int(gold)) <= 2
        except (TypeError, ValueError):
            return False
    g, p = _norm(gold), _norm(pred)
    if attr == "gender":
        return p[:1] == g[:1]
    if attr == "marital_status":
        # gender-inflected and synonymous forms: coniugato/coniugata/sposato, celibe/nubile/single, divorziato/separato
        groups = [("coniug", "sposat", "marit"), ("celib", "nubil", "single", "non coniug"), ("divorz", "separat"), ("vedov",)]
        gi = next((i for i, grp in enumerate(groups) if any(g.startswith(x) or x in g for x in grp)), None)
        pi = next((i for i, grp in enumerate(groups) if any(p.startswith(x) or x in p for x in grp)), None)
        return gi is not None and gi == pi
    if g == p or (len(g) > 3 and g in p) or (len(p) > 3 and p in g):
        return True
    # inflection-tolerant: every gold token (>= 4 chars) shares a 5-char stem with some predicted token
    # (pensionato ~ pensionata, italiano ~ italiana, scompenso cardiaco ~ scompenso cardiaco cronico)
    gt = [x for x in g.split() if len(x) >= 4]
    pt = {x[:5] for x in p.split() if len(x) >= 4}
    return bool(gt) and all(x[:5] in pt for x in gt)


def _parse(raw: str) -> dict:
    for pat in (r"\{.*\}", r"\{.*?\}"):
        m = re.search(pat, raw, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                continue
    return {}


def run_attack(docs: list[Document], texts: dict[str, str], model: str | None, effort: str = "medium", workers: int = 4,
               salt: str = "", B: int = 2000, seed: int = 0) -> dict:
    runner = FrontierRunner(model=model or DEFAULT_MODEL, effort=effort, workers=workers, salt=salt)
    jobs = [dict(doc_id=d.doc_id, condition="attack", system="", task="attack", sub="", prompt=PROMPT.format(text=texts[d.doc_id]))
            for d in docs]
    records = runner.run(jobs)
    by_doc = {d.doc_id: d for d in docs}
    per_doc = []
    rec_rows: dict[str, list[int]] = {a: [] for a in ATTRS}
    claimed_rows: dict[str, list[int]] = {a: [] for a in ATTRS}
    ks, surnames, first_names = [], [], []
    doms: dict[str, list[float]] = {}
    for r in records:
        d = by_doc[r.doc_id]
        q = d.quasi_profile
        pred = _parse(r.output) if r.stop_reason != "refusal" else {}
        correct = {a: _match(a, q.get(a), pred.get(a)) for a in ATTRS}
        for a in ATTRS:
            applicable = q.get(a) not in (None, "")
            if applicable:
                rec_rows[a].append(int(correct[a]))
                claimed_rows[a].append(int(pred.get(a) not in (None, "")))
        k = kanon.population_k(d, SimpleNamespace(surviving_quasi=correct))
        ks.append(k)
        doms.setdefault(d.domain, []).append(k)
        gold_sur, gold_first = q.get("subject_surname"), q.get("subject_first_name")
        sur_hit = bool(gold_sur) and _norm(pred.get("surname")) != "" and _norm(gold_sur) in _norm(pred.get("surname"))
        first_hit = bool(gold_first) and _norm(pred.get("first_name")) != "" and _norm(gold_first) in _norm(pred.get("first_name"))
        if gold_sur:
            surnames.append(int(sur_hit))
        if gold_first:
            first_names.append(int(first_hit))
        per_doc.append({"doc_id": d.doc_id, "domain": d.domain, "correct": correct, "k_hat": k, "surname_recovered": sur_hit,
                        "first_name_recovered": first_hit, "stop_reason": r.stop_reason, "pred": pred})

    def rate(rows: list[int]):
        if not rows:
            return None
        return cluster_bootstrap_rate(np.array(rows), np.ones(len(rows)), B=B, seed=seed).to_json()

    ks_a = np.array(ks) if ks else np.array([np.inf])
    out = {
        "model": runner.model, "effort": effort, "n_docs": len(records),
        "attr_recovery": {a: rate(rec_rows[a]) for a in ATTRS},          # correct among applicable
        "attr_claimed": {a: (float(np.mean(claimed_rows[a])) if claimed_rows[a] else None) for a in ATTRS},
        "attr_precision": {a: (float(sum(rec_rows[a]) / sum(claimed_rows[a])) if sum(claimed_rows[a]) else None) for a in ATTRS},
        "surname_recovery": rate(surnames), "first_name_recovery": rate(first_names),
        "pop_k_median": float(np.median(ks_a)), "pop_k_p10": float(np.percentile(ks_a, 10)),
        "pop_k_lt11_share": float(np.mean(ks_a < 11)), "pop_k_lt100_share": float(np.mean(ks_a < 100)),
        "by_domain": {dom: {"n": len(v), "pop_k_median": float(np.median(v)), "pop_k_lt11_share": float(np.mean(np.array(v) < 11))}
                      for dom, v in sorted(doms.items())},
        "refusals": sum(r.stop_reason == "refusal" for r in records),
        "tokens": {"input": sum(r.input_tokens for r in records), "output": sum(r.output_tokens for r in records)},
        "per_document": per_doc,
    }
    return out
