"""LLM-judged grammatical agreement (complement to the heuristic checker in metrics/agreement.py).

The heuristic checker only sees articles, titles and a lexicon of participles within a short window and
cannot resolve which noun a participle refers to; its error rate is therefore an upper bound with
unknown slack. Here a judge model receives the pseudonymized output and the gold gender of every
person pseudonym and reports, per pseudonym, whether the text contains gendered agreement with it and
whether that agreement is consistent. Error rate = inconsistent / (consistent + inconsistent), cluster
bootstrap over documents. The judge never sees the real names.
"""
from __future__ import annotations

import json
import re

import numpy as np

from ..metrics.stats import cluster_bootstrap_rate
from ..schema import Document, Mapping
from .downstream import CallRecord, FrontierRunner, gold_gender_table, DEFAULT_MODEL

PROMPT = """Sei un revisore linguistico. Nel testo seguente alcune persone sono indicate da segnaposto. Per ogni segnaposto ti è dato il genere reale della persona.

Segnaposto e genere:
{table}

Per ciascun segnaposto, verifica se nel testo ci sono elementi con accordo di genere riferiti a quella persona (articoli e preposizioni articolate: il/la, del/della, al/alla; titoli esplicitamente marcati: sig./sig.ra, dott.ssa, prof.ssa, avvocata; participi passati e aggettivi predicativi; pronomi) e se tale accordo è coerente con il genere indicato. NON considerare come marcati di genere i titoli e i nomi epiceni usati nel registro formale per entrambi i generi: "dott.", "avv.", "ing.", "prof.", "giudice", "notaio", "medico", "testimone", "parte", "paziente", "dipendente", né i nomi di ruolo usati al maschile generico nel registro burocratico ("il gestore", "il titolare", "il responsabile", "il consulente", "il legale rappresentante", "geom.", "rag."), né gli articoli che si accordano con questi nomi anziché con la persona (es. "il giudice dott. PER_f_0003" e "il gestore, geom. PER_f_0004" NON sono errori). Conta come "incoerente" SOLO un elemento che si riferisce direttamente alla persona e ne contraddice il genere in modo inequivocabile: un titolo marcato del genere sbagliato ("il sig. PER_f_0001", "la sig.ra PER_m_0002"), un participio o aggettivo predicativo riferito alla persona con desinenza del genere sbagliato ("PER_f_0001 è stato convocato"), un pronome del genere sbagliato. In caso di dubbio rispondi "nessun_accordo". Rispondi SOLO con un oggetto JSON che associa a ogni segnaposto uno di questi valori: "coerente" (c'è almeno un accordo di genere e tutti sono coerenti), "incoerente" (almeno un accordo contraddice il genere), "nessun_accordo" (nel testo non compare alcun elemento che esprima il genere di quella persona).

Testo:
<<<
{text}
>>>"""


def _first_json_object(text: str) -> dict:
    """Greedy match first (handles nested braces), then the first balanced object if the model appended more text."""
    for pat in (r"\{.*\}", r"\{.*?\}"):
        m = re.search(pat, text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except json.JSONDecodeError:
                continue
    return {}


def judge_agreement_jobs(docs: list[Document], records: list[CallRecord], mappings: dict[tuple[str, str], Mapping],
                         condition: str) -> list[dict]:
    by_doc = {d.doc_id: d for d in docs}
    jobs = []
    for r in records:
        if r.task != "summarize" or r.condition != condition or r.doc_id not in by_doc:
            continue
        mp = mappings.get((r.doc_id, condition))
        if mp is None:
            continue
        table = {k: v for k, v in gold_gender_table(by_doc[r.doc_id], mp).items() if v in ("m", "f") and k in r.output}
        if not table:
            continue
        lines = "\n".join(f"- {k}: {'uomo' if g == 'm' else 'donna'}" for k, g in sorted(table.items()))
        jobs.append(dict(doc_id=r.doc_id, condition=condition, system=r.system, task="agreement_judge", sub="",
                         prompt=PROMPT.format(table=lines, text=r.output), _table=table))
    return jobs


def score_agreement_judge(jobs: list[dict], records: list[CallRecord], B: int = 2000, seed: int = 0) -> dict:
    tables = {j["doc_id"]: j["_table"] for j in jobs}
    per_doc_err, per_doc_ctx = {}, {}
    unparsed = 0
    for r in records:
        if r.task != "agreement_judge":
            continue
        verdict = _first_json_object(r.output)
        if not isinstance(verdict, dict) or not verdict:
            unparsed += 1
            continue
        err = ctx = 0
        for k in tables.get(r.doc_id, {}):
            v = str(verdict.get(k, "")).lower()
            if v.startswith("coer"):
                ctx += 1
            elif v.startswith("incoer"):
                ctx += 1
                err += 1
        per_doc_err[r.doc_id] = err
        per_doc_ctx[r.doc_id] = ctx
    ids = sorted(per_doc_ctx)
    e = np.array([per_doc_err[i] for i in ids], dtype=float)
    c = np.array([per_doc_ctx[i] for i in ids], dtype=float)
    rate = cluster_bootstrap_rate(e, c, B=B, seed=seed).to_json() if ids else None
    total = sum(len(tables.get(i, {})) for i in ids)
    return {"n_docs": len(ids), "pseudonyms_total": total, "share_with_agreement": (float(c.sum()) / total if total else None),
            "pseudonyms_with_agreement": int(c.sum()) if ids else 0,
            "pseudonyms_inconsistent": int(e.sum()) if ids else 0, "error_rate": rate, "unparsed": unparsed}


def run_agreement_judge(docs, records, mappings, conditions: list[str], model: str | None = None, effort: str = "low",
                        workers: int = 8, salt: str = "", B: int = 2000) -> dict:
    runner = FrontierRunner(model=model or DEFAULT_MODEL, effort=effort, workers=workers, salt=salt)
    out = {"judge_model": runner.model, "effort": effort, "by_condition": {}}
    all_jobs: list[dict] = []
    for cond in conditions:
        all_jobs += judge_agreement_jobs(docs, records, mappings, cond)
    jrec = runner.run([{k: v for k, v in j.items() if k != "_table"} for j in all_jobs])
    for cond in conditions:
        cj = [j for j in all_jobs if j["condition"] == cond]
        cr = [r for r in jrec if r.condition == cond]
        out["by_condition"][cond] = score_agreement_judge(cj, cr, B=B)
    out["tokens"] = {"input": sum(r.input_tokens for r in jrec), "output": sum(r.output_tokens for r in jrec)}
    return out, jrec
