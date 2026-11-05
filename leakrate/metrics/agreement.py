"""Grammatical-agreement errors around pseudonyms in Italian model output (METHODS.md §8).

For each pseudonym occurrence in y' (before reverse mapping) we collect agreement contexts:
the preceding article/title and the first participle / gendered adjective within 6 tokens after it,
and compare their gender with the ground-truth gender of the referent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

import numpy as np

from .stats import cluster_bootstrap_rate, RateCI

TYPED_RE = re.compile(r"\bPER_([mfx])_(\d{4})\b")
OPAQUE_RE = re.compile(r"\[\[([0-9a-f]{8})\]\]")

ART_F = {"la", "una", "alla", "della", "dalla", "nella", "sulla", "l'", "sig.ra", "sig.na", "signora", "dott.ssa", "dottoressa", "prof.ssa", "avvocatessa", "ricorrente", "lavoratrice", "dipendente"}
ART_M = {"il", "lo", "un", "uno", "al", "allo", "del", "dello", "dal", "dallo", "nel", "nello", "sul", "sullo", "sig.", "signor", "signore", "dottor", "dottore", "professor", "lavoratore", "avvocato"}
# words that are gender-neutral in -e are excluded from both sets (dipendente, ricorrente are kept
# only in ART_F when preceded by "la"; handled by requiring the article to be the decisive cue)
# epicene titles and nouns of the formal register: no gender cue by themselves ("il giudice dott. PER_f_0003" is not an error)
NEUTRAL = {"ricorrente", "dipendente", "ing.", "avv.", "rag.", "geom.", "dr.", "dott.", "prof.", "arch.", "paziente", "cliente",
           "giudice", "notaio", "medico", "testimone", "parte", "controparte", "convenuto", "legale", "difensore", "titolare"}
AUX = {"è", "era", "fu", "sarà", "viene", "veniva", "verrà", "risulta", "risultava", "essendo", "non", "già", "poi", "quindi", "anche"}
STOP_AFTER = {"il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "del", "della", "dei", "delle", "al", "alla", "dal", "dalla",
              "nel", "nella", "con", "in", "a", "da", "per", "di", "che", "e", "ed", "o", "su", "tra", "fra", "presso", "contro"}

PARTICIPLE_RE = re.compile(r"^(?:[a-zà-ü]+?)(at|ut|it|os|iv|an|on|ier|ic|al|ar|es|ut)([oaie])$")
PART_STEMS = ("convocat", "assunt", "nat", "residen", "licenziat", "dimess", "ricoverat", "condannat", "trasferit", "informat",
              "domiciliat", "rappresentat", "difes", "assentat", "citat", "obbligat", "tenut", "stat", "giunt", "sottopost",
              "visitat", "valutat", "inquadrat", "contestat", "sospes", "reintegrat", "risarcit", "indicat", "iscritt", "coniugat",
              "divorziat", "separat", "vedov", "celib", "nubil", "pensionat", "disoccupat", "impiegat", "operai", "infermier",
              "stess", "medesim", "sudett", "suddett", "predett", "quest", "quell", "prim", "second", "ultim", "nuov", "vecchi", "titolar")


def _gender_of_word(w: str) -> str | None:
    w = w.lower().strip(",;:()\"'")
    if w in NEUTRAL:
        return None
    if w in ART_F:
        return "f"
    if w in ART_M:
        return "m"
    for stem in PART_STEMS:
        if w.startswith(stem) and len(w) == len(stem) + 1:
            e = w[-1]
            if e in ("o", "i"):
                return "m"
            if e in ("a", "e") and stem not in ("residen", "titolar"):
                return "f"
    return None


@dataclass
class AgreementResult:
    contexts: int
    errors: int
    rate: RateCI
    unknown_gender_pseudonyms: int

    def to_json(self) -> dict:
        return {"contexts": self.contexts, "errors": self.errors, "rate": self.rate.to_json(), "unknown_gender_pseudonyms": self.unknown_gender_pseudonyms}


def agreement_contexts(text: str, pseudonym_gender: dict[str, str]) -> tuple[int, int, int]:
    """Return (contexts, errors, occurrences with unknown gold gender) for one output text.

    Before-cue: the token immediately before the pseudonym when it is an article, articulated preposition
    or gender-marked title (``la``, ``dalla``, ``sig.ra``); an epicene title (``dott.``, ``avv.``) or a bare
    preposition gives no cue. After-cue: only when the pseudonym is in subject position (sentence start, or
    preceded by an article / gender-marked title), the first participle or predicative adjective within
    5 tokens, skipping auxiliaries and stopping at any new noun phrase or punctuation.
    """
    toks = [t for t in text.replace("\n", " ").split(" ") if t]
    contexts = errors = unknown = 0
    for i, tok in enumerate(toks):
        key = tok.strip(",.;:()\"'")
        m = TYPED_RE.fullmatch(key) or OPAQUE_RE.fullmatch(key)
        if not m:
            continue
        gold = pseudonym_gender.get(key)
        if gold not in ("m", "f"):
            unknown += 1
            continue
        cues = []
        prev = toks[i - 1] if i > 0 else ""
        prev_w = prev.split("'")[-1].lower().strip(",;:()\"") if prev else ""
        subject = i == 0 or prev.endswith((".", ";", ":")) or prev_w in ART_M or prev_w in ART_F
        if prev_w and prev_w not in NEUTRAL:
            g = _gender_of_word(prev_w)
            if g:
                cues.append(g)
        if subject and not tok.endswith((".", ";")):
            for j in range(i + 1, min(len(toks), i + 6)):
                w = toks[j].lower()
                core = w.strip(",;:()\"'")
                if core in STOP_AFTER or w.startswith(("l'", "dell'", "all'", "dall'", "nell'")):
                    break
                if core in AUX:
                    if toks[j].endswith((".", ";")):
                        break
                    continue
                g = _gender_of_word(core)
                if g:
                    cues.append(g)
                    break
                if toks[j].endswith((".", ";", ":")) or core.startswith("per_") or core.startswith("[["):
                    break
        for g in cues:
            contexts += 1
            errors += int(g != gold)
    return contexts, errors, unknown


def evaluate_agreement(outputs: list[tuple[str, dict[str, str]]], B: int = 2000, seed: int = 0) -> AgreementResult:
    """outputs: list of (model_output_text, {pseudonym_token: gold_gender}) per document."""
    c = np.array([agreement_contexts(t, g)[0] for t, g in outputs])
    e = np.array([agreement_contexts(t, g)[1] for t, g in outputs])
    u = sum(agreement_contexts(t, g)[2] for t, g in outputs)
    return AgreementResult(int(c.sum()), int(e.sum()), cluster_bootstrap_rate(e, c, B, seed), int(u))
