"""The frontier model used as its own anonymizer (`llm_self`, METHODS.md §9).

This is the comparison condition, not a deployable option: the provider sees the raw text.
Its leak rate tells us how much the local models lose relative to the strongest detector.

Implementation: one Messages API call per document asking for a JSON list of identifier mentions;
mentions are located in the text by exact match (all occurrences). The model under test is fixed
and refusals are recorded as outcomes — server-side fallbacks are deliberately NOT enabled, because
an experiment must not silently switch models.
"""
from __future__ import annotations

import json
import os
import re

from ..schema import Span
from .base import Pseudonymizer

DEFAULT_MODEL = os.environ.get("LEAKRATE_MODEL", "claude-opus-5-5")

SYSTEM = """Sei un sistema di pseudonimizzazione per documenti italiani (legali, bancari, HR, clinici).
Individua OGNI menzione di identificatori personali nel testo e restituisci SOLO un array JSON di oggetti
{"text": <stringa esatta come appare nel testo>, "category": <categoria>}.
Categorie ammesse: PER (nome, cognome o entrambi di una persona fisica, in qualunque forma), CF (codice fiscale),
IBAN, PIVA (partita IVA), EMAIL, PHONE, ADDRESS (via/piazza e numero civico), DOC_ID (numero di carta d'identità,
passaporto, patente), PLATE (targa), DATE_BIRTH (data di nascita), ORG (azienda, ente, ospedale, banca, studio),
LOC (città, comune, nazionalità), DATE (altre date), AGE (età o anno di nascita), PROFESSION (mestiere), HEALTH (diagnosi, farmaci).
Includi ogni variante della stessa persona (nome completo, solo cognome, COGNOME Nome, iniziale puntata).
Non aggiungere commenti: solo l'array JSON."""


class LLMPseudonymizer(Pseudonymizer):
    name = "llm_self"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = os.environ.get("LEAKRATE_EFFORT", "low"), **kw):
        super().__init__(**kw)
        # the system id carries model and effort so that eval files of different frontier models never collide
        self.name = f"llm_self-{model.replace('claude-', '')}-{effort}"
        import anthropic
        self.client = anthropic.Anthropic(base_url=os.environ.get("LEAKRATE_BASE_URL", "https://api.anthropic.com"), max_retries=10)
        self.model = model
        self.effort = effort
        self.last_stop_reason = None

    def _call(self, text: str) -> str:
        import hashlib
        from pathlib import Path
        cache = Path(os.environ.get("LEAKRATE_CACHE", "results/cache"))
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / (hashlib.sha256(f"{self.model}|{self.effort}|{SYSTEM}|{text}".encode()).hexdigest() + ".json")
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            self.last_stop_reason = d["stop_reason"]
            return d["output"] if d["stop_reason"] != "refusal" else "[]"
        from ..eval.downstream import _with_backoff
        resp = _with_backoff(lambda: self.client.messages.create(
            model=self.model, max_tokens=16000, system=SYSTEM,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": f"Testo:\n<<<\n{text}\n>>>"}],
        ))
        self.last_stop_reason = resp.stop_reason
        out = "" if resp.stop_reason == "refusal" else "".join(b.text for b in resp.content if b.type == "text")
        f.write_text(json.dumps({"model": self.model, "effort": self.effort, "kind": "llm_self", "output": out, "stop_reason": resp.stop_reason,
                                 "in": resp.usage.input_tokens, "out": resp.usage.output_tokens}, ensure_ascii=False), encoding="utf-8")
        if resp.stop_reason == "refusal":
            return "[]"
        return out

    def detect(self, text: str) -> list[Span]:
        raw = self._call(text)
        m = re.search(r"\[.*\]", raw, re.S)
        items = []
        if m:
            try:
                items = json.loads(m.group(0))
            except json.JSONDecodeError:
                items = []
        spans: list[Span] = []
        for it in items:
            s, cat = str(it.get("text", "")).strip(), str(it.get("category", "")).upper()
            if len(s) < 2 or cat not in {"PER", "CF", "IBAN", "PIVA", "EMAIL", "PHONE", "ADDRESS", "DOC_ID", "PLATE",
                                         "DATE_BIRTH", "ORG", "LOC", "DATE", "AGE", "PROFESSION", "HEALTH"}:
                continue
            for mm in re.finditer(re.escape(s), text):
                spans.append(Span(mm.start(), mm.end(), cat, 0.9, s))
        return spans
