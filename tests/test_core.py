import random
from datetime import date

import numpy as np
import pytest

from leakrate.data.italian_ids import codice_fiscale, validate_cf, iban_it, validate_iban, partita_iva, validate_piva
from leakrate.data.generator import generate_document, generate_corpus
from leakrate.pseudonymizers import get_pseudonymizer
from leakrate.pseudonymizers.base import OraclePseudonymizer, IdentityPseudonymizer, link_entities, infer_gender
from leakrate.schema import Span, Mapping, MappingEntry
from leakrate.metrics.leak import evaluate_leak, NormalizedText, strict_leak, partial_leak
from leakrate.metrics.stats import cluster_bootstrap_rate, wilson, paired_cluster_bootstrap_diff
from leakrate.metrics.kanon import evaluate_kanon
from leakrate.metrics.agreement import agreement_contexts
from leakrate.eval.reverse import reverse_map
from leakrate.schema import Entity


# ----------------------------------------------------------------------------- identifiers

def test_cf_known_example():
    assert validate_cf("RSSMRA85T10A562S")
    assert codice_fiscale("Rossi", "Mario", date(1985, 12, 10), False, "A562") == "RSSMRA85T10A562S"
    assert not validate_cf("RSSMRA85T10A562X")


def test_iban_piva_checksums():
    rng = random.Random(0)
    for _ in range(200):
        assert validate_iban(iban_it(rng))
        assert validate_piva(partita_iva(rng))
    assert validate_piva("00743110157")
    assert not validate_piva("00743110158")


# ----------------------------------------------------------------------------- generator

def test_generator_spans_match_and_deterministic():
    d1 = generate_document("legal", 7, "test", 42)
    d2 = generate_document("legal", 7, "test", 42)
    assert d1.text == d2.text and len(d1.entities) == len(d2.entities)
    for e in d1.entities:
        assert d1.text[e.start:e.end] == e.surface
    assert any(e.category == "PER" for e in d1.entities)
    assert any(e.category == "CF" for e in d1.entities)


def test_splits_use_disjoint_surnames():
    tr = generate_corpus(20, seed=1)
    train_sur = {e.canonical.split()[-1] for d in tr if d.split == "train" for e in d.entities if e.category == "PER"}
    test_sur = {e.canonical.split()[-1] for d in tr if d.split == "test" for e in d.entities if e.category == "PER"}
    assert not (train_sur & test_sur)


def test_train_never_uses_holdout_templates():
    docs = generate_corpus(30, seed=3)
    assert all("holdout" not in d.template_id for d in docs if d.split == "train")
    assert any("holdout" in d.template_id for d in docs if d.split == "test")


# ----------------------------------------------------------------------------- pseudonymizers

def test_identity_leaks_everything_and_oracle_leaks_nothing():
    docs = generate_corpus(10, seed=5)
    docs = [d for d in docs if d.split == "test"]
    ident = [IdentityPseudonymizer().pseudonymize(d.text, d.doc_id) for d in docs]
    rep, _ = evaluate_leak(docs, ident, B=200)
    assert rep.doc_leak_rate.rate == 1.0
    assert rep.direct_mention_leak.rate == 1.0
    oracle = OraclePseudonymizer(gold_by_doc={d.doc_id: d.entities for d in docs}, policy="all")
    outs = [oracle.pseudonymize(d.text, d.doc_id) for d in docs]
    rep, _ = evaluate_leak(docs, outs, B=200)
    assert rep.doc_leak_rate.rate == 0.0
    for c, v in rep.per_category.items():
        assert v["strict"].rate == 0.0, c
    assert rep.over_redaction_per_1k_tokens == 0.0


def test_regex_catches_structured_ids_but_leaks_names():
    docs = [d for d in generate_corpus(10, seed=8) if d.split == "test"]
    outs = [get_pseudonymizer("regex").pseudonymize(d.text, d.doc_id) for d in docs]
    rep, _ = evaluate_leak(docs, outs, B=200)
    assert rep.per_category["CF"]["strict"].rate == 0.0
    assert rep.per_category["IBAN"]["strict"].rate == 0.0
    assert rep.per_category["EMAIL"]["strict"].rate == 0.0
    assert rep.per_category["PER"]["strict"].rate == 1.0


def test_coreference_and_typed_pseudonyms():
    text = "La sig.ra Giulia Ferraro, residente a Bergamo. La Ferraro ha dichiarato. FERRARO Giulia firma. G. Ferraro."
    spans = []
    for surf in ("Giulia Ferraro", "La Ferraro", "FERRARO Giulia", "G. Ferraro"):
        i = text.index(surf)
        if surf.startswith("La "):
            i += 3
            surf = surf[3:]
        spans.append(Span(i, i + len(surf), "PER"))
    groups = link_entities(text, spans)
    assert len(groups) == 1
    assert infer_gender(text, spans[0]) == "f"
    assert infer_gender(text, spans[1]) == "f"      # surname only, resolved via article "La"


def test_gender_not_inferred_from_surname_ending():
    text = "comunica alla sig.ra Greco quanto segue"
    assert infer_gender(text, Span(21, 26, "PER")) == "f"
    text2 = "il collega Greco"
    assert infer_gender(text2, Span(11, 16, "PER")) in ("m", None)


def test_reverse_mapping_tolerates_mutations():
    mp = Mapping("x", "typed", [MappingEntry("PER_f_0001", "PER", "Giulia Ferraro", ["Giulia Ferraro"], {"gender": "f"}),
                                MappingEntry("ORG_0002", "ORG", "Acme S.r.l.", ["Acme S.r.l."])])
    out = reverse_map("Riassunto: per_f_0001 lavora per ORG 0002; PER_f_0001's contratto. ORG_0009 ignota.", mp)
    assert "Giulia Ferraro lavora per Acme S.r.l." in out.text
    assert out.replaced == 3 and out.unresolved == ["ORG_0009"] and out.residual_pseudonyms == 1


# ----------------------------------------------------------------------------- metrics

def test_strict_and_partial_leak_rules():
    e_iban = Entity("e1", "IBAN", "direct", 0, 0, "IT60 X054 2811 1010 0000 0123 456", "IT60X0542811101000000123456")
    assert strict_leak(e_iban, NormalizedText("conto IT60X0542811101000000123456 ok"))
    assert strict_leak(e_iban, NormalizedText("conto it60 x054 2811 1010 0000 0123 456 ok"))
    assert not strict_leak(e_iban, NormalizedText("conto IBAN_0001 ok"))
    assert partial_leak(e_iban, NormalizedText("conto ...101000000123456"))
    e_per = Entity("e2", "PER", "direct", 0, 0, "Giulia Ferraro", "Giulia Ferraro", {"gender": "f"})
    assert not strict_leak(e_per, NormalizedText("PER_f_0001 ha firmato"))
    assert not strict_leak(e_per, NormalizedText("Ferraro ha firmato"))     # strict requires the full surface
    assert partial_leak(e_per, NormalizedText("Ferraro ha firmato"))        # surname fragment survives
    assert not partial_leak(e_per, NormalizedText("Ferrarotti ha firmato"))  # token-bounded
    e_dob = Entity("e3", "DATE_BIRTH", "direct", 0, 0, "12/03/1980", "1980-03-12")
    assert strict_leak(e_dob, NormalizedText("nato il 12 marzo 1980"))
    assert partial_leak(e_dob, NormalizedText("classe 1980"))


def test_bootstrap_and_wilson():
    num = np.array([1, 0, 0, 0, 2, 0, 0, 0, 0, 0])
    den = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    ci = cluster_bootstrap_rate(num, den, B=500, seed=1)
    assert ci.rate == pytest.approx(0.15)
    assert ci.lo <= ci.rate <= ci.hi
    lo, hi = wilson(3, 20)
    assert ci.lo <= lo + 0.05  # clustered CI at least as wide as Wilson (allow slack)
    z = cluster_bootstrap_rate(np.zeros(10), den, B=100)
    assert z.rate == 0 and z.rule_of_three == pytest.approx(3 / 20)
    d = paired_cluster_bootstrap_diff(num, den, np.zeros(10), den, B=300)
    assert d["lo"] <= d["diff"] <= d["hi"] and d["diff"] == pytest.approx(0.15)


def test_kanon_more_survival_means_smaller_k():
    docs = [d for d in generate_corpus(20, seed=11) if d.split == "test"]
    ident = [IdentityPseudonymizer().pseudonymize(d.text, d.doc_id) for d in docs]
    _, rows_i = evaluate_leak(docs, ident, B=50)
    k_i = evaluate_kanon(docs, rows_i)
    oracle = OraclePseudonymizer(gold_by_doc={d.doc_id: d.entities for d in docs}, policy="all")
    outs = [oracle.pseudonymize(d.text, d.doc_id) for d in docs]
    _, rows_o = evaluate_leak(docs, outs, B=50)
    k_o = evaluate_kanon(docs, rows_o)
    assert k_i.pop_k_median < k_o.pop_k_median
    assert k_i.attr_survival["city"] > k_o.attr_survival["city"]


def test_agreement_detection():
    table = {"PER_f_0001": "f", "PER_m_0002": "m"}
    c, e, u = agreement_contexts("La PER_f_0001 è stata convocata. Il PER_m_0002 è stato assunto.", table)
    assert c == 4 and e == 0 and u == 0
    c, e, u = agreement_contexts("Il PER_f_0001 è stato convocato.", table)
    assert c == 2 and e == 2
    c, e, u = agreement_contexts("[[abcdef12]] è stato convocato.", {"[[abcdef12]]": "f"})
    assert c == 1 and e == 1


def test_downstream_scorers():
    from leakrate.eval.downstream import score_extract, score_classify, _num, judge_score, CallRecord
    gold = {"tipo_atto": "decreto_ingiuntivo", "importo_euro": "234.900,00", "rg": "853/2025", "esito": None}
    out = '```json\n{"tipo_atto": "Ricorso per decreto ingiuntivo", "importo_euro": 234900.00, "rg": "853/2025", "esito": null}\n```'
    assert score_extract(gold, out) == 1.0
    assert score_extract({"esito": "accoglie parzialmente"}, '{"esito": "Accoglimento parziale"}') == 1.0
    assert score_extract({"esito": "rigetta"}, '{"esito": "accoglimento"}') == 0.0
    assert score_extract({"tipo_atto": "verbale_sinistro", "rg": "1/2025"}, '{"tipo_atto": "Verbale", "rg": "1/2025"}',
                         skip_values={"verbale_sinistro"}) == 1.0
    assert _num("1.234,50") == _num("1,234.50") == _num("1234.5") == _num(1234.5) == "1234.50"
    assert _num("234.900") == "234900.00" and _num("12,5") == "12.50"
    assert score_classify("decreto_ingiuntivo", "Decreto ingiuntivo.") == 1.0
    recs = [CallRecord("d1", "typed", "s", "judge", "noswap", "h", "**B**", "end_turn", 1, 1),
            CallRecord("d1", "typed", "s", "judge", "swap", "h", "Risposta: A", "end_turn", 1, 1),
            CallRecord("d2", "typed", "s", "judge", "noswap", "h", "TIE", "end_turn", 1, 1)]
    js = judge_score(recs)
    assert js["n"] == 2 and abs(js["win_rate_vs_raw"] - 0.75) < 1e-9
    from leakrate.eval.downstream import parse_verdict
    assert parse_verdict("Entrambi i riassunti riportano i fatti. A risulta leggermente più completo rispetto a B.  **A**") == "A"
    assert parse_verdict("Il riassunto B è più fedele. Nel complesso B.\n\nB") == "B"
    assert parse_verdict("TIE") == "TIE" and parse_verdict("Risposta: A") == "A"
    assert parse_verdict("Entrambi i riassunti sono equivalenti.") == "TIE"


def test_stated_gold_filter():
    from leakrate.eval.downstream import stated_in
    text = "Lettera di assunzione. CCNL Studi professionali, 6° livello, RAL euro 20.000,00 lordi annui. Data 04/03/2025."
    assert stated_in("Studi professionali", text) and stated_in("6°", text) and stated_in("20.000,00", text)
    assert not stated_in("4°", text) and not stated_in("Metalmeccanici", text) and not stated_in("22.000,00", text)
    assert not stated_in(4, text)  # a lone digit inside a date is not evidence
    assert stated_in(4, "Degenza di 4 giorni, dimissione il 04/03/2025.") and not stated_in(3, "dimissione il 04/03/2025.")


def test_attacker_matching():
    from leakrate.eval.attacker import _match
    assert _match("profession", "pensionato", "pensionata") and _match("profession", "operaio", "operaia specializzata")
    assert not _match("profession", "operaio", "impiegato")
    assert _match("marital_status", "celibe", "nubile") and _match("marital_status", "coniugato", "sposata")
    assert not _match("marital_status", "coniugato", "divorziata")
    assert _match("nationality", "italiana", "italiano") and _match("age", 42, "43") and not _match("age", 42, 46)
    assert _match("diagnosis", "scompenso cardiaco", "Scompenso cardiaco cronico") and not _match("city", "Siena", "Verona")


def test_case_normalization_projection():
    from leakrate.pseudonymizers.casenorm import normalize, project
    from leakrate.schema import Span
    text = "Il sig. ROSSI Mario, nato a MILANO, con IBAN IT60X0542811101000000123456."
    norm, back = normalize(text)
    assert "Mario Rossi" in norm and "Milano" in norm and "IBAN" in norm and len(norm) == len(text)
    i = norm.index("Mario Rossi")
    spans = project([Span(i, i + len("Mario Rossi"), "PER", 0.9, "Mario Rossi")], back, text)
    assert sorted(s.text for s in spans) == ["Mario", "ROSSI"]
    j = norm.index("Milano")
    assert project([Span(j, j + 6, "LOC", 0.9, "Milano")], back, text)[0].text == "MILANO"
