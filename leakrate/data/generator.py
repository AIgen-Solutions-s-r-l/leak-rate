"""Synthetic Italian benchmark generator with character-level ground truth.

Design (METHODS.md §2, §10):
* every document is assembled by a `Builder` that records the span of each identifier it emits;
* a document = header + 3–5 body paragraphs sampled from a domain pool + closing;
* lexicons (names, cities, organisations) are disjoint across train/dev/test;
* the last two paragraph variants of every pool are *holdout* variants, never used in train
  documents, so the test split contains templates a fine-tuned model has never seen.

Everything is seeded: the same (seed, n) produces byte-identical corpora.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

from . import lexicons as L
from .italian_ids import (codice_fiscale, iban_it, format_iban, partita_iva, phone_it, plate_it,
                          carta_identita, passaporto, email_for, format_date_it)
from ..schema import Document, Entity, DOMAINS


# ============================================================================ entities

@dataclass
class Org:
    name: str
    piva: str
    city: tuple
    address: str


@dataclass
class Person:
    first: str
    last: str
    gender: str
    birth: date
    birth_city: tuple
    res_city: tuple
    street: str
    cap: str
    cf: str
    phone: str
    email: str
    profession_m: str
    profession_f: str
    prof_share: float
    nationality: str
    marital_m: str
    marital_f: str
    doc_id: str
    iban: str
    plate: str
    employer: Org | None = None

    @property
    def full(self) -> str:
        return f"{self.first} {self.last}"

    @property
    def profession(self) -> str:
        return self.profession_f if self.gender == "f" else self.profession_m

    @property
    def marital(self) -> str:
        return self.marital_f if self.gender == "f" else self.marital_m

    @property
    def address(self) -> str:
        return f"{self.street}, {self.cap} {self.res_city[0]}"

    def age_on(self, d: date) -> int:
        return d.year - self.birth.year - ((d.month, d.day) < (self.birth.month, self.birth.day))


class Sampler:
    """Split-aware sampler of persons and organisations."""

    def __init__(self, rng: random.Random, split: str):
        self.rng = rng
        self.split = split
        self.first_m = L.by_split(L.FIRST_NAMES_M, split)
        self.first_f = L.by_split(L.FIRST_NAMES_F, split)
        self.surnames = L.by_split(L.SURNAMES, split)
        self.cities = L.by_split(L.CITIES, split)
        self.orgs = L.by_split(L.ORG_STEMS, split)

    def city(self) -> tuple:
        return self.rng.choice(self.cities)

    def street(self, city: tuple) -> str:
        names = [s for s in L.STREET_NAMES if s != city[0]]
        return f"{self.rng.choice(L.STREET_TYPES)} {self.rng.choice(names)} {self.rng.randint(1, 180)}"

    def cap(self, city: tuple) -> str:
        return f"{self.rng.randint(10, 98)}{self.rng.randint(100, 199)}"

    def org(self, kind: str = "company") -> Org:
        rng = self.rng
        if kind == "bank":
            name = rng.choice(L.BANKS)
        elif kind == "hospital":
            city = self.city()
            name = f"{rng.choice(L.HOSPITALS)} di {city[0]}"
            return Org(name, partita_iva(rng), city, f"{self.street(city)}, {self.cap(city)} {city[0]}")
        elif kind == "lawfirm":
            name = f"{rng.choice(L.LAW_FIRMS)} {rng.choice(self.surnames)}"
        else:
            name = f"{rng.choice(self.orgs)} {rng.choice(L.ORG_SUFFIXES)}".strip()
        city = self.city()
        return Org(name, partita_iva(rng), city, f"{self.street(city)}, {self.cap(city)} {city[0]}")

    def person(self, ref: date, min_age: int = 18, max_age: int = 80, gender: str | None = None) -> Person:
        rng = self.rng
        g = gender or rng.choice("mf")
        first = rng.choice(self.first_f if g == "f" else self.first_m)
        last = rng.choice(self.surnames)
        age = rng.randint(min_age, max_age)
        birth = date(ref.year - age, rng.randint(1, 12), rng.randint(1, 28))
        bcity, rcity = self.city(), self.city()
        prof = rng.choices(L.PROFESSIONS, weights=[p[2] for p in L.PROFESSIONS])[0]
        nat = rng.choices(L.NATIONALITIES, weights=[n[1] for n in L.NATIONALITIES])[0][0]
        mar = rng.choices(L.MARITAL, weights=[m[2] for m in L.MARITAL])[0]
        return Person(first=first, last=last, gender=g, birth=birth, birth_city=bcity, res_city=rcity,
                      street=self.street(rcity), cap=self.cap(rcity),
                      cf=codice_fiscale(last, first, birth, g == "f", bcity[1]),
                      phone=phone_it(rng), email=email_for(first, last, rng),
                      profession_m=prof[0], profession_f=prof[1], prof_share=prof[2], nationality=nat,
                      marital_m=mar[0], marital_f=mar[1],
                      doc_id=carta_identita(rng) if rng.random() < 0.8 else passaporto(rng),
                      iban=iban_it(rng), plate=plate_it(rng))


# ============================================================================ builder

class Builder:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.parts: list[str] = []
        self.pos = 0
        self.entities: list[Entity] = []
        self._ids: dict[tuple, str] = {}
        self.facts: dict = {}

    # -- raw text
    def t(self, s: str) -> "Builder":
        self.parts.append(s)
        self.pos += len(s)
        return self

    def nl(self, n: int = 1) -> "Builder":
        return self.t("\n" * n)

    def _eid(self, category: str, canonical: str) -> str:
        key = (category, canonical)
        if key not in self._ids:
            self._ids[key] = f"e{len(self._ids) + 1}"
        return self._ids[key]

    def ent(self, category: str, kind: str, surface: str, canonical: str | None = None, attrs: dict | None = None) -> "Builder":
        canonical = canonical if canonical is not None else surface
        self.entities.append(Entity(id=self._eid(category, canonical), category=category, kind=kind,
                                    start=self.pos, end=self.pos + len(surface), surface=surface,
                                    canonical=canonical, attrs=attrs or {}))
        return self.t(surface)

    # -- typed helpers
    def per(self, p: Person, form: str = "full", title: bool = False) -> "Builder":
        attrs = {"gender": p.gender}
        if title:
            self.t(self.rng.choice(L.TITLES_F if p.gender == "f" else L.TITLES_M) + " ")
        if form == "full":
            return self.ent("PER", "direct", p.full, p.full, attrs)
        if form == "inverted":
            return self.ent("PER", "direct", f"{p.last.upper()} {p.first}", p.full, attrs)
        if form == "surname":
            return self.ent("PER", "direct", p.last, p.full, attrs)
        if form == "first":
            return self.ent("PER", "direct", p.first, p.full, attrs)
        if form == "initial":
            return self.ent("PER", "direct", f"{p.first[0]}. {p.last}", p.full, attrs)
        if form == "upper":
            return self.ent("PER", "direct", p.full.upper(), p.full, attrs)
        raise ValueError(form)

    def cf(self, p: Person) -> "Builder":
        return self.ent("CF", "direct", p.cf)

    def iban(self, p: Person) -> "Builder":
        return self.ent("IBAN", "direct", format_iban(p.iban, self.rng), p.iban)

    def piva(self, o: Org) -> "Builder":
        surf = o.piva if self.rng.random() < 0.7 else f"IT{o.piva}"
        return self.ent("PIVA", "direct", surf, o.piva)

    def email(self, p: Person) -> "Builder":
        return self.ent("EMAIL", "direct", p.email)

    def phone(self, p: Person) -> "Builder":
        return self.ent("PHONE", "direct", p.phone, "".join(ch for ch in p.phone if ch.isdigit())[-9:])

    def address(self, p: Person) -> "Builder":
        # street+number is the direct identifier; the city inside is also a LOC quasi mention
        self.ent("ADDRESS", "direct", p.street)
        self.t(f", {p.cap} ")
        return self.loc(p.res_city)

    def org_address(self, o: Org) -> "Builder":
        street, rest = o.address.split(", ", 1)
        self.ent("ADDRESS", "direct", street)
        cap, _ = rest.split(" ", 1)
        self.t(f", {cap} ")
        return self.loc(o.city)

    def doc_id(self, p: Person) -> "Builder":
        kind = "carta d'identità n. " if p.doc_id.startswith("C") else "passaporto n. "
        self.t(kind)
        return self.ent("DOC_ID", "direct", p.doc_id)

    def plate(self, p: Person) -> "Builder":
        surf = p.plate if self.rng.random() < 0.6 else f"{p.plate[:2]} {p.plate[2:5]} {p.plate[5:]}"
        return self.ent("PLATE", "direct", surf, p.plate)

    def birth_date(self, p: Person) -> "Builder":
        return self.ent("DATE_BIRTH", "direct", format_date_it(p.birth, self.rng), p.birth.isoformat())

    def date(self, d: date) -> "Builder":
        return self.ent("DATE", "quasi", format_date_it(d, self.rng), d.isoformat())

    def loc(self, city: tuple) -> "Builder":
        return self.ent("LOC", "quasi", city[0])

    def org(self, o: Org) -> "Builder":
        return self.ent("ORG", "quasi", o.name)

    def age(self, p: Person, ref: date) -> "Builder":
        a = p.age_on(ref)
        return self.ent("AGE", "quasi", f"{a} anni", str(a))

    def profession(self, p: Person) -> "Builder":
        return self.ent("PROFESSION", "quasi", p.profession, p.profession_m)

    def health(self, s: str) -> "Builder":
        return self.ent("HEALTH", "quasi", s)

    def g(self, p: Person, m: str, f: str) -> "Builder":
        return self.t(f if p.gender == "f" else m)

    def text(self) -> str:
        return "".join(self.parts)


# ============================================================================ context

@dataclass
class Ctx:
    rng: random.Random
    S: Sampler
    ref: date
    subj: Person
    others: dict = field(default_factory=dict)
    orgs: dict = field(default_factory=dict)
    label: str = ""
    extra: dict = field(default_factory=dict)


def _amount(rng: random.Random, lo: int, hi: int, step: int = 100) -> str:
    v = rng.randrange(lo, hi, step)
    return f"{v:,}".replace(",", ".") + ",00"


def _rand_date(rng: random.Random, ref: date, back_days: int, fwd_days: int = 0) -> date:
    return ref + timedelta(days=rng.randint(-back_days, fwd_days))


# ============================================================================ HR

def hr_setup(c: Ctx):
    c.orgs["employer"] = c.S.org()
    c.subj.employer = c.orgs["employer"]
    c.others["hr"] = c.S.person(c.ref, 30, 62)
    c.label = c.rng.choice(["assunzione", "contestazione_disciplinare", "valutazione", "licenziamento", "trasferimento"])
    c.extra["ral"] = _amount(c.rng, 18000, 75000, 500)
    c.extra["level"] = c.rng.choice(["1°", "2°", "3°", "4°", "5°", "6°", "7°"])
    c.extra["ccnl"] = c.rng.choice(["Commercio", "Metalmeccanici", "Terziario", "Edilizia", "Turismo", "Studi professionali", "Logistica"])
    c.extra["start"] = _rand_date(c.rng, c.ref, 30, 60)
    c.extra["events"] = sorted(_rand_date(c.rng, c.ref, 60) for _ in range(2))


def hr_header(b: Builder, c: Ctx):
    o = c.orgs["employer"]
    b.org(o).nl().t("Sede legale: ").org_address(o).nl().t("P.IVA ").piva(o).nl(2)
    b.date(c.ref).t(f" — Prot. n. {c.rng.randint(100, 9999)}/{c.ref.year}").nl(2)
    b.t("Alla c.a. ").g(c.subj, "del sig. ", "della sig.ra ").per(c.subj, c.rng.choice(["full", "inverted", "upper"])).nl()
    b.address(c.subj).nl(2)
    subj = {"assunzione": "Lettera di assunzione", "contestazione_disciplinare": "Contestazione disciplinare ai sensi dell'art. 7 L. 300/1970",
            "valutazione": "Esito valutazione annuale delle prestazioni", "licenziamento": "Comunicazione di licenziamento",
            "trasferimento": "Comunicazione di trasferimento di sede"}[c.label]
    b.t(f"Oggetto: {subj}").nl(2)


def hr_p_intro_a(b: Builder, c: Ctx):
    b.t("Con la presente la società ").org(c.orgs["employer"]).t(" comunica ").g(c.subj, "al sig. ", "alla sig.ra ").per(c.subj, "surname")
    b.t(", ").g(c.subj, "nato", "nata").t(" a ").loc(c.subj.birth_city).t(" il ").birth_date(c.subj).t(", codice fiscale ").cf(c.subj)
    b.t(", ").g(c.subj, "residente", "residente").t(" in ").address(c.subj).t(", quanto segue.").nl(2)


def hr_p_intro_b(b: Builder, c: Ctx):
    b.t("Egregi").g(c.subj, "o sig. ", "a sig.ra ").per(c.subj, "full").t(",").nl().t("facendo seguito ai colloqui intercorsi, Le confermiamo i termini di seguito riportati. ")
    b.t("I Suoi dati anagrafici risultano i seguenti: C.F. ").cf(c.subj).t(", ").g(c.subj, "nato", "nata").t(" il ").birth_date(c.subj).t(" a ").loc(c.subj.birth_city)
    b.t(", documento ").doc_id(c.subj).t(".").nl(2)


def hr_p_intro_c(b: Builder, c: Ctx):
    b.t("Il presente documento è indirizzato ").g(c.subj, "al dipendente ", "alla dipendente ").per(c.subj, "inverted")
    b.t(" (matricola ").t(str(c.rng.randint(1000, 99999))).t("), ").profession(c.subj).t(" presso la sede di ").loc(c.orgs["employer"].city)
    b.t(", di ").age(c.subj, c.ref).t(", cittadinanza ").ent("LOC", "quasi", c.subj.nationality, c.subj.nationality).t(", stato civile ").t(c.subj.marital).t(".").nl(2)


def hr_p_body_assunzione(b: Builder, c: Ctx):
    b.t("Il rapporto di lavoro, a tempo indeterminato e a tempo pieno, avrà inizio il ").date(c.extra["start"])
    b.t(f" con inquadramento al {c.extra['level']} livello del CCNL {c.extra['ccnl']}, mansione di ").profession(c.subj)
    b.t(f". La retribuzione annua lorda è fissata in euro {c.extra['ral']}, corrisposta in 14 mensilità. ")
    b.t("È previsto un periodo di prova di ").t(c.rng.choice(["60", "90", "180"])).t(" giorni di effettivo servizio.").nl(2)


def hr_p_body_contestazione(b: Builder, c: Ctx):
    e1, e2 = c.extra["events"]
    b.t("Le contestiamo formalmente che in data ").date(e1).t(" e in data ").date(e2)
    b.t(" Lei si è ").g(c.subj, "assentato", "assentata").t(" dal posto di lavoro senza preavviso né giustificazione, ")
    b.t("in violazione dell'art. ").t(c.rng.choice(["215", "220", "225", "7"])).t(f" del CCNL {c.extra['ccnl']}. ")
    b.t("La condotta è stata segnalata dal responsabile di reparto ").per(c.others["hr"], "full", title=True)
    b.t(". Ai sensi dell'art. 7 della L. 300/1970 Lei potrà presentare giustificazioni scritte entro 5 giorni dal ricevimento della presente.").nl(2)


def hr_p_body_valutazione(b: Builder, c: Ctx):
    score = c.rng.choice(["insufficiente", "adeguato", "buono", "eccellente"])
    c.extra["score"] = score
    b.t(f"La valutazione complessiva per l'anno {c.ref.year - 1} risulta: {score}. ")
    b.t("Il colloquio di feedback si è svolto il ").date(c.extra["events"][0]).t(" con ").per(c.others["hr"], "full", title=True)
    b.t(", responsabile Risorse Umane. Nel corso dell'anno ").g(c.subj, "il sig. ", "la sig.ra ").per(c.subj, "surname")
    b.t(" ha svolto la mansione di ").profession(c.subj).t(f" con una retribuzione annua lorda di euro {c.extra['ral']}.").nl(2)


def hr_p_body_licenziamento(b: Builder, c: Ctx):
    b.t("Le comunichiamo la risoluzione del rapporto di lavoro per ").t(c.rng.choice(["giustificato motivo oggettivo", "giusta causa", "superamento del periodo di comporto"]))
    b.t(" con effetto dal ").date(c.extra["start"]).t(". Le competenze di fine rapporto, pari a euro ").t(_amount(c.rng, 1500, 20000)).t(", saranno accreditate sul conto corrente IBAN ").iban(c.subj)
    b.t(" entro 30 giorni. Il TFR maturato sarà liquidato nei termini di legge.").nl(2)


def hr_p_body_trasferimento(b: Builder, c: Ctx):
    new = c.S.city()
    c.extra["new_city"] = new[0]
    b.t("A decorrere dal ").date(c.extra["start"]).t(" Lei è ").g(c.subj, "trasferito", "trasferita").t(" presso l'unità produttiva di ").loc(new)
    b.t(", per comprovate ragioni tecniche e organizzative ex art. 2103 c.c. Mansione e inquadramento (")
    b.t(f"{c.extra['level']} livello CCNL {c.extra['ccnl']}) restano invariati.").nl(2)


def hr_p_contacts(b: Builder, c: Ctx):
    b.t("Per ogni chiarimento può contattare l'ufficio del personale (").per(c.others["hr"], "initial").t(") all'indirizzo ")
    b.ent("EMAIL", "direct", c.others["hr"].email).t(" o al numero ").phone(c.others["hr"]).t(".").nl(2)


def hr_p_personal(b: Builder, c: Ctx):
    b.t("Dati per la busta paga: recapito telefonico ").phone(c.subj).t(", e-mail ").email(c.subj)
    b.t(", IBAN per l'accredito ").iban(c.subj).t(".").nl(2)


def hr_p_holdout_1(b: Builder, c: Ctx):  # holdout variant
    b.t("Si prende atto che ").per(c.subj, "first").t(" ").per(c.subj, "surname").t(", ").age(c.subj, c.ref).t(", ").profession(c.subj)
    b.t(f", è in forza presso l'azienda dal {c.rng.randint(2005, c.ref.year - 1)}. Il presente provvedimento è adottato ai sensi del CCNL {c.extra['ccnl']} e del regolamento aziendale vigente.").nl(2)


def hr_p_holdout_2(b: Builder, c: Ctx):  # holdout variant
    b.t("N.B. Copia della presente viene trasmessa alla RSU e conservata nel fascicolo personale n. ").t(str(c.rng.randint(100, 9999)))
    b.t(" intestato a ").per(c.subj, "upper").t(" (C.F. ").cf(c.subj).t(").").nl(2)


def hr_closing(b: Builder, c: Ctx):
    b.t("Distinti saluti.").nl(2).org(c.orgs["employer"]).nl().t("Il Responsabile Risorse Umane").nl().per(c.others["hr"], "full", title=True).nl()


# ============================================================================ LEGAL

def legal_setup(c: Ctx):
    c.others["avv"] = c.S.person(c.ref, 30, 70)
    c.others["judge"] = c.S.person(c.ref, 40, 70)
    c.others["counter"] = c.S.person(c.ref, 25, 75)
    c.orgs["counter_org"] = c.S.org()
    c.orgs["court_city"] = c.S.city()
    c.label = c.rng.choice(["atto_di_citazione", "decreto_ingiuntivo", "sentenza", "ricorso_lavoro", "verbale_sinistro"])
    c.extra["amount"] = _amount(c.rng, 1500, 250000)
    c.extra["rg"] = f"{c.rng.randint(100, 9999)}/{c.ref.year - c.rng.randint(0, 2)}"
    c.extra["hearing"] = _rand_date(c.rng, c.ref, 0, 180)
    c.extra["contract"] = _rand_date(c.rng, c.ref, 1500, 0)
    c.extra["outcome"] = c.rng.choice(["accoglie", "rigetta", "accoglie parzialmente"])


def legal_header(b: Builder, c: Ctx):
    court = c.rng.choice(L.COURTS)
    b.t(f"{court.upper()} DI ").ent("LOC", "quasi", c.orgs["court_city"][0].upper(), c.orgs["court_city"][0]).nl()
    b.t(c.rng.choice(["Sezione Civile", "Sezione Lavoro", "Seconda Sezione Civile", "Sezione Prima"])).nl()
    b.t(f"R.G. n. {c.extra['rg']}").nl(2)
    title = {"atto_di_citazione": "ATTO DI CITAZIONE", "decreto_ingiuntivo": "RICORSO PER DECRETO INGIUNTIVO", "sentenza": "SENTENZA",
             "ricorso_lavoro": "RICORSO EX ART. 414 C.P.C.", "verbale_sinistro": "VERBALE DI CONSTATAZIONE AMICHEVOLE E RELAZIONE"}[c.label]
    b.t(title).nl(2)


def legal_p_parties_a(b: Builder, c: Ctx):
    s, a = c.subj, c.others["avv"]
    b.g(s, "Il sig. ", "La sig.ra ").per(s, "full").t(" (C.F. ").cf(s).t("), ").g(s, "nato", "nata").t(" a ").loc(s.birth_city).t(" il ").birth_date(s)
    b.t(", ").g(s, "residente", "residente").t(" in ").address(s).t(", ").g(s, "rappresentato e difeso", "rappresentata e difesa").t(" dall'avv. ").per(a, "full")
    b.t(" del Foro di ").loc(a.res_city).t(" (C.F. ").cf(a).t(", PEC ").ent("EMAIL", "direct", a.email).t("), ")
    b.t("elettivamente ").g(s, "domiciliato", "domiciliata").t(" presso il suo studio in ").ent("ADDRESS", "direct", a.street).t(f", {a.cap} ").loc(a.res_city).t(";").nl(2)


def legal_p_parties_b(b: Builder, c: Ctx):
    s = c.subj
    b.t("RICORRENTE: ").per(s, "upper").t(", ").profession(s).t(", di ").age(s, c.ref).t(", C.F. ").cf(s).t(", ").g(s, "residente", "residente").t(" a ").loc(s.res_city)
    b.t(", ").address(s).t(", tel. ").phone(s).t(", e-mail ").email(s).t(";").nl(2)


def legal_p_counter_person(b: Builder, c: Ctx):
    k = c.others["counter"]
    b.t("CONTRO").nl().g(k, "il sig. ", "la sig.ra ").per(k, "full").t(", C.F. ").cf(k).t(", ").g(k, "residente", "residente").t(" in ").address(k).t(", CONVENUT").g(k, "O", "A").t(";").nl(2)


def legal_p_counter_org(b: Builder, c: Ctx):
    o = c.orgs["counter_org"]
    b.t("CONTRO").nl().org(o).t(", P.IVA ").piva(o).t(", con sede legale in ").org_address(o).t(", in persona del legale rappresentante pro tempore, CONVENUTA;").nl(2)


def legal_p_facts_contract(b: Builder, c: Ctx):
    b.t("FATTO").nl().t("In data ").date(c.extra["contract"]).t(" le parti stipulavano un contratto di ")
    b.t(c.rng.choice(["locazione", "appalto", "compravendita", "fornitura", "prestazione d'opera"])).t(f". Il corrispettivo pattuito ammontava a euro {c.extra['amount']}. ")
    b.t("Nonostante i solleciti inviati, anche a mezzo PEC all'indirizzo ").email(c.subj).t(", la controparte non ha adempiuto.").nl(2)


def legal_p_facts_sinistro(b: Builder, c: Ctx):
    k = c.others["counter"]
    b.t("In data ").date(c.extra["contract"]).t(", in ").loc(c.orgs["court_city"]).t(", ").ent("ADDRESS", "direct", c.S.street(c.orgs["court_city"]))
    b.t(", il veicolo targato ").plate(c.subj).t(", condotto ").g(c.subj, "dal sig. ", "dalla sig.ra ").per(c.subj, "surname")
    b.t(", veniva urtato dal veicolo targato ").ent("PLATE", "direct", k.plate).t(" condotto ").g(k, "dal sig. ", "dalla sig.ra ").per(k, "full")
    b.t(f". I danni accertati ammontano a euro {c.extra['amount']}.").nl(2)


def legal_p_facts_lavoro(b: Builder, c: Ctx):
    o = c.orgs["counter_org"]
    b.g(c.subj, "Il ricorrente", "La ricorrente").t(" ha prestato attività lavorativa come ").profession(c.subj).t(" presso ").org(o)
    b.t(" dal ").date(c.extra["contract"]).t(f" fino al licenziamento. Chiede la condanna al pagamento di euro {c.extra['amount']} a titolo di differenze retributive e indennità.").nl(2)


def legal_p_request(b: Builder, c: Ctx):
    b.t("P.Q.M.").nl().t("Voglia l'Ill.mo Tribunale, contrariis reiectis, condannare la parte convenuta al pagamento di euro ")
    b.t(c.extra["amount"]).t(", oltre interessi legali e spese di lite, da accreditare sul conto IBAN ").iban(c.subj).t(" intestato ").g(c.subj, "al sig. ", "alla sig.ra ").per(c.subj, "surname").t(".").nl(2)


def legal_p_hearing(b: Builder, c: Ctx):
    b.t("Si invita la parte convenuta a comparire all'udienza fissata per il giorno ").date(c.extra["hearing"])
    b.t(" dinanzi al Giudice designato, dott. ").per(c.others["judge"], "full").t(", con avvertimento che in difetto si procederà in sua contumacia.").nl(2)


def legal_p_sentence(b: Builder, c: Ctx):
    b.t("Il Giudice, dott. ").per(c.others["judge"], "full").t(", definitivamente pronunciando, ").t(c.extra["outcome"]).t(" la domanda proposta da ")
    b.per(c.subj, "full").t(f" e condanna la parte soccombente al pagamento delle spese di lite, liquidate in euro {_amount(c.rng, 800, 9000)}.").nl(2)


def legal_p_holdout_1(b: Builder, c: Ctx):  # holdout
    b.t("Si allega procura alle liti rilasciata ").g(c.subj, "dal sig. ", "dalla sig.ra ").per(c.subj, "inverted").t(" (").doc_id(c.subj).t(", rilasciata dal Comune di ").loc(c.subj.res_city).t(").").nl(2)


def legal_p_holdout_2(b: Builder, c: Ctx):  # holdout
    b.t("Il presente atto viene notificato a mezzo PEC all'indirizzo ").ent("EMAIL", "direct", c.others["counter"].email).t(" e per conoscenza al difensore, avv. ")
    b.per(c.others["avv"], "initial").t(", tel. ").ent("PHONE", "direct", c.others["avv"].phone, "".join(ch for ch in c.others["avv"].phone if ch.isdigit())[-9:]).t(".").nl(2)


def legal_closing(b: Builder, c: Ctx):
    b.loc(c.orgs["court_city"]).t(", ").date(c.ref).nl(2).t("Avv. ").per(c.others["avv"], "full").nl()


# ============================================================================ FINANCIAL

def fin_setup(c: Ctx):
    c.orgs["bank"] = c.S.org("bank")
    c.subj.employer = c.S.org()
    c.orgs["employer"] = c.subj.employer
    c.others["officer"] = c.S.person(c.ref, 28, 60)
    c.label = c.rng.choice(["richiesta_mutuo", "segnalazione_operazione_sospetta", "reclamo", "apertura_conto", "richiesta_prestito"])
    c.extra["amount"] = _amount(c.rng, 5000, 400000, 1000)
    c.extra["years"] = c.rng.choice([5, 10, 15, 20, 25, 30])
    c.extra["rate"] = f"{c.rng.randint(2, 6)},{c.rng.randint(0, 99):02d}%"
    c.extra["income"] = _amount(c.rng, 15000, 90000, 500)
    c.extra["op_date"] = _rand_date(c.rng, c.ref, 90)


def fin_header(b: Builder, c: Ctx):
    o = c.orgs["bank"]
    b.org(o).t(" — Filiale di ").loc(o.city).nl().org_address(o).nl(2)
    b.t(f"Pratica n. {c.rng.randint(10000, 999999)} del ").date(c.ref).nl(2)
    b.t("Oggetto: ").t({"richiesta_mutuo": "Richiesta di mutuo ipotecario", "segnalazione_operazione_sospetta": "Segnalazione di operazione sospetta (D.Lgs. 231/2007)",
                        "reclamo": "Reclamo del cliente", "apertura_conto": "Apertura rapporto di conto corrente", "richiesta_prestito": "Richiesta di prestito personale"}[c.label]).nl(2)


def fin_p_customer_a(b: Builder, c: Ctx):
    s = c.subj
    b.t("Cliente: ").per(s, "full").t(", ").g(s, "nato", "nata").t(" a ").loc(s.birth_city).t(" il ").birth_date(s).t(", C.F. ").cf(s)
    b.t(", ").g(s, "residente", "residente").t(" in ").address(s).t(", documento di identità: ").doc_id(s).t(". Recapiti: ").phone(s).t(", ").email(s).t(".").nl(2)


def fin_p_customer_b(b: Builder, c: Ctx):
    s = c.subj
    b.t("Dati anagrafici del richiedente").nl().t("Cognome e nome: ").per(s, "inverted").nl().t("Codice fiscale: ").cf(s).nl()
    b.t("Data di nascita: ").birth_date(s).nl().t("Residenza: ").address(s).nl().t("Stato civile: ").t(s.marital).nl().t("Cittadinanza: ").ent("LOC", "quasi", s.nationality, s.nationality).nl(2)


def fin_p_customer_c(b: Builder, c: Ctx):
    s = c.subj
    b.g(s, "Il sig. ", "La sig.ra ").per(s, "surname").t(", ").age(s, c.ref).t(", ").profession(s).t(" presso ").org(c.orgs["employer"])
    b.t(f" con reddito annuo netto dichiarato di euro {c.extra['income']}, è titolare del conto corrente IBAN ").iban(s).t(" acceso presso la nostra filiale di ").loc(c.orgs["bank"].city).t(".").nl(2)


def fin_p_mutuo(b: Builder, c: Ctx):
    b.t(f"Si richiede un mutuo ipotecario di euro {c.extra['amount']} della durata di {c.extra['years']} anni a tasso fisso {c.extra['rate']}, ")
    b.t("finalizzato all'acquisto dell'immobile sito in ").ent("ADDRESS", "direct", c.S.street(c.subj.res_city)).t(", ").loc(c.subj.res_city)
    b.t(". La rata mensile stimata è pari a euro ").t(_amount(c.rng, 300, 2500, 10)).t(".").nl(2)


def fin_p_sos(b: Builder, c: Ctx):
    b.t("In data ").date(c.extra["op_date"]).t(f" il cliente ha disposto un bonifico di euro {c.extra['amount']} dal conto IBAN ").iban(c.subj)
    b.t(" verso un conto estero non coerente con il profilo economico dichiarato (").profession(c.subj).t(f", reddito euro {c.extra['income']}). ")
    b.t("L'operazione è stata segnalata dal gestore ").per(c.others["officer"], "full", title=True).t(" per approfondimento ai sensi dell'art. 35 D.Lgs. 231/2007.").nl(2)


def fin_p_reclamo(b: Builder, c: Ctx):
    b.g(c.subj, "Il sig. ", "La sig.ra ").per(c.subj, "full").t(" lamenta l'addebito non autorizzato di euro ").t(_amount(c.rng, 50, 5000, 10)).t(" in data ").date(c.extra["op_date"])
    b.t(" sulla carta collegata al conto ").iban(c.subj).t(". Chiede il rimborso e la chiusura della carta. Il reclamo è stato inoltrato via e-mail da ").email(c.subj).t(".").nl(2)


def fin_p_conto(b: Builder, c: Ctx):
    b.t("Viene aperto il rapporto di conto corrente n. ").t(str(c.rng.randint(100000, 9999999))).t(" con IBAN ").iban(c.subj)
    b.t(" intestato a ").per(c.subj, "full").t(". Il cliente dichiara di essere ").profession(c.subj).t(" e di avere un reddito annuo di euro ").t(c.extra["income"]).t(". Firma apposta in filiale il ").date(c.ref).t(".").nl(2)


def fin_p_prestito(b: Builder, c: Ctx):
    b.t(f"Richiesta di prestito personale di euro {c.extra['amount']} in {c.extra['years'] * 12} rate mensili al TAN {c.extra['rate']}. ")
    b.t("Finalità dichiarata: ").t(c.rng.choice(["acquisto auto", "ristrutturazione", "spese mediche", "consolidamento debiti", "arredamento"]))
    b.t(". Il richiedente è ").g(c.subj, "dipendente", "dipendente").t(" di ").org(c.orgs["employer"]).t(" dal ").date(_rand_date(c.rng, c.ref, 4000, -365)).t(".").nl(2)


def fin_p_holdout_1(b: Builder, c: Ctx):  # holdout
    b.t("Verifica adeguata della clientela eseguita da ").per(c.others["officer"], "initial").t(" il ").date(c.ref).t(": documento ").doc_id(c.subj)
    b.t(", titolare effettivo ").per(c.subj, "upper").t(", C.F. ").cf(c.subj).t(", professione ").profession(c.subj).t(", nessuna PEP.").nl(2)


def fin_p_holdout_2(b: Builder, c: Ctx):  # holdout
    b.t("Contatti del cliente aggiornati in anagrafe: cellulare ").phone(c.subj).t(", e-mail ").email(c.subj).t(", domicilio ").address(c.subj).t(".").nl(2)


def fin_closing(b: Builder, c: Ctx):
    b.t("Il gestore della relazione").nl().per(c.others["officer"], "full", title=True).nl().org(c.orgs["bank"]).nl()


# ============================================================================ CLINICAL

def clin_setup(c: Ctx):
    c.orgs["hospital"] = c.S.org("hospital")
    c.others["doc"] = c.S.person(c.ref, 30, 68)
    c.others["gp"] = c.S.person(c.ref, 35, 68)
    c.label = c.rng.choice(L.WARDS)
    dx = c.rng.sample(L.DIAGNOSES, 2)
    c.extra["dx"], c.extra["dx2"] = dx[0][0], dx[1][0]
    c.extra["dx_share"] = dx[0][1]
    c.extra["drugs"] = c.rng.sample(L.DRUGS, 2)
    c.extra["admit"] = _rand_date(c.rng, c.ref, 20, -2)
    c.extra["los"] = (c.ref - c.extra["admit"]).days


def clin_header(b: Builder, c: Ctx):
    o = c.orgs["hospital"]
    b.org(o).nl().t("U.O. di ").t(c.label).nl().org_address(o).nl(2)
    b.t(c.rng.choice(["LETTERA DI DIMISSIONE", "RELAZIONE CLINICA", "REFERTO DI VISITA SPECIALISTICA", "VERBALE DI PRONTO SOCCORSO"])).nl(2)


def clin_p_patient_a(b: Builder, c: Ctx):
    s = c.subj
    b.t("Paziente: ").per(s, "inverted").t(", ").g(s, "nato", "nata").t(" il ").birth_date(s).t(" a ").loc(s.birth_city).t(" (").age(s, c.ref).t("), C.F. ").cf(s)
    b.t(", ").g(s, "residente", "residente").t(" in ").address(s).t(", tel. ").phone(s).t(".").nl(2)


def clin_p_patient_b(b: Builder, c: Ctx):
    s = c.subj
    b.g(s, "Il sig. ", "La sig.ra ").per(s, "full").t(", ").age(s, c.ref).t(", ").profession(s).t(", ").t(s.marital).t(", di nazionalità ").ent("LOC", "quasi", s.nationality, s.nationality)
    b.t(", giunge alla nostra attenzione in data ").date(c.extra["admit"]).t(" inviat").g(s, "o", "a").t(" dal MMG dott. ").per(c.others["gp"], "full").t(".").nl(2)


def clin_p_anamnesi(b: Builder, c: Ctx):
    b.t("Anamnesi: ").g(c.subj, "paziente", "paziente").t(" con storia di ").health(c.extra["dx2"]).t(" in terapia domiciliare con ").health(c.extra["drugs"][1])
    b.t(". Nega allergie. ").g(c.subj, "Fumatore", "Fumatrice").t(c.rng.choice([" attivo", " in passato", ": no"])).t(". ")
    b.g(c.subj, "Il sig. ", "La sig.ra ").per(c.subj, "surname").t(" vive a ").loc(c.subj.res_city).t(" con ").t(c.rng.choice(["il coniuge", "i familiari", "da solo/a", "la famiglia"])).t(".").nl(2)


def clin_p_diagnosis(b: Builder, c: Ctx):
    b.t("Diagnosi di dimissione: ").health(c.extra["dx"]).t(f". Degenza di {c.extra['los']} giorni dal ").date(c.extra["admit"]).t(" al ").date(c.ref)
    b.t(". Decorso clinico ").t(c.rng.choice(["regolare", "complicato da episodio febbrile", "favorevole", "caratterizzato da lento miglioramento"])).t(".").nl(2)


def clin_p_therapy(b: Builder, c: Ctx):
    b.t("Terapia consigliata alla dimissione: ").health(c.extra["drugs"][0]).t(" 1 cp/die, ").health(c.extra["drugs"][1]).t(" come da schema. ")
    b.t("Controllo ambulatoriale il ").date(_rand_date(c.rng, c.ref, 0, 60)).t(" presso l'ambulatorio di ").t(c.label).t(". Referente: dott. ").per(c.others["doc"], "surname").t(".").nl(2)


def clin_p_exams(b: Builder, c: Ctx):
    b.t("Esami eseguiti: emocromo, funzionalità renale (creatinina ").t(f"{c.rng.randint(6, 25) / 10:.1f}".replace(".", ",")).t(" mg/dl), ECG, ")
    b.t(c.rng.choice(["RX torace", "TC encefalo", "ecografia addome", "ecocardiogramma", "RM rachide lombare"])).t(". Pressione arteriosa ")
    b.t(f"{c.rng.randint(110, 170)}/{c.rng.randint(65, 100)} mmHg, peso {c.rng.randint(50, 110)} kg.").nl(2)


def clin_p_holdout_1(b: Builder, c: Ctx):  # holdout
    b.t("Persona di riferimento: ").per(c.S.person(c.ref, 25, 75), "full").t(", recapito ").ent("PHONE", "direct", phone_it(c.rng))
    b.t(". Il/la paziente ").per(c.subj, "first").t(" è ").g(c.subj, "stato informato", "stata informata").t(" della diagnosi di ").health(c.extra["dx"]).t(" e ha espresso consenso al trattamento.").nl(2)


def clin_p_holdout_2(b: Builder, c: Ctx):  # holdout
    b.t("Tessera sanitaria / C.F.: ").cf(c.subj).t(". Esenzione ticket per patologia. Copia inviata al MMG dott. ").per(c.others["gp"], "initial")
    b.t(" all'indirizzo ").ent("EMAIL", "direct", c.others["gp"].email).t(".").nl(2)


def clin_closing(b: Builder, c: Ctx):
    b.loc(c.orgs["hospital"].city).t(", ").date(c.ref).nl(2).t("Il Dirigente Medico").nl().t("Dott.").g(c.others["doc"], " ", "ssa ").per(c.others["doc"], "full").nl()


# ============================================================================ pools

POOLS = {
    "hr": dict(setup=hr_setup, header=hr_header, closing=hr_closing,
               intro=[hr_p_intro_a, hr_p_intro_b, hr_p_intro_c],
               body={"assunzione": hr_p_body_assunzione, "contestazione_disciplinare": hr_p_body_contestazione, "valutazione": hr_p_body_valutazione,
                     "licenziamento": hr_p_body_licenziamento, "trasferimento": hr_p_body_trasferimento},
               extra=[hr_p_contacts, hr_p_personal], holdout=[hr_p_holdout_1, hr_p_holdout_2]),
    "legal": dict(setup=legal_setup, header=legal_header, closing=legal_closing,
                  intro=[legal_p_parties_a, legal_p_parties_b],
                  body={"atto_di_citazione": legal_p_facts_contract, "decreto_ingiuntivo": legal_p_facts_contract, "sentenza": legal_p_sentence,
                        "ricorso_lavoro": legal_p_facts_lavoro, "verbale_sinistro": legal_p_facts_sinistro},
                  extra=[legal_p_request, legal_p_hearing], holdout=[legal_p_holdout_1, legal_p_holdout_2]),
    "financial": dict(setup=fin_setup, header=fin_header, closing=fin_closing,
                      intro=[fin_p_customer_a, fin_p_customer_b, fin_p_customer_c],
                      body={"richiesta_mutuo": fin_p_mutuo, "segnalazione_operazione_sospetta": fin_p_sos, "reclamo": fin_p_reclamo,
                            "apertura_conto": fin_p_conto, "richiesta_prestito": fin_p_prestito},
                      extra=[fin_p_customer_c], holdout=[fin_p_holdout_1, fin_p_holdout_2]),
    "clinical": dict(setup=clin_setup, header=clin_header, closing=clin_closing,
                     intro=[clin_p_patient_a, clin_p_patient_b],
                     body={w: clin_p_diagnosis for w in L.WARDS},
                     extra=[clin_p_anamnesi, clin_p_therapy, clin_p_exams], holdout=[clin_p_holdout_1, clin_p_holdout_2]),
}


def _tasks_for(c: Ctx, domain: str) -> dict:
    """Gold for downstream tasks: only non-identifier fields (METHODS.md §7)."""
    e = c.extra
    if domain == "hr":
        qa = [("Qual è la retribuzione annua lorda indicata?", f"euro {e['ral']}"), ("A quale CCNL fa riferimento la lettera?", e["ccnl"])]
        extract = {"tipo_documento": c.label, "ccnl": e["ccnl"], "livello": e["level"], "ral_euro": e["ral"]}
        labels = list(POOLS["hr"]["body"].keys())
    elif domain == "legal":
        qa = [("Qual è l'importo richiesto o liquidato in euro?", e["amount"]), ("Qual è il numero di ruolo generale?", e["rg"])]
        extract = {"tipo_atto": c.label, "importo_euro": e["amount"], "rg": e["rg"], "esito": e["outcome"] if c.label == "sentenza" else None}
        labels = list(POOLS["legal"]["body"].keys())
    elif domain == "financial":
        qa = [("Qual è l'importo dell'operazione o della richiesta in euro?", e["amount"]), ("Qual è il reddito annuo dichiarato in euro?", e["income"])]
        extract = {"tipo_pratica": c.label, "importo_euro": e["amount"], "reddito_euro": e["income"], "durata_anni": e["years"] if c.label in ("richiesta_mutuo", "richiesta_prestito") else None, "tasso": e["rate"] if c.label in ("richiesta_mutuo", "richiesta_prestito") else None}
        labels = list(POOLS["financial"]["body"].keys())
    else:
        qa = [("Qual è la diagnosi di dimissione?", e["dx"]), ("Quanti giorni è durata la degenza?", str(e["los"]))]
        extract = {"reparto": c.label, "diagnosi": e["dx"], "giorni_degenza": e["los"], "terapia": e["drugs"][0]}
        labels = list(L.WARDS)
    return {"classify": {"label": c.label, "label_set": labels},
            "extract": extract,
            "qa": [{"q": q, "a": a} for q, a in qa]}


def generate_document(domain: str, idx: int, split: str, seed: int, ref: date = date(2025, 6, 15)) -> Document:
    rng = random.Random(f"{seed}-{domain}-{split}-{idx}")
    S = Sampler(rng, split)
    ref = ref - timedelta(days=rng.randint(0, 540))
    subj = S.person(ref, 18, 85)
    c = Ctx(rng=rng, S=S, ref=ref, subj=subj)
    pool = POOLS[domain]
    pool["setup"](c)
    b = Builder(rng)
    pool["header"](b, c)
    intro = rng.choice(pool["intro"])
    intro(b, c)
    pool["body"][c.label](b, c)
    extras = list(pool["extra"])
    if split != "train":
        extras += pool["holdout"]
    k = rng.randint(1, min(2, len(extras)))
    used = rng.sample(extras, k)
    for fn in used:
        fn(b, c)
    pool["closing"](b, c)
    text = b.text()
    # sanity: every recorded span matches the text
    for e in b.entities:
        assert text[e.start:e.end] == e.surface, (e, text[e.start:e.end])
    profile = {"gender": subj.gender, "birth_year": subj.birth.year, "age": subj.age_on(ref), "city": subj.res_city[0],
               "city_population": subj.res_city[2], "profession": subj.profession_m, "profession_share": subj.prof_share,
               "employer": subj.employer.name if subj.employer else None, "nationality": subj.nationality,
               "marital_status": subj.marital_m, "diagnosis": c.extra.get("dx"), "diagnosis_share": c.extra.get("dx_share"),
               # the subject's name, for the LLM-attacker experiment (never used by the k-anonymity estimate)
               "subject_first_name": subj.first, "subject_surname": subj.last}
    template_id = f"{domain}:{intro.__name__}+{pool['body'][c.label].__name__}+{'+'.join(f.__name__ for f in used)}"
    return Document(doc_id=f"{domain}_{split}_{idx:05d}", domain=domain, template_id=template_id, split=split, text=text,
                    entities=b.entities, quasi_profile=profile, tasks=_tasks_for(c, domain))


def generate_corpus(n_per_domain: int, seed: int = 20251105, split_fracs=(0.7, 0.1, 0.2)) -> list[Document]:
    docs = []
    for domain in DOMAINS:
        n_train = int(n_per_domain * split_fracs[0])
        n_dev = int(n_per_domain * split_fracs[1])
        n_test = n_per_domain - n_train - n_dev
        for split, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
            for i in range(n):
                docs.append(generate_document(domain, i, split, seed))
    return docs
