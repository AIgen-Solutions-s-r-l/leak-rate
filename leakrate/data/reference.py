"""Reference marginals for the population-k estimator (METHODS.md §6), from ISTAT where ISTAT publishes them.

Sources (all public, consulted 2026-11-05; values rounded to the precision used here):
- population and sex: ISTAT, "Indicatori demografici, anno 2024/2025" and demo.istat.it, resident population
  on 1 January 2025 = 58 934 000 (provisional); females 51.1 %, males 48.9 %.
- age structure: ISTAT indicators, 1 January 2025: 0-14 = 12.0 %, 15-64 = 63.3 %, 65+ = 24.7 % (80+ ≈ 8 %);
  turned into a per-year-of-age share by band (flat within band).
- civil status: ISTAT, "Popolazione residente per stato civile", 1 January 2024 (as republished by UrbiStat):
  celibi/nubili 44.1 %, coniugati/e 44.8 %, divorziati/e 3.8 %, vedovi/e 7.3 %. ISTAT counts the legally
  separated among the married; the 4 % "separato" share is an approximation from ISTAT's separations series.
- citizenship: ISTAT, "Stranieri residenti al 1° gennaio 2025": 5 371 251 foreign residents = 9.1 % of the
  population; the per-country shares below are rounded from the same table.
- professions: shares of the resident population aged 18+, derived from ISTAT "Rilevazione sulle forze di
  lavoro 2024" (23.9 M employed, by occupation group), INPS (16.1 M pensioners) and ISTAT inactive-population
  categories (housework ≈ 7 M, students 18+ ≈ 2 M, unemployed 1.6 M). Fine occupations (e.g. farmacista)
  are order-of-magnitude values from the professional registers; they are marked ~ below.
- municipalities: ISTAT resident population by comune, 1 January 2024, rounded (the values stored in the lexicon).
- diagnoses: ISTAT "Aspetti della vita quotidiana 2023" for hypertension (18.6 % of the population) and
  diabetes (6.4 %); other prevalences are order-of-magnitude values from Italian epidemiological reports.
Every value is used only under the independence assumption of §6; the distribution shape, not the point
values, is the finding. `export_json()` writes the table used for a run next to the benchmark.
"""
from __future__ import annotations

from . import lexicons as L

SOURCE_DATE = "2026-11-05"
ITALY_POPULATION = 58_934_000                      # ISTAT, 1 Jan 2025, provisional
GENDER_SHARE = {"m": 0.489, "f": 0.511}            # ISTAT, 1 Jan 2025

# ISTAT age structure, 1 Jan 2025: 0-14 12.0 %, 15-64 63.3 %, 65-79 16.7 %, 80+ 8.0 % (flat within band)
AGE_BANDS = [(0, 14, 0.120 / 15), (15, 64, 0.633 / 50), (65, 79, 0.167 / 15), (80, 110, 0.080 / 31)]
SHARE_PER_YEAR_OF_AGE = 0.633 / 50               # kept for callers that need a single number (adult band)
SHARE_PER_5Y_BAND = 5 * SHARE_PER_YEAR_OF_AGE


def share_of_age(age) -> float:
    """Share of the resident population with exactly this age in years (ISTAT bands, flat within band)."""
    try:
        a = int(age)
    except (TypeError, ValueError):
        return SHARE_PER_YEAR_OF_AGE
    for lo, hi, s in AGE_BANDS:
        if lo <= a <= hi:
            return s
    return SHARE_PER_YEAR_OF_AGE


CITY_POPULATION = {c[0]: c[2] for c in L.CITIES}   # ISTAT 1 Jan 2024, rounded

# shares of the population aged 18+ (ISTAT RFL 2024, INPS 2024, ISTAT inactive categories); ~ = register-based estimate
PROFESSION_SHARE = {
    "pensionato": 0.32, "casalingo": 0.14, "operaio": 0.15, "impiegato": 0.14, "studente": 0.045, "disoccupato": 0.032,
    "libero professionista": 0.030, "imprenditore": 0.025, "artigiano": 0.025, "insegnante": 0.020, "commesso": 0.015,
    "addetto alle pulizie": 0.012, "agricoltore": 0.012, "consulente": 0.010, "infermiere": 0.009, "programmatore": 0.008,
    "autista": 0.008, "cameriere": 0.008, "magazziniere": 0.006, "cuoco": 0.006, "medico": 0.005, "avvocato": 0.005,
    "ingegnere": 0.005, "tecnico informatico": 0.005, "cassiere": 0.004, "barista": 0.004, "elettricista": 0.004,
    "responsabile vendite": 0.004, "militare": 0.004, "idraulico": 0.003, "architetto": 0.003, "poliziotto": 0.003,
    "educatore": 0.003, "commercialista": 0.002, "geometra": 0.002, "farmacista": 0.002, "agente immobiliare": 0.002,
    "ricercatore": 0.002, "psicologo": 0.002, "fisioterapista": 0.001,
}

# ISTAT 1 Jan 2025, foreign residents by citizenship, rounded to the share of the whole resident population
NATIONALITY_SHARE = {
    "italiana": 0.909, "rumena": 0.0183, "albanese": 0.0071, "marocchina": 0.0071, "cinese": 0.0053, "ucraina": 0.0046,
    "bangladese": 0.0032, "indiana": 0.0029, "egiziana": 0.0027, "filippina": 0.0025, "nigeriana": 0.0020,
    "senegalese": 0.0019, "tunisina": 0.0017, "moldava": 0.0017, "peruviana": 0.0014,
}

# ISTAT civil status, 1 Jan 2024 (married includes the legally separated; "separato" is an approximation)
MARITAL_SHARE = {"celibe": 0.441, "coniugato": 0.408, "separato": 0.040, "divorziato": 0.038, "vedovo": 0.073}

DIAGNOSIS_SHARE = dict(L.DIAGNOSES)
DIAGNOSIS_SHARE.update({"ipertensione arteriosa": 0.186, "diabete mellito di tipo 2": 0.064})   # ISTAT AVQ 2023

# a named employer identifies a workforce: ISTAT "Imprese e addetti" 2022, mean size of firms with 10+ employees ≈ 40
EMPLOYER_SIZE = 40


def export_json() -> dict:
    return {"source_date": SOURCE_DATE, "italy_population": ITALY_POPULATION, "gender_share": GENDER_SHARE,
            "age_bands": AGE_BANDS, "share_per_year_of_age_adult": SHARE_PER_YEAR_OF_AGE,
            "city_population": CITY_POPULATION, "profession_share": PROFESSION_SHARE,
            "nationality_share": NATIONALITY_SHARE, "marital_share": MARITAL_SHARE,
            "diagnosis_share": DIAGNOSIS_SHARE, "employer_size": EMPLOYER_SIZE,
            "note": "ISTAT-derived marginals (rounded); see module docstring for the source of each table."}
