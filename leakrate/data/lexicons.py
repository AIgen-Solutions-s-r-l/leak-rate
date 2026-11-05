"""Italian lexicons used by the synthetic generator and by the gender inference of pseudonymizers.

Lists are split-partitioned (train / dev / test) by index so that a model fine-tuned on the
train split never sees a test surname, first name, city or organisation (METHODS.md §10).
"""
from __future__ import annotations

FIRST_NAMES_M = [
    "Marco", "Luca", "Andrea", "Francesco", "Alessandro", "Giuseppe", "Matteo", "Davide", "Lorenzo", "Simone",
    "Giovanni", "Antonio", "Stefano", "Roberto", "Paolo", "Fabio", "Riccardo", "Federico", "Alberto", "Claudio",
    "Pietro", "Salvatore", "Vincenzo", "Massimo", "Gabriele", "Nicola", "Daniele", "Emanuele", "Michele", "Tommaso",
    "Enrico", "Giorgio", "Mario", "Luigi", "Angelo", "Carlo", "Domenico", "Filippo", "Franco", "Giacomo",
    "Leonardo", "Sergio", "Umberto", "Vittorio", "Pier Luigi", "Gian Marco", "Raffaele", "Cristian", "Diego", "Edoardo",
    "Ettore", "Fausto", "Gennaro", "Ivan", "Jacopo", "Lucio", "Mattia", "Nunzio", "Oreste", "Renato",
]
FIRST_NAMES_F = [
    "Giulia", "Francesca", "Chiara", "Sara", "Martina", "Valentina", "Alessia", "Elena", "Laura", "Anna",
    "Maria", "Silvia", "Elisa", "Federica", "Paola", "Roberta", "Simona", "Monica", "Claudia", "Alessandra",
    "Barbara", "Daniela", "Cristina", "Stefania", "Lucia", "Giovanna", "Rosa", "Angela", "Teresa", "Antonella",
    "Beatrice", "Camilla", "Debora", "Eleonora", "Fabiana", "Gaia", "Ilaria", "Lorena", "Marta", "Noemi",
    "Ornella", "Patrizia", "Rita", "Serena", "Tiziana", "Veronica", "Maria Grazia", "Anna Maria", "Rossella", "Sabrina",
    "Giada", "Irene", "Letizia", "Michela", "Nadia", "Olga", "Raffaella", "Sofia", "Vanessa", "Viola",
]
SURNAMES = [
    "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco",
    "Bruno", "Gallo", "Conti", "De Luca", "Mancini", "Costa", "Giordano", "Rizzo", "Lombardi", "Moretti",
    "Barbieri", "Fontana", "Santoro", "Mariani", "Rinaldi", "Caruso", "Ferrara", "Galli", "Martini", "Leone",
    "Longo", "Gentile", "Martinelli", "Vitale", "Lombardo", "Serra", "Coppola", "De Santis", "D'Angelo", "Marchetti",
    "Parisi", "Villa", "Conte", "Ferraro", "Ferri", "Fabbri", "Bianco", "Marini", "Grasso", "Valentini",
    "Messina", "Sala", "De Angelis", "Gatti", "Pellegrini", "Palumbo", "Sanna", "Farina", "Rizzi", "Monti",
    "Cattaneo", "Morelli", "Amato", "Silvestri", "Mazza", "Testa", "Grassi", "Pellegrino", "Carbone", "Giuliani",
    "Benedetti", "Barone", "Rossetti", "Caputo", "Montanari", "Guerra", "Palmieri", "Bernardi", "Martino", "Fiore",
    "De Rosa", "Ferretti", "Bellini", "Basile", "Riva", "Donati", "Piras", "Vitali", "Battaglia", "Sartori",
    "Neri", "Costantini", "Milani", "Pagano", "Ruggiero", "Sorrentino", "D'Amico", "Orlando", "Damico", "Negri",
    "Franchi", "Zanetti", "Brambilla", "Fiorentino", "Piazza", "Olivieri", "Cavallo", "Lamberti", "Massari", "Tedesco",
    "Volpe", "Marchese", "Cicero", "Spina", "Poli", "Baldi", "Pini", "Zito", "Bruni", "Falcone",
    "Castelli", "Pastore", "Sacco", "Toscano", "Ventura", "Benvenuti", "Cortese", "Napolitano", "Pozzi", "Riccardi",
    "Agostini", "Salvi", "Vaccaro", "Nardi", "Catalano", "Costanzo", "Guidi", "Mele", "Rocca", "Valente",
]

# (name, Belfiore code, resident population ~2024, region). Population figures are
# ISTAT-order-of-magnitude values used ONLY for the illustrative population-k estimator.
CITIES = [
    ("Roma", "H501", 2750000, "Lazio"), ("Milano", "F205", 1370000, "Lombardia"), ("Napoli", "F839", 910000, "Campania"),
    ("Torino", "L219", 850000, "Piemonte"), ("Palermo", "G273", 630000, "Sicilia"), ("Genova", "D969", 560000, "Liguria"),
    ("Bologna", "A944", 390000, "Emilia-Romagna"), ("Firenze", "D612", 360000, "Toscana"), ("Bari", "A662", 316000, "Puglia"),
    ("Catania", "C351", 298000, "Sicilia"), ("Verona", "L781", 256000, "Veneto"), ("Venezia", "L736", 250000, "Veneto"),
    ("Messina", "F158", 218000, "Sicilia"), ("Padova", "G224", 208000, "Veneto"), ("Trieste", "L424", 200000, "Friuli-Venezia Giulia"),
    ("Brescia", "B157", 197000, "Lombardia"), ("Parma", "G337", 196000, "Emilia-Romagna"), ("Prato", "G999", 195000, "Toscana"),
    ("Modena", "F257", 185000, "Emilia-Romagna"), ("Reggio Calabria", "H224", 170000, "Calabria"), ("Reggio Emilia", "H223", 170000, "Emilia-Romagna"),
    ("Perugia", "G478", 162000, "Umbria"), ("Ravenna", "H199", 157000, "Emilia-Romagna"), ("Livorno", "E625", 152000, "Toscana"),
    ("Cagliari", "B354", 149000, "Sardegna"), ("Rimini", "H294", 149000, "Emilia-Romagna"), ("Foggia", "D643", 146000, "Puglia"),
    ("Salerno", "H703", 127000, "Campania"), ("Ferrara", "D548", 129000, "Emilia-Romagna"), ("Latina", "E472", 127000, "Lazio"),
    ("Sassari", "I452", 121000, "Sardegna"), ("Monza", "F704", 122000, "Lombardia"), ("Bergamo", "A794", 120000, "Lombardia"),
    ("Pescara", "G482", 118000, "Abruzzo"), ("Trento", "L378", 118000, "Trentino-Alto Adige"), ("Vicenza", "L840", 110000, "Veneto"),
    ("Bolzano", "A952", 107000, "Trentino-Alto Adige"), ("Novara", "F952", 101000, "Piemonte"), ("Udine", "L483", 98000, "Friuli-Venezia Giulia"),
    ("Ancona", "A271", 99000, "Marche"), ("Lecce", "E506", 94000, "Puglia"), ("Pisa", "G702", 90000, "Toscana"),
    ("Como", "C933", 84000, "Lombardia"), ("Lucca", "E715", 89000, "Toscana"), ("Treviso", "L407", 85000, "Veneto"),
    ("Varese", "L682", 79000, "Lombardia"), ("Pavia", "G388", 71000, "Lombardia"), ("Cremona", "D150", 71000, "Lombardia"),
    ("Mantova", "E897", 49000, "Lombardia"), ("Siena", "I726", 53000, "Toscana"), ("Matera", "F052", 59000, "Basilicata"),
    ("Potenza", "G942", 64000, "Basilicata"), ("Catanzaro", "C352", 85000, "Calabria"), ("L'Aquila", "A345", 69000, "Abruzzo"),
    ("Aosta", "A326", 33000, "Valle d'Aosta"), ("Campobasso", "B519", 47000, "Molise"), ("Asti", "A479", 74000, "Piemonte"),
    ("Alessandria", "A182", 92000, "Piemonte"), ("Arezzo", "A390", 97000, "Toscana"), ("Terni", "L117", 107000, "Umbria"),
]

STREET_TYPES = ["Via", "Via", "Via", "Viale", "Piazza", "Corso", "Largo", "Vicolo"]
STREET_NAMES = [
    "Roma", "Garibaldi", "Mazzini", "Cavour", "Dante", "Verdi", "Manzoni", "Leopardi", "Marconi", "Matteotti",
    "della Repubblica", "della Libertà", "XX Settembre", "IV Novembre", "Vittorio Veneto", "dei Mille", "Gramsci",
    "San Francesco", "Sant'Antonio", "Santa Maria", "Petrarca", "Carducci", "Pascoli", "Volta", "Galilei", "Michelangelo",
    "Leonardo da Vinci", "Giotto", "Donatello", "Raffaello", "Buonarroti", "Colombo", "Marco Polo", "Toscanini", "Puccini",
    "Rossini", "Bellini", "Monteverdi", "Piave", "Isonzo", "Adige", "Po", "Tevere", "Arno", "dei Platani", "delle Querce",
    "del Lavoro", "dell'Industria", "Europa", "Aldo Moro", "Kennedy", "Trieste", "Trento", "Firenze", "Torino",
]

# (masculine, feminine, rough share of the resident adult population — illustrative, for k̂ only)
PROFESSIONS = [
    ("impiegato", "impiegata", 0.14), ("operaio", "operaia", 0.11), ("insegnante", "insegnante", 0.03),
    ("infermiere", "infermiera", 0.012), ("medico", "medico", 0.008), ("avvocato", "avvocata", 0.005),
    ("ingegnere", "ingegnera", 0.012), ("commercialista", "commercialista", 0.003), ("architetto", "architetta", 0.003),
    ("geometra", "geometra", 0.002), ("farmacista", "farmacista", 0.002), ("programmatore", "programmatrice", 0.01),
    ("autista", "autista", 0.01), ("magazziniere", "magazziniera", 0.01), ("cassiere", "cassiera", 0.008),
    ("commesso", "commessa", 0.02), ("cuoco", "cuoca", 0.008), ("cameriere", "cameriera", 0.012),
    ("elettricista", "elettricista", 0.006), ("idraulico", "idraulica", 0.004), ("agente immobiliare", "agente immobiliare", 0.003),
    ("consulente", "consulente", 0.01), ("responsabile vendite", "responsabile vendite", 0.005),
    ("addetto alle pulizie", "addetta alle pulizie", 0.015), ("pensionato", "pensionata", 0.26),
    ("studente", "studentessa", 0.06), ("libero professionista", "libera professionista", 0.03),
    ("imprenditore", "imprenditrice", 0.03), ("artigiano", "artigiana", 0.02), ("agricoltore", "agricoltrice", 0.015),
    ("barista", "barista", 0.006), ("tecnico informatico", "tecnica informatica", 0.006), ("ricercatore", "ricercatrice", 0.002),
    ("disoccupato", "disoccupata", 0.05), ("casalingo", "casalinga", 0.04), ("militare", "militare", 0.005),
    ("poliziotto", "poliziotta", 0.003), ("educatore", "educatrice", 0.004), ("fisioterapista", "fisioterapista", 0.002),
    ("psicologo", "psicologa", 0.002),
]

ORG_STEMS = [
    "Tecnoservice", "Edilcasa", "Logistica Padana", "Meccanica Precisa", "Alimentari Bellavista", "Studio Tecnico Delta",
    "Officine Riunite", "Impianti Elettrici Nord", "Cooperativa Sociale Il Faro", "Autotrasporti Veloci", "Digital Solutions",
    "Consulting Partners", "Metalmeccanica Adriatica", "Pulizie Industriali Globo", "Ristorazione Collettiva Sapori",
    "Farmacia Centrale", "Tessile Lombarda", "Costruzioni Generali", "Infissi Moderni", "Agrifood Italia",
    "Grafica Creativa", "Arredamenti Design", "Energia Pulita", "Trasporti Internazionali Sud", "Supermercati Risparmio",
    "Elettronica Avanzata", "Servizi Assicurativi Uniti", "Centro Medico Salus", "Istituto Vendite Giudiziarie", "Hotel Belvedere",
    "Carpenteria Rossa", "Calzaturificio Marche", "Mobilificio Veneto", "Vetreria Artistica", "Cantina Sociale",
    "Software House Innova", "Agenzia Viaggi Orizzonti", "Lavanderia Industriale Blu", "Studio Dentistico Sorriso", "Palestra Fitness Club",
]
ORG_SUFFIXES = ["S.r.l.", "S.p.A.", "S.r.l.", "S.n.c.", "S.a.s.", "S.r.l.s.", "Soc. Coop.", ""]

HOSPITALS = [
    "Ospedale Maggiore", "Policlinico Universitario", "Ospedale San Giovanni", "Azienda Ospedaliera Santa Croce",
    "Presidio Ospedaliero San Carlo", "Ospedale Civile", "Istituto Clinico Humanitas", "Ospedale Sant'Anna",
    "Ospedale Papa Giovanni XXIII", "Ospedale Niguarda", "Ospedale Molinette", "Ospedale Cardarelli",
    "Ospedale San Raffaele", "Ospedale Careggi", "Ospedale Sant'Orsola", "Ospedale Riuniti",
]
BANKS = ["Banca Popolare del Territorio", "Credito Cooperativo Valle", "Banca di Credito Regionale", "Istituto Bancario Nazionale",
         "Cassa di Risparmio Provinciale", "Banca Mediolanea", "Banca Unione", "Credito Agricolo Italiano"]
LAW_FIRMS = ["Studio Legale Associato", "Studio Legale", "Studio Legale e Tributario", "Avvocati Associati"]
COURTS = ["Tribunale Ordinario", "Tribunale", "Corte d'Appello", "Giudice di Pace", "Tribunale del Lavoro"]

# (diagnosis, approximate adult prevalence — illustrative, for k̂ only)
DIAGNOSES = [
    ("ipertensione arteriosa", 0.30), ("diabete mellito di tipo 2", 0.06), ("frattura del collo del femore", 0.002),
    ("polmonite comunitaria", 0.01), ("BPCO riacutizzata", 0.03), ("fibrillazione atriale", 0.02),
    ("episodio depressivo maggiore", 0.05), ("carcinoma mammario", 0.01), ("appendicite acuta", 0.001),
    ("ernia discale L4-L5", 0.02), ("insufficienza renale cronica stadio 3", 0.03), ("asma bronchiale", 0.06),
    ("colecistite acuta litiasica", 0.003), ("infarto miocardico acuto NSTEMI", 0.004), ("ictus ischemico", 0.005),
    ("epilessia focale", 0.007), ("celiachia", 0.01), ("ipotiroidismo", 0.04), ("artrite reumatoide", 0.005),
    ("infezione da HIV", 0.002), ("disturbo bipolare", 0.01), ("carcinoma del colon", 0.005), ("scompenso cardiaco", 0.02),
    ("sindrome ansioso-depressiva", 0.06), ("obesità di II grado", 0.05), ("gastrite erosiva", 0.03),
]
DRUGS = ["metformina 850 mg", "ramipril 5 mg", "atorvastatina 20 mg", "enoxaparina 4000 UI", "amoxicillina/acido clavulanico 1 g",
         "pantoprazolo 40 mg", "sertralina 50 mg", "levotiroxina 75 mcg", "bisoprololo 2,5 mg", "warfarin 5 mg",
         "furosemide 25 mg", "paracetamolo 1000 mg", "acido acetilsalicilico 100 mg", "insulina glargine", "prednisone 25 mg"]
WARDS = ["Medicina Interna", "Cardiologia", "Ortopedia e Traumatologia", "Chirurgia Generale", "Pneumologia", "Neurologia",
         "Psichiatria", "Oncologia", "Nefrologia", "Geriatria", "Pronto Soccorso", "Endocrinologia"]

NATIONALITIES = [("italiana", 0.91), ("rumena", 0.018), ("albanese", 0.007), ("marocchina", 0.007), ("cinese", 0.005),
                 ("ucraina", 0.004), ("filippina", 0.003), ("peruviana", 0.002), ("egiziana", 0.002), ("bangladese", 0.003),
                 ("senegalese", 0.002), ("tunisina", 0.002), ("nigeriana", 0.002), ("moldava", 0.002), ("indiana", 0.003)]
# (masculine, feminine, share)
MARITAL = [("celibe", "nubile", 0.35), ("coniugato", "coniugata", 0.47), ("divorziato", "divorziata", 0.06),
           ("vedovo", "vedova", 0.08), ("separato", "separata", 0.04)]

TITLES_M = ["sig.", "Sig.", "dott.", "Dott.", "ing.", "avv.", "rag.", "geom.", "prof."]
TITLES_F = ["sig.ra", "Sig.ra", "dott.ssa", "Dott.ssa", "ing.", "avv.", "rag.", "geom.", "prof.ssa"]


# --------------------------------------------------------------------- splits

def split_of_index(i: int) -> str:
    r = i % 10
    return "train" if r < 7 else ("dev" if r == 7 else "test")


def by_split(items: list, split: str) -> list:
    return [x for i, x in enumerate(items) if split_of_index(i) == split]


# --------------------------------------------------------------------- gender lexicon

_M_SET = {n.split()[0] for n in FIRST_NAMES_M} | set(FIRST_NAMES_M)
_F_SET = {n.split()[0] for n in FIRST_NAMES_F} | set(FIRST_NAMES_F)


def gender_of_first_name(name: str, lexicon_only: bool = False) -> str | None:
    """Return 'm' | 'f' | None. Used by pseudonymizers (deployment-realistic: lexicon + ending heuristic)."""
    n = name.strip().split()[0] if name.strip() else ""
    cap = n[:1].upper() + n[1:].lower()
    if cap in _M_SET:
        return "m"
    if cap in _F_SET:
        return "f"
    if lexicon_only:
        return None
    low = n.lower()
    if low.endswith(("a", "ella", "ina")) and low not in {"andrea", "luca", "nicola", "mattia", "elia", "tobia"}:
        return "f"
    if low.endswith(("o", "e", "i")):
        return "m"
    return None
