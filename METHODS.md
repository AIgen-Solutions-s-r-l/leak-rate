# Methods specification: Leak Rate, Not Loss

This file is the single source of truth for the definitions used by the code
(`leakrate/`) and the paper (`paper/main.tex`). If they disagree, this file wins.

## 1. Threat model and pipeline

```
raw document d  ──►  local pseudonymizer P  ──►  d' = P(d), mapping M  ──►  frontier LLM F  ──►  y' = F(d', task)
                                                                                                  │
                                                       y = M⁻¹(y')   ◄────────────────────────────┘
```

* The frontier provider sees only `d'` and `y'`. The mapping `M` never leaves the premises.
* The adversary is the provider (or anyone with access to the provider's logs). It sees `d'`
  and whatever `F` emits. It does not see `M`.
* **Leak** = any ground-truth identifier mention of `d` whose surface form (or an informative
  fragment of it) is still present in `d'`.

## 2. Ground-truth annotation schema (dataset)

Each document is a JSON object:

```json
{
  "doc_id": "hr_000123",
  "domain": "hr",                      // legal | financial | hr | clinical
  "template_id": "hr_contestazione_v2",
  "text": "…",
  "entities": [
    {
      "id": "e1",                      // entity (coreference) id, stable across mentions
      "category": "PER",               // see §3
      "kind": "direct",                // direct | quasi
      "start": 112, "end": 128,        // character offsets, [start, end)
      "surface": "Giulia Ferraro",
      "canonical": "Giulia Ferraro",   // normalized value used for leak matching
      "attrs": {"gender": "f", "number": "sg"}   // only for PER
    }
  ],
  "quasi_profile": {                   // per-document quasi-identifier tuple of the data subject
    "gender": "f", "birth_year": 1987, "city": "Bergamo", "profession": "infermiera",
    "employer": "…", "nationality": "italiana", "marital_status": "coniugata"
  }
}
```

## 3. Identifier categories

| Code | Kind | Description | Italian specifics |
|---|---|---|---|
| PER | direct | person full name, surname alone, first name alone (each mention annotated) | gender attribute for agreement |
| CF | direct | codice fiscale (16 chars, valid checksum) | |
| IBAN | direct | Italian IBAN (IT + 2 check + CIN + ABI + CAB + account), valid mod-97 | |
| PIVA | direct | partita IVA (11 digits, valid Luhn-like check) | |
| EMAIL | direct | e-mail address | |
| PHONE | direct | mobile/landline in Italian formats (+39, 3xx xxx xxxx, 0xx …) | |
| ADDRESS | direct | street + number (+ CAP + city) | |
| DOC_ID | direct | carta d'identità (CA12345AB), passaporto (YA1234567), patente | |
| PLATE | direct | targa auto (AB123CD) | |
| DATE_BIRTH | direct | full date of birth | GDPR treats as direct when combined with name |
| ORG | quasi | employer / counterparty organisation name | |
| LOC | quasi | city of residence/birth | |
| AGE | quasi | explicit age or birth year | |
| PROFESSION | quasi | job title | |
| HEALTH | quasi (special category, Art. 9) | diagnosis, drug, procedure | clinical domain only |
| DATE | quasi | other dates (hearing, admission, hire) | |

"direct" = identifies the data subject alone; "quasi" = identifies in combination.

## 4. Pseudonym mapping schema

Typed pseudonyms carry the information the downstream model needs to produce
grammatical Italian **without** carrying identity:

```
<TYPE>_<attrs>_<counter>        e.g.  PER_f_0012   PER_m_0003   ORG_0002   LOC_0007   CF_0001
```

* `TYPE` ∈ category codes of §3.
* `attrs` only for PER: gender `m|f` (needed for article/adjective/participle agreement:
  "la dott.ssa PER_f_0012 è stata convocata"). Optional `pl` for plural referents.
* `counter` is per-document, per-type, zero-padded, assigned in order of first appearance.
  It is **not** a global identifier: the same person in two documents gets independent counters
  unless the deployment chooses *consistent* mode (see below).

Mapping file `M` (JSON, stays on-premise):

```json
{
  "doc_id": "hr_000123",
  "mode": "per_document",            // per_document | consistent (cross-document, salted HMAC key)
  "entries": [
    {"pseudonym": "PER_f_0012", "category": "PER", "attrs": {"gender": "f"},
     "canonical": "Giulia Ferraro",
     "surfaces": ["Giulia Ferraro", "Ferraro", "Giulia", "G. Ferraro", "FERRARO Giulia"]}
  ]
}
```

Baseline for the agreement experiment: **opaque pseudonyms** `[[a3f9c2e1]]` (8-hex UUID
fragment) carrying no type and no gender.

Reverse mapping `M⁻¹` on the LLM output: longest-match replacement of every pseudonym token,
tolerant to common model mutations (trailing punctuation, possessive, lowercase, lost
underscores). Unresolvable pseudonyms are counted as **reverse-mapping failures** (reported).

## 5. Leak rate

Let `G_c` be the multiset of ground-truth mentions of category `c` over the corpus and
`d'_i` the pseudonymized text of document `i`.

A mention `m ∈ G_c` with surface `s` and canonical value `v` **leaks** in `d'_i` if:

* **strict leak**: `norm(s)` occurs in `norm(d'_i)` as a token-bounded substring, where `norm`
  lowercases, strips diacritics, and collapses whitespace/punctuation inside numeric identifiers
  (so `IT60 X054 2811 1010 0000 0123 456` matches `IT60X0542811101000000123456`). The surface, not
  the canonical value, is tested: a surname-only mention leaks strictly when the surname survives,
  a full-name mention does not leak strictly if only the surname survives (that is a partial leak).
  Dates are tested against every common Italian rendering of the canonical date;
* **partial leak** (reported separately): a *discriminating fragment* of `v` survives:
  for PER the surname token alone (≥4 chars); for CF the first 6 letters (surname+name
  encoding) or the last 5 chars (place+check); for IBAN the last 12 digits (account);
  for PHONE the last 7 digits; for EMAIL the local part itself (as a contiguous string, so that a
  surviving "F. Rossi" is counted once, under PER, and not again under EMAIL); for ADDRESS street
  name + number; for DATE_BIRTH day+month or the year alone (year alone is also counted under AGE).

Also reported for PER: strict leak rate **by surface form** (full name, SURNAME Name, surname only,
first name only, initial + surname, ALL CAPS), which is where the error analysis lives.

```
leak_rate(c) = |{ m ∈ G_c : m leaks (strict) }| / |G_c|
partial_leak_rate(c) = |{ m ∈ G_c : m leaks (strict ∨ partial) }| / |G_c|
doc_leak_rate = |{ i : ∃ direct mention in d_i that leaks (strict) }| / N
```

`doc_leak_rate` is the number that matters operationally: the fraction of documents for which
the provider receives at least one direct identifier.

Also reported: **entity-level leak rate** (an entity leaks if any of its mentions leaks), which
is the right unit for re-identification, and **over-redaction rate** (tokens replaced that are not
inside any ground-truth span, normalized per 1 000 tokens).

### 5.1 Confidence intervals

Mentions inside a document are not independent (the same surname recurs; a pseudonymizer that
misses an entity usually misses all its mentions). Therefore:

* **Primary CI**: cluster bootstrap over documents, B = 2 000 resamples, percentile 95 % CI,
  seeded. Reported for every rate above.
* **Reference CI**: Wilson score interval at the mention level, reported in the appendix only
  to show how much it *understates* uncertainty (anti-conservative under clustering).
* Pairwise comparison between pseudonymizers: paired cluster bootstrap on the difference of
  rates (same document resample for both systems).

Zero leaks in `n` mentions is reported as `0 % (95 % CI 0 – 3/n)` (rule of three) in addition to
the bootstrap, since the bootstrap collapses to a point.

## 6. Re-identification risk from surviving quasi-identifiers

After pseudonymization, a document still exposes a quasi-identifier tuple
`q(d'_i) ⊆ {gender, birth_year or age band, city, profession, employer, nationality, marital_status, diagnosis}`
restricted to the attributes **that survived** in `d'_i` (detected by the same leak matcher on
the quasi-identifier canonical values of `quasi_profile`).

* **Corpus k**: `k_i = |{ j : q(d'_j) = q(d'_i) }|`, i.e. equivalence class size within the
  released corpus. Standard k-anonymity (Sweeney 2002) with the corpus as the population.
* **Population k̂**: an estimate of how many Italian residents share `q(d'_i)`, computed from
  marginal reference tables (`leakrate/data/reference_population.json`: resident population per
  comune, age-band and gender shares, profession prevalence). Independence between attributes is
  assumed (documented limitation; it over-estimates k̂ for correlated attributes, so k̂ is an
  optimistic upper bound and corpus k is the conservative lower bound).
* Reported: distribution of `k` and `k̂` (median, 10th percentile, share of documents with
  `k̂ < 5`, `< 11`, `< 100`), per domain and per pseudonymizer.
* `l`-diversity of the sensitive attribute HEALTH within each equivalence class (clinical domain).

Interpretation threshold: `k̂ < 11` is flagged (the threshold used by several European
statistical offices for microdata release); `k̂ < 5` is "singling out plausible".

## 7. Downstream quality

Tasks `T` (per domain), each run with raw text, typed-pseudonymized text and opaque-pseudonymized
text through the same frontier model, same prompt, default sampling (current Claude models reject
`temperature`; run-to-run variance is estimated by repeat runs with a fresh cache salt, §12.3, and is
reported alongside Δ). Pseudonymized prompts carry one extra sentence telling the model to copy
placeholders verbatim:

| Task | Domain | Output | Scoring |
|---|---|---|---|
| summarize | all | 3-sentence Italian summary | LLM-judge pairwise (raw vs pseudo, position-swapped), plus ROUGE-L vs reference |
| extract | all | JSON of key fields (dates, amounts, parties, outcome) | exact / numeric / 5-char-stem match on **non-identifier** fields against template ground truth; label-set fields skipped (§12.8) |
| classify | legal, hr, clinical | label from closed set | accuracy |
| qa | all | answer to 2 template-generated questions whose answer is not a PII | exact match after normalization |

Primary estimand: `Δ = quality(typed) − quality(raw)` with paired bootstrap CI over documents.
Non-inferiority margin pre-registered at 2 points (absolute). `n = N × |T| × 3 conditions`.

Note on `extract`: fields that *are* identifiers are scored after reverse mapping `M⁻¹`
(the deployment-realistic measurement) and also reported before reverse mapping as the
reverse-mapping failure rate.

## 8. Grammatical agreement (typed vs opaque)

On the frontier output `y'` *before* reverse mapping, for every pseudonym occurrence find the
agreement context: preceding determiner/article (`il, lo, la, i, gli, le, un, una, uno, del,
della, dei, delle, al, alla, dal, dalla, nel, nella`), preceding title (`sig., sig.ra, dott.,
dott.ssa, avv., ing., prof., prof.ssa`), and the first past participle / predicative adjective
in the same clause within 6 tokens (`-o/-a/-i/-e` endings on a lexicon of common participles:
convocato, assunto, nato, residente, licenziato, dimesso, ricoverato, condannato, …).
An **agreement error** is a context whose gender disagrees with the ground-truth gender of the
entity. Rate = errors / contexts, cluster-bootstrap CI over documents. Reported for typed and
opaque; the abstract's `z` = relative reduction.

*Refinement after the pilot (2026-11-05).* Compared with the LLM judge of §12.5 on the 50 pilot
summaries, the first version of the heuristic over-counted by an order of magnitude (29 % vs 0 %):
it took `dott.` as masculine, read the participle of the previous clause (`difeso dall'avv. PER_f`)
as agreeing with the pseudonym, and took articles of the *next* noun phrase as cues. The checker now
(i) treats epicene titles and nouns (`dott.`, `avv.`, `giudice`, `parte`, …) as giving no cue,
(ii) uses only the token immediately before the pseudonym as the before-cue, and (iii) looks for an
after-cue only when the pseudonym is in subject position, skipping auxiliaries and stopping at any
new noun phrase. It remains a secondary, conservative measure; the judged rate is primary for claim 3.

## 9. Pseudonymizers under test

| id | description | runs locally | notes |
|---|---|---|---|
| `regex` | pattern-only baseline (CF, IBAN, PIVA, EMAIL, PHONE, PLATE, DOC_ID, dates, CAP) | yes | expected to leak all PER/ORG/LOC |
| `presidio` | Microsoft Presidio, `it_core_news_lg` NER + built-in and custom Italian recognizers | yes | custom recognizers for CF, PIVA, PLATE, DOC_ID added |
| `gliner` | GLiNER zero-shot NER (`urchade/gliner_multi-v2.1`) with Italian labels + `regex` | yes (torch) | labels prompted in Italian |
| `finetuned` | small encoder fine-tuned on the train split of the benchmark | yes | reported as the upper bound for local models |
| `llm_self` | the frontier model prompted to pseudonymize its own input | no (API) | the "trust the provider anyway" comparison; its leak rate is informational since the provider sees the raw text |

All systems emit the same mapping schema (§4) through a common `Pseudonymizer` interface. Two
reference systems bound the scale: `identity` (replaces nothing; leak rate must be 100 %) and
`oracle` (replaces the gold spans; leak rate must be 0 % for replaced categories).

**Replacement policy.** Which detected categories are replaced is a deployment choice recorded with
every result: `direct_only`, `direct+org+loc` (default: the data subject's name, identifiers,
employer and city go; dates, age, profession and diagnosis stay for utility), or `all`.
**Overlap resolution.** When two detections overlap, a direct-identifier detection wins over a
quasi-identifier one, then the longer span, then the higher score. Without the first rule an IBAN
that spaCy also tags as ORGANIZATION survives under `direct_only`; we observed exactly this.

## 10. Splits and leakage control

Templates are partitioned into train / dev / test **by template family**, so that the fine-tuned
model never sees the test templates. Names, cities and organisations are sampled from disjoint
lists per split. `N` documents per domain; seed recorded in the dataset card.

## 11. Pre-specified claims (to be filled with measured values)

*Status of the pre-specification.* These claims were written in this file before any result was
computed, and the file's edit history is in the repository, but there is no public timestamp (no
OSF registration). The paper therefore calls them *pre-specified*, not pre-registered.

1. `doc_leak_rate(presidio)` and `doc_leak_rate(gliner)` are both > 5 % on the test split
   (i.e. "off-the-shelf local pseudonymization is not negligible-risk").
2. `Δ quality(typed) − quality(raw)` is within the 2-point non-inferiority margin on every task.
3. Agreement-error rate(typed) < agreement-error rate(opaque), relative reduction > 50 %.
4. Median population `k̂` after `presidio` is < 100 for the HR and clinical domains.

**Outcome (2026-11-05, test split, N = 400).**
1. **Supported.** Presidio + regex 41.2 % [36.5, 46.2]; GLiNER + regex 93.8 % [91.5, 96.0].
   The fine-tuned encoder and every `llm_self` variant are at 0.0–0.2 %.
2. **Not supported for extraction and classification; supported for QA.** Opus 5.5 medium, typed:
   QA −0.4 [−1.0, 0.0], extraction −2.5 [−4.0, −1.0], classification −3.8 [−6.2, −1.2] points.
   The `oracle` condition gives extraction 0.0 [−0.4, 0.4] and classification +1.8 [0.0, +3.8]:
   the loss comes from detection errors (over- and under-redaction), not from pseudonymization.
   Repeat runs agree (rep3: −2.4 / −3.0). Sonnet and Haiku lose 4–6 points on extraction.
3. **Not supported as stated.** Judged error rates are ≤ 2 % in both conditions (Sonnet: 0.6 % typed
   vs 1.0 % opaque; Haiku 0.6 % vs 1.9 %); the heuristic gives 2.6 % vs 4.9 % for Opus. The
   measurable effect of typed pseudonyms is that the model commits to a gender at all: 41–43 % of
   person pseudonyms receive gender-marked agreement with typed pseudonyms, 22–24 % with opaque ones.
4. **Not supported.** With ISTAT marginals, median `k̂` after Presidio + regex is ≈ 1 160 (clinical)
   and ≈ 420 (HR); the risk is in the tail: 12.5 % of subjects (surface) and 14.0 % (LLM attacker)
   have `k̂ < 11` (raw text 64 %, oracle 1.8–2.0 %). Case normalization does not change this;
   WikiNEuRal + case normalization brings it to 8.2 %.
Unregistered findings: the LLM attacker recovers the surname from 35 % of typed documents (0 %
after the oracle, 100 % on raw text); the pairwise judge prefers the pseudonymized summaries
(win rate 0.65–0.75) because, after reverse mapping, they carry more identifiers back
(48 % vs 31 % of gold direct identifiers).

## 12a. Reviewer round 1 (2026-11-05, after the first full draft)

Added after an external review of the first PDF, before the corresponding results were inspected:
1. **Case-normalization baseline**: `presidio+casenorm+regex` (and `+casenorm` without regex): detector
   input rewritten (`COGNOME Nome` → `Nome Cognome`; all-caps words ≥3 letters title-cased, acronyms
   excluded) with a char map back to the original; regex and output on the original text.
2. **Second supervised off-the-shelf NER**: `wikineural+regex` (Babelscape/wikineural-multilingual-ner),
   with and without case normalization.
3. **Cross-template fine-tuning**: train on two domains, evaluate on the other two, both folds
   (`scripts/cross_template.sh`, system `finetuned-xt+regex`); the in-distribution 0 % is reported
   only as such and removed from the abstract, key numbers and conclusion.
4. **Marginals**: ISTAT tables replace the illustrative values (`leakrate/data/reference.py`, sources in
   the module docstring; `data/reference_population.json`). All k̂ figures recomputed.
5. **Second-house attacker**: an open-weight model (Qwen2.5-14B-Instruct via Ollama, greedy decoding,
   local GPU) runs the attacker prompt on the typed condition; `--model ollama:<name>`.
6. **Judge win rate** removed from the downstream table and kept in the text as a diagnosed artifact;
   a judge instructed to ignore identifier completeness is planned when credits allow.
7. "Pre-registered" → "pre-specified" throughout (§11); `llm_self` reframed as deployable under a
   processor contract; intro claim count corrected; judge/attacker/consumer single-provider limitation
   acknowledged and partially addressed by item 5.
Outstanding and needing the author: real documents (even 30 HR letters the company may use).

**Outcomes (2026-11-05 evening).** (1) Presidio + case norm. + regex: doc leak 19.5 % [15.5, 23.2]
(from 41.2), direct-mention 2.8 %, COGNOME Nome 20.9 → 0.0 %, ALL CAPS 32.8 → 5.2 %, over-redaction
15.9 → 17.3. (2) WikiNEuRal + regex 56.5 %; + case norm. 11.0 % [8.2, 14.2], direct-mention 1.2 %,
over-redaction 5.1: best off-the-shelf configuration. (3) Cross-template fine-tuned (2 folds): doc
leak 0.0 % again, over-redaction 1.4 → 31.0 per 1k tokens (mostly ORG and PER false positives on
function words and acronyms): forms transfer, precision does not. (4) ISTAT marginals: k̂<11 12.5 %
(Presidio + regex), 8.2 % (WikiNEuRal + case norm.), 1.8 % oracle, 63.7 % identity; attacker 14.0 %
typed, 64 % raw, 2 % oracle. (5) Qwen2.5-14B (local, Ollama): surname 31.5 % typed / 100 % raw /
0 % oracle, k̂<11 10.8 %. Qwen2.5-72B-Instruct-AWQ (vLLM on a RunPod H100, greedy) on all six
conditions: surname 100 / 33.8 / 34.8 / 45.0 / 33.5 / 0.0 % (raw, typed, opaque, direct_only, all,
oracle), k̂<11 64.2 / 12.0 / 14.0 / 44.5 / 12.0 / 1.8 %: within two points of Opus everywhere.
(6) Judge on the Opus medium run with Qwen2.5-72B: default prompt 0.66 typed / 0.66 opaque
(reproduces Sonnet 0.68 / 0.73 and Opus 0.67 / 0.72, so the preference is the prompt's, not the
judge's); completeness-blind prompt 0.39 typed / 0.40 opaque / 0.485 oracle (n = 400 each). The
default preference is an artifact; debiased, pseudonymized summaries are ~10 points below parity and
the oracle is at parity, so the deficit comes from detector errors, as for extraction and classification.

## 12. Extended frontier protocol (added 2026-11-05, before any extended result was inspected)

The 50-document pilot (§7, Opus 5.5) cost 0.6 ¢ per call, so the frontier budget allows a wider
design than the one above. All additions below were specified before the corresponding runs were
scored; every call is cached by (model, effort, salt, prompt), so re-scoring is free and a run can be
resumed after an interruption without re-billing.

**12.1 Policy sweep (privacy–utility frontier).** Downstream quality is measured not only for the
default policy but for `direct_only`, `direct+org+loc`, `all` (Presidio + regex, typed) and for the
`oracle` pseudonymizer under the default policy. Together with the leak rates of §5 this gives one
point per policy on the (leak rate, Δ quality) plane; `oracle` isolates the cost of pseudonymization
itself from the cost of detection errors.

**12.2 Consumer-model sweep.** The same raw / typed / opaque protocol is run with Opus 5.5 (effort
low, medium and high), Sonnet 5.5 and Haiku 5.5 as the consumer. Pre-registered expectation: the
typed-vs-opaque gap in agreement errors is largest for the smallest model; Δ quality is within the
margin for all of them.

**12.3 Run-to-run variance.** The typed and raw conditions are repeated with a fresh cache salt on
a stratified 100-document subsample (two extra runs) and once on the full test split (one extra
run). The spread of Δ across runs is reported next to the bootstrap CI of the primary run; if the
spread exceeds the CI half-width, the CI is widened accordingly in the paper (reported both ways).

**12.4 Cross-judge.** The pairwise summary judge (Sonnet 5.5 by default) is repeated with Opus 5.5
on the typed and opaque conditions of the Opus consumer run; judge agreement (fraction of documents
with the same per-document preference) is reported.

**12.5 LLM-judged agreement.** Besides the heuristic checker of §8, a judge model receives each
pseudonymized summary plus the gold gender of every person pseudonym and reports, per pseudonym,
`coerente` / `incoerente` / `nessun_accordo`. Error rate = incoerente / (coerente + incoerente),
cluster bootstrap over documents. The heuristic and judged rates are both reported; the judged rate
is primary for claim 3 because the heuristic cannot resolve which noun a participle modifies.
*Prompt revisions.* (1) After the pilot: epicene titles (`dott.`, `avv.`, `giudice`) are not gender
cues. (2) After the Sonnet/Haiku runs: generic masculine role nouns of the bureaucratic register
(`il gestore`, `il titolare`, `geom.`, `rag.`) are not gender cues either, and only an element that
refers directly to the person and contradicts the gender unambiguously (wrong marked title, wrong
participle/adjective ending, wrong pronoun) counts as `incoerente`; in doubt, `nessun_accordo`.
Inspection of the first-version verdicts showed 'il gestore, geom. PER_f_0004' flagged as an error.
Both versions' aggregate rates are kept in the results log; the paper reports the final version.

**12.6 LLM attacker (inference, not surface survival).** For each condition (raw text, typed,
opaque, `direct_only`, `all`, `oracle`) the frontier model is prompted as a re-identification
analyst and asked to infer the data subject's gender, age, city, profession, employer, nationality,
marital status, diagnosis and, if reconstructible, surname and first name. Scoring against the gold
`quasi_profile`: exact normalized match for categorical attributes, ±2 years for age; recall is
`correct / applicable`, precision is `correct / claimed`. The population `k̂` of §6 is recomputed on
the tuple of *correctly* recovered attributes with the same reference marginals, so the inference-
based `k̂` distribution is directly comparable to the surface-survival one. Surname recovery on the
pseudonymized conditions is the leak rate seen through the attacker's eyes and must be ≤ the
document leak rate of §5. The attack is also run with Sonnet 5.5 on the typed condition to show how
much the result depends on attacker strength. Raw text is the ceiling.

**12.7 `llm_self` sweep.** The frontier-as-anonymizer condition is run with Opus 5.5 (effort low and
medium), Sonnet 5.5 and Haiku 5.5, each on the full test split, and evaluated with §5–§6 like any
local system. Each system id carries model and effort (`llm_self-opus-5-5-low`).

**12.8 Scoring fixes from the pilot** (applied before scoring any extended run): numbers are
canonicalized from Italian, English or numeric form; gold slugs (`decreto_ingiuntivo`) match
their spaced form; `esito` fields match on 5-character stems (`rigetta` ≈ `rigetto`); extraction
fields whose gold value belongs to the classify label set are skipped (they are measured by
`classify`); judge verdicts are parsed as the first `A` / `B` / `TIE` token, so `**B**` counts.
Two further fixes after the first full-split run (Sonnet 5.5) was inspected, before any Δ was read:
(a) judges that reason before answering put the verdict *last*, and Italian prose contains the words
*a* and *e*, so the verdict is now the last standalone `A`/`B`/`TIE` token (first token only for
bare answers; "entrambi … equivalenti" = TIE); (b) the template generator records extraction fields
that the chosen paragraph variant did not render (e.g. CCNL level in a dismissal letter: 64 % of
clinical `terapia`, 54 % of HR `ral_euro`, 41 % of financial `importo_euro` values are not in the
text). Extraction is scored only on fields whose gold value is *stated* in the document (verbatim,
or as a ≥4-character normalized string, or as a ≥3-digit number among the document's numbers);
documents with no stated field are excluded from the extraction task. The same rule applies to QA:
a question whose gold answer is not stated in the text is not scored (the raw-text accuracy of
0.804 for both Opus and Sonnet was exactly the share of stated answers). Both fixes apply to raw and
pseudonymized conditions alike, so Δ is unaffected in expectation; absolute raw accuracy is not.
(c) Attacker matching is inflection-tolerant for categorical attributes (5-character stems:
*pensionato* ≈ *pensionata*; *celibe* ≈ *nubile*; *coniugato* ≈ *sposata*), after the raw-text
ceiling showed profession recall of 0.36 with precision 0.55 for what were gendered forms of the
same word. All of (a)–(c) are scoring changes; no call was repeated.

**12.9 Budget.** Credits of 300 USD / 30 days. Planned spend ≈ 250 USD: Opus consumer runs ≈ 130,
attacker ≈ 25, `llm_self` ≈ 20, Sonnet/Haiku lanes ≈ 30, judges and agreement judge ≈ 20, variance
≈ 25. `scripts/cost.py` sums actual spend from the cache at list prices.
*Actual run (2026-11-05).* Cache-based spend at the end of all lanes: 272.86 USD (Opus 5.5 ≈ 150,
Sonnet 5.5 ≈ 55, Haiku 5.5 ≈ 1.5) before the final agreement judge (≈ 3). The Opus lane was
relaunched once with a revised stage order; because the first launch kept running for about
25 minutes after it was believed stopped, stages A5–A6 ran twice concurrently and a few thousand
calls were billed twice but cached once, so `cost.py` undercounts by an estimated 15–25 USD. The
high-effort consumer run (A7) was limited to 100 stratified documents for this reason. The
provider's console, not the cache, is the ground truth for spend.
