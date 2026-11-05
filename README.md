# Leak Rate, Not Loss

Measuring what actually escapes when a local model pseudonymizes text for frontier LLMs.

This repository holds the benchmark generator, the pseudonymizer adapters, the metrics
(leak rate with cluster-bootstrap CIs, k-anonymity on surviving quasi-identifiers, grammatical
agreement), the downstream-quality harness and the paper draft.

* `METHODS.md`: the normative definitions (metrics, schema, pre-registered claims). Code and paper follow it.
* `leakrate/`: Python package (see module docstrings).
* `paper/main.tex`, `paper/references.bib`: paper draft.
* `data/`: generated benchmark (`benchmark.jsonl`), dataset card, reference marginals.
* `results/`: pseudonymized outputs, per-system evaluations, `summary.md` with all tables.
* `tests/`: 14 tests covering checksums, generator integrity, sanity bounds (identity = 100 % leak, oracle = 0 %), metrics.

## Setup

The code needs Python ≥ 3.11. On Windows the virtual environment must live in a **short path**
(spaCy's compiled extensions fail to load from deep directories):

```bash
python -m venv C:/Users/<you>/.venvs/leakrate
C:/Users/<you>/.venvs/leakrate/Scripts/python -m pip install -e ".[presidio,dev]"
C:/Users/<you>/.venvs/leakrate/Scripts/python -m spacy download it_core_news_lg
```

Optional extras: `.[gliner]` (GLiNER + torch), `.[llm]` (Anthropic SDK for `llm_self` and the downstream experiments).
Set `PYTHONUTF8=1` on Windows so Italian text prints correctly.

## Reproduce the local pipeline (no API access needed)

```bash
python -m leakrate generate --n 500                       # 2 000 docs, 70/10/20 split, seed 20251105
python -m leakrate pseudonymize --system presidio+regex  # test split, typed pseudonyms, policy direct+org+loc
python -m leakrate pseudonymize --system presidio+regex --mode opaque
python -m leakrate pseudonymize --system regex
python -m leakrate pseudonymize --system oracle           # sanity ceiling (uses gold spans)
python -m leakrate pseudonymize --system identity         # sanity floor
python -m leakrate evaluate --pseudo results/pseudo/presidio-regex__typed__direct-org-loc__test.jsonl
python -m leakrate report                                 # results/summary.md
```

Systems: `regex`, `presidio`, `presidio+regex`, `gliner+regex` (needs torch), `llm_self` (needs API), `oracle`, `identity`.
Policies (`--policy`): `direct_only`, `direct+org+loc` (default), `all`. Modes: `typed` (PER_f_0012 …), `opaque` ([[a3f9c2e1]]).

## Building the PDF

MiKTeX (user install, `winget install MiKTeX.MiKTeX --scope user`, missing packages auto-installed):

```bash
cd paper && pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main
```

Regenerate `paper/tables.tex`, `paper/tables_frontier.tex` and the figures in `paper/figures/` first
(`python scripts/make_tables.py`, `python scripts/make_frontier_tables.py`, `python scripts/make_figures.py`)
whenever results change; wide tables are shrunk with `adjustbox`. House style (AIgen Solutions working
paper): sans-serif blue headings, abstract box, key-numbers box, running headers; the pipeline diagram
is TikZ inside `main.tex`. Style rule: no em dashes anywhere in the text.

## Frontier-model experiments (need an Anthropic API key)

Status 2026-11-05: the full protocol below has been run once (≈ 57 000 calls, ≈ 281 USD at list prices
from the cache; the credits of the key used were exhausted at the very end, during the Opus agreement
judge, which is therefore reported only for Sonnet and Haiku). Results: `results/frontier/*.json`,
`results/frontier_summary.md`, `paper/tables_frontier.tex`. Re-running any command re-uses the cache.

Put the key in the environment (`export ANTHROPIC_API_KEY=...`) or in a `.env` file in the project
root (`ANTHROPIC_API_KEY=...`; the file is git-ignored and read by every `leakrate` command). Then the
whole pilot (downstream quality + agreement on typed vs opaque pseudonyms, `llm_self` as anonymizer,
summary and paper tables) is one command:

```bash
bash scripts/run_frontier_pilot.sh          # N=50 test documents by default; N=400 for the full split
```

The individual steps:

```bash
python -m leakrate downstream --pseudo-typed results/pseudo/presidio-regex__typed__direct-org-loc__test.jsonl \
    --pseudo-opaque results/pseudo/presidio-regex__opaque__direct-org-loc__test.jsonl \
    --limit 50 --dry-run            # prints calls and ~input tokens, bills nothing
python -m leakrate downstream ... --model claude-opus-5-5 --out results/downstream.json
python -m leakrate pseudonymize --system llm_self        # frontier model as its own anonymizer
```

### Extended protocol (METHODS.md §12)

```bash
LANE=opus  W=12 bash scripts/run_frontier_full.sh   # Opus 5.5: 6-condition policy sweep, low/high effort, attacker, llm_self, variance, cross-judge
LANE=small W=8  bash scripts/run_frontier_full.sh   # Sonnet 5.5 and Haiku 5.5 as consumer and anonymizer, Sonnet attacker
bash scripts/run_frontier_extra.sh                  # re-score from cache, LLM-judged agreement, tables (paper/tables_frontier.tex)
python scripts/cost.py                              # spend so far, from the cache, at list prices
```

New subcommands: `downstream --pseudo label=path` (any number of conditions, `--run-tag` for repeat runs that
bypass the cache), `attack --pseudo <jsonl|raw>` (LLM attacker inferring the data subject's quasi-identifiers
and name; inference-based k̂), `agreement-judge --records <run>.records.jsonl` (LLM-judged grammatical
agreement), `pseudonymize --system llm_self --llm-model ... --llm-effort ... --workers N`. Both lanes can run
at the same time; interrupted lanes resume from the cache. Output: `results/frontier/*.json`,
`results/frontier_summary.md`.

Every call is cached under `results/cache/` keyed by (model, effort, prompt), so re-running the
analysis is free. Refusals are recorded as outcomes and scored as failures; server-side model
fallbacks are deliberately disabled so the model under test never changes silently. The full
protocol (400 test documents × 4 tasks × 3 conditions ≈ 6 000 calls plus the pairwise judge) costs
in the order of a few tens of euros at Opus 5.5 rates ($4/$20 per MTok); the 50-document pilot is 750 calls, ≈263k input tokens plus 100 judge calls (Sonnet 5.5, $2/$10), i.e. roughly 10–20 USD depending on how much the model thinks at effort `medium`. Use `--limit` for a pilot first. For the
full run the Message Batches API (50 % discount) is the better transport; the runner is
deliberately simple and synchronous.

## Reviewer round 1 (METHODS.md §12a)

Local, no API credits needed: `presidio+casenorm+regex` and `wikineural+casenorm+regex` (case normalization
of the detector input, `leakrate/pseudonymizers/casenorm.py`; WikiNEuRal via `transformers`),
`scripts/cross_template.sh` (fine-tuned encoder trained on two domains, evaluated on the other two),
ISTAT marginals in `leakrate/data/reference.py` (sources in the docstring), and a second-house attacker
through Ollama: `python -m leakrate attack --model ollama:qwen2.5:14b --pseudo <jsonl> --out ...`
(install Ollama, `ollama pull qwen2.5:14b`; the harness caches these calls like the API ones). Any
OpenAI-compatible server works as `--model openai:<served-name>` with `LEAKRATE_OPENAI_BASE_URL` (and
`LEAKRATE_OPENAI_API_KEY` if needed); `scripts/runpod_jobs.sh` is the script that ran Qwen2.5-72B-Instruct-AWQ
on a RunPod H100 (vLLM) as attacker on all six conditions and as summary judge with the default and the
completeness-blind prompt (`python -m leakrate judge --blind --records <run>.records.jsonl ...`), then
terminated the pod. Note: the RunPod proxy rejects urllib's default User-Agent with 403, so the backend sets one.

## Status of the pre-specified claims (METHODS.md §11)

| # | claim | status |
|---|---|---|
| 1 | off-the-shelf local pseudonymization leaks in > 5 % of documents | **supported**: presidio+regex 41.2 % [36.5, 46.2]; GLiNER+regex 93.8 %; WikiNEuRal+regex 56.5 %. Case normalization of the detector input (reviewer baseline) halves Presidio to 19.5 % and brings WikiNEuRal to 11.0 % [8.2, 14.2] with no training |
| 2 | typed pseudonyms are non-inferior to raw text on downstream tasks | **supported for QA only** (Opus 5.5: QA −0.4 [−1.0, 0.0]; extraction −2.5 [−4.0, −1.0]; classification −3.8 [−6.2, −1.2] points). The `oracle` condition is at 0.0 / +1.8: the loss is detection error, not pseudonymization. Replicated (rep3: −2.4 / −3.0). Sonnet/Haiku lose 4–6 points on extraction |
| 3 | typed pseudonyms halve agreement errors vs opaque | **not supported as stated**: judged errors ≤ 2 % in both conditions (Sonnet 0.6 vs 1.0 %, Haiku 0.6 vs 1.9 %; heuristic on Opus 2.6 vs 4.9 %). The real effect: 41–43 % of person pseudonyms get gender-marked agreement with typed pseudonyms vs 22–24 % with opaque ones |
| 4 | median population k̂ < 100 for HR and clinical after presidio | **not supported** at the default policy (k̂ median ≈ 1 100 clinical / ≈ 840 HR); the 10th percentile is ≈ 1 and 10–21 % of documents have k̂ < 11: the paper reports the distribution, not the median |

## Replacing synthetic data with real documents

The generator exists because real Italian legal / HR / clinical documents cannot be released.
The annotation schema (`METHODS.md` §2) is deliberately generic: a real, consented corpus annotated
with the same categories can be dropped into `data/benchmark.jsonl` and the whole pipeline runs unchanged.
