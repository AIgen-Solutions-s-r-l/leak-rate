#!/usr/bin/env bash
# Full frontier protocol on the 400-document test split. Two independent lanes so that the per-model
# rate limits are used in parallel:  LANE=opus  (Opus 5.5 as consumer, attacker and anonymizer)
#                                    LANE=small (Sonnet 5.5 and Haiku 5.5 as consumer and anonymizer)
# Every call is cached in results/cache/, so a lane can be re-run after an interruption at no cost.
# Spend is printed after every stage (scripts/cost.py). Needs ANTHROPIC_API_KEY in env or ./.env.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ] && [ -f .venv-path ]; then eval "$(grep '^PY=' .venv-path)"; fi
PY="${PY:-python}"
export PYTHONUTF8=1
LANE="${LANE:-opus}"
W="${W:-12}"                      # parallel requests per stage
P=results/pseudo
TYPED=$P/presidio-regex__typed__direct-org-loc__test.jsonl
OPAQUE=$P/presidio-regex__opaque__direct-org-loc__test.jsonl
DONLY=$P/presidio-regex__typed__direct_only__test.jsonl
ALL=$P/presidio-regex__typed__all__test.jsonl
ORACLE=$P/oracle__typed__direct-org-loc__test.jsonl
mkdir -p results/frontier results/logs

stage() { echo; echo "=================== [$(date +%H:%M:%S)] $*"; }
cost()  { "$PY" scripts/cost.py | tail -1; }

if [ "$LANE" = "opus" ]; then
  stage "A1  Opus 5.5 consumer, medium effort, 6 conditions (raw/typed/opaque/direct_only/all/oracle)"
  "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" \
     --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --pseudo direct_only=$DONLY --pseudo all=$ALL --pseudo oracle=$ORACLE \
     --out results/frontier/downstream__opus-5-5__medium.json
  cost

  stage "A2  Opus 5.5 consumer, low effort (raw/typed/opaque)"
  "$PY" -m leakrate downstream --model claude-opus-5-5 --effort low --workers "$W" \
     --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out results/frontier/downstream__opus-5-5__low.json
  cost

  stage "A3  LLM attacker (Opus 5.5, medium) on raw and on every pseudonymized condition"
  "$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo raw     --out results/frontier/attack__opus-5-5__raw.json
  "$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $TYPED  --out results/frontier/attack__opus-5-5__typed.json
  "$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $OPAQUE --out results/frontier/attack__opus-5-5__opaque.json
  "$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $DONLY  --out results/frontier/attack__opus-5-5__direct_only.json
  "$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $ALL    --out results/frontier/attack__opus-5-5__all.json
  "$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $ORACLE --out results/frontier/attack__opus-5-5__oracle.json
  cost

  stage "A4  Opus 5.5 as its own anonymizer (llm_self), low and medium effort"
  for EFF in low medium; do
    "$PY" -m leakrate pseudonymize --system llm_self --llm-model claude-opus-5-5 --llm-effort $EFF --workers "$W" --split test
    "$PY" -m leakrate evaluate --pseudo $P/llm_self-opus-5-5-${EFF}__typed__direct-org-loc__test.jsonl
  done
  cost

  stage "A5  Run-to-run variance: typed vs raw, Opus medium, 100 docs, two extra runs with a fresh cache salt"
  for TAG in rep1 rep2; do
    "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" --limit 100 --run-tag $TAG --no-judge \
       --pseudo typed=$TYPED --out results/frontier/downstream__opus-5-5__medium__${TAG}.json
  done
  cost

  stage "A6  Cross-judge: Opus 5.5 as the summary judge for the Opus consumer run (typed/opaque)"
  "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" --judge-model claude-opus-5-5 \
     --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out results/frontier/downstream__opus-5-5__medium__opusjudge.json
  cost

  stage "A7  Opus 5.5 consumer, high effort (raw/typed) — does more thinking help with pseudonyms?"
  "$PY" -m leakrate downstream --model claude-opus-5-5 --effort high --workers "$W" --no-judge \
     --pseudo typed=$TYPED --out results/frontier/downstream__opus-5-5__high.json
  cost

  stage "A8  Full-split replicate of the typed condition (Opus medium, rep3) for the headline Δ"
  "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" --run-tag rep3 --no-judge \
     --pseudo typed=$TYPED --out results/frontier/downstream__opus-5-5__medium__rep3.json
  cost
fi

if [ "$LANE" = "small" ]; then
  for M in claude-sonnet-5-5 claude-haiku-5-5; do
    SHORT=${M#claude-}
    stage "B1  $M consumer (raw/typed/opaque), default-ish effort"
    "$PY" -m leakrate downstream --model $M --effort medium --workers "$W" \
       --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out results/frontier/downstream__${SHORT}__medium.json
    cost
    stage "B2  $M as its own anonymizer (llm_self, low effort)"
    "$PY" -m leakrate pseudonymize --system llm_self --llm-model $M --llm-effort low --workers "$W" --split test
    "$PY" -m leakrate evaluate --pseudo $P/llm_self-${SHORT}-low__typed__direct-org-loc__test.jsonl
    cost
  done
  stage "B3  LLM attacker with Sonnet 5.5 on the typed condition (attacker-strength check)"
  "$PY" -m leakrate attack --model claude-sonnet-5-5 --workers "$W" --pseudo $TYPED --out results/frontier/attack__sonnet-5-5__typed.json
  cost
fi

stage "done ($LANE)"
cost
