#!/usr/bin/env bash
# Opus lane, second launch (resumes from the cache): same stages as run_frontier_full.sh LANE=opus, with the
# high-effort run limited to 100 stratified documents to stay inside the 300 USD budget, and ordered so that
# the cheap, essential stages come first.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ] && [ -f .venv-path ]; then eval "$(grep '^PY=' .venv-path)"; fi
PY="${PY:-python}"
export PYTHONUTF8=1
W="${W:-12}"
P=results/pseudo
TYPED=$P/presidio-regex__typed__direct-org-loc__test.jsonl
OPAQUE=$P/presidio-regex__opaque__direct-org-loc__test.jsonl
DONLY=$P/presidio-regex__typed__direct_only__test.jsonl
ALL=$P/presidio-regex__typed__all__test.jsonl
ORACLE=$P/oracle__typed__direct-org-loc__test.jsonl
mkdir -p results/frontier results/logs
stage() { echo; echo "=================== [$(date +%H:%M:%S)] $*"; }
cost()  { "$PY" scripts/cost.py | tail -1; }

stage "A1  Opus 5.5 consumer, medium, 6 conditions (resume: consumer calls cached, judge finishes)"
"$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --pseudo direct_only=$DONLY --pseudo all=$ALL --pseudo oracle=$ORACLE \
   --out results/frontier/downstream__opus-5-5__medium.json > /dev/null
cost

stage "A3  LLM attacker (Opus 5.5, medium) on raw and on every pseudonymized condition"
"$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo raw     --out results/frontier/attack__opus-5-5__raw.json > /dev/null
"$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $TYPED  --out results/frontier/attack__opus-5-5__typed.json > /dev/null
"$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $OPAQUE --out results/frontier/attack__opus-5-5__opaque.json > /dev/null
"$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $DONLY  --out results/frontier/attack__opus-5-5__direct_only.json > /dev/null
"$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $ALL    --out results/frontier/attack__opus-5-5__all.json > /dev/null
"$PY" -m leakrate attack --model claude-opus-5-5 --workers "$W" --pseudo $ORACLE --out results/frontier/attack__opus-5-5__oracle.json > /dev/null
cost

stage "A4  Opus 5.5 as its own anonymizer (llm_self), low and medium effort"
for EFF in low medium; do
  "$PY" -m leakrate pseudonymize --system llm_self --llm-model claude-opus-5-5 --llm-effort $EFF --workers "$W" --split test
  "$PY" -m leakrate evaluate --pseudo $P/llm_self-opus-5-5-${EFF}__typed__direct-org-loc__test.jsonl | head -3
done
cost

stage "A5  Run-to-run variance: typed vs raw, Opus medium, 100 stratified docs, two extra runs"
for TAG in rep1 rep2; do
  "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" --limit 100 --run-tag $TAG --no-judge \
     --pseudo typed=$TYPED --out results/frontier/downstream__opus-5-5__medium__${TAG}.json > /dev/null
done
cost

stage "A2  Opus 5.5 consumer, low effort (raw/typed/opaque)"
"$PY" -m leakrate downstream --model claude-opus-5-5 --effort low --workers "$W" \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out results/frontier/downstream__opus-5-5__low.json > /dev/null
cost

stage "A6  Cross-judge: Opus 5.5 as the summary judge (typed/opaque)"
"$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" --judge-model claude-opus-5-5 \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out results/frontier/downstream__opus-5-5__medium__opusjudge.json > /dev/null
cost

stage "A8  Full-split replicate of the typed condition (Opus medium, rep3)"
"$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" --run-tag rep3 --no-judge \
   --pseudo typed=$TYPED --out results/frontier/downstream__opus-5-5__medium__rep3.json > /dev/null
cost

stage "A7  Opus 5.5 consumer, high effort (raw/typed), 100 stratified docs"
"$PY" -m leakrate downstream --model claude-opus-5-5 --effort high --workers "$W" --no-judge --limit 100 \
   --pseudo typed=$TYPED --out results/frontier/downstream__opus-5-5__high.json > /dev/null
cost

stage "done (opus2)"
cost
