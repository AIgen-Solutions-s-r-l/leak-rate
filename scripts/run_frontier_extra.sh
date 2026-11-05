#!/usr/bin/env bash
# Post-processing after the two lanes of run_frontier_full.sh: LLM-judged agreement on every consumer run,
# re-scoring of every run from the cache with the current scorers (free), summary tables.
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
F=results/frontier

stage() { echo; echo "=================== [$(date +%H:%M:%S)] $*"; }

stage "C1  re-score every consumer run from the cache (current scorers)"
[ -f $F/downstream__opus-5-5__medium.json ] && "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --pseudo direct_only=$DONLY --pseudo all=$ALL --pseudo oracle=$ORACLE \
   --out $F/downstream__opus-5-5__medium.json > /dev/null
[ -f $F/downstream__opus-5-5__low.json ] && "$PY" -m leakrate downstream --model claude-opus-5-5 --effort low --workers "$W" \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out $F/downstream__opus-5-5__low.json > /dev/null
[ -f $F/downstream__opus-5-5__high.json ] && "$PY" -m leakrate downstream --model claude-opus-5-5 --effort high --workers "$W" --no-judge --limit 100 \
   --pseudo typed=$TYPED --out $F/downstream__opus-5-5__high.json > /dev/null
for M in sonnet-5-5 haiku-5-5; do
  [ -f $F/downstream__${M}__medium.json ] && "$PY" -m leakrate downstream --model claude-$M --effort medium --workers "$W" \
     --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out $F/downstream__${M}__medium.json > /dev/null
done
for TAG in rep1 rep2; do
  [ -f $F/downstream__opus-5-5__medium__${TAG}.json ] && "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" \
     --limit 100 --run-tag $TAG --no-judge --pseudo typed=$TYPED --out $F/downstream__opus-5-5__medium__${TAG}.json > /dev/null
done
[ -f $F/downstream__opus-5-5__medium__rep3.json ] && "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" \
   --run-tag rep3 --no-judge --pseudo typed=$TYPED --out $F/downstream__opus-5-5__medium__rep3.json > /dev/null
[ -f $F/downstream__opus-5-5__medium__opusjudge.json ] && "$PY" -m leakrate downstream --model claude-opus-5-5 --effort medium --workers "$W" \
   --judge-model claude-opus-5-5 --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out $F/downstream__opus-5-5__medium__opusjudge.json > /dev/null

stage "C2  LLM-judged agreement (Sonnet 5.5 judge) on the summaries of every consumer run"
# typed and opaque only (the budget-relevant comparison); Sonnet/Haiku judged earlier with the same prompt version
[ -f $F/downstream__opus-5-5__medium.records.jsonl ] && "$PY" -m leakrate agreement-judge --workers "$W" \
   --records $F/downstream__opus-5-5__medium.records.jsonl --pseudo typed=$TYPED --pseudo opaque=$OPAQUE \
   --out $F/agreement_judge__opus-5-5__medium.json > /dev/null
for M in sonnet-5-5 haiku-5-5; do
  [ -f $F/agreement_judge__${M}__medium.json ] || "$PY" -m leakrate agreement-judge --workers "$W" \
     --records $F/downstream__${M}__medium.records.jsonl --pseudo typed=$TYPED --pseudo opaque=$OPAQUE \
     --out $F/agreement_judge__${M}__medium.json > /dev/null
done

stage "C3  tables"
"$PY" -m leakrate report
"$PY" scripts/make_tables.py
"$PY" scripts/make_frontier_tables.py
"$PY" scripts/check_tex.py
"$PY" scripts/cost.py
