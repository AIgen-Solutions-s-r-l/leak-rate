#!/usr/bin/env bash
# One-shot frontier pilot: downstream quality + agreement (typed vs opaque) and llm_self as anonymizer.
# Needs ANTHROPIC_API_KEY in the environment or in ./.env. Calls are cached in results/cache/, so re-runs are free.
set -euo pipefail
cd "$(dirname "$0")/.."
# .venv-path holds a line "PY=/path/to/venv/Scripts/python"; fall back to the active python
if [ -z "${PY:-}" ] && [ -f .venv-path ]; then eval "$(grep '^PY=' .venv-path)"; fi
PY="${PY:-python}"
"$PY" -c "import anthropic" 2>/dev/null || { echo "python at $PY lacks the anthropic SDK: pip install anthropic"; exit 2; }
N="${N:-50}"
export PYTHONUTF8=1

echo "== downstream quality + agreement on $N test documents"
"$PY" -m leakrate downstream \
  --pseudo-typed  results/pseudo/presidio-regex__typed__direct-org-loc__test.jsonl \
  --pseudo-opaque results/pseudo/presidio-regex__opaque__direct-org-loc__test.jsonl \
  --limit "$N" --out results/downstream.json

echo "== frontier model as its own anonymizer (llm_self) on $N test documents"
"$PY" -m leakrate pseudonymize --system llm_self --mode typed --policy direct+org+loc --split test --limit "$N"
"$PY" -m leakrate evaluate --pseudo results/pseudo/llm_self-opus-5-5-low__typed__direct-org-loc__test.jsonl

echo "== regenerate summary and paper tables"
"$PY" -m leakrate report
"$PY" scripts/make_tables.py
"$PY" scripts/check_tex.py
