#!/usr/bin/env bash
# Cross-template evaluation of the fine-tuned encoder: train on two domains, evaluate on the other two,
# both folds, then concatenate the held-out-domain outputs into one test-split file (system finetuned-xt+regex).
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ] && [ -f .venv-path ]; then eval "$(grep '^PY=' .venv-path)"; fi
PY="${PY:-python}"
export PYTHONUTF8=1
P=results/pseudo
OUT=$P/finetuned-xt-regex__typed__direct-org-loc__test.jsonl
rm -f "$OUT"
for FOLD in "A hr,clinical legal financial" "B legal,financial hr clinical"; do
  set -- $FOLD; NAME=$1; TRAIN=$2; shift 2
  echo "== fold $NAME: train on $TRAIN, evaluate on $*"
  "$PY" scripts/train_finetuned.py --domains "$TRAIN" --epochs 2 --batch 16 --out "models/finetuned-xt-$NAME" 2>&1 | grep -E "epoch|saved|windows"
  for D in "$@"; do
    LEAKRATE_FINETUNED_DIR="models/finetuned-xt-$NAME" LEAKRATE_FINETUNED_NAME="finetuned-xt" \
      "$PY" -m leakrate pseudonymize --system finetuned+regex --split test --domain "$D" --out "$P/xt_${NAME}_${D}.jsonl" 2>&1 | grep -v -i "warn\|Loading"
    cat "$P/xt_${NAME}_${D}.jsonl" >> "$OUT"
  done
done
"$PY" -m leakrate evaluate --pseudo "$OUT" | head -12
