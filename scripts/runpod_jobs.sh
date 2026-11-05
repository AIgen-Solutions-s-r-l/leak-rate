#!/usr/bin/env bash
# Second-house attacker (Qwen2.5-72B-Instruct-AWQ on a RunPod H100 via vLLM) on every condition, and the
# completeness-blind summary judge on the Opus consumer run. Waits for the vLLM server, runs, then
# terminates the pod whatever happens. Usage: POD_ID=<id> bash scripts/runpod_jobs.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ] && [ -f .venv-path ]; then eval "$(grep '^PY=' .venv-path)"; fi
PY="${PY:-python}"
export PYTHONUTF8=1
POD_ID="${POD_ID:?set POD_ID}"
K=$(grep RUNPOD_API_KEY .env | cut -d= -f2)
export LEAKRATE_OPENAI_BASE_URL="https://${POD_ID}-8000.proxy.runpod.net/v1"
MODEL="openai:qwen72b"
P=results/pseudo; F=results/frontier
TYPED=$P/presidio-regex__typed__direct-org-loc__test.jsonl
OPAQUE=$P/presidio-regex__opaque__direct-org-loc__test.jsonl
DONLY=$P/presidio-regex__typed__direct_only__test.jsonl
ALL=$P/presidio-regex__typed__all__test.jsonl
ORACLE=$P/oracle__typed__direct-org-loc__test.jsonl
W="${W:-16}"

terminate() { echo "== terminating pod $POD_ID"; curl -s --max-time 60 -X DELETE -H "Authorization: Bearer $K" "https://rest.runpod.io/v1/pods/$POD_ID" | head -c 200; echo; }
trap terminate EXIT

echo "== waiting for vLLM at $LEAKRATE_OPENAI_BASE_URL"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 20 "$LEAKRATE_OPENAI_BASE_URL/models" || true)
  if [ "$code" = "200" ]; then echo "ready after $((i * 20))s"; break; fi
  sleep 20
done
[ "$code" = "200" ] || { echo "server never became ready"; exit 1; }
curl -s --max-time 120 "$LEAKRATE_OPENAI_BASE_URL/chat/completions" -H "Content-Type: application/json" \
  -d '{"model":"qwen72b","messages":[{"role":"user","content":"Rispondi solo: ok"}],"max_tokens":5,"temperature":0}' | head -c 300; echo

echo "== attacker, 6 conditions"
for C in "raw raw" "typed $TYPED" "opaque $OPAQUE" "direct_only $DONLY" "all $ALL" "oracle $ORACLE"; do
  set -- $C
  "$PY" -m leakrate attack --model $MODEL --workers "$W" --pseudo $2 --out $F/attack__qwen2.5-72b__$1.json > /dev/null 2>$F/.err_attack_$1 && echo "attack $1 done" || { echo "attack $1 FAILED"; tail -3 $F/.err_attack_$1; }
done

echo "== completeness-blind judge on the Opus medium run (typed / opaque / oracle)"
"$PY" -m leakrate judge --blind --model $MODEL --workers "$W" --records $F/downstream__opus-5-5__medium.records.jsonl \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --pseudo oracle=$ORACLE --out $F/judge_blind__qwen2.5-72b__opus-medium.json > /dev/null 2>$F/.err_judge && echo "judge blind done" || { echo "judge blind FAILED"; tail -3 $F/.err_judge; }
echo "== default-prompt judge with the same model (judge-model comparison), typed / opaque"
"$PY" -m leakrate judge --model $MODEL --workers "$W" --records $F/downstream__opus-5-5__medium.records.jsonl \
   --pseudo typed=$TYPED --pseudo opaque=$OPAQUE --out $F/judge_default__qwen2.5-72b__opus-medium.json > /dev/null 2>$F/.err_judge2 && echo "judge default done" || { echo "judge default FAILED"; tail -3 $F/.err_judge2; }

"$PY" - <<'EOF'
import json, pathlib
F = pathlib.Path("results/frontier")
for c in ["raw", "typed", "opaque", "direct_only", "all", "oracle"]:
    f = F / f"attack__qwen2.5-72b__{c}.json"
    if f.exists():
        a = json.loads(f.read_text(encoding="utf-8"))
        print(f"attack72 {c:12s} surname {100*a['surname_recovery']['rate']:.1f}%  k<11 {100*a['pop_k_lt11_share']:.1f}%  median {a['pop_k_median']:,.0f}")
for n in ["judge_blind__qwen2.5-72b__opus-medium", "judge_default__qwen2.5-72b__opus-medium"]:
    f = F / f"{n}.json"
    if f.exists():
        j = json.loads(f.read_text(encoding="utf-8"))
        print(n, {k: (round(v.get("win_rate_vs_raw", float("nan")), 3), round(v.get("lo", 0), 3), round(v.get("hi", 0), 3), v.get("n")) for k, v in j["judge_vs_raw"].items()})
EOF
echo "== done"
