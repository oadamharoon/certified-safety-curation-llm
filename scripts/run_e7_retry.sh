#!/bin/bash
# E7 retry after run_llm_rest.sh (its E7 step OOM'd on the full-vocabulary logits tensor; the
# judge pass now keeps last-position logits only, batch 8). Runs after LLM REST DONE, before the
# RL F2 resume, which waits for E7 RETRY DONE.
PY="${PY:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
until grep -q "LLM REST DONE" logs/run_llm_rest.out 2>/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done
env PYTHONNOUSERSITE=1 ${PY} scripts/e7_quality.py > logs/e7_quality.log 2>&1 && echo "[$(date +%m/%d-%H:%M)] E7 DONE" || { echo "[$(date +%m/%d-%H:%M)] E7 FAILED"; tail -n 3 logs/e7_quality.log; }
echo "E7 RETRY DONE"
