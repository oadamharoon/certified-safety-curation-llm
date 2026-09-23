#!/bin/bash
# E20's q = 0.30 level failed on an arm-name mismatch (selection files are safedemo8_0.3_s*.json,
# the runner passed safedemo8_0.30). Rerun after the running CDT cells finish; CDT xargs continued after.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$W"
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
export PYTHONNOUSERSITE=1 LAMBDA=0.25
# Wait on the GPU's own process list, never on pgrep -f (the first version's pgrep -f matched the
# shell that launched the waiter and it slept 1.7 h on an idle GPU, 2026-09-18).
cdt_running () { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_cdt.py && return 0; done; return 1; }
while cdt_running; do sleep 120; done
for s in 0 1 2; do gate; ARM=safedemo8_0.3 SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_safedemo8_0.3_s${s}_0.25.log 2>&1 && echo "[$(date +%m/%d-%H:%M)] done safedemo8_0.3 s$s" || echo "FAIL safedemo8_0.3 s$s"; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e20b_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e20b_0.25.log 2>&1
kill -CONT 3769291; echo "[$(date +%m/%d-%H:%M)] E20 q=0.30 done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
echo "E20FIX DONE"
