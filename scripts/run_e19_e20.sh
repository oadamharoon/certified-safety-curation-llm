#!/bin/bash
# E19 (Llama Guard 3 through the gate / label-complexity sweep) then E20 (8B dose-response), after
# E18; the CDT xargs (3769291) stays stopped until both finish (run_e17_wait.sh continues it).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
export PYTHONNOUSERSITE=1 LAMBDA=0.25
until grep -q "E18 DONE" logs/run_e18.out 2>/dev/null; do sleep 120; done
gate; $PY1 scripts/e19_lg_gate.py > logs/e19_lg_gate.log 2>&1 && say "E19 DONE" || { say "E19 FAILED"; tail -n 3 logs/e19_lg_gate.log; }
$PY1 scripts/e20_dose8b.py > logs/e20_select.log 2>&1 && grep -v Warn logs/e20_select.log || { say "E20 SELECT FAILED"; tail -n 3 logs/e20_select.log; }
for q in 0.30 0.37 0.45 0.545; do for s in 0 1 2; do gate; ARM=safedemo8_$q SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_safedemo8_${q}_s${s}_0.25.log 2>&1 && say "done safedemo8_$q s$s" || say "FAIL safedemo8_$q s$s"; done; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e20_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e20_0.25.log 2>&1
say "E20 DONE"; echo "E19E20 DONE"
