#!/bin/bash
# SAFT reimplementation + certified-SAFT at lambda .25 on Llama-3.1-8B-Instruct. Waits for the
# lambda .10 re-judge (end of the current LLM chain) so the 8B embedding pass and the SFT runs
# never overlap another large GPU job.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
until grep -q "JUDGE010 DONE" $W/logs/e3/run_judge010.out 2>/dev/null; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
export LAMBDA=0.25
env PYTHONNOUSERSITE=1 $PY2 scripts/saft_select.py > logs/e3/saft_select_0.25.log 2>&1 || { echo "SAFT SELECT FAILED"; exit 1; }
for s in 0 1 2; do for arm in saft saft15 certsaft; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"
done; done
env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_saft_0.25.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_saft_0.25.log 2>&1
grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_saft_0.25.log | tail -n 16
echo SAFT DONE
