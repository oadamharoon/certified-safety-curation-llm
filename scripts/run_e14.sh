#!/bin/bash
# E14: SAFT at rho .10 and .30 so the contamination sweep carries the same baselines as Table 5.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
until grep -q "E13 DONE" $W/logs/run_e13.out 2>/dev/null; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
for L in 0.10 0.30; do
  export LAMBDA=$L
  env PYTHONNOUSERSITE=1 $PY2 scripts/saft_select.py > logs/e3/saft_select_$L.log 2>&1 || { echo "SAFT SELECT $L FAILED"; continue; }
  for s in 0 1 2; do for arm in saft saft15 certsaft; do
    env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_$L.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s $L" || echo "FAIL $arm s$s $L"
  done; done
  env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_saft_$L.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_saft_$L.log 2>&1
done
echo E14 DONE
