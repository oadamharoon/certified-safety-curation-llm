#!/bin/bash
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
for L in 0.10 0.30; do
  export LAMBDA=$L
  [ -f selections_e3/lambda$L/summary.json ] || env PYTHONNOUSERSITE=1 $PY1 scripts/e3_select.py > logs/e3/select_$L.log 2>&1 || { echo "SELECT $L FAILED"; continue; }
  for s in 0 1 2; do for arm in full prompting random certlg certstrat2 oracle; do
    env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_$L.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s lambda $L" || echo "FAIL $arm s$s $L"
  done; done
  env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_$L.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_$L.log 2>&1
  grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_$L.log | tail -n 10
done
echo E3 SWEEP2 DONE
