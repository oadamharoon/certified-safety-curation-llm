#!/bin/bash
# E16 after E15: lgfilter and labels arms at 8B (selection, 6 LoRA SFTs, gen, judge).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
export PYTHONNOUSERSITE=1 LAMBDA=0.25
until grep -q "E15 DONE" logs/run_e15.out 2>/dev/null; do sleep 120; done
$PY1 scripts/e16_arms8.py > logs/e16_select.log 2>&1 || { say "E16 SELECT FAILED"; tail -n 3 logs/e16_select.log; echo "E16 DONE"; exit 1; }
grep -v Warn logs/e16_select.log
for s in 0 1 2; do for arm in lgfilter labels; do
  gate; ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && say "done $arm s$s" || say "FAIL $arm s$s"
done; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e16_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e16_0.25.log 2>&1
gate; $PY2 scripts/e15_harmsets.py gen8 > logs/e15_gen8_e16.log 2>&1; gate; $PY1 scripts/e15_harmsets.py judge > logs/e15_judge_e16.log 2>&1
echo "E16 DONE"
