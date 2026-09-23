#!/bin/bash
# E11, budget-matched selector: selections seal_long / certseal_long, SFT, generation, judging.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
until grep -q "SEAL LONG DONE" $W/logs/seal/run_seal_long.out 2>/dev/null; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
export LAMBDA=0.25
env PYTHONNOUSERSITE=1 SELECTOR=ours_lambda025_long_softmax.pt ARM_SUFFIX=_long $PY1 scripts/seal_select.py > logs/seal/select_long.log 2>&1 || { echo "SEAL LONG SELECT FAILED"; exit 1; }
grep -v Warn logs/seal/select_long.log | head -n 3
for s in 0 1 2; do for arm in seal_long certseal_long; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"
done; done
env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_seal_long_0.25.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_seal_long_0.25.log 2>&1
grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_seal_long_0.25.log | tail -n 18
echo SEAL LONG ARMS DONE
