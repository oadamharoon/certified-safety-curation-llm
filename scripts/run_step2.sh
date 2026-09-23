#!/bin/bash
# Step 2: 6 arms x 3 seeds of SFT, then evaluation. Idempotent (skips finished models).
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY=${PY}; cd $W; mkdir -p logs/sft
for s in 0 1 2; do
  for arm in cert400 cert200 heur labels oracle full; do
    env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 \
      && echo "[$(date +%H:%M)] done $arm s$s: $(grep -c step logs/sft/${arm}_s$s.log) log lines" || echo "FAIL $arm s$s"
  done
done
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval.py > logs/step2_eval.log 2>&1 && tail -n 22 logs/step2_eval.log || echo "EVAL FAILED"
echo STEP2 DONE
