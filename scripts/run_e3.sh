#!/bin/bash
# E3 at one contamination level. Waits for E10d to release the GPU, then: selections (judges,
# under PY) -> smoke test -> 18 LoRA runs (under PY_LLM) -> generation -> judging.
# (E10d wait removed on relaunch 2026-09-13)
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
export LAMBDA=${LAMBDA:-0.25}; mkdir -p logs/e3
env PYTHONNOUSERSITE=1 $PY1 scripts/e3_select.py > logs/e3/select_$LAMBDA.log 2>&1 || { echo "SELECT FAILED"; tail -5 logs/e3/select_$LAMBDA.log; exit 1; }
grep -v -i "warn\|Loading\|Fetching" logs/e3/select_$LAMBDA.log
S=${TMPDIR:-/tmp}/e3smoke
env PYTHONNOUSERSITE=1 ARM=full SEED=0 SFT_SMOKE=$S $PY2 scripts/e3_lora_sft.py > logs/e3/smoke.log 2>&1 && echo "smoke PASS" || { echo "smoke FAIL"; tail -8 logs/e3/smoke.log; exit 1; }
for s in 0 1 2; do for arm in full prompting random certlg certstrat2 oracle; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_$LAMBDA.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"
done; done
env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_$LAMBDA.log 2>&1 || echo "GEN FAILED"
env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_$LAMBDA.log 2>&1 || echo "JUDGE FAILED"
grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_$LAMBDA.log | tail -n 12
echo E3 DONE
