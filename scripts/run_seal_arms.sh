#!/bin/bash
# E11 arms (seal, certseal) at lambda .25: SFT with the E3 recipe, generation, judging. Waits for
# the SAFT stage so the GPU is never shared with another 8B job.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
until grep -q "SAFT DONE" $W/logs/e3/run_saft.out 2>/dev/null; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
export LAMBDA=0.25
for s in 0 1 2; do for arm in seal certseal; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"
done; done
env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_seal_0.25.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_seal_0.25.log 2>&1
grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_seal_0.25.log | tail -n 16
echo SEAL ARMS DONE
