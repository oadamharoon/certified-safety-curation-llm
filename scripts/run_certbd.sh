#!/bin/bash
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
for s in 0 1 2; do env PYTHONNOUSERSITE=1 ARM=certbd400 SEED=$s $PY scripts/step2_sft.py > logs/sft/certbd400_s$s.log 2>&1 && echo "done s$s" || echo "FAIL s$s"; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=certbd400_s0,certbd400_s1,certbd400_s2 $PY scripts/step2_eval.py > logs/step2_eval_certbd.log 2>&1
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py 2>&1 | grep -v -i "warn\|deprecat\|Loading\|Fetching\|legacy" > logs/step2_eval2.log; cat logs/step2_eval2.log; echo CERTBD DONE
