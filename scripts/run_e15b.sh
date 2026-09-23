#!/bin/bash
# E15-b after E16: every remaining arm of Tables 1 and 3 on the field harm sets (E15_ALL=1; caches
# make the E15 models no-ops). Then the RL F2 resume waits for E15B DONE.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
export PYTHONNOUSERSITE=1 E15_ALL=1
until grep -q "E16 DONE" logs/run_e16.out 2>/dev/null; do sleep 120; done
gate; $PY1 scripts/e15_harmsets.py gen05 > logs/e15b_gen05.log 2>&1 && say "E15b gen05 done" || { say "E15B GEN05 FAILED"; tail -n 3 logs/e15b_gen05.log; }
gate; $PY2 scripts/e15_harmsets.py gen8 > logs/e15b_gen8.log 2>&1 && say "E15b gen8 done" || { say "E15B GEN8 FAILED"; tail -n 3 logs/e15b_gen8.log; }
gate; $PY1 scripts/e15_harmsets.py judge > logs/e15b_judge.log 2>&1 && say "E15b judge done" || { say "E15B JUDGE FAILED"; tail -n 3 logs/e15b_judge.log; }
grep -v Warn logs/e15b_judge.log | tail -n 40
echo "E15B DONE"
