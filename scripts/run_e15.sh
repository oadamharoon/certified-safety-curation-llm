#!/bin/bash
# E15 (field harm sets) after the E7 retry; GPU gate before each phase; smoke test first (the
# smoke generations go to a scratch cache dir so they never mix with the real cache).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
export PYTHONNOUSERSITE=1
until grep -q "E7 RETRY DONE" logs/run_e7_retry.out 2>/dev/null; do sleep 120; done
gate; E15_SMOKE=1 $PY1 scripts/e15_harmsets.py gen05 > logs/e15_smoke.log 2>&1 && say "E15 smoke gen05 ok" || { say "E15 SMOKE FAILED"; tail -n 3 logs/e15_smoke.log; echo "E15 DONE"; exit 1; }
rm -f results/gen_harmsets/05b/*.json   # smoke output (6 prompts) must not be mistaken for a real cache
gate; $PY1 scripts/e15_harmsets.py gen05 > logs/e15_gen05.log 2>&1 && say "E15 gen05 done" || { say "E15 GEN05 FAILED"; tail -n 3 logs/e15_gen05.log; }
gate; $PY2 scripts/e15_harmsets.py gen8 > logs/e15_gen8.log 2>&1 && say "E15 gen8 done" || { say "E15 GEN8 FAILED"; tail -n 3 logs/e15_gen8.log; }
gate; $PY1 scripts/e15_harmsets.py judge > logs/e15_judge.log 2>&1 && say "E15 judge done" || { say "E15 JUDGE FAILED"; tail -n 3 logs/e15_judge.log; }
grep -v Warn logs/e15_judge.log | tail -n 20
echo "E15 DONE"
