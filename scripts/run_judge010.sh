#!/bin/bash
# Re-judge the lambda=0.10 8B arms: the 2026-09-14 22:23 judge pass OOM'd when the RL paper's
# regeneration queue shared the GPU. Waits for the SEAL stage so the 17 GB judge never overlaps it.
PY="${PY:-python}"
until grep -q "SEAL DONE" $W/logs/seal/run_seal.out 2>/dev/null; do sleep 120; done
cd $W
env PYTHONNOUSERSITE=1 LAMBDA=0.10 ${PY} scripts/e3_eval.py judge > logs/e3/judge_0.10_rerun.log 2>&1
grep -v -i "warn\|Loading\|Fetching\|legacy" logs/e3/judge_0.10_rerun.log | tail -n 10
echo JUDGE010 DONE
