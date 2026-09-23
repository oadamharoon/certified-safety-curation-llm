#!/bin/bash
# E23: 0.5B dose-response extension (3 new levels x 3 seeds + seeds 3-4 on all 7 levels = 23 runs),
# beside the RL CDT cells (a 0.5B run takes ~5 GB). Then generation, the four judges (cached per model).
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 18000 ]; do sleep 60; done; }
export PYTHONNOUSERSITE=1
$PY scripts/e23_dose_extend.py > logs/e23_select.log 2>&1 || { say "E23 SELECT FAILED"; exit 1; }
names=$(tail -1 logs/e23_select.log); say "selections: $names"
for a in $names; do arm=${a%_s*}; s=${a##*_s}; gate
  [ -f results/models/${a}/config.json ] && { say "skip $a (exists)"; continue; }
  ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${a}.log 2>&1 && say "done $a" || say "FAIL $a"; done
gate; EVAL_ONLY=$(echo $names | tr ' ' ',') $PY scripts/step2_eval.py > logs/e23_gen.log 2>&1 || say "GEN FAILED"
gate; $PY scripts/step2_eval2.py > logs/e23_eval2.log 2>&1 || say "EVAL2 FAILED"
gate; $PY scripts/e1_evaluators.py > logs/e23_mdj.log 2>&1 || say "MDJ FAILED"
gate; $PY scripts/e1_panel2.py > logs/e23_panel2.log 2>&1 || say "PANEL2 FAILED"
say "E23 DONE"; echo "E23 DONE"
