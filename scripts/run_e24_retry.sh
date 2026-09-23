#!/bin/bash
# E24 retry: TinyLlama's tokenizer has no chat template; step2_sft now assigns Qwen2.5-0.5B's ChatML
# template (format only, identical prompt format across base models) and saves it with the model.
PY="${PY:-python}"
L="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; W="$(dirname "$L")"; cd $L
PY1=${PY}; say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
cdt_running () { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_cdt.py && return 0; done; return 1; }
export PYTHONNOUSERSITE=1
kill -STOP 3769291 && say "CDT xargs SIGSTOP"; while cdt_running; do sleep 60; done; say "GPU free"
names=""
for arm in judgetop05 judgeharm05; do for s in 0 1 2; do gate
  n=${arm}_s$s; [ -f results/models/$n/config.json ] && { say "skip $n"; names="$names,$n"; continue; }
  ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/$n.log 2>&1 && { say "done $n"; names="$names,$n"; } || { say "FAIL $n"; tail -2 logs/sft/$n.log; }; done; done
for arm in full random1400 cert400 anti_qwen; do for s in 0 1 2; do gate
  n=tl_${arm}_s$s; [ -f results/models/$n/config.json ] && { say "skip $n"; names="$names,$n"; continue; }
  SFT_BASE=TinyLlama/TinyLlama_v1.1 SFT_TAG=tl_ ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/$n.log 2>&1 && { say "done $n"; names="$names,$n"; } || { say "FAIL $n"; tail -2 logs/sft/$n.log; }; done; done
gate; EVAL_ONLY=${names#,} $PY1 scripts/step2_eval.py > logs/e24_gen.log 2>&1 || say "GEN FAILED"
gate; $PY1 scripts/step2_eval2.py > logs/e24_eval2.log 2>&1 || say "EVAL2 FAILED"
gate; $PY1 scripts/e1_evaluators.py > logs/e24_mdj.log 2>&1 || say "MDJ FAILED"
gate; $PY1 scripts/e1_panel2.py > logs/e24_panel2.log 2>&1 || say "PANEL2 FAILED"
gate; $PY1 scripts/e6_utility.py > logs/e24_e6.log 2>&1 || say "E6 FAILED"
say "E24 E27 DONE"; kill -CONT 3769291 && say "CDT xargs SIGCONT"; echo "[$(date +%m/%d-%H:%M)] E24 retry done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
