#!/bin/bash
# Review-driven experiments E23 (0.5B dose extension), E21 (size-matched oracle, 8B), E22 (which harmful
# examples, 8B). The RL CDT xargs (3769291) is SIGSTOPped now so no new cell starts; the three running
# cells finish first (~02:50), then the bundle runs on the free GPU, then the xargs is SIGCONTed.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
L="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; W="$(dirname "$L")"; cd $L
PY1=${PY}; PY2=${PY_LLM}
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
cdt_running () { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_cdt.py && return 0; done; return 1; }
export PYTHONNOUSERSITE=1 LAMBDA=0.25
kill -STOP 3769291 && say "CDT xargs 3769291 SIGSTOP" || say "WARN: could not stop xargs"
while cdt_running; do sleep 120; done; say "CDT cells finished, GPU free"
# ---- E23 (0.5B)
names=$(tail -1 logs/e23_select.log)
for a in $names; do arm=${a%_s*}; s=${a##*_s}; gate
  [ -f results/models/${a}/config.json ] && { say "skip $a"; continue; }
  ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/${a}.log 2>&1 && say "done $a" || say "FAIL $a"; done
gate; EVAL_ONLY=$(echo $names | tr ' ' ',') $PY1 scripts/step2_eval.py > logs/e23_gen.log 2>&1 || say "E23 GEN FAILED"
gate; $PY1 scripts/step2_eval2.py > logs/e23_eval2.log 2>&1 || say "E23 EVAL2 FAILED"
gate; $PY1 scripts/e1_evaluators.py > logs/e23_mdj.log 2>&1 || say "E23 MDJ FAILED"
gate; $PY1 scripts/e1_panel2.py > logs/e23_panel2.log 2>&1 || say "E23 PANEL2 FAILED"
say "E23 DONE"
# ---- E21 + E22 (8B)
$PY1 scripts/e21_e22_select.py > logs/e21_e22_select.log 2>&1 && grep -v Warn logs/e21_e22_select.log || say "E21E22 SELECT FAILED"
for arm in oracle_matched harmtop8 harmmid8 harmbot8; do for s in 0 1 2; do gate
  [ -f results/models_e3/lambda0.25/${arm}_s${s}/adapter_config.json ] && { say "skip $arm s$s"; continue; }
  ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && say "done $arm s$s" || say "FAIL $arm s$s"; done; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e21e22_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e21e22_0.25.log 2>&1
say "E21 E22 DONE"
kill -CONT 3769291 && say "CDT xargs 3769291 SIGCONT" ; echo "[$(date +%m/%d-%H:%M)] review experiments E21-E23 done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
echo "E21E23 DONE"
