#!/bin/bash
# E26b (8B, 6 LoRA) + E28 (Qwen2.5-0.5B-Instruct, 24 runs) after the running CDT cells; ~1.5 h.
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
L="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; W="$(dirname "$L")"; cd $L
PY1=${PY}; PY2=${PY_LLM}
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
cdt_running () { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_cdt.py && return 0; done; return 1; }
export PYTHONNOUSERSITE=1 LAMBDA=0.25
kill -STOP 3769291 && say "CDT xargs SIGSTOP"; while cdt_running; do sleep 60; done; say "GPU free"
$PY1 scripts/e26b_e28_select.py > logs/e26b_select.log 2>&1 && grep -v Warn logs/e26b_select.log || say "SELECT FAILED"
for arm in labels800_8 randdemo8; do for s in 0 1 2; do gate
  [ -f results/models_e3/lambda0.25/${arm}_s${s}/adapter_config.json ] && { say "skip $arm s$s"; continue; }
  ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && say "done $arm s$s" || say "FAIL $arm s$s"; done; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e26b_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e26b_0.25.log 2>&1; say "E26B DONE"
names=""
for arm in lgfilter certlg400 certstrat2_lg800; do for s in 0 1 2; do gate
  n=tl_${arm}_s$s; [ -f results/models/$n/config.json ] && { say "skip $n"; names="$names,$n"; continue; }
  SFT_BASE=TinyLlama/TinyLlama_v1.1 SFT_TAG=tl_ ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/$n.log 2>&1 && { say "done $n"; names="$names,$n"; } || { say "FAIL $n"; tail -2 logs/sft/$n.log; }; done; done
say "E24B TRAINED"
for arm in full random1400 cert400 anti_qwen certlg400 certstrat2_lg800 safedemo0.1 safedemo0.54; do for s in 0 1 2; do gate
  n=qi_${arm}_s$s; [ -f results/models/$n/config.json ] && { say "skip $n"; names="$names,$n"; continue; }
  SFT_BASE=Qwen/Qwen2.5-0.5B-Instruct SFT_TAG=qi_ ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/$n.log 2>&1 && { say "done $n"; names="$names,$n"; } || { say "FAIL $n"; tail -2 logs/sft/$n.log; }; done; done
gate; EVAL_ONLY=${names#,} $PY1 scripts/step2_eval.py > logs/e28_gen.log 2>&1 || say "GEN FAILED"
gate; $PY1 scripts/step2_eval2.py > logs/e28_eval2.log 2>&1 || say "EVAL2 FAILED"
gate; $PY1 scripts/e1_evaluators.py > logs/e28_mdj.log 2>&1 || say "MDJ FAILED"
gate; $PY1 scripts/e1_panel2.py > logs/e28_panel2.log 2>&1 || say "PANEL2 FAILED"
gate; $PY1 scripts/e6_utility.py > logs/e28_e6.log 2>&1 || say "E6 FAILED"
say "E24B E28 DONE"; kill -CONT 3769291 && say "CDT xargs SIGCONT"; echo "[$(date +%m/%d-%H:%M)] E26b/E28 done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
