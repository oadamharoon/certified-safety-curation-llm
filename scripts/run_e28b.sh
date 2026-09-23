#!/bin/bash
# E28b: qi_safedemo0.25 x3, plus the untouched TinyLlama and Qwen2.5-0.5B-Instruct as tl_base_s0 / qi_base_s0
# (weights copied into results/models so the shared generation and judge scripts pick them up). ~25 min.
PY="${PY:-python}"
L="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; W="$(dirname "$L")"; cd $L
PY1=${PY}; say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
cdt_running () { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_cdt.py && return 0; done; return 1; }
export PYTHONNOUSERSITE=1
kill -STOP 3769291 && say "CDT xargs SIGSTOP"; while cdt_running; do sleep 60; done; say "GPU free"
names=""
for s in 0 1 2; do n=qi_safedemo0.25_s$s; [ -f results/models/$n/config.json ] && { names="$names,$n"; continue; }
  SFT_BASE=Qwen/Qwen2.5-0.5B-Instruct SFT_TAG=qi_ ARM=safedemo0.25 SEED=$s $PY1 scripts/step2_sft.py > logs/sft/$n.log 2>&1 && { say "done $n"; names="$names,$n"; } || say "FAIL $n"; done
$PY1 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
for name, base in (("tl_base_s0", "TinyLlama/TinyLlama_v1.1"), ("qi_base_s0", "Qwen/Qwen2.5-0.5B-Instruct")):
    out=f"{W}/results/models/{name}"
    if os.path.exists(f"{out}/config.json"): continue
    tok=AutoTokenizer.from_pretrained(base); tok.pad_token = tok.pad_token or tok.eos_token
    if tok.chat_template is None: tok.chat_template = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").chat_template
    m=AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16); m.save_pretrained(out); tok.save_pretrained(out); print("saved", name)
PY
names="$names,tl_base_s0,qi_base_s0"
gate; EVAL_ONLY=${names#,} $PY1 scripts/step2_eval.py > logs/e28b_gen.log 2>&1 || say "GEN FAILED"
gate; $PY1 scripts/step2_eval2.py > logs/e28b_eval2.log 2>&1 || say "EVAL2 FAILED"
gate; $PY1 scripts/e1_evaluators.py > logs/e28b_mdj.log 2>&1 || say "MDJ FAILED"
gate; $PY1 scripts/e1_panel2.py > logs/e28b_panel2.log 2>&1 || say "PANEL2 FAILED"
gate; $PY1 scripts/e6_utility.py > logs/e28b_e6.log 2>&1 || say "E6 FAILED"
say "E28B DONE"; kill -CONT 3769291 && say "CDT xargs SIGCONT"; echo "[$(date +%m/%d-%H:%M)] E28b done, CDT xargs 3769291 SIGCONT" >> $W/runs/logs/v2f2/progress.log
