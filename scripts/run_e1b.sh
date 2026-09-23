#!/bin/bash
# waits for the pool-refusal job (last GPU user in the chain), then runs the two controls
PY="${PY:-python}"
until grep -q "POOLREF DONE" /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/tasks/bj8o8ifpd.output 2>/dev/null; do sleep 30; done
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
for s in 0 1 2; do env PYTHONNOUSERSITE=1 ARM=random1400 SEED=$s $PY scripts/step2_sft.py > logs/sft/random1400_s$s.log 2>&1 && echo "done random1400 s$s" || echo "FAIL random1400 s$s"; done
# base reference: symlink the base checkpoint into results/models so every eval script treats it as an arm
mkdir -p results/models/base_s0 && env PYTHONNOUSERSITE=1 $PY - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype="bfloat16"); m.save_pretrained("results/models/base_s0"); AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained("results/models/base_s0")
PY
env PYTHONNOUSERSITE=1 EVAL_ONLY=random1400_s0,random1400_s1,random1400_s2,base_s0 $PY scripts/step2_eval.py > logs/e1b_gen.log 2>&1 || echo "GEN FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e1b_eval2.log 2>&1 || echo "EVAL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e1b_mdj.log 2>&1 || echo "MDJ FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e1b_panel2.log 2>&1; grep -v -i "warn\|Fetching\|Loading\|legacy" logs/e1b_panel2.log
echo E1B DONE
