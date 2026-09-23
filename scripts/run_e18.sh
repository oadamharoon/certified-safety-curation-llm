#!/bin/bash
# E18 after E17: certstrat_prompt (8B, 3 LoRA) and certstrat_saft05 (0.5B, 3 SFT), standard evals,
# and the 8B condition through the harm sets. Run with the CDT xargs paused (run_e17_wait.sh).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 7000 ]; do sleep 60; done; }
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
export PYTHONNOUSERSITE=1 LAMBDA=0.25
for s in 0 1 2; do gate; ARM=certstrat_prompt SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_certstrat_prompt_s${s}_0.25.log 2>&1 && say "done certstrat_prompt s$s" || say "FAIL certstrat_prompt s$s"; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e18_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e18_0.25.log 2>&1
names=""
for s in 0 1 2; do gate; ARM=certstrat_saft05 SEED=$s $PY1 scripts/step2_sft.py > logs/sft/certstrat_saft05_s$s.log 2>&1 && say "done certstrat_saft05 s$s" || say "FAIL certstrat_saft05 s$s"; names="$names,certstrat_saft05_s$s"; done
gate; EVAL_ONLY=${names#,} $PY1 scripts/step2_eval.py > logs/e18_gen.log 2>&1; gate; $PY1 scripts/step2_eval2.py > logs/e18_eval2.log 2>&1
gate; $PY1 scripts/e1_evaluators.py > logs/e18_mdj.log 2>&1; gate; $PY1 scripts/e1_panel2.py > logs/e18_panel2.log 2>&1; gate; $PY1 scripts/e6_utility.py > logs/e18_e6.log 2>&1
export E15_ALL=1
gate; $PY2 scripts/e15_harmsets.py gen8 > logs/e18_hs_gen8.log 2>&1; gate; $PY1 scripts/e15_harmsets.py gen05 > logs/e18_hs_gen05.log 2>&1; gate; $PY1 scripts/e15_harmsets.py judge > logs/e18_hs_judge.log 2>&1
echo "E18 DONE"
