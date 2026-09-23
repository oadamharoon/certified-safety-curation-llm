#!/bin/bash
# Remaining LLM GPU passes in one sequential chain, each behind a GPU gate (the earlier per-experiment
# chains OOM'd against the RL regeneration jobs). Order: E8 utility + summary, E7, E13 (8B then 0.5B),
# E12 categories, E14. Replaces run_e7/e12/e13/e14.sh (E13 selections already done: logs/e13_select.log).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
gate () { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt "${1:-7000}" ]; do sleep 60; done; }
say () { echo "[$(date +%m/%d-%H:%M)] $*"; }
export PYTHONNOUSERSITE=1

# ---- E8: utility for the p2 models (OOM'd), then the summary table
gate; $PY1 scripts/e6_utility.py > logs/e8_e6.log 2>&1 && say "E8 utility done" || say "E8 UTILITY FAILED"
sed -n '/^env PYTHONNOUSERSITE=1 \$PY - <<.PY.$/,/^PY$/p' scripts/run_e8.sh | sed '1d;$d' | $PY1 - && say "E8 SUMMARY DONE"

# ---- E7: low-quality property
gate; $PY1 scripts/e7_quality.py > logs/e7_quality.log 2>&1 && say "E7 DONE" || { say "E7 FAILED"; tail -n 3 logs/e7_quality.log; }

# ---- E13: 8B arms (certprompt, certstrat_saft, certstrat_seal), then 0.5B (saft05, certsaft05)
export LAMBDA=0.25
for s in 0 1 2; do for arm in certprompt certstrat_saft certstrat_seal; do
  gate; ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && say "done $arm s$s" || say "FAIL $arm s$s"
done; done
gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_e13_0.25.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_e13_0.25.log 2>&1
names=""
for arm in saft05 certsaft05; do for s in 0 1 2; do
  gate; ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && say "done $arm s$s" || say "FAIL $arm s$s"; names="$names,${arm}_s$s"
done; done
gate; EVAL_ONLY=${names#,} $PY1 scripts/step2_eval.py > logs/e13_gen.log 2>&1; gate; $PY1 scripts/step2_eval2.py > logs/e13_eval2.log 2>&1
gate; $PY1 scripts/e1_evaluators.py > logs/e13_mdj.log 2>&1; gate; $PY1 scripts/e1_panel2.py > logs/e13_panel2.log 2>&1; gate; $PY1 scripts/e6_utility.py > logs/e13_e6.log 2>&1
say "E13 DONE"

# ---- E12: beaver-dam category flags (all models, so after E13)
gate; $PY1 scripts/e12_categories.py > logs/e12_categories.log 2>&1 && say "E12 DONE" || { say "E12 FAILED"; tail -n 3 logs/e12_categories.log; }

# ---- E14: SAFT at rho .10 / .30
for L in 0.10 0.30; do
  export LAMBDA=$L
  gate; $PY2 scripts/saft_select.py > logs/e3/saft_select_$L.log 2>&1 || { say "SAFT SELECT $L FAILED"; continue; }
  for s in 0 1 2; do for arm in saft saft15 certsaft; do
    gate; ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_$L.log 2>&1 && say "done $arm s$s $L" || say "FAIL $arm s$s $L"
  done; done
  gate; $PY2 scripts/e3_eval.py gen > logs/e3/gen_saft_$L.log 2>&1; gate; $PY1 scripts/e3_eval.py judge > logs/e3/judge_saft_$L.log 2>&1
done
say "E14 DONE"
say "LLM REST DONE"
