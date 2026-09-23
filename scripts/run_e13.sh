#!/bin/bash
# E13 completeness arms: after the category pass (last GPU job of the earlier chain).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
until grep -q "E12 DONE\|E12 FAILED" $W/logs/run_e12.out 2>/dev/null; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W
PY1=${PY}; PY2=${PY_LLM}
env PYTHONNOUSERSITE=1 $PY2 scripts/e13_completeness.py > logs/e13_select.log 2>&1 || { echo "E13 SELECT FAILED"; tail -n 5 logs/e13_select.log; exit 1; }
grep -v Warn logs/e13_select.log | grep "AUROC\|SEAL scores" | head -n 3
export LAMBDA=0.25
for s in 0 1 2; do for arm in certprompt certstrat_saft certstrat_seal; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY2 scripts/e3_lora_sft.py > logs/e3/sft_${arm}_s${s}_0.25.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"
done; done
env PYTHONNOUSERSITE=1 $PY2 scripts/e3_eval.py gen > logs/e3/gen_e13_0.25.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e3_eval.py judge > logs/e3/judge_e13_0.25.log 2>&1
names=""
for arm in saft05 certsaft05; do for s in 0 1 2; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY1 scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"; names="$names,${arm}_s$s"
done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY1 scripts/step2_eval.py > logs/e13_gen.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/step2_eval2.py > logs/e13_eval2.log 2>&1
env PYTHONNOUSERSITE=1 $PY1 scripts/e1_evaluators.py > logs/e13_mdj.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e1_panel2.py > logs/e13_panel2.log 2>&1; env PYTHONNOUSERSITE=1 $PY1 scripts/e6_utility.py > logs/e13_e6.log 2>&1
echo E13 DONE
