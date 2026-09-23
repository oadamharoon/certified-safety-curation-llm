#!/bin/bash
# E8-lite: second pool (PKU-SafeRLHF) at 0.5B. Waits for the SEAL long arms (GPU).
PY="${PY:-python}"
until grep -q "SEAL LONG ARMS DONE" $W/logs/seal/run_seal_long_arms.out 2>/dev/null; do sleep 120; done; until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 9000 ]; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; cd $W; PY=${PY}
env PYTHONNOUSERSITE=1 $PY scripts/e8_pool2.py > logs/e8_select.log 2>&1 || { echo "E8 SELECT FAILED"; tail -n 5 logs/e8_select.log; exit 1; }
grep -v Warn logs/e8_select.log | grep "pool2\|PKU" | head -n 3
names=""
for arm in p2_full p2_random p2_cert400 p2_certlg400 p2_certstrat2 p2_oracle; do for s in 0 1 2; do
  env PYTHONNOUSERSITE=1 POOL_JSON=$W/results/pool2.json ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && echo "[$(date +%H:%M)] done $arm s$s" || echo "FAIL $arm s$s"; names="$names,${arm}_s$s"
done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e8_gen.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e8_eval2.log 2>&1
env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e8_mdj.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e8_panel2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e6_utility.py > logs/e8_e6.log 2>&1
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); sel=json.load(open(f"{W}/results/step2_selections.json")); u=json.load(open(f"{W}/results/e6_utility.json"))
print(f"{'arm':14s} {'unsafe':>6s} {'demos':>6s} {'cert':>5s} {'beaver-dam':>11s} {'MD-Judge':>11s} {'WildGuard':>11s} {'LlamaGuard3':>11s} {'XSTest-unsafe':>13s} {'ROUGE-L':>8s}")
for a in ["p2_full","p2_random","p2_cert400","p2_certlg400","p2_certstrat2","p2_oracle"]:
    ms=[f"{a}_s{s}" for s in range(3)]; m=sel[a]
    c=f"{sum(bool(m[s].get('certified',False)) for s in m)}/3" if any('certified' in m[s] for s in m) else "--"
    print(f"{a:14s} {np.mean([m[s]['unsafe_frac'] for s in m]):6.3f} {np.mean([m[s]['safety_demo_share'] for s in m]):6.3f} {c:>5s} {np.mean([ev[n]['beaverdam_flag'] for n in ms]):11.3f} {np.mean([np.mean(md[n]['flag']) for n in ms]):11.3f} {np.mean([np.mean(wg[n]['harmful_response']) for n in ms]):11.3f} {np.mean([np.mean(lg[n]['flag']) for n in ms]):11.3f} {np.mean([u[n]['xstest_unsafe_harmful'] for n in ms]):13.3f} {np.mean([u[n]['rouge_l'] for n in ms]):8.3f}")
PY
echo E8 DONE
