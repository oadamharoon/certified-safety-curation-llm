#!/bin/bash
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
names=""
for arm in certlg400 certlg200 lgfilter; do for s in 0 1 2; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && echo "done $arm s$s" || echo "FAIL $arm s$s"; names="$names,${arm}_s$s"; done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e2_gen.log 2>&1 || echo "GEN FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e2_eval2.log 2>&1 || echo "EVAL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e2_mdj.log 2>&1 || echo "MDJ FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e2_panel2.log 2>&1 || echo "PANEL2 FAILED"
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); sel=json.load(open(f"{W}/results/step2_selections.json"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.3f}" if len(v)>1 else f"{np.mean(v):.3f}"
print(f"{'arm':12s} {'unsafe':>6s} {'demos':>6s} {'beaver-dam':>12s} {'MD-Judge':>12s} {'WildGuard':>12s} {'LlamaGuard3*':>13s}")
for a in ["full","lgfilter","certlg200","certlg400","cert400","certbd400","oracle"]:
    ms=[n for n in wg if n.startswith(a+"_")]; u=np.mean([sel[a][s]["unsafe_frac"] for s in sel[a]]); d=np.mean([sel[a][s].get("safety_demo_share",float("nan")) for s in sel[a]])
    print(f"{a:12s} {u:6.3f} {d:6.3f} {cell([ev[n]['beaverdam_flag'] for n in ms]):>12s} {cell([np.mean(md[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>12s} {cell([np.mean(lg[n]['flag']) for n in ms]):>13s}")
print("* Llama Guard 3 is the scorer of the lg arms; its column is scorer-shared there")
PY
echo E2 DONE
