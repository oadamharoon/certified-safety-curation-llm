#!/bin/bash
PY="${PY:-python}"
until grep -q "E9 DONE" /tmp/claude-1001/-home-omniverse-workspace-safevlmcpl/cbe3ff25-bd02-4cf4-9f36-173bf5fa270c/tasks/b1dt40m5x.output 2>/dev/null; do sleep 30; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
env PYTHONNOUSERSITE=1 $PY scripts/e10_two_property.py 2>&1 | grep -v -i warn | tee logs/e10_select.log
names=""
for arm in certstrat_qwen certstrat_lg cert2p_qwen cert2p_lg; do for s in 0 1 2; do
  [ -f selections/${arm}_s$s.json ] || continue
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && echo "done $arm s$s" || echo "FAIL $arm s$s"; names="$names,${arm}_s$s"; done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e10_gen.log 2>&1 || echo "GEN FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e10_eval2.log 2>&1 || echo "EVAL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e10_mdj.log 2>&1 || echo "MDJ FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e10_panel2.log 2>&1 || echo "PANEL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e6_utility.py > logs/e10_e6.log 2>&1 || echo "E6 FAILED"
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); sel=json.load(open(f"{W}/results/step2_selections.json")); u6=json.load(open(f"{W}/results/e6_utility.json"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.3f}" if len(v)>1 else (f"{np.mean(v):.3f}" if v else "  --  ")
print(f"{'arm':16s} {'unsafe':>6s} {'demos':>6s} {'beaver-dam':>12s} {'MD-Judge':>12s} {'WildGuard':>12s} {'LlamaGuard3':>12s} {'XSTest-unsafe':>13s} {'ROUGE-L':>8s}")
for a in ["full","cert400","certstrat_qwen","cert2p_qwen","certlg400","certstrat_lg","cert2p_lg","certbd400","oracle"]:
    ms=[n for n in wg if n.startswith(a+"_")]
    if not ms: print(f"{a:16s}   (no trained models: refused)"); continue
    u=np.mean([sel[a][s]["unsafe_frac"] for s in sel[a] if "unsafe_frac" in sel[a][s]]); d=np.mean([sel[a][s]["safety_demo_share"] for s in sel[a] if "safety_demo_share" in sel[a][s]])
    print(f"{a:16s} {u:6.3f} {d:6.3f} {cell([ev[n]['beaverdam_flag'] for n in ms]):>12s} {cell([np.mean(md[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>12s} {cell([np.mean(lg[n]['flag']) for n in ms]):>12s} {cell([u6[n]['xstest_unsafe_harmful'] for n in ms]):>13s} {cell([u6[n]['rouge_l'] for n in ms]):>8s}")
PY
echo E10 DONE
