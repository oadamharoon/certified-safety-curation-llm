#!/bin/bash
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
names=""
for q in 0.1 0.25 0.4 0.54; do for s in 0 1 2; do
  env PYTHONNOUSERSITE=1 ARM=safedemo$q SEED=$s $PY scripts/step2_sft.py > logs/sft/safedemo${q}_s$s.log 2>&1 && echo "done safedemo$q s$s" || echo "FAIL safedemo$q s$s"
  names="$names,safedemo${q}_s$s"; done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e4b_gen.log 2>&1 || echo "GEN FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e4b_eval2.log 2>&1 || echo "EVAL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e4b_mdj.log 2>&1 || echo "MDJ FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e4b_panel2.log 2>&1 || echo "PANEL2 FAILED"
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.3f}" if len(v)>1 else f"{np.mean(v):.3f}"
print(f"{'arm':14s} {'beaver-dam':>12s} {'MD-Judge':>12s} {'WildGuard':>12s} {'LlamaGuard3':>12s} {'WG refusal':>11s}")
for a in ["safedemo0.1","safedemo0.25","safedemo0.4","safedemo0.54","cert400","refshare0.1","full","random1400","certbd400","oracle","base"]:
    ms=[n for n in wg if n.startswith(a+"_")]
    print(f"{a:14s} {cell([ev[n]['beaverdam_flag'] for n in ms]):>12s} {cell([np.mean(md[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>12s} {cell([np.mean(lg[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['refusal']) for n in ms]):>11s}")
B=np.concatenate([np.array(json.load(open(f"{W}/results/gen_scores/beaverdam.json"))[n])>0.5 for n in wg]); L=np.concatenate([np.array(lg[n]["flag"]) for n in wg]); Wg=np.concatenate([np.array(wg[n]["harmful_response"]) for n in wg]); M=np.concatenate([np.array(md[n]["flag"]) for n in wg])
print("agreement with LlamaGuard3: bd %.3f md %.3f wg %.3f; LG flag rate %.3f"%(np.mean(B==L),np.mean(M==L),np.mean(Wg==L),L.mean()))
PY
echo E4B DONE
