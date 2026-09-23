#!/bin/bash
# Control: naive inversion of the judge filter (bottom 1400 by Qwen score = cert400's size).
# Waits for E3's smoke test line so it does not collide with E3's judge scoring; then runs
# between E3 LoRA runs is not possible, so it waits for E3 DONE instead.
PY="${PY:-python}"
until [ -f $W/results/e3_eval_lambda0.25.json ]; do sleep 60; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, sys, numpy as np; sys.path.insert(0, "scripts"); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open("results/scores_wildguard_pool.json"))["harmful_request"]); demo = req & ~unsafe
g = np.array(json.load(open("results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json"))); summ = json.load(open("results/step2_selections.json"))
for s in range(3):
    idx = [int(i) for i in np.argsort(g)[:1400]]
    meta = {"rule": "bottom 1400 by Qwen judge score (naive inversion)", "n": 1400, "unsafe_frac": float(unsafe[idx].mean()), "safety_demo_share": float(demo[idx].mean())}
    json.dump({"idx": idx, "meta": meta}, open(f"selections/anti_qwen_s{s}.json", "w")); summ.setdefault("anti_qwen", {})[str(s)] = meta
print("anti_qwen: unsafe %.3f demos %.3f" % (meta["unsafe_frac"], meta["safety_demo_share"]))
json.dump(summ, open("results/step2_selections.json", "w"), indent=1)
PY
for s in 0 1 2; do env PYTHONNOUSERSITE=1 ARM=anti_qwen SEED=$s $PY scripts/step2_sft.py > logs/sft/anti_qwen_s$s.log 2>&1 && echo "done s$s"; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=anti_qwen_s0,anti_qwen_s1,anti_qwen_s2 $PY scripts/step2_eval.py > logs/anti_gen.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/anti_eval2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/anti_mdj.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/anti_panel2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e6_utility.py > logs/anti_e6.log 2>&1
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); u6=json.load(open(f"{W}/results/e6_utility.json"))
for a in ["anti_qwen","full","cert400","random1400"]:
    ms=[n for n in wg if n.startswith(a+"_")]
    print(f"{a:12s} beaver-dam {np.mean([ev[n]['beaverdam_flag'] for n in ms]):.3f} WildGuard {np.mean([wg[n]['harmful_response'] and np.mean(wg[n]['harmful_response']) for n in ms]):.3f} LlamaGuard {np.mean([np.mean(lg[n]['flag']) for n in ms]):.3f} XSTest-unsafe {np.mean([u6[n]['xstest_unsafe_harmful'] for n in ms]):.3f}")
PY
echo ANTI DONE
