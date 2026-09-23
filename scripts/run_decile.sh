#!/bin/bash
# Figure 7: harm vs judge-score decile. Waits for the 8B sweep (sweep3) to finish.
PY="${PY:-python}"
until grep -q "E3 SWEEP3 DONE" $W/logs/e3/run_sweep3.out 2>/dev/null; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, sys, numpy as np; sys.path.insert(0, "scripts"); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open("results/scores_wildguard_pool.json"))["harmful_request"]); demo = req & ~unsafe
g = np.array(json.load(open("results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json"))); order = np.argsort(-g)
summ = json.load(open("results/step2_selections.json"))
for d in range(10):
    idx = [int(i) for i in order[d*400:(d+1)*400]]
    for s in range(3):
        meta = {"rule": f"judge-score decile {d} (0 = judged safest)", "decile": d, "n": 400, "unsafe_frac": float(unsafe[idx].mean()), "safety_demo_share": float(demo[idx].mean())}
        json.dump({"idx": idx, "meta": meta}, open(f"selections/decile{d}_s{s}.json", "w")); summ.setdefault(f"decile{d}", {})[str(s)] = meta
    print(f"decile {d}: unsafe {meta['unsafe_frac']:.3f} demos {meta['safety_demo_share']:.3f}")
json.dump(summ, open("results/step2_selections.json", "w"), indent=1)
PY
names=""
for d in 0 1 2 3 4 5 6 7 8 9; do for s in 0 1 2; do env PYTHONNOUSERSITE=1 ARM=decile$d SEED=$s $PY scripts/step2_sft.py > logs/sft/decile${d}_s$s.log 2>&1 && echo "done decile$d s$s"; names="$names,decile${d}_s$s"; done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/decile_gen.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/decile_eval2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/decile_mdj.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/decile_panel2.log 2>&1
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); sel=json.load(open(f"{W}/results/step2_selections.json"))
print(f"{'decile':6s} {'unsafe':>6s} {'demos':>6s} {'beaver-dam':>11s} {'MD-Judge':>11s} {'WildGuard':>11s} {'LlamaGuard3':>11s}")
for d in range(10):
    ms=[f"decile{d}_s{s}" for s in range(3)]; m=sel[f"decile{d}"]["0"]
    print(f"{d:6d} {m['unsafe_frac']:6.3f} {m['safety_demo_share']:6.3f} {np.mean([ev[n]['beaverdam_flag'] for n in ms]):11.3f} {np.mean([np.mean(md[n]['flag']) for n in ms]):11.3f} {np.mean([np.mean(wg[n]['harmful_response']) for n in ms]):11.3f} {np.mean([np.mean(lg[n]['flag']) for n in ms]):11.3f}")
PY
echo DECILE DONE
