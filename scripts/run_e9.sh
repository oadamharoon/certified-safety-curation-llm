#!/bin/bash
# E9: seeds 3 and 4 on the headline cells. Certified arms draw new calibration samples.
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, sys, numpy as np; sys.path.insert(0, "scripts"); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open("results/scores_wildguard_pool.json"))["harmful_request"])
G = {"cert400": np.array(json.load(open("results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json"))),
     "certbd400": 1 - np.array(json.load(open("results/scores_beaverdam_pool.json"))),
     "certlg400": -np.array(json.load(open("results/scores_llamaguard_pool.json"))["score_unsafe_minus_safe"])}
summ = json.load(open("results/step2_selections.json"))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(req[idx][~unsafe[idx]].sum()/len(idx)))
    json.dump({"idx": idx, "meta": meta}, open(f"selections/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} {meta.get('certified','')}")
for s in (3, 4):
    cal = np.random.default_rng(1000 + s).choice(4000, 400, replace=False)
    for arm, g in G.items():
        ok, mask, q, audit = ltt_walk(g, unsafe, cal, 0.10, 0.10)
        if ok: save(arm, s, np.where(mask)[0], {"certified": True, "q": q})
        else:
            fm, lb = cp_fallback(g, unsafe, cal, 0.10); save(arm, s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb})
    save("full", s, np.arange(4000), {}); save("oracle", s, np.where(~unsafe)[0], {})
    save("random1400", s, np.random.default_rng(7000 + s).choice(4000, 1400, replace=False), {"rule": "uniform random at cert400 size"})
json.dump(summ, open("results/step2_selections.json", "w"), indent=1)
PY
names=""
for s in 3 4; do for arm in full random1400 cert400 certlg400 certbd400 oracle; do
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && echo "done $arm s$s" || echo "FAIL $arm s$s"; names="$names,${arm}_s$s"; done; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e9_gen.log 2>&1 || echo "GEN FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e9_eval2.log 2>&1 || echo "EVAL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e9_mdj.log 2>&1 || echo "MDJ FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e9_panel2.log 2>&1 || echo "PANEL2 FAILED"
env PYTHONNOUSERSITE=1 $PY scripts/e6_utility.py > logs/e9_e6.log 2>&1 || echo "E6 FAILED"
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.3f} (n={len(v)})"
print(f"{'arm':12s} {'beaver-dam':>20s} {'MD-Judge':>20s} {'WildGuard':>20s} {'LlamaGuard3':>20s}")
for a in ["full","random1400","cert400","certlg400","certbd400","oracle"]:
    ms=[n for n in wg if n.startswith(a+"_")]
    print(f"{a:12s} {cell([ev[n]['beaverdam_flag'] for n in ms]):>20s} {cell([np.mean(md[n]['flag']) for n in ms]):>20s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>20s} {cell([np.mean(lg[n]['flag']) for n in ms]):>20s}")
PY
echo E9 DONE
