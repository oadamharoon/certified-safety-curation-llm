#!/bin/bash
# E10d: stratified certified curation with STRATUM-SPECIFIC targets fixed before calibration:
# harmful-prompt stratum alpha_h = 0.20, benign-prompt stratum alpha_b = 0.15, delta/2 each,
# n = 800, Llama Guard 3 scorer. Overall bound = mixture of the two targets by selection shares.
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
env PYTHONNOUSERSITE=1 $PY - <<'PY' 2>&1 | grep -v -i warn | tee logs/e10d_select.log
import json, sys, numpy as np; sys.path.insert(0, "scripts"); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool]); N = 4000
req = np.array(json.load(open("results/scores_wildguard_pool.json"))["harmful_request"]); demo = req & ~unsafe
g = -np.array(json.load(open("results/scores_llamaguard_pool.json"))["score_unsafe_minus_safe"])
A = {"h": 0.20, "b": 0.15}; DELTA = 0.10; summ = json.load(open("results/step2_selections.json"))
def strat(cal):
    keep = np.zeros(N, bool); ok_all = True; bound = 0.0
    for key, stratum in (("h", req), ("b", ~req)):
        ids = np.where(stratum)[0]; pos = {i: j for j, i in enumerate(ids)}; cl = np.array([pos[c] for c in cal if stratum[c]])
        ok, mask, q, _ = ltt_walk(g[ids], unsafe[ids], cl, A[key], DELTA / 2)
        if ok: keep[ids[mask]] = True
        else: ok_all = False; fm, lb = cp_fallback(g[ids], unsafe[ids], cl, DELTA / 2); keep[ids[fm]] = True
    sh = req[keep].mean(); return keep, ok_all, A["h"] * sh + A["b"] * (1 - sh)
r = np.random.default_rng(1); nc = nf = 0
for _ in range(200):
    keep, ok, bound = strat(r.choice(N, 800, replace=False)); nc += ok; nf += ok and unsafe[keep].mean() > bound
print(f"stratified certificate (alpha_h .20, alpha_b .15, n 800): rate {nc/200:.2f}, false {nf/200:.3f}")
for s in range(3):
    keep, ok, bound = strat(np.random.default_rng(1000 + s).choice(N, 800, replace=False)); idx = [int(i) for i in np.where(keep)[0]]
    meta = {"certified": ok, "overall_bound": float(bound), "alpha_h": .20, "alpha_b": .15, "n": len(idx), "unsafe_frac": float(unsafe[idx].mean()), "safety_demo_share": float(demo[idx].mean()), "scorer": "Llama-Guard-3-8B"}
    json.dump({"idx": idx, "meta": meta}, open(f"selections/certstrat2_lg800_s{s}.json", "w")); summ.setdefault("certstrat2_lg800", {})[str(s)] = meta
    print(f"  certstrat2_lg800 s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} (bound {bound:.3f}) demos={meta['safety_demo_share']:.3f} certified={ok}")
json.dump(summ, open("results/step2_selections.json", "w"), indent=1)
PY
names=""
for s in 0 1 2; do env PYTHONNOUSERSITE=1 ARM=certstrat2_lg800 SEED=$s $PY scripts/step2_sft.py > logs/sft/certstrat2_lg800_s$s.log 2>&1 && echo "done s$s" || echo "FAIL s$s"; names="$names,certstrat2_lg800_s$s"; done
env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e10d_gen.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e10d_eval2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e10d_mdj.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e10d_panel2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e6_utility.py > logs/e10d_e6.log 2>&1
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); sel=json.load(open(f"{W}/results/step2_selections.json")); u6=json.load(open(f"{W}/results/e6_utility.json"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.3f}"
print(f"{'arm':18s} {'unsafe':>6s} {'demos':>6s} {'cert':>5s} {'beaver-dam':>12s} {'MD-Judge':>12s} {'WildGuard':>12s} {'LlamaGuard3':>12s} {'XSTest-unsafe':>13s} {'ROUGE-L':>8s}")
for a in ["full","certlg400","certstrat_lg800","certstrat2_lg800","certbd400","oracle"]:
    ms=[n for n in wg if n.startswith(a+"_")]; u=np.mean([sel[a][s]["unsafe_frac"] for s in sel[a]]); d=np.mean([sel[a][s].get("safety_demo_share",float('nan')) for s in sel[a]]); c=sum(bool(sel[a][s].get("certified",False)) for s in sel[a])
    print(f"{a:18s} {u:6.3f} {d:6.3f} {c:>3d}/{len(sel[a])} {cell([ev[n]['beaverdam_flag'] for n in ms]):>12s} {cell([np.mean(md[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>12s} {cell([np.mean(lg[n]['flag']) for n in ms]):>12s} {cell([u6[n]['xstest_unsafe_harmful'] for n in ms]):>13s} {cell([u6[n]['rouge_l'] for n in ms]):>8s}")
PY
echo E10D DONE
