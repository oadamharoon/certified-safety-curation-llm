#!/bin/bash
# E10c: stratified certified curation with Llama Guard at n=800 (a feasible budget), and the
# two-property certificate at n=800 with beta = 0.30, to find the operating point at which
# the method certifies rather than falls back. 3 seeds, alpha .10.
PY="${PY:-python}"
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; PY=${PY}; cd $W
env PYTHONNOUSERSITE=1 $PY - <<'PY' 2>&1 | grep -v -i warn | tee logs/e10c_select.log
import json, sys, numpy as np; sys.path.insert(0, "scripts"); from common import *
from scipy.stats import hypergeom
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool]); N = 4000
req = np.array(json.load(open("results/scores_wildguard_pool.json"))["harmful_request"]); demo = req & ~unsafe
g = -np.array(json.load(open("results/scores_llamaguard_pool.json"))["score_unsafe_minus_safe"])
summ = json.load(open("results/step2_selections.json")); ALPHA, DELTA = 0.10, 0.10
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"selections/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:18s} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} certified={meta['certified']}")
def strat(cal):
    keep = np.zeros(N, bool); ok_all = True
    for stratum in (req, ~req):
        ids = np.where(stratum)[0]; pos = {i: j for j, i in enumerate(ids)}; cal_local = np.array([pos[c] for c in cal if stratum[c]])
        ok, mask, q, _ = ltt_walk(g[ids], unsafe[ids], cal_local, ALPHA, DELTA / 2)
        if ok: keep[ids[mask]] = True
        else: ok_all = False; fm, lb = cp_fallback(g[ids], unsafe[ids], cal_local, DELTA / 2); keep[ids[fm]] = True
    return keep, ok_all
def hyp_p_lower(k, m, n_sel, beta):
    kd = int(np.ceil(beta * n_sel)) - 1
    return 1.0 if kd < 0 else float(hypergeom.sf(k - 1, n_sel, kd, m))
def walk2(cal, beta):
    best = None
    for q in QS:
        mask = g >= np.quantile(g, q); sc = mask[cal]; m = int(sc.sum()); n_sel = int(mask.sum())
        p1 = hyp_p(int(unsafe[cal][sc].sum()), m, n_sel, ALPHA) if m else 1.0; p2 = hyp_p_lower(int(demo[cal][sc].sum()), m, n_sel, beta) if m else 1.0
        if p1 <= DELTA / 2 and p2 <= DELTA / 2: best = (mask, q)
        else: break
    return best
for n in (800,):
    r = np.random.default_rng(1); nc_s = nc_2 = 0
    for _ in range(200):
        cal = r.choice(N, n, replace=False); nc_s += strat(cal)[1]; nc_2 += walk2(cal, 0.30) is not None
    print(f"n={n}: stratified (both strata certify) rate {nc_s/200:.2f}; two-property (alpha .10, beta .30) rate {nc_2/200:.2f}")
    for s in range(3):
        cal = np.random.default_rng(1000 + s).choice(N, n, replace=False)
        keep, ok = strat(cal); save(f"certstrat_lg{n}", s, np.where(keep)[0], {"certified": ok, "scorer": "Llama-Guard-3-8B"})
        b = walk2(cal, 0.30)
        if b is not None: save(f"cert2p_lg{n}", s, np.where(b[0])[0], {"certified": True, "q": b[1], "beta": 0.30})
        else: print(f"  cert2p_lg{n} s{s}: REFUSED")
json.dump(summ, open("results/step2_selections.json", "w"), indent=1)
PY
names=""
for arm in certstrat_lg800 cert2p_lg800; do for s in 0 1 2; do
  [ -f selections/${arm}_s$s.json ] || continue
  env PYTHONNOUSERSITE=1 ARM=$arm SEED=$s $PY scripts/step2_sft.py > logs/sft/${arm}_s$s.log 2>&1 && echo "done $arm s$s" || echo "FAIL $arm s$s"; names="$names,${arm}_s$s"; done; done
[ -n "$names" ] && { env PYTHONNOUSERSITE=1 EVAL_ONLY=${names#,} $PY scripts/step2_eval.py > logs/e10c_gen.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/step2_eval2.py > logs/e10c_eval2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_evaluators.py > logs/e10c_mdj.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e1_panel2.py > logs/e10c_panel2.log 2>&1; env PYTHONNOUSERSITE=1 $PY scripts/e6_utility.py > logs/e10c_e6.log 2>&1; }
env PYTHONNOUSERSITE=1 $PY - <<'PY'
import json, numpy as np
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ev=json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md=json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg=json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg=json.load(open(f"{W}/results/gen_scores/llamaguard.json")); sel=json.load(open(f"{W}/results/step2_selections.json")); u6=json.load(open(f"{W}/results/e6_utility.json"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v,ddof=1)/np.sqrt(len(v)):.3f}" if len(v)>1 else (f"{np.mean(v):.3f}" if v else "  --  ")
print(f"{'arm':16s} {'unsafe':>6s} {'demos':>6s} {'cert':>5s} {'beaver-dam':>12s} {'MD-Judge':>12s} {'WildGuard':>12s} {'LlamaGuard3':>12s} {'XSTest-unsafe':>13s}")
for a in ["full","certlg400","certstrat_lg","certstrat_lg800","cert2p_lg800","certbd400","oracle"]:
    ms=[n for n in wg if n.startswith(a+"_")]
    if not ms: print(f"{a:16s}   (no trained models)"); continue
    u=np.mean([sel[a][s]["unsafe_frac"] for s in sel[a]]); d=np.mean([sel[a][s].get("safety_demo_share",float('nan')) for s in sel[a]]); c=sum(bool(sel[a][s].get("certified",False)) for s in sel[a])
    print(f"{a:16s} {u:6.3f} {d:6.3f} {c:>3d}/{len(sel[a])} {cell([ev[n]['beaverdam_flag'] for n in ms]):>12s} {cell([np.mean(md[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>12s} {cell([np.mean(lg[n]['flag']) for n in ms]):>12s} {cell([u6[n]['xstest_unsafe_harmful'] for n in ms]):>13s}")
PY
echo E10C DONE
