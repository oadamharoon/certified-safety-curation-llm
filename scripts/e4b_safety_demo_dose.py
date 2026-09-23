"""E4b: causal test of the refined mechanism. Fixed size 1400 and composition 6.6 percent;
vary the share of SAFETY DEMONSTRATIONS (harmful prompt by WildGuard's label, human-safe
response) among kept safe examples, q in {0.10, 0.25, 0.40, 0.54} = cert400 / full / oracle
levels plus a lower point, by stratified uniform sampling with the natural refusal mix inside
each stratum. Three seeds. Selections safedemo{q}_s{s}."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"])
demo = np.where(~unsafe & req)[0]; benign = np.where(~unsafe & ~req)[0]; uns = np.where(unsafe)[0]
print(f"strata: safety-demo {len(demo)}, benign-prompt safe {len(benign)}, unsafe {len(uns)}")
summ = json.load(open(f"{W}/results/step2_selections.json"))
for q in [0.10, 0.25, 0.40, 0.54]:
    for s in range(3):
        rng = np.random.default_rng(6000 + int(q * 100) * 10 + s); n_safe = 1400 - 92; k = int(round(q * n_safe))
        idx = np.r_[rng.choice(demo, k, replace=False), rng.choice(benign, n_safe - k, replace=False), rng.choice(uns, 92, replace=False)]
        idx = [int(i) for i in rng.permutation(idx)]
        meta = {"n": 1400, "unsafe_frac": float(unsafe[idx].mean()), "safety_demo_share_safe": q, "rule": "stratified uniform by WildGuard harmful-prompt label"}
        json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/safedemo{q}_s{s}.json", "w")); summ.setdefault(f"safedemo{q}", {})[s] = meta
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1); print("written")
