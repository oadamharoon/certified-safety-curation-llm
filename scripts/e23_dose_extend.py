"""E23 (pre-registered 2026-09-19 00:40, README): extend the 0.5B dose-response of E4b (Fig. 3 right,
Table 7) with three intermediate demonstration-share levels, q in {0.17, 0.33, 0.47}, and raise every
level to five seeds (seeds 3 and 4 added to the existing q in {0.10, 0.25, 0.40, 0.54}). Same rule as
E4b: fixed size 1400 and composition 6.6 percent (92 harmful), the share q of safety demonstrations
among the 1308 kept safe examples, stratified uniform sampling by WildGuard's prompt label with the
natural refusal mix inside each stratum, no scorer. Selections safedemo{q}_s{s}; rng seed
6000 + int(q*100)*10 + s as in E4b. Idempotent: existing selection files are kept."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"])
demo = np.where(~unsafe & req)[0]; benign = np.where(~unsafe & ~req)[0]; uns = np.where(unsafe)[0]
summ = json.load(open(f"{W}/results/step2_selections.json")); new = []
for q in [0.1, 0.17, 0.25, 0.33, 0.4, 0.47, 0.54]:
    for s in range(5):
        f = f"{W}/selections/safedemo{q}_s{s}.json"
        if os.path.exists(f): continue
        rng = np.random.default_rng(6000 + int(round(q * 100)) * 10 + s); n_safe = 1400 - 92; k = int(round(q * n_safe))
        idx = np.r_[rng.choice(demo, k, replace=False), rng.choice(benign, n_safe - k, replace=False), rng.choice(uns, 92, replace=False)]
        idx = [int(i) for i in rng.permutation(idx)]
        meta = {"n": 1400, "unsafe_frac": float(unsafe[idx].mean()), "safety_demo_share_safe": q, "rule": "stratified uniform by WildGuard harmful-prompt label"}
        json.dump({"idx": idx, "meta": meta}, open(f, "w")); summ.setdefault(f"safedemo{q}", {})[str(s)] = meta; new.append(f"safedemo{q}_s{s}")
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1)
print("new selections:", len(new)); print(" ".join(new))
