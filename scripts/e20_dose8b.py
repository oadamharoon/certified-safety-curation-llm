"""E20: the dose-response of Section 6 at 8B (pre-registered 2026-09-18 01:00, README). Fixed size
1584 (the certified Llama Guard 3 set's mean size in Table 3) and composition 8.3 percent harmful
(132 harmful examples); the share of safety demonstrations among the kept safe examples varies over
q in {0.30, 0.37, 0.45, 0.545} (the smallest share the benign-prompt stratum allows, the certified
Llama Guard 3 set's share, an intermediate point, and the pool's share) by stratified uniform
sampling with no scorer, three seeds. Selections safedemo8_{q}_s{s} in selections_e3/lambda0.25."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import W
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); S = json.load(open(f"{D}/scores.json"))
unsafe = np.array([not r["is_safe"] for r in pool]); req = np.array(S["harmful_request"])
demo = np.where(~unsafe & req)[0]; benign = np.where(~unsafe & ~req)[0]; uns = np.where(unsafe)[0]
print(f"8B strata: safety-demo {len(demo)}, benign-prompt safe {len(benign)}, unsafe {len(uns)}")
summ = json.load(open(f"{D}/summary.json")); NSEL, NUNS = 1584, 132; n_safe = NSEL - NUNS
for q in [0.30, 0.37, 0.45, 0.545]:
    for s in range(3):
        rng = np.random.default_rng(8000 + int(q * 1000) * 10 + s); k = int(round(q * n_safe))
        idx = np.r_[rng.choice(demo, k, replace=False), rng.choice(benign, n_safe - k, replace=False), rng.choice(uns, NUNS, replace=False)]
        idx = [int(i) for i in rng.permutation(idx)]
        meta = {"n": NSEL, "unsafe_frac": float(unsafe[idx].mean()), "safety_demo_share": float((req[idx] & ~unsafe[idx]).mean()), "safety_demo_share_safe": q, "n_harmful": NUNS, "rule": "stratified uniform by WildGuard harmful-prompt label, no scorer"}
        json.dump({"idx": idx, "meta": meta}, open(f"{D}/safedemo8_{q}_s{s}.json", "w")); summ.setdefault(f"safedemo8_{q}", {})[str(s)] = meta
        print(f"  safedemo8_{q} s{s}: n={NSEL} unsafe={meta['unsafe_frac']:.3f} demo share of set={meta['safety_demo_share']:.3f}")
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1); print("E20 SELECT DONE")
