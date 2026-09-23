"""E4: causal test of the style-entanglement mechanism.

Fix size (1400) and composition (92 unsafe = 6.6 percent, cert400 s0's values) and vary
ONLY the refusal share of the kept safe examples, p in {0.05, 0.10, 0.17, 0.25}, by
stratified uniform sampling from the pool's safe-refusal and safe-engaged strata (never by
score, so the scorer plays no role). Unsafe examples are sampled uniformly. Three seeds.
Refusal style is the regex used throughout (a limitation stated in the paper). Writes
selections/refshare{p}_s{s}.json; training and evaluation reuse step2_sft / step2_eval /
step2_eval2 / e1_evaluators unchanged.
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

REFUSAL = re.compile(r"^\s*(I'?m sorry|I cannot|I can'?t|I am not able|I'm not able|No[,.]|It is not|I do not|I don'?t think|Sorry|I would not|I will not|That is not|It's not appropriate|I am unable|Unfortunately)", re.I)
N_SEL, N_UNSAFE = 1400, 92
SHARES = [0.05, 0.10, 0.17, 0.25]
pool = build_pool(load_split("train"), 4000, 0.25, 0)
unsafe = np.array([not r["is_safe"] for r in pool]); ref = np.array([bool(REFUSAL.match(r["response"])) for r in pool])
safe_ref = np.where(~unsafe & ref)[0]; safe_eng = np.where(~unsafe & ~ref)[0]; uns = np.where(unsafe)[0]
print(f"strata: safe-refusal {len(safe_ref)}, safe-engaged {len(safe_eng)}, unsafe {len(uns)}")
summ = json.load(open(f"{W}/results/step2_selections.json"))
for p in SHARES:
    for s in range(3):
        rng = np.random.default_rng(5000 + int(p * 100) * 10 + s)
        n_safe = N_SEL - N_UNSAFE; n_ref = int(round(p * n_safe))
        idx = np.r_[rng.choice(safe_ref, n_ref, replace=False), rng.choice(safe_eng, n_safe - n_ref, replace=False), rng.choice(uns, N_UNSAFE, replace=False)]
        idx = [int(i) for i in rng.permutation(idx)]
        meta = {"n": len(idx), "unsafe_frac": float(unsafe[idx].mean()), "refusal_share_safe": p, "rule": "stratified uniform, no scorer"}
        json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/refshare{p}_s{s}.json", "w"))
        summ.setdefault(f"refshare{p}", {})[s] = meta
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1)
print("selections written")
