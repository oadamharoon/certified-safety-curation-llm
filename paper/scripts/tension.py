"""Lemma 1 (composition-covariate tension) on the 0.5B judge-certified arm: per seed, the
retention rates R_u (harmful), R_1 (safe, harmful prompt), R_0 (safe, benign prompt) of the
cert400 selection, the constant c = R_1 / R_u, the realized safety-demonstration share among the
selection's safe examples pi_lambda, and the lemma's bound c * pi * alpha/(1-alpha) * (1-rho)/rho.
Writes results/tension.json (the numbers quoted after the lemma)."""
import json, sys, os
import numpy as np
W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{W}/scripts")
from common import build_pool, load_split
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"]); safe1 = ~unsafe & req; safe0 = ~unsafe & ~req
pi = safe1.sum() / (~unsafe).sum(); rho = unsafe.mean(); alpha = 0.10; out = {"pi": float(pi), "rho": float(rho), "seeds": {}}
for s in range(5):
    idx = np.array(json.load(open(f"{W}/selections/cert400_s{s}.json"))["idx"]); k = np.zeros(len(pool), bool); k[idx] = True
    Ru, R1, R0 = (k & unsafe).sum() / unsafe.sum(), (k & safe1).sum() / safe1.sum(), (k & safe0).sum() / safe0.sum()
    c = R1 / Ru; pil = (k & safe1).sum() / (k & ~unsafe).sum()
    out["seeds"][str(s)] = {"R_u": float(Ru), "R_1": float(R1), "R_0": float(R0), "c": float(c), "pi_lambda": float(pil),
                            "bound": float(c * pi * alpha / (1 - alpha) * (1 - rho) / rho), "ahat": float(unsafe[idx].mean())}
json.dump(out, open(f"{W}/results/tension.json", "w"), indent=1)
for s, v in out["seeds"].items(): print(s, {k: round(x, 3) for k, x in v.items()})
