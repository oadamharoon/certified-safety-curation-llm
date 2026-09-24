"""E19: Llama Guard 3 through the certification gate and the label-complexity sweep (pre-registered
2026-09-18 01:00, README), so every scorer of the paper passes the same validity check as the two
Qwen judges. Scores the 0.10 and natural pools with Llama Guard 3 (the 0.25 pool's scores exist
from E2), then runs E5's sweep (200 draws, n grid, alpha .10/.25, closed form) for the LG scorer and
appends to results/e5_label_complexity.json under keys '<pool>|LG|a<alpha>'; writes the gate
row (n = 200) into results/llm_certify_LlamaGuard3_<pool>.json in step 1's format. Runs under PY."""
import os
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from cert_rate_theory import pr_certify
from sklearn.metrics import roc_auc_score
rows = load_split("train"); NS = [50, 100, 200, 300, 400, 600, 800]; DRAWS = 200
POOLS = [("nat", None), ("u0.25", 0.25), ("u0.10", 0.10)]
out = json.load(open(f"{W}/results/e5_label_complexity.json"))
for ptag, pu in POOLS:
    rng = np.random.default_rng(0)
    pool = build_pool(rows, 4000, pu, 0) if pu is not None else [rows[i] for i in rng.choice(len(rows), 4000, replace=False)]
    unsafe = np.array([not r["is_safe"] for r in pool]); N = len(pool)
    sp = f"{W}/results/scores_llamaguard_N4000_{ptag}.json"
    if ptag == "u0.25":
        g = -np.array(json.load(open(f"{W}/results/scores_llamaguard_pool.json"))["score_unsafe_minus_safe"])
    elif os.path.exists(sp):
        g = np.array(json.load(open(sp)))
    else:
        sc, fl = llama_guard_scores(pool); g = -sc; json.dump(g.tolist(), open(sp, "w"))
    auc = float(roc_auc_score(~unsafe, g)); mask1 = g >= np.quantile(g, QS[0]); N1 = int(mask1.sum()); K1 = int(unsafe[mask1].sum())
    gate = {"model": "meta-llama/Llama-Guard-3-8B", "N": N, "pool_unsafe": float(unsafe.mean()), "auc": auc, "pool_tag": ptag}
    for alpha in (0.10, 0.25):
        key = f"{ptag}|LG|a{alpha}"
        rec = {"N": N, "N1": N1, "K1": K1, "u1": K1 / N1, "margin": alpha - K1 / N1, "n": NS, "rate": [], "false": [], "pred": []}
        for n in NS:
            r2 = np.random.default_rng(1); nc = nf = 0
            for _ in range(DRAWS):
                ok, m, q, _ = ltt_walk(g, unsafe, r2.choice(N, n, replace=False), alpha, 0.10)
                if ok: nc += 1; nf += unsafe[m].mean() > alpha
            rec["rate"].append(nc / DRAWS); rec["false"].append(nf / DRAWS); rec["pred"].append(pr_certify(N, N1, K1, n, alpha=alpha, delta=0.10))
        out[key] = rec; i200 = NS.index(200)
        gate[f"alpha{alpha:g}"] = {"cert_rate": rec["rate"][i200], "false_cert": rec["false"][i200]}
        print(f"{key:20s} auc={auc:.3f} u1={rec['u1']:.3f} margin={rec['margin']:+.3f} | " + " ".join(f"n{n}:{r:.2f}/{p:.2f}" for n, r, p in zip(NS, rec["rate"], rec["pred"])) + f" | worst false {max(rec['false']):.3f}", flush=True)
    json.dump(gate, open(f"{W}/results/llm_certify_LlamaGuard3_{ptag}.json", "w"), indent=1)
json.dump(out, open(f"{W}/results/e5_label_complexity.json", "w"), indent=1); print("E19 DONE")
