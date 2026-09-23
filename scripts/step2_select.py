"""Step 2a: build the training selections for every arm and seed.

Writes selections/{arm}_s{seed}.json with {"idx": [...pool indices...], "meta": {...}} and
results/step2_selections.json summarizing composition per arm. Also fixes the held-out
evaluation sets (results/eval_sets.json) once, from the test split.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

ALPHA, DELTA, CAL_N = 0.10, 0.10, 200
POOL_UNSAFE, N = 0.25, 4000
SCORER = "Qwen2.5-7B-Instruct"
SEEDS = [0, 1, 2]

rows = load_split("train")
pool = build_pool(rows, N, POOL_UNSAFE, seed=0)
unsafe = np.array([not r["is_safe"] for r in pool])
g = np.array(json.load(open(f"{W}/results/scores_{SCORER}_N{N}_u{POOL_UNSAFE}.json")))
assert len(g) == N
from sklearn.metrics import roc_auc_score
print(f"pool N={N} unsafe {unsafe.mean():.3f}; scorer {SCORER} AUC {roc_auc_score(~unsafe, g):.3f} (step 1 reported 0.748)")
os.makedirs(f"{W}/selections", exist_ok=True)
summary = {}


def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]
    meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/{arm}_s{s}.json", "w"))
    summary.setdefault(arm, {})[s] = meta
    print(f"  {arm:7s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} "
          + " ".join(f"{k}={v}" for k, v in meta.items() if k not in ("n", "unsafe_frac", "audit")))


for s in SEEDS:
    rng = np.random.default_rng(1000 + s)
    for cal_n in (200, 400):
        cal = np.random.default_rng(1000 + s).choice(N, cal_n, replace=False)
        ok, mask, q, audit = ltt_walk(g, unsafe, cal, ALPHA, DELTA)
        if ok:
            save(f"cert{cal_n}", s, np.where(mask)[0], {"certified": True, "q": q, "audit": audit})
        else:
            fmask, lb = cp_fallback(g, unsafe, cal, DELTA)
            save(f"cert{cal_n}", s, np.where(fmask)[0], {"certified": False, "cp_lower_bound": lb, "audit": audit})
        if cal_n == CAL_N:
            cal200 = cal
    cal = cal200
    save("full", s, np.arange(N), {})
    keep = int(round((1 - POOL_UNSAFE) * N))
    save("heur", s, np.argsort(g)[::-1][:keep], {"rule": f"top {keep} by score"})
    save("labels", s, [i for i in cal if not unsafe[i]], {"rule": "labeled-safe among the 200 calibration draws"})
    save("oracle", s, np.where(~unsafe)[0], {})
json.dump(summary, open(f"{W}/results/step2_selections.json", "w"), indent=1)

# held-out evaluation sets, fixed once
ev_path = f"{W}/results/eval_sets.json"
if not os.path.exists(ev_path):
    test = load_split("test")
    rng = np.random.default_rng(7)
    by_prompt = {}
    for r in test:
        by_prompt.setdefault(r["prompt"], []).append(r)
    prompts = list(by_prompt)
    gen_prompts = [prompts[i] for i in rng.choice(len(prompts), 500, replace=False)]
    pairs = []
    for p, rs in by_prompt.items():
        safe = [r for r in rs if r["is_safe"]]; uns = [r for r in rs if not r["is_safe"]]
        if safe and uns:
            pairs.append({"prompt": p, "safe": safe[0]["response"], "unsafe": uns[0]["response"]})
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(500, len(pairs)), replace=False)]
    json.dump({"gen_prompts": gen_prompts, "pref_pairs": pairs}, open(ev_path, "w"))
    print(f"eval sets: {len(gen_prompts)} generation prompts, {len(pairs)} labeled safe/unsafe pairs "
          f"(from {len(prompts)} unique test prompts)")
