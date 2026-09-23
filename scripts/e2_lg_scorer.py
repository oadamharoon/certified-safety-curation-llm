"""E2: a deployable scorer. Llama Guard 3 (not trained on BeaverTails) scores the pool;
the identical certificate (alpha .10, delta .10, n in {200, 400}) selects; plus the field's
fixed-cutoff practice with the same scorer (keep everything Llama Guard does not flag, no
labels). Selections: certlg{n}_s{s}, lgfilter_s{s}."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from sklearn.metrics import roc_auc_score
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
path = f"{W}/results/scores_llamaguard_pool.json"
if os.path.exists(path):
    d = json.load(open(path))
else:
    sc, fl = llama_guard_scores(pool); d = {"score_unsafe_minus_safe": sc.tolist(), "flag": fl.tolist()}; json.dump(d, open(path, "w"))
g = -np.array(d["score_unsafe_minus_safe"]); flag = np.array(d["flag"])       # g: higher = safer
wgp = json.load(open(f"{W}/results/scores_wildguard_pool.json")); req = np.array(wgp["harmful_request"]); ref = np.array(wgp["refusal"])
sf = ~unsafe
print(f"Llama Guard 3 on the pool: AUC {roc_auc_score(~unsafe, g):.3f}; flags {flag[~unsafe].mean():.3f} of human-safe, {flag[unsafe].mean():.3f} of human-unsafe")
print(f"style bias among human-safe: score benign-prompt {g[sf & ~req].mean():.2f} vs harmful-prompt {g[sf & req].mean():.2f}; refusal {g[sf & ref].mean():.2f} vs engaged {g[sf & ~ref].mean():.2f}")
summ = json.load(open(f"{W}/results/step2_selections.json"))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()),
                                            safety_demo_share=float(req[idx][~unsafe[idx]].sum() / len(idx)))
    json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[s] = meta
    print(f"  {arm:10s} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} safety-demo {meta['safety_demo_share']:.3f} " + " ".join(f"{k}={v}" for k, v in meta.items() if k in ("certified", "q")))
for s in range(3):
    for n in (200, 400):
        cal = np.random.default_rng(1000 + s).choice(4000, n, replace=False)
        ok, mask, q, audit = ltt_walk(g, unsafe, cal, 0.10, 0.10)
        if ok: save(f"certlg{n}", s, np.where(mask)[0], {"certified": True, "q": q, "scorer": "Llama-Guard-3-8B"})
        else:
            fm, lb = cp_fallback(g, unsafe, cal, 0.10); save(f"certlg{n}", s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb, "scorer": "Llama-Guard-3-8B"})
    save("lgfilter", s, np.where(~flag)[0], {"rule": "keep all not flagged by Llama Guard 3, no labels", "scorer": "Llama-Guard-3-8B"})
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1)
