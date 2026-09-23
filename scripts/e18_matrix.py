"""E18: the last two cells of the scorer x certificate design matrix (pre-registered 2026-09-17
23:00, README). Every scorer at each scale should carry a fixed cutoff, the pooled certificate and
the stratified certificate at the attainable targets (.20/.15, n 800). Missing were the stratified
certificate on the Prompting judge score at 8B (certstrat_prompt) and on SAFT's score at 0.5B
(certstrat_saft05). Selections only; the runners fine-tune and evaluate. certcurate-llm env."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import W, build_pool, load_split, ltt_walk, cp_fallback


def strat(g, unsafe, req, s, N):   # verbatim from e13_completeness.py (alpha_h .20, alpha_b .15, delta/2 each, n 800, rng 1000+s)
    cal = np.random.default_rng(1000 + s).choice(N, 800, replace=False); keep = np.zeros(N, bool); ok_all = True
    for stratum, a in ((req, 0.20), (~req, 0.15)):
        ids = np.where(stratum)[0]; pos = {i: j for j, i in enumerate(ids)}; cl = np.array([pos[c] for c in cal if stratum[c]])
        ok, mask, q, _ = ltt_walk(g[ids], unsafe[ids], cl, a, 0.05)
        if ok: keep[ids[mask]] = True
        else: ok_all = False; fm, lb = cp_fallback(g[ids], unsafe[ids], cl, 0.05); keep[ids[fm]] = True
    sh = req[keep].mean(); return keep, {"certified": ok_all, "overall_bound": float(0.20 * sh + 0.15 * (1 - sh))}


def saver(D, summ, unsafe, demo):
    def save(arm, s, idx, meta):
        idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
        json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
        print(f"  {arm:15s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} {meta.get('certified','')}", flush=True)
    return save
# 8B: Prompting judge score
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); N = len(pool)
unsafe = np.array([not r["is_safe"] for r in pool]); S = json.load(open(f"{D}/scores.json")); req = np.array(S["harmful_request"]); demo = req & ~unsafe
summ = json.load(open(f"{D}/summary.json")); save = saver(D, summ, unsafe, demo); gp = -np.array(S["prompting_yes_logodds"])
for s in range(3):
    keep, meta = strat(gp, unsafe, req, s, N); save("certstrat_prompt", s, np.where(keep)[0], dict(meta, scorer="Prompting judge"))
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1)
# 0.5B: SAFT score (validated k of seed s, as certsaft05)
pool = build_pool(load_split("train"), 4000, 0.25, 0); N = len(pool); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"]); demo = req & ~unsafe
Z = np.load(f"{W}/results/saft05_embeddings.npz")["layer"]
def saft_scores(Z, k):
    Zc = Z - Z.mean(0, keepdims=True); _, _, Vt = np.linalg.svd(Zc, full_matrices=False); return (Zc @ Vt[:k].T) ** 2 @ np.ones(k) / k
summ2 = json.load(open(f"{W}/results/step2_selections.json")); save2 = saver(f"{W}/selections", summ2, unsafe, demo)
for s in range(3):
    k = summ2["certsaft05"][str(s)]["k"]; sc = -saft_scores(Z, k)
    keep, meta = strat(sc, unsafe, req, s, N); save2("certstrat_saft05", s, np.where(keep)[0], dict(meta, k=k))
json.dump(summ2, open(f"{W}/results/step2_selections.json", "w"), indent=1); print("E18 SELECT DONE")
