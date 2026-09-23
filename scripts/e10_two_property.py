"""E10: certify the property the learner needs.

(a) Two-property certificate on one threshold walk: at each grid point test BOTH
    H1: harmful fraction of S(lambda) > alpha      (exact hypergeometric, as before)
    H2: safety-demonstration share of S(lambda) < beta  (exact hypergeometric, count of
        calibration examples in S that are harmful-prompt AND safe-response vs the least
        favorable null beta*N_j - 1)
    Reject a grid point only if both p-values <= delta/2 (Bonferroni over the two risks);
    fixed-sequence walk as before. Labels: the calibration sample supplies both bits
    (human is_safe; prompt harmfulness from WildGuard's label, standing in for a second
    human bit). beta is set to the pool's own safety-demo share among safe examples times
    (1 - alpha), i.e. "keep at least the pool's proportion".
(b) Stratified certified curation: split the pool by WildGuard prompt-harmfulness, run the
    single-property walk within each stratum with delta/2 each, take the union; the
    stratum proportions of the pool are preserved by construction.
Selections: cert2p_{scorer}_s{s} (may be all refusals) and certstrat_{scorer}_s{s}.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool]); N = len(pool)
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"])
demo = req & ~unsafe
G = {"qwen": np.array(json.load(open(f"{W}/results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json"))),
     "lg": -np.array(json.load(open(f"{W}/results/scores_llamaguard_pool.json"))["score_unsafe_minus_safe"])}
ALPHA, DELTA = 0.10, 0.10
BETA = float(demo.sum() / N) * 0.9      # pool has 40.7% safety demos; require >= 36.6%
print(f"pool: unsafe {unsafe.mean():.3f}, safety-demo share {demo.mean():.3f}; beta = {BETA:.3f}")
summ = json.load(open(f"{W}/results/step2_selections.json"))


def hyp_p_lower(k, m, n_sel, beta):
    """P(>= k demos among m draws | selection has < beta*n_sel demos), least favorable null."""
    from scipy.stats import hypergeom
    kd = int(np.ceil(beta * n_sel)) - 1
    if kd < 0: return 1.0
    return float(hypergeom.sf(k - 1, n_sel, kd, m))


def walk2(g, cal):
    best = None
    for q in QS:
        mask = g >= np.quantile(g, q); sc = mask[cal]; m = int(sc.sum())
        k1 = int(unsafe[cal][sc].sum()); k2 = int(demo[cal][sc].sum()); n_sel = int(mask.sum())
        p1 = hyp_p(k1, m, n_sel, ALPHA) if m else 1.0; p2 = hyp_p_lower(k2, m, n_sel, BETA) if m else 1.0
        if p1 <= DELTA / 2 and p2 <= DELTA / 2: best = (mask, q)
        else: break
    return best


def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:16s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} {meta.get('certified','')}")


rates = {}
for name, g in G.items():
    # 200-draw certification rate of the two-property certificate
    r2 = np.random.default_rng(1); nc = 0
    for _ in range(200):
        if walk2(g, r2.choice(N, 400, replace=False)) is not None: nc += 1
    rates[name] = nc / 200
    print(f"{name}: two-property certification rate at n=400: {nc/200:.2f}")
    for s in range(3):
        cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False)
        b = walk2(g, cal)
        if b is not None: save(f"cert2p_{name}", s, np.where(b[0])[0], {"certified": True, "q": b[1], "beta": BETA})
        else: print(f"  cert2p_{name:5s} s{s}: REFUSED (no threshold satisfies both)"); summ.setdefault(f"cert2p_{name}", {})[str(s)] = {"certified": False, "n": 0}
        # stratified: walk within each prompt stratum with delta/2
        keep = np.zeros(N, bool); ok_all = True
        for stratum in (req, ~req):
            ids = np.where(stratum)[0]; cal_s = np.array([c for c in cal if stratum[c]])
            pos = {i: j for j, i in enumerate(ids)}; cal_local = np.array([pos[c] for c in cal_s])
            ok, mask, q, _ = ltt_walk(g[ids], unsafe[ids], cal_local, ALPHA, DELTA / 2)
            if ok: keep[ids[mask]] = True
            else:
                ok_all = False; fm, lb = cp_fallback(g[ids], unsafe[ids], cal_local, DELTA / 2); keep[ids[fm]] = True
        save(f"certstrat_{name}", s, np.where(keep)[0], {"certified": ok_all, "rule": "per-prompt-stratum walk, delta/2 each; fallback in a refusing stratum"})
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1)
json.dump({"beta": BETA, "two_property_cert_rate_n400": rates}, open(f"{W}/results/e10_rates.json", "w"), indent=1)
