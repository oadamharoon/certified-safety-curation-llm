"""E25 (8B) and E26 (0.5B) selections. Pre-registered in README before launch (2026-09-21 03:30).
E25 safetop8_s{s}: size 1584, 132 harmful from the top third of Llama Guard 3's harmful ranking (as
harmtop8), demonstration share .37 among the 1452 safe examples, the safe examples drawn from the TOP of
Llama Guard 3's within-safe ranking inside each prompt stratum (top k by score, no randomness beyond the
tie order). safelen8_s{s}: same, but the safe examples sampled score-blind to match the certified set's
response-length distribution (stratified by length decile of certlg_s{s}'s safe examples).
E26 labels800_s{s}: the human-safe examples among the 800-label calibration draw of the stratified
certificate (rng 1000+s, as run_e10d.sh); randdemo_s{s}: random1400 seed s plus every safety
demonstration among those 800 labels."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import W, build_pool, load_split
# ---- E25 (8B pool)
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); S = json.load(open(f"{D}/scores.json"))
unsafe = np.array([not r["is_safe"] for r in pool]); req = np.array(S["harmful_request"]); g = np.array(S["lg_safe_score"])
rl = np.array([len(r["response"]) for r in pool]); summ = json.load(open(f"{D}/summary.json"))
def save8(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), n_harmful=int(unsafe[idx].sum()), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float((req[idx] & ~unsafe[idx]).mean()),
                                            safe_lg_mean=float(g[idx][~unsafe[idx]].mean()), safe_len_mean=float(rl[idx][~unsafe[idx]].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:10s} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} safeLG={meta['safe_lg_mean']:.2f} safeLen={meta['safe_len_mean']:.0f}")
uns = np.where(unsafe)[0]; order = uns[np.argsort(-g[uns])]; top_third = order[:len(order) // 3]
demo = np.where(~unsafe & req)[0]; benign = np.where(~unsafe & ~req)[0]; n_safe = 1584 - 132; k = int(round(0.37 * n_safe))
for s in range(3):
    rng = np.random.default_rng(9300 + s); harm = rng.choice(top_third, 132, replace=False)   # same rng as harmtop8's harmful draw
    # safetop8: top-k by LG score within each stratum
    top_demo = demo[np.argsort(-g[demo])][:k]; top_ben = benign[np.argsort(-g[benign])][:n_safe - k]
    save8("safetop8", s, rng.permutation(np.r_[top_demo, top_ben, harm]), {"rule": "safe examples = top of Llama Guard 3's within-safe ranking per stratum; harmful = top third", "safety_demo_share_safe": 0.37})
    # safelen8: score-blind, length-matched to certlg_s{s}'s safe examples (decile-stratified)
    cert = np.array(json.load(open(f"{D}/certlg_s{s}.json"))["idx"]); cert_safe_len = rl[cert[~unsafe[cert]]]
    edges = np.quantile(cert_safe_len, np.linspace(0, 1, 11)); edges[0], edges[-1] = -1, 1e9
    rng2 = np.random.default_rng(9400 + s); picked = []
    for stratum, n_want in ((demo, k), (benign, n_safe - k)):
        per = np.histogram(cert_safe_len, bins=edges)[0] / len(cert_safe_len) * n_want; per = np.round(per).astype(int); per[-1] += n_want - per.sum()
        for b in range(10):
            cand = stratum[(rl[stratum] > edges[b]) & (rl[stratum] <= edges[b + 1])]
            picked.extend(rng2.choice(cand, min(per[b], len(cand)), replace=False))
    picked = np.array(picked); short = n_safe - len(picked)
    if short > 0: picked = np.r_[picked, rng2.choice(np.setdiff1d(np.r_[demo, benign], picked), short, replace=False)]
    save8("safelen8", s, rng2.permutation(np.r_[picked, harm]), {"rule": "safe examples score-blind, length-matched to certlg's safe examples by decile per stratum; harmful = top third", "safety_demo_share_safe": 0.37})
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1)
# ---- E26 (0.5B pool)
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool]); N = 4000
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"])
sel = json.load(open(f"{W}/results/step2_selections.json"))
def save05(arm, s, idx, meta):
    idx = sorted(int(i) for i in set(idx)); meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float((req[idx] & ~unsafe[idx]).mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/{arm}_s{s}.json", "w")); sel.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:10s} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f}")
for s in range(3):
    cal = np.random.default_rng(1000 + s).choice(N, 800, replace=False)
    save05("labels800", s, [i for i in cal if not unsafe[i]], {"rule": "human-safe among the 800 calibration draws of certstrat2_lg800"})
    r14 = json.load(open(f"{W}/selections/random1400_s{s}.json"))["idx"]
    save05("randdemo", s, list(r14) + [int(i) for i in cal if not unsafe[i] and req[i]], {"rule": "random1400 plus the safety demonstrations among the 800 labels"})
json.dump(sel, open(f"{W}/results/step2_selections.json", "w"), indent=1); print("E25E26 SELECT DONE")
