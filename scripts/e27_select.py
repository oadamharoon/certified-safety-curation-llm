"""E27 (pre-registered 2026-09-21 13:40, README): the within-stratum test of E22/E25 at 0.5B, for
symmetry with 8B. Fixed size 1400 and 92 harmful (the judge-certified set's first-seed composition, as
E4b), demonstration share .25 among the 1308 safe examples (the certified set's level), and:
judgetop05 = harmful examples from the top third of the 7B judge's harmful ranking (rated safest) AND
safe examples from the top of the judge's within-safe ranking per stratum; judgeharm05 = the same
harmful examples with the safe examples drawn uniformly per stratum (the E4b q=.25 rule), isolating
the harmful-example step. Three seeds. Selections judgetop05/judgeharm05_s{s}."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
req = np.array(json.load(open(f"{W}/results/scores_wildguard_pool.json"))["harmful_request"])
g = np.array(json.load(open(f"{W}/results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json")))
demo = np.where(~unsafe & req)[0]; benign = np.where(~unsafe & ~req)[0]; uns = np.where(unsafe)[0]
order = uns[np.argsort(-g[uns])]; top_third = order[:len(order) // 3]
summ = json.load(open(f"{W}/results/step2_selections.json")); n_safe = 1400 - 92; k = int(round(0.25 * n_safe))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), n_harmful=int(unsafe[idx].sum()), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float((req[idx] & ~unsafe[idx]).mean()),
                                            safe_judge_mean=float(g[idx][~unsafe[idx]].mean()), harm_judge_mean=float(g[idx][unsafe[idx]].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:11s} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} safeJ={meta['safe_judge_mean']:.1f} harmJ={meta['harm_judge_mean']:.1f}")
for s in range(3):
    rng = np.random.default_rng(6500 + s); harm = rng.choice(top_third, 92, replace=False)
    top_demo = demo[np.argsort(-g[demo])][:k]; top_ben = benign[np.argsort(-g[benign])][:n_safe - k]
    save("judgetop05", s, rng.permutation(np.r_[top_demo, top_ben, harm]), {"rule": "harmful from the judge's top third; safe from the top of the judge's within-safe ranking per stratum; share .25", "safety_demo_share_safe": 0.25})
    rng2 = np.random.default_rng(6600 + s); safe_part = np.r_[rng2.choice(demo, k, replace=False), rng2.choice(benign, n_safe - k, replace=False)]
    save("judgeharm05", s, rng2.permutation(np.r_[safe_part, harm]), {"rule": "harmful from the judge's top third; safe uniform per stratum; share .25", "safety_demo_share_safe": 0.25})
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1); print("E27 SELECT DONE")
