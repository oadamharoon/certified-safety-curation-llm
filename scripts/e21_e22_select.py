"""E21 + E22 selections at 8B (pre-registered 2026-09-19 00:45, README).
E21 oracle_matched_s{s}: uniform random subset of the 2250 human-safe examples, of the size of
certlg_s{s} (2176 / 464 / 2112), no scorer: the size-matched oracle the certified set's margin needs.
E22 harmtop8 / harmmid8 / harmbot8 _s{s}: size 1584, 132 harmful, demonstration share .37 among the
1452 safe examples (E20's certified-set level, stratified uniform by WildGuard's prompt label), and the
132 harmful examples drawn uniformly from the top / middle / bottom third of the 750 harmful examples
ranked by Llama Guard 3's safe score (top = rated safest, what a certified selection keeps).
Selections in selections_e3/lambda0.25 with summary.json entries, as every E3 condition."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import W
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); S = json.load(open(f"{D}/scores.json"))
unsafe = np.array([not r["is_safe"] for r in pool]); req = np.array(S["harmful_request"]); g = np.array(S["lg_safe_score"])
summ = json.load(open(f"{D}/summary.json"))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float((req[idx] & ~unsafe[idx]).mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:15s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f}", flush=True)
safe = np.where(~unsafe)[0]; demo = np.where(~unsafe & req)[0]; benign = np.where(~unsafe & ~req)[0]; uns = np.where(unsafe)[0]
order = uns[np.argsort(-g[uns])]   # harmful examples, safest-rated first
third = len(order) // 3; bands = {"harmtop8": order[:third], "harmmid8": order[third:2 * third], "harmbot8": order[2 * third:]}
for s in range(3):
    rng = np.random.default_rng(9100 + s); nsel = summ["certlg"][str(s)]["n"]
    save("oracle_matched", s, rng.choice(safe, nsel, replace=False), {"rule": "uniform among human-safe examples at certlg's size, no scorer"})
    n_safe = 1584 - 132; k = int(round(0.37 * n_safe))
    for arm, band in bands.items():
        rng = np.random.default_rng(9200 + s)   # same safe part across the three bands, per seed
        safe_part = np.r_[rng.choice(demo, k, replace=False), rng.choice(benign, n_safe - k, replace=False)]
        rng2 = np.random.default_rng(9300 + s + {"harmtop8": 0, "harmmid8": 10, "harmbot8": 20}[arm])
        idx = rng2.permutation(np.r_[safe_part, rng2.choice(band, 132, replace=False)])
        save(arm, s, idx, {"rule": f"stratified uniform safe part (demo share .37), harmful examples from the {arm[4:7]} third of the harmful stratum by Llama Guard 3 score", "safety_demo_share_safe": 0.37, "n_harmful": 132,
                           "harmful_lg_score_mean": float(g[idx][unsafe[idx]].mean())})
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1); print("E21E22 SELECT DONE")
