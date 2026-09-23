"""E26b (8B) and E28 (0.5B aligned) selections. Pre-registered in README 2026-09-21 17:00.
E26b labels800_8_s{s}: human-safe examples among the 800-label calibration draw of the 8B stratified
certificate (rng 1000+s, as e3_select.py); randdemo8_s{s}: the size-matched random set (random_s{s})
plus every safety demonstration among those 800 labels. E28 reuses the 0.5B selection files with a
new base model (Qwen2.5-0.5B-Instruct), so it needs no selections; listed here for the record."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import W
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); S = json.load(open(f"{D}/scores.json"))
unsafe = np.array([not r["is_safe"] for r in pool]); req = np.array(S["harmful_request"]); N = len(pool); summ = json.load(open(f"{D}/summary.json"))
def save(arm, s, idx, meta):
    idx = sorted(int(i) for i in set(idx)); meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float((req[idx] & ~unsafe[idx]).mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:12s} s{s}: n={len(idx)} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f}")
for s in range(3):
    cal = np.random.default_rng(1000 + s).choice(N, 800, replace=False)
    save("labels800_8", s, [i for i in cal if not unsafe[i]], {"rule": "human-safe among the 800 calibration draws of certstrat2"})
    r = json.load(open(f"{D}/random_s{s}.json"))["idx"]
    save("randdemo8", s, list(r) + [int(i) for i in cal if not unsafe[i] and req[i]], {"rule": "random (certlg size) plus the safety demonstrations among the 800 labels"})
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1); print("E26B SELECT DONE")
