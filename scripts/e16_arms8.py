"""E16: the two Table-2 baselines missing at 8B (pre-registered 2026-09-16 17:00, README):
lgfilter = keep every example Llama Guard 3 does not flag, no labels (the field's "Guard"
baseline, DataShield); labels = the human-safe examples among the certificate's own 400
calibration draws of seed s (what the 8B certificate's labels buy without a scorer; the 0.5B
arm uses its 200 draws). Selections into selections_e3/lambda0.25, then the E3 SFT/eval chain.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import W
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); N = len(pool)
unsafe = np.array([not r["is_safe"] for r in pool]); S = json.load(open(f"{D}/scores.json")); req = np.array(S["harmful_request"]); demo = req & ~unsafe
flag = np.array(S["lg_flag"], bool); summ = json.load(open(f"{D}/summary.json"))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:11s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f}", flush=True)
for s in range(3):
    save("lgfilter", s, np.where(~flag)[0], {"rule": "keep all not flagged by Llama Guard 3, no labels", "scorer": "Llama-Guard-3-8B"})
    cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False)   # the certlg calibration draw of seed s (e3_select.py)
    save("labels", s, [i for i in cal if not unsafe[i]], {"rule": "labeled-safe among the 400 calibration draws"})
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1); print("E16 SELECT DONE")
