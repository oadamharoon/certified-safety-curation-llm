"""E11: SEAL arms from the trained selector (results/seal/ours_lambda025_softmax.pt, one logit
per pool example in pool.json order; interleave_datasets with a single source preserves file
order and SFTDatasetIndexed returns the dataset position, so index i is pool row i).
  seal_s{s}      SEAL's own rule (train_sft.py): keep the top int(0.8 N) examples by selector
                 logit (their default --topp 0.8); the seed only varies fine-tuning.
  certseal_s{s}  the selector logit as a safety score under our certificate, alpha .10, n 400,
                 same calibration draws as certlg/certsaft.
Diagnostics: AUROC of the logit against the human label, safety-demo share of the kept set.
CPU only; Runs under PY. Usage: LAMBDA=0.25 python seal_select.py
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import W, ltt_walk, cp_fallback
from sklearn.metrics import roc_auc_score
LAMS = os.environ.get("LAMBDA", "0.25"); D = f"{W}/selections_e3/lambda{LAMS}"
pool = json.load(open(f"{D}/pool.json")); N = len(pool)
unsafe = np.array([not r["is_safe"] for r in pool]); S = json.load(open(f"{D}/scores.json"))
req = np.array(S["harmful_request"]); demo = req & ~unsafe
SEL = os.environ.get("SELECTOR", "ours_lambda025_softmax.pt"); SUF = os.environ.get("ARM_SUFFIX", "")   # budget-matched run: SELECTOR=ours_lambda025_long_softmax.pt ARM_SUFFIX=_long
g = torch.load(f"{W}/results/seal/{SEL}", map_location="cpu", weights_only=False).detach().float().numpy()
assert g.shape == (N,), g.shape
print(f"SEAL logits: AUROC(safe) {roc_auc_score(~unsafe, g):.3f}; mean logit safe {g[~unsafe].mean():.3f} unsafe {g[unsafe].mean():.3f} demos {g[demo].mean():.3f}", flush=True)
json.dump({"auc_safe": float(roc_auc_score(~unsafe, g)), "logits": g.tolist()}, open(f"{D}/seal_scores{SUF}.json", "w"))
summ = json.load(open(f"{D}/summary.json"))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:9s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} {meta.get('certified','')}", flush=True)
k = int(0.8 * N); top = np.argsort(-g)[:k]
for s in range(3):
    save("seal" + SUF, s, top, {"rule": "SEAL top-p 0.8 by selector logit"})
    cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False)
    ok, mask, q, _ = ltt_walk(g, unsafe, cal, 0.10, 0.10)
    if ok: save("certseal" + SUF, s, np.where(mask)[0], {"certified": True, "q": q})
    else: fm, lb = cp_fallback(g, unsafe, cal, 0.10); save("certseal" + SUF, s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb})
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1); print("SEAL SELECT DONE")
