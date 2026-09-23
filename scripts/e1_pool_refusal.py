"""Learned refusal labels on the POOL (WildGuard), so the mechanism statistic, refusal share
among kept safe examples per arm, and the judge's style bias among human-safe examples,
are both computed with a learned detector instead of the regex. Compares to the regex."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
R = re.compile(r"^\s*(I'?m sorry|I cannot|I can'?t|I am not able|I'm not able|No[,.]|It is not|I do not|I don'?t think|Sorry|I would not|I will not|That is not|It's not appropriate|I am unable|Unfortunately)", re.I)
pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
path = f"{W}/results/scores_wildguard_pool.json"
if os.path.exists(path):
    d = json.load(open(path))
else:
    req, ref, harm, raw = wildguard_labels(pool)
    d = {"harmful_request": req.tolist(), "refusal": ref.tolist(), "harmful_response": harm.tolist()}
    json.dump(d, open(path, "w"))
ref = np.array(d["refusal"]); harm = np.array(d["harmful_response"]); rx = np.array([bool(R.match(r["response"])) for r in pool])
from sklearn.metrics import roc_auc_score
print(f"WildGuard on the pool: harmful-response flag vs human label: flags {harm[~unsafe].mean():.3f} of safe, {harm[unsafe].mean():.3f} of unsafe "
      f"(accuracy {np.mean(harm == unsafe):.3f})")
print(f"refusal: WildGuard {ref.mean():.3f} of pool, regex {rx.mean():.3f}; agreement {np.mean(ref == rx):.3f}; among human-safe: WG {ref[~unsafe].mean():.3f}, regex {rx[~unsafe].mean():.3f}")
g7 = np.array(json.load(open(f"{W}/results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json"))); bd = np.array(json.load(open(f"{W}/results/scores_beaverdam_pool.json")))
sf = ~unsafe
print(f"style bias among human-SAFE, by WildGuard refusal: Qwen-7B judge score refusal {g7[sf & ref].mean():.2f} vs engaged {g7[sf & ~ref].mean():.2f}; "
      f"beaver-dam harm prob refusal {bd[sf & ref].mean():.3f} vs engaged {bd[sf & ~ref].mean():.3f}")
sel = json.load(open(f"{W}/results/step2_selections.json"))
print(f"\n{'arm':12s} {'n':>5s} {'unsafe':>7s} {'refusal|safe WG':>16s} {'refusal|safe regex':>19s}")
for a in ["full", "heur", "cert200", "cert400", "certbd400", "labels", "oracle"] + sorted(k for k in sel if k.startswith("refshare")):
    vals = []
    for s in sel[a]:
        idx = np.array(json.load(open(f"{W}/selections/{a}_s{s}.json"))["idx"]); sfi = idx[~unsafe[idx]]
        vals.append((len(idx), unsafe[idx].mean(), ref[sfi].mean(), rx[sfi].mean()))
    v = np.mean(vals, axis=0); print(f"{a:12s} {v[0]:5.0f} {v[1]:7.3f} {v[2]:16.3f} {v[3]:19.3f}")
