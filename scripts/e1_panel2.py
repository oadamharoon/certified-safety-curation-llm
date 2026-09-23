"""E1 continued: Llama Guard 3 and WildGuard on every generation (both free of BeaverTails
training data), cached per model in results/gen_scores/{llamaguard,wildguard}.json, then
the full evaluator panel table plus WildGuard's learned refusal rate per arm.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

names = sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(f"{W}/results/models/*/config.json"))
gens = {n: json.load(open(f"{W}/results/gen/{n}.json")) for n in names}


def cached(key, fn, fields):
    path = f"{W}/results/gen_scores/{key}.json"
    d = json.load(open(path)) if os.path.exists(path) else {}
    todo = [n for n in names if n not in d]
    if todo:
        res = fn([g for n in todo for g in gens[n]])
        for i, n in enumerate(todo):
            d[n] = {f: np.asarray(r[i*500:(i+1)*500]).tolist() for f, r in zip(fields, res)}
        json.dump(d, open(path, "w"))
    return d


lg = cached("llamaguard", llama_guard_scores, ["score", "flag"])
wg = cached("wildguard", wildguard_labels, ["harmful_request", "refusal", "harmful_response", "raw"])
ev2 = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; mdj = json.load(open(f"{W}/results/gen_scores/mdjudge.json"))
sel = json.load(open(f"{W}/results/step2_selections.json"))
arms = [a for a in ["full", "heur", "cert200", "cert400", "certbd400", "labels", "oracle"] + sorted(k for k in sel if k.startswith("refshare"))]
def cell(v): return f"{np.mean(v):.3f}±{np.std(v, ddof=1)/np.sqrt(len(v)):.3f}"
print(f"{'arm':14s} {'unsafe':>6s} {'beaver-dam':>12s} {'MD-Judge':>12s} {'LlamaGuard3':>12s} {'WildGuard':>12s} {'WG refusal':>12s}")
for a in arms:
    ms = [n for n in names if n.startswith(a + "_")]
    if not ms: continue
    u = np.mean([sel[a][s]["unsafe_frac"] for s in sel[a]])
    print(f"{a:14s} {u:6.3f} {cell([ev2[n]['beaverdam_flag'] for n in ms]):>12s} {cell([np.mean(mdj[n]['flag']) for n in ms]):>12s} "
          f"{cell([np.mean(lg[n]['flag']) for n in ms]):>12s} {cell([np.mean(wg[n]['harmful_response']) for n in ms]):>12s} {cell([np.mean(wg[n]['refusal']) for n in ms]):>12s}")
B = np.concatenate([np.array(json.load(open(f"{W}/results/gen_scores/beaverdam.json"))[n]) > 0.5 for n in names])
M = np.concatenate([np.array(mdj[n]["flag"]) for n in names]); L = np.concatenate([np.array(lg[n]["flag"]) for n in names]); Wg = np.concatenate([np.array(wg[n]["harmful_response"]) for n in names])
print("\npairwise flag agreement:", {k: round(float(np.mean(x == y)), 3) for k, (x, y) in
      {"bd-md": (B, M), "bd-lg": (B, L), "bd-wg": (B, Wg), "md-lg": (M, L), "md-wg": (M, Wg), "lg-wg": (L, Wg)}.items()})
print("flag rates:", {k: round(float(v.mean()), 3) for k, v in {"beaver-dam": B, "MD-Judge": M, "LlamaGuard3": L, "WildGuard": Wg}.items()})
