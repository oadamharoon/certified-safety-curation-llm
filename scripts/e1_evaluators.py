"""E1: evaluator panel on every generation. Adds MD-Judge (and Llama Guard 3 / WildGuard
when the HF token is present) to the cached per-model scores, then prints the arm table
with every evaluator side by side. Each evaluator's training overlap with BeaverTails is
stated in RELATED_WORK.md section E/H and must accompany any use of its numbers.
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *

names = sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(f"{W}/results/models/*/config.json"))
gens = {n: json.load(open(f"{W}/results/gen/{n}.json")) for n in names}


def cached(key, fn, extra_flag=False):
    path = f"{W}/results/gen_scores/{key}.json"
    d = json.load(open(path)) if os.path.exists(path) else {}
    todo = [n for n in names if n not in d]
    if todo:
        res = fn([g for n in todo for g in gens[n]])
        sc, fl = (res if extra_flag else (res, None))
        for i, n in enumerate(todo):
            d[n] = {"score": sc[i*500:(i+1)*500].tolist()} | ({"flag": fl[i*500:(i+1)*500].tolist()} if fl is not None else {})
        json.dump(d, open(path, "w"))
    return d


mdj = cached("mdjudge", md_judge_scores, extra_flag=True)
ev2 = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]
sel = json.load(open(f"{W}/results/step2_selections.json"))
arms = ["full", "heur", "cert200", "cert400", "certbd400", "labels", "oracle"]
print(f"{'arm':10s} {'train unsafe':>12s} {'beaver-dam':>11s} {'MD-Judge':>11s} {'Qwen-1.5B':>11s}   (fraction of 500 generations flagged, mean±SE over seeds)")
for a in arms:
    ms = [n for n in names if n.startswith(a + "_")]
    u = np.mean([sel[a][s]["unsafe_frac"] for s in sel[a]])
    def cell(vals): return f"{np.mean(vals):.3f}±{np.std(vals, ddof=1)/np.sqrt(len(vals)):.3f}"
    bd = [ev2[n]["beaverdam_flag"] for n in ms]; q = [ev2[n]["qwen1.5b_flag"] for n in ms]
    md = [np.mean(mdj[n]["flag"]) for n in ms]
    print(f"{a:10s} {u:12.3f} {cell(bd):>11s} {cell(md):>11s} {cell(q):>11s}")
# evaluator agreement on the pooled generations
allbd = np.concatenate([np.array(json.load(open(f"{W}/results/gen_scores/beaverdam.json"))[n]) > 0.5 for n in names])
allmd = np.concatenate([np.array(mdj[n]["flag"]) for n in names])
print(f"\nbeaver-dam vs MD-Judge flag agreement on {len(allbd)} generations: {np.mean(allbd == allmd):.3f}; "
      f"beaver-dam flags {allbd.mean():.3f}, MD-Judge flags {allmd.mean():.3f}")
