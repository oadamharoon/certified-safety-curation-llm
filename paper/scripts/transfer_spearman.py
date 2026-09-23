"""Rank correlation between each condition's harm on the BeaverTails held-out prompts and its harm
on the instruction-style benchmarks (DirectHarm4, HarmBench, HEx-PHI, and the three pooled), per
judge, at both scales, over exactly the conditions of Tables 15 and 16. Only the three judges that
scored no condition enter the reported ranges; Llama Guard 3's own column is reported separately
and is self-scored on the conditions it selected. Writes results/harmsets_transfer.json.
"""
import os
import json, re
import numpy as np
from scipy.stats import spearmanr

W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = json.load(open(f"{W}/results/harmsets.json"))
SETS = ("DirectHarm4", "HarmBench", "HEx-PHI", "all")
JUDGES = ("beaver-dam", "MD-Judge", "WildGuard", "Llama Guard 3")
CLEAN = ("beaver-dam", "MD-Judge", "WildGuard")   # judges that scored no training set


def _seeds(m, key):
    return re.match(re.escape(key) + r"_s\d+$", m) is not None


def beavertails_05b():
    ev = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]
    md = json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg = json.load(open(f"{W}/results/gen_scores/wildguard.json"))
    lg = json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
    return {"beaver-dam": lambda m: ev[m]["beaverdam_flag"], "MD-Judge": lambda m: float(np.mean(md[m]["flag"])),
            "WildGuard": lambda m: float(np.mean(wg[m]["harmful_response"])), "Llama Guard 3": lambda m: float(np.mean(lg[m]["flag"]))}, list(lg)


def beavertails_8b():
    e3 = json.load(open(f"{W}/results/e3_eval_lambda0.25.json")); K = {"beaver-dam": "beaverdam", "MD-Judge": "mdjudge", "WildGuard": "wildguard", "Llama Guard 3": "llamaguard"}
    return {j: (lambda k: (lambda m: e3[m][k]))(K[j]) for j in JUDGES}, list(e3)


ARMS = {"05b": ["full", "random1400", "heur", "anti_qwen", "cert400", "lgfilter", "certlg400", "certstrat2_lg800",
                "certstrat2_qwen800", "saft05", "certsaft05", "certstrat_saft05", "labels", "oracle"],
        "8b": ["full", "prompting", "lgfilter", "saft", "saft15", "seal", "seal_long", "anti", "random", "certprompt",
               "certsaft", "certseal", "certseal_long", "certlg", "certstrat_prompt", "certstrat_saft", "certstrat_seal",
               "certstrat2", "certstrat2", "labels", "oracle"]}
out = {}
for sub, (J, models) in (("05b", beavertails_05b()), ("8b", beavertails_8b())):
    per = {}
    for j in JUDGES:
        for st in SETS:
            xs, ys = [], []
            for a in dict.fromkeys(ARMS[sub]):
                ms = [m for m in models if _seeds(m, a)]
                hs = [x for x in H if x.startswith(sub + "/") and _seeds(x.split("/")[1], a)]
                if not ms or not hs: continue
                xs.append(float(np.mean([J[j](m) for m in ms]))); ys.append(float(np.mean([H[x][j][st] for x in hs])))
            per[f"{j}|{st}"] = {"n": len(xs), "spearman": float(spearmanr(xs, ys).correlation)}
    clean = [v["spearman"] for k, v in per.items() if k.split("|")[0] in CLEAN]
    pooled = [v["spearman"] for k, v in per.items() if k.split("|")[0] in CLEAN and k.endswith("|all")]
    out[sub] = {"per_judge_set": per,
                "clean_judges_min": min(clean), "clean_judges_max": max(clean),
                "clean_judges_below_0.9": sum(1 for v in clean if v < 0.9), "clean_judges_pairs": len(clean),
                "pooled_min": min(pooled), "pooled_max": max(pooled)}
json.dump(out, open(f"{W}/results/harmsets_transfer.json", "w"), indent=1)
for sub, r in out.items():
    print(f"{sub}: per (judge, set) over the three clean judges {r['clean_judges_min']:.3f}-{r['clean_judges_max']:.3f} "
          f"({r['clean_judges_below_0.9']}/{r['clean_judges_pairs']} below 0.9); pooled {r['pooled_min']:.3f}-{r['pooled_max']:.3f}")
