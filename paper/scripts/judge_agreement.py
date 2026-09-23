"""Appendix: per-generation agreement of the four judges (pairwise agreement and Cohen's kappa
over every 0.5B generation judged by all four) and arm-level agreement (Spearman rank correlation of
per-model harm rates). Writes results/judge_agreement.json and paper/data/tables/judges.tex."""
import os
import json, itertools, numpy as np
from scipy.stats import spearmanr
W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
bd = json.load(open(f"{W}/results/gen_scores/beaverdam.json")); md = json.load(open(f"{W}/results/gen_scores/mdjudge.json"))
wg = json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg = json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
def kappa(a, b):
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()); return (po - pe) / (1 - pe)


def analyse(J, models):
    F = {k: np.concatenate([f(m) for m in models]) for k, f in J.items()}; n = len(F["beaver-dam"])
    rates = {k: np.array([f(m).mean() for m in models]) for k, f in J.items()}
    out = {"n_generations": n, "n_models": len(models), "flag_rate": {k: float(v.mean()) for k, v in F.items()},
           "all_four_agree": float(np.mean([len(set(F[k][i] for k in J)) == 1 for i in range(n)])), "pairs": {}}
    for a, b in itertools.combinations(J, 2):
        out["pairs"][f"{a} | {b}"] = {"agreement": float((F[a] == F[b]).mean()), "kappa": float(kappa(F[a], F[b])), "arm_spearman": float(spearmanr(rates[a], rates[b]).statistic)}
    return out


models = [m for m in wg if m in bd and m in md and m in lg]
J05 = {"beaver-dam": lambda m: np.array(bd[m]) > 0.5, "MD-Judge": lambda m: np.array(md[m]["flag"], bool),
       "WildGuard": lambda m: np.array(wg[m]["harmful_response"], bool), "Llama Guard 3": lambda m: np.array(lg[m]["flag"], bool)}
res = {"0.5B": analyse(J05, models)}
# 8B: every lambda's score file
S = {l: json.load(open(f"{W}/results/e3_scores_lambda{l}.json")) for l in ("0.10", "0.25", "0.30")}
m8 = [(l, m) for l in S for m in S[l]["beaverdam"] if m != "base_s0" and m in S[l]["mdjudge"] and m in S[l]["wildguard"] and m in S[l]["llamaguard"]]
J8 = {"beaver-dam": lambda lm: np.array(S[lm[0]]["beaverdam"][lm[1]]["score"]) > 0.5, "MD-Judge": lambda lm: np.array(S[lm[0]]["mdjudge"][lm[1]]["flag"], bool),
      "WildGuard": lambda lm: np.array(S[lm[0]]["wildguard"][lm[1]]["harmful_response"], bool), "Llama Guard 3": lambda lm: np.array(S[lm[0]]["llamaguard"][lm[1]]["flag"], bool)}
res["8B"] = analyse(J8, m8)
json.dump(res, open(f"{W}/results/judge_agreement.json", "w"), indent=1)
rows = []
for a, b in itertools.combinations(J05, 2):
    p5, p8 = res["0.5B"]["pairs"][f"{a} | {b}"], res["8B"]["pairs"][f"{a} | {b}"]
    rows.append(f"{a} & {b} & {p5['agreement']:.3f} & {p5['kappa']:.2f} & {p5['arm_spearman']:.3f} & {p8['agreement']:.3f} & {p8['kappa']:.2f} & {p8['arm_spearman']:.3f} \\\\")
body = ("\\begin{tabular}{llcccccc}\n\\toprule\n & & \\multicolumn{3}{c}{Qwen2.5-0.5B (" + f"{res['0.5B']['n_models']}" + " models)} & \\multicolumn{3}{c}{Llama-3.1-8B-Instruct (" + f"{res['8B']['n_models']}" + " models)} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
        "judge & judge & agreement & $\\kappa$ & condition Spearman & agreement & $\\kappa$ & condition Spearman \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(f"{W}/paper/data/tables/judges.tex", "w").write(body)
for k, v in res.items(): print(k, v["n_generations"], v["n_models"], round(v["all_four_agree"], 3), {a: round(b, 3) for a, b in v["flag_rate"].items()})
