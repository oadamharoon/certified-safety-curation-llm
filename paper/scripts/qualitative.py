"""Qualitative example box (appendix), the convention of SAFT Table 5 / Qi et al. Fig. 4 / Bianchi et al.
Fig. 2. Selection rule, fixed so the box is reproducible and not cherry-picked by hand: at each scale,
the FIRST held-out prompt (in evaluation order) whose prompt WildGuard labels harmful, for which the
seed-0 no-filter model's response is flagged by beaver-dam and MD-Judge, the seed-0 stratified
certified model's response is flagged by neither, and every printed response is at most 320
characters and contains no sexual content. Writes paper/data/tables/qualitative_05b.tex and
qualitative_8b.tex; the chosen indices go to results/qualitative_examples.json."""
import os
import json, os, re
import numpy as np
W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); T = f"{W}/paper/data/tables"
SKIP = ("sex", "cock", "penis", "rape", "porn")


def esc(t):
    t = t.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$").replace("#", "\\#").replace("_", "\\_").replace("{", "\\{").replace("}", "\\}").replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    return re.sub(r"\s+", " ", t).strip()


def box(name, prompt, rows, caption_note):
    lines = ["\\begin{tabular}{p{0.15\\linewidth}p{0.565\\linewidth}cc}", "\\toprule",
             f"\\textbf{{prompt}} & {esc(prompt)} & beaver-dam & MD-Judge \\\\", "\\midrule"]
    for arm, resp, (fb, fm) in rows:
        mk = lambda f: "$\\times$" if f else "$\\checkmark$"
        lines.append(f"{arm} & {esc(resp)} & {mk(fb)} & {mk(fm)} \\\\[5pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{T}/{name}.tex", "w").write("\n".join(lines) + "\n")


out = {}
# ---- 0.5B
ev = json.load(open(f"{W}/results/eval_sets.json"))["gen_prompts"]
ARMS05 = (("full", "no filter"), ("heur", "fixed cutoff, judge"), ("anti_qwen", "judge's rejects"), ("cert400", "certified, judge"), ("random1400", "random, size of certified"), ("lgfilter", "fixed cutoff, Llama Guard 3"), ("certlg400", "certified, Llama Guard 3"), ("certstrat2_lg800", "stratified certified"), ("certstrat2_qwen800", "stratified certified, judge"), ("saft05", "SAFT"), ("certsaft05", "certified, SAFT score"), ("certstrat_saft05", "stratified certified, SAFT score"), ("labels", "labeled only"), ("oracle", "oracle"))
gens = {a: json.load(open(f"{W}/results/gen/{a}_s0.json")) for a, _ in ARMS05}
bd = json.load(open(f"{W}/results/gen_scores/beaverdam.json")); md = json.load(open(f"{W}/results/gen_scores/mdjudge.json"))
wgp = json.load(open(f"{W}/results/gen_scores/wildguard.json"))
def flagged(model, i): return bd[model][i] > 0.5, bool(md[model]["flag"][i])
pick = None
for i, p in enumerate(ev):
    if not wgp["full_s0"]["harmful_request"][i]: continue
    f_full = flagged("full_s0", i); f_strat = flagged("certstrat2_lg800_s0", i)
    if all(f_full) and not any(f_strat) and all(len(gens[a][i]["response"]) <= 320 for a in gens) and not any(k in (p + "".join(gens[a][i]["response"] for a in gens)).lower() for k in SKIP):
        pick = i; break
rows = []
for a, lab in ARMS05:
    rows.append((lab, gens[a][pick]["response"], flagged(f"{a}_s0", pick)))
box("qualitative_05b", ev[pick], rows, ""); out["0.5B"] = {"index": pick, "prompt": ev[pick]}
# ---- 8B
ARMS8 = (("full", "no filter"), ("prompting", "Prompting filter"), ("lgfilter", "fixed cutoff, Llama Guard 3"), ("labels", "labeled only"), ("saft", "SAFT"), ("seal", "SEAL"), ("anti", "judge's rejects"), ("random", "random"), ("certprompt", "certified, Prompting judge"), ("certstrat_prompt", "stratified certified, Prompting judge"), ("certlg", "certified, Llama Guard 3"), ("certsaft", "certified, SAFT score"), ("certseal", "certified, SEAL score"), ("certstrat_saft", "stratified certified, SAFT score"), ("certstrat_seal", "stratified certified, SEAL score"), ("certstrat2", "stratified certified"), ("oracle", "oracle"))
ARMS8 = tuple(x for x in ARMS8 if os.path.exists(f"{W}/results/gen_e3/lambda0.25/{x[0]}_s0.json"))   # arms still in the queue (E16) appear once run
G = {a: json.load(open(f"{W}/results/gen_e3/lambda0.25/{a}_s0.json"))["held"] for a, _ in ARMS8}
S = json.load(open(f"{W}/results/e3_scores_lambda0.25.json"))
def flagged8(model, i): return S["beaverdam"][model][i] > 0.5, bool(S["mdjudge"][model][i]["flag"][i] if isinstance(S["mdjudge"][model], dict) else S["mdjudge"][model][i])
def flagged8(model, i): return S["beaverdam"][model]["score"][i] > 0.5, bool(S["mdjudge"][model]["flag"][i])
pick8 = None
for i, p in enumerate(ev):
    if not S["wildguard"]["full_s0"]["harmful_request"][i]: continue
    if all(flagged8("full_s0", i)) and not any(flagged8("certstrat2_s0", i)) and all(len(G[a][i]["response"]) <= 320 for a in G) and not any(k in (p + "".join(G[a][i]["response"] for a in G)).lower() for k in SKIP):
        pick8 = i; break
rows = []
for a, lab in ARMS8:
    rows.append((lab, G[a][pick8]["response"], flagged8(f"{a}_s0", pick8)))
box("qualitative_8b", ev[pick8], rows, ""); out["8B"] = {"index": pick8, "prompt": ev[pick8]}
json.dump(out, open(f"{W}/results/qualitative_examples.json", "w"), indent=1); print(out)
