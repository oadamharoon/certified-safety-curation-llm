"""Regenerate paper/data/tables/*.tex from certified-data-curation-llm/results/*.json.

Every number in the paper's tables comes from here; nothing is typed by hand.
"""
import os
import json, os
import numpy as np
import re


def _seeds(m, key):
    """m is a seed model of arm key: <key>_s<digits>. A prefix test would let seal_ match seal_long_."""
    return re.match(re.escape(key) + r"_s\d+$", m) is not None


W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
T = f"{W}/paper/data/tables"
os.makedirs(T, exist_ok=True)


def gate_table():
    """Step 1: certification gate grid (pool contamination x judge)."""
    rows = []
    for judge, jl in (("Qwen2.5-1.5B-Instruct", "1.5B"), ("Qwen2.5-7B-Instruct", "7B"), ("LlamaGuard3", "Llama Guard 3")):
        for tag, label in (("nat", "0.57"), ("u0.25", "0.25"), ("u0.10", "0.10")):
            d = json.load(open(f"{W}/results/llm_certify_{judge}_{tag}.json"))
            a25, a10 = d["alpha0.25"], d["alpha0.1"]
            rows.append(f"{label} & {jl} & {d['auc']:.2f} & "
                        f"{a25['cert_rate']:.2f} & {a25['false_cert']:.3f} & {a10['cert_rate']:.2f} & {a10['false_cert']:.3f} \\\\")
    body = ("\\begin{tabular}{llccccc}\n\\toprule\n"
            "pool unsafe & scorer & AUC & \\multicolumn{2}{c}{$\\alpha=0.25$} & \\multicolumn{2}{c}{$\\alpha=0.10$} \\\\\n"
            "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & & & cert.\\ rate & false cert.\\ & cert.\\ rate & false cert.\\ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/gate.tex", "w").write(body)


def _judges():
    ev = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]
    md = json.load(open(f"{W}/results/gen_scores/mdjudge.json")); wg = json.load(open(f"{W}/results/gen_scores/wildguard.json"))
    lg = json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
    return {"beaver-dam": lambda m: ev[m]["beaverdam_flag"], "MD-Judge": lambda m: float(np.mean(md[m]["flag"])),
            "WildGuard": lambda m: float(np.mean(wg[m]["harmful_response"])), "Llama Guard 3": lambda m: float(np.mean(lg[m]["flag"]))}


def _cell(vals):
    return f"{np.mean(vals):.3f}" if len(vals) == 1 else f"{np.mean(vals):.3f} $\\pm$ {np.std(vals, ddof=1) / np.sqrt(len(vals)):.3f}"


LG_SCORED = {"lgfilter", "certlg400", "certstrat2_lg800", "certstrat_lg", "certstrat_lg800", "certlg", "certstrat2",
             "p2_certlg400", "p2_certstrat2",
             "harmtop8", "harmmid8", "harmbot8", "safetop8", "safelen8"}   # conditions whose training set Llama Guard 3 scored
# The judges that score training sets and are also evaluation judges are Llama Guard 3 alone: the Qwen
# judges, SAFT's spectral score and SEAL's selector never evaluate, so no other column can be self-scored.


def _bold_min(rows, keys, cols, exclude=("oracle", "base"), self_scored=None):
    """rows: list of (key, [cell strings for cols], [mean values for cols]); returns cells with the
    per-column minimum among rows whose key is not excluded wrapped in \\textbf. One rule for every
    harm table: bold marks the safest condition per judge, the oracle (full labels) and the untuned base
    excluded because neither is a curation method. self_scored: {row key: column index} of cells where
    the evaluating judge is the row's own scorer; those cells are printed gray and never bolded."""
    out = {}; self_scored = self_scored or {}
    for j in range(len(cols)):
        cand = [(r[2][j], r[0]) for r in rows if r[0] not in exclude and r[2][j] is not None and self_scored.get(r[0]) != j]
        best = min(cand)[1] if cand else None
        for r in rows:
            out.setdefault(r[0], list(r[1]))
            if r[0] == best: out[r[0]][j] = "\\textbf{" + r[1][j] + "}"
            elif self_scored.get(r[0]) == j: out[r[0]][j] = "\\textcolor{gray}{" + r[1][j] + "}"
    return out


def _lg_col(cols):
    """index of the Llama Guard 3 judge column, for the self-scored rule"""
    return next(i for i, c in enumerate(cols) if c == "llamaguard" or c == "Llama Guard 3")


def step2_table():
    """Main table: harm after SFT per arm under four judges, plus training-set statistics."""
    J = _judges(); sel = json.load(open(f"{W}/results/step2_selections.json"))
    wgp = json.load(open(f"{W}/results/scores_wildguard_pool.json")); req = np.array(wgp["harmful_request"])
    import sys; sys.path.insert(0, f"{W}/scripts"); from common import main_pool
    pool = main_pool(); unsafe = np.array([not r["is_safe"] for r in pool])
    def demo_share(arm):
        out = []
        for s_ in sel[arm]:
            idx = np.array(json.load(open(f"{W}/selections/{arm}_s{s_}.json"))["idx"])
            out.append(req[idx][~unsafe[idx]].sum() / len(idx))   # harmful prompt with a safe response, as a share of the set
        return np.mean(out)
    arms = [("base", "No fine-tuning", "--", "--"), ("full", "Full pool", "--", "0"), ("random1400", "Random, size of certified", "--", "0"),
            ("heur", "Top 75\\% by judge (fixed cutoff)", "Qwen-7B", "0"), ("anti_qwen", "Bottom 35\\% by judge (its rejects)", "Qwen-7B", "0"),
            ("cert400", "Certified, $\\alpha{=}.10$, $n{=}400$", "Qwen-7B", "400"), ("certstrat2_qwen800", "Stratified certified, $n{=}800$", "Qwen-7B", "800"),
            ("lgfilter", "Fixed cutoff", "Llama Guard 3", "0"), ("certlg400", "Certified, $\\alpha{=}.10$, $n{=}400$", "Llama Guard 3", "400"),
            ("certstrat2_lg800", "Stratified certified, $n{=}800$", "Llama Guard 3", "800"),
            ("saft05", "SAFT (spectral score, tuned $\\tau$)", "0.5B embeddings", "100"), ("certsaft05", "Certified, $\\alpha{=}.10$, $n{=}400$", "SAFT score", "400"),
            ("certstrat_saft05", "Stratified certified, $n{=}800$", "SAFT score", "800"),
            ("labels", "Labeled-safe only", "--", "200"), ("oracle", "Oracle, all safe", "--", "4000")]
    models = json.load(open(f"{W}/results/gen_scores/wildguard.json")).keys()
    rows, recs = [], []
    for key, name, scorer, labels in arms:
        ms = [m for m in models if _seeds(m, key)]
        if key == "base":
            comp = "--"; demo = "--"; cert = "--"
        else:
            comp = f"{np.mean([sel[key][s_]['unsafe_frac'] for s_ in sel[key]]):.3f}"; demo = f"{demo_share(key):.2f}"
            cert = "--" if not any("certified" in sel[key][s_] for s_ in sel[key]) else f"{sum(bool(sel[key][s_].get('certified')) for s_ in sel[key])}/{len(sel[key])}"
        recs.append((key, [_cell([J[j](m) for m in ms]) for j in J], [np.mean([J[j](m) for m in ms]) for j in J], f"{name} & {scorer} & {labels} & {comp} & {demo} & {cert}"))
    lg = _lg_col(list(J)); B = _bold_min([(k, c, v) for k, c, v, _ in recs], None, list(J), self_scored={k: lg for k, *_ in recs if k in LG_SCORED})
    groups = {"heur", "lgfilter", "saft05", "labels"}   # first row of each scorer block gets a rule above it
    for k, c, v, head in recs:
        if k in groups: rows.append("\\midrule")
        rows.append(f"{head} & " + " & ".join(B[k]) + " \\\\")
    body = ("\\begin{tabular}{llrccccccc}\n\\toprule\n"
            "training set & scorer & labels & harmful & safety & cert. & \\multicolumn{4}{c}{harmful generations} \\\\\n"
            "\\cmidrule(lr){7-10}\n & & & frac. & demos & & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/step2_main.tex", "w").write(body)


def basemodel_table(tag, fname, arms):
    """E24/E24b/E28: the Table 1 conditions on a second base model; models {tag}{arm}_s{seed} share the 0.5B selection files."""
    J = _judges(); sel = json.load(open(f"{W}/results/step2_selections.json")); u = json.load(open(f"{W}/results/e6_utility.json"))
    wgp = json.load(open(f"{W}/results/scores_wildguard_pool.json")); req = np.array(wgp["harmful_request"])
    import sys; sys.path.insert(0, f"{W}/scripts"); from common import main_pool
    pool = main_pool(); unsafe = np.array([not r["is_safe"] for r in pool])
    models = json.load(open(f"{W}/results/gen_scores/wildguard.json")).keys(); rows, recs = [], []
    bm = [m for m in models if _seeds(m, tag + "base")]   # the untouched model, as Tables 1 and 3 carry it
    if bm:
        xs = np.mean([u[m]["xstest_unsafe_harmful"] for m in bm]); rl = np.mean([u[m]["rouge_l"] for m in bm])
        cells = " & ".join(f"{np.mean([J[j](m) for m in bm]):.3f}" for j in J)
        rows.append(f"No fine-tuning & -- & -- & -- & -- & -- & {cells} & {xs:.3f} & {rl:.3f} \\\\")
    for key, name, scorer, labels in arms:
        ms = [m for m in models if _seeds(m, tag + key)]
        if not ms: continue
        comp = np.mean([sel[key][x]["unsafe_frac"] for x in sel[key]]); demo = np.mean([req[np.array(json.load(open(f"{W}/selections/{key}_s{x}.json"))["idx"])][~unsafe[np.array(json.load(open(f"{W}/selections/{key}_s{x}.json"))["idx"])]].sum() / sel[key][x]["n"] for x in sel[key]])
        cert = "--" if not any("certified" in sel[key][x] for x in sel[key]) else f"{sum(bool(sel[key][x].get('certified')) for x in sel[key])}/{len(sel[key])}"
        xv = [u[m]["xstest_unsafe_harmful"] for m in ms]; rl = [u[m]["rouge_l"] for m in ms]
        recs.append((key, [_cell([J[j](m) for m in ms]) for j in J] + [_cell(xv)], [np.mean([J[j](m) for m in ms]) for j in J] + [np.mean(xv)], f"{name} & {scorer} & {labels} & {comp:.3f} & {demo:.2f} & {cert}", _cell(rl)))
    lg = _lg_col(list(J)); B = _bold_min([(k, c, v) for k, c, v, _, _ in recs], None, list(J) + ["xs"], self_scored={k: lg for k, *_ in recs if k in LG_SCORED})
    for k, c, v, head, rl in recs: rows.append(f"{head} & " + " & ".join(B[k]) + f" & {rl} \\\\")
    body = ("\\begin{tabular}{llrccccccccc}\n\\toprule\ntraining set & scorer & labels & harmful & demos & cert. & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 & XSTest-unsafe & ROUGE-L \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/{fname}.tex", "w").write(body)


def dose_table():
    """E4b: harm against the safety-demonstration share at fixed size and composition."""
    J = _judges(); models = json.load(open(f"{W}/results/gen_scores/wildguard.json")).keys()
    rows = []
    for q in ("0.1", "0.17", "0.25", "0.33", "0.4", "0.47", "0.54"):
        ms = [m for m in models if _seeds(m, f"safedemo{q}")]
        rows.append(f"{float(q):.2f} & " + " & ".join(_cell([J[j](m) for m in ms]) for j in J) + " \\\\")
    body = ("\\begin{tabular}{lcccc}\n\\toprule\nsafety-demo share & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/dose.tex", "w").write(body)


def within05_table():
    """E27: the within-stratum test at 0.5B (judgeharm05, judgetop05) beside the q=.25 sample and the certified set."""
    J = _judges(); models = json.load(open(f"{W}/results/gen_scores/wildguard.json")).keys(); rows = []
    for key, name in (("safedemo0.25", "stratified random, no scorer (share $0.25$)"), ("judgeharm05", "harmful examples from the judge's top third"), ("judgetop05", "top third, safe examples from the top of the within-safe ranking"), ("cert400", "certified, judge (Table~\\ref{tab:step2})")):
        ms = [m for m in models if _seeds(m, key)]; rows.append(f"{name} & " + " & ".join(_cell([J[j](m) for m in ms]) for j in J) + " \\\\")
    body = ("\\begin{tabular}{lcccc}\n\\toprule\ntraining set & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/within05.tex", "w").write(body)


def dose8_table():
    """E20: the same dose-response on Llama-3.1-8B-Instruct at the certified Llama Guard 3 set's size and
    composition (1584, 8.3 percent harmful), with the certified and random conditions of Table 3 for reference."""
    res = json.load(open(f"{W}/results/e3_eval_lambda0.25.json")); summ = json.load(open(f"{W}/selections_e3/lambda0.25/summary.json"))
    KS = ("beaverdam", "mdjudge", "wildguard", "llamaguard", "xstest_unsafe_harmful"); rows = []
    def row(name, key, share):
        ms = [m for m in res if _seeds(m, key)]
        cells = [_cell([res[m][k] for m in ms]) for k in KS]
        if key in LG_SCORED:   # Llama Guard 3 ranked this row's training set, so its column is self-scored
            cells[KS.index("llamaguard")] = "\\textcolor{gray}{" + cells[KS.index("llamaguard")] + "}"
        return f"{name} & {share} & " + " & ".join(cells) + " \\\\"
    for q in ("0.3", "0.37", "0.45", "0.545"):
        rows.append(row("stratified random, no scorer", f"safedemo8_{q}", f"{float(q):.2f}"))
    rows.append("\\midrule")
    for key, name in (("harmtop8", "harmful examples from the top third by Llama Guard 3 score"), ("harmmid8", "middle third"), ("harmbot8", "bottom third")):
        rows.append(row(name, key, "0.37"))
    rows.append("\\midrule")
    rows.append(row("top third, safe examples from the top of the within-safe ranking", "safetop8", "0.37"))
    rows.append(row("top third, safe examples length-matched to the certified set", "safelen8", f"{np.mean([summ['safelen8'][x]['safety_demo_share'] for x in summ['safelen8']]) / (1 - 0.0833):.2f}"))
    rows.append("\\midrule"); demo = lambda k: np.mean([summ[k][x]["safety_demo_share"] for x in summ[k]]) / (1 - np.mean([summ[k][x]["unsafe_frac"] for x in summ[k]]))
    rows.append(row("certified, Llama Guard 3 (Table~\\ref{tab:field})", "certlg", f"{demo('certlg'):.2f}"))
    rows.append(row("random, pool composition (Table~\\ref{tab:field})", "random", f"{demo('random'):.2f}"))
    rows.append(row("oracle at the certified set's size (Table~\\ref{tab:field})", "oracle_matched", f"{demo('oracle_matched'):.2f}"))
    body = ("\\begin{tabular}{llccccc}\n\\toprule\ntraining set & demo share & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 & XSTest-unsafe \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/dose8.tex", "w").write(body)


def utility_table():
    u = json.load(open(f"{W}/results/e6_utility.json")); models = list(u)
    arms = [("base", "No fine-tuning"), ("full", "Full pool"), ("random1400", "Random, size of certified"), ("heur", "Fixed cutoff, judge"), ("anti_qwen", "Judge's rejects"),
            ("cert400", "Certified, judge"), ("certstrat2_qwen800", "Stratified certified, judge"), ("lgfilter", "Fixed cutoff, Llama Guard 3"), ("certlg400", "Certified, Llama Guard 3"),
            ("certstrat2_lg800", "Stratified certified, Llama Guard 3"), ("saft05", "SAFT"), ("certsaft05", "Certified, SAFT score"), ("certstrat_saft05", "Stratified certified, SAFT score"),
            ("labels", "Labeled-safe only"), ("oracle", "Oracle")]
    rows = []
    for key, name in arms:
        ms = [m for m in models if _seeds(m, key)]
        rows.append(f"{name} & " + " & ".join(_cell([u[m][k] for m in ms]) for k in ("rouge_l", "xstest_safe_refusal", "xstest_unsafe_harmful")) + " \\\\")
    body = ("\\begin{tabular}{lccc}\n\\toprule\ntraining set & ROUGE-L & XSTest safe: refusal & XSTest unsafe: harmful \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/utility.tex", "w").write(body)




def e3_table(lam="0.25", arms=None, fname=None, exclude=("oracle", "oracle_matched", "base", "saft", "saft15")):
    """Field protocol: Llama-3.1-8B-Instruct, SAFT's recipe, all baselines, four judges + XSTest + ROUGE-L.
    The SEAL 3-epoch rows live in the App. L table (seal_table) to keep Table 3 on the page budget."""
    res = json.load(open(f"{W}/results/e3_eval_lambda{lam}.json")); summ = json.load(open(f"{W}/selections_e3/lambda{lam}/summary.json"))
    arms = arms or [("full", "No filter (SFT)", "0"), ("prompting", "Prompting filter (SAFT's judge)", "0"), ("lgfilter", "Fixed cutoff, Llama Guard 3", "0"),
            ("saft", "SAFT (spectral score, tuned $\\tau$)", "100"),
            ("seal_long", "SEAL selector, top 80\\%, 30 epochs", "0$^\\dagger$"),
            ("anti", "Prompting judge's rejects", "0"), ("random", "Random, size of certified", "0"),
            ("certsaft", "Certified, SAFT score", "400"), ("certseal_long", "Certified, SEAL score, 30 epochs", "400"),
            ("certprompt", "Certified, Prompting judge score", "400"), ("certstrat_prompt", "Stratified certified, Prompting judge score", "800"), ("certlg", "Certified, Llama Guard 3", "400"),
            ("certstrat_saft", "Stratified certified, SAFT score", "800"), ("certstrat_seal", "Stratified certified, SEAL score, 30 epochs", "800"),
            ("certstrat2", "Stratified certified, Llama Guard 3", "800"), ("labels", "Labeled-safe only", "400"), ("labels800_8", "Labeled-safe only", "800"), ("randdemo8", "Random plus the labeled demonstrations", "800"), ("oracle", "Oracle, all safe", "3000"), ("oracle_matched", "Oracle, random subset at the certified size", "3000")]
    rows, recs = [], []; KS = ("beaverdam", "mdjudge", "wildguard", "llamaguard", "xstest_unsafe_harmful")
    if fname is None:   # the untouched aligned model, as Table 1 shows the untouched base model
        rows.append("No fine-tuning & -- & -- & -- & -- & -- & " + " & ".join(f"{res['base_s0'][k]:.3f}" for k in KS) + f" & {res['base_s0']['rouge_l']:.3f} \\\\")
    for key, name, labels in arms:
        if key not in summ: continue
        ms = [m for m in res if _seeds(m, key)]
        n = np.mean([summ[key][x]["n"] for x in summ[key]]); comp = np.mean([summ[key][x]["unsafe_frac"] for x in summ[key]])
        demo = np.mean([summ[key][x]["safety_demo_share"] for x in summ[key]])
        c = "--" if not any("certified" in summ[key][x] for x in summ[key]) else f"{sum(bool(summ[key][x].get('certified')) for x in summ[key])}/{len(summ[key])}"
        recs.append((key, [_cell([res[m][k] for m in ms]) for k in KS], [np.mean([res[m][k] for m in ms]) for k in KS], f"{name} & {labels} & {n:.0f} & {comp:.3f} & {demo:.2f} & {c}", _cell([res[m]["rouge_l"] for m in ms])))
    lg = _lg_col(list(KS)); B = _bold_min([(k, c, v) for k, c, v, _, _ in recs], None, list(KS), exclude=exclude, self_scored={k: lg for k, *_ in recs if k in LG_SCORED})
    for k, c, v, head, rl in recs:
        rows.append(f"{head} & " + " & ".join(B[k]) + f" & {rl} \\\\")
    body = ("\\begin{tabular}{lrrccccccccc}\n\\toprule\n"
            "training set & labels & size & harmful & demos & cert. & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 & XSTest-unsafe & ROUGE-L \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/{fname or 'e3_lambda' + lam.replace('.', '')}.tex", "w").write(body)
    if fname: return
    # per-seed SAFT detail
    rows = []
    for key in ("saft15", "saft"):
        if key not in summ: continue
        for x in sorted(summ[key]):
            m = summ[key][x]; r = res[f"{key}_s{x}"]
            rows.append(f"{'layer 15 (as published)' if key == 'saft15' else 'layer validated'} & {x} & {m['layer']} & {m['k']} & {m['val_f1']:.2f} & {m['n']} & {m['unsafe_frac']:.3f} & {r['beaverdam']:.3f} & {r['rouge_l']:.3f} \\\\")
    body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\nvariant & seed & layer & $d$ & val.\\ F1 & kept & harmful & beaver-dam & ROUGE-L \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/saft_seeds.tex", "w").write(body)


def lambda_table():
    """Contamination sweep at 8B: beaver-dam and Llama Guard 3 harm per arm and lambda."""
    arms = [("full", "No filter"), ("prompting", "Prompting filter"), ("saft15", "SAFT, layer 15 (as published)"), ("saft", "SAFT, validated layer"),
            ("random", "Random, matched"), ("certsaft", "Certified, SAFT score"), ("certlg", "Certified, Llama Guard 3"),
            ("certstrat2", "Stratified certified, Llama Guard 3"), ("oracle", "Oracle")]
    lams = ("0.10", "0.25", "0.30"); R = {l: json.load(open(f"{W}/results/e3_eval_lambda{l}.json")) for l in lams}
    S = {l: json.load(open(f"{W}/selections_e3/lambda{l}/summary.json")) for l in lams}
    rows, recs = [], []
    for key, name in arms:
        cells, vals = [], []
        for l in lams:
            ms = [m for m in R[l] if _seeds(m, key)]
            c = "" if not any("certified" in S[l][key][x] for x in S[l][key]) else f" ({sum(bool(S[l][key][x].get('certified')) for x in S[l][key])}/3)"
            cells.append(_cell([R[l][m]["beaverdam"] for m in ms]) + c); vals.append(np.mean([R[l][m]["beaverdam"] for m in ms]))
        recs.append((key, cells, vals, name))
    B = _bold_min([(k, c, v) for k, c, v, _ in recs], None, list(lams), exclude=("oracle", "base", "saft", "saft15"))
    for k, c, v, name in recs:
        rows.append(f"{name} & " + " & ".join(B[k]) + " \\\\")
    body = ("\\begin{tabular}{lccc}\n\\toprule\ntraining set & $\\rho = 0.10$ & $\\rho = 0.25$ & $\\rho = 0.30$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/lambda_sweep.tex", "w").write(body)


def strat_table():
    """E10: certifying the property the learner needs (0.5B, Llama Guard scorer)."""
    J = _judges(); sel = json.load(open(f"{W}/results/step2_selections.json")); u = json.load(open(f"{W}/results/e6_utility.json"))
    wgp = json.load(open(f"{W}/results/scores_wildguard_pool.json")); req = np.array(wgp["harmful_request"])
    import sys; sys.path.insert(0, f"{W}/scripts"); from common import main_pool
    pool = main_pool(); unsafe = np.array([not r["is_safe"] for r in pool])
    models = json.load(open(f"{W}/results/gen_scores/wildguard.json")).keys()
    arms = [("full", "Full pool"), ("certlg400", "Certified, $\\alpha{=}.10$, $n{=}400$"), ("certstrat_lg", "Stratified, $\\alpha{=}.10$ both strata, $n{=}400$"),
            ("certstrat_lg800", "Stratified, $\\alpha{=}.10$ both strata, $n{=}800$"), ("certstrat2_lg800", "Stratified, $\\alpha_h{=}.20$, $\\alpha_b{=}.15$, $n{=}800$"),
            ("labels800", "Labeled-safe only, $n{=}800$, no scorer"), ("randdemo", "Random plus the labeled demonstrations, $n{=}800$"), ("oracle", "Oracle")]
    rows, recs = [], []
    for key, name in arms:
        ms = [m for m in models if _seeds(m, key)]
        comp = np.mean([sel[key][x]["unsafe_frac"] for x in sel[key]])
        demo = np.mean([sel[key][x]["safety_demo_share"] if "safety_demo_share" in sel[key][x] else
                        (lambda idx: req[idx][~unsafe[idx]].sum() / len(idx))(np.array(json.load(open(f"{W}/selections/{key}_s{x}.json"))["idx"])) for x in sel[key]])
        c = "--" if not any("certified" in sel[key][x] for x in sel[key]) else f"{sum(bool(sel[key][x].get('certified')) for x in sel[key])}/{len(sel[key])}"
        xv = [u[m]["xstest_unsafe_harmful"] for m in ms if m in u]
        cells = [_cell([J[j](m) for m in ms]) for j in J] + [_cell(xv) if xv else "--"]
        vals = [np.mean([J[j](m) for m in ms]) for j in J] + [np.mean(xv) if xv else None]
        recs.append((key, cells, vals, f"{name} & {comp:.3f} & {demo:.2f} & {c}"))
    lg = _lg_col(list(J)); B = _bold_min([(k, c, v) for k, c, v, _ in recs], None, list(J) + ["xs"], self_scored={k: lg for k, *_ in recs if k in LG_SCORED})
    for k, c, v, head in recs:
        rows.append(f"{head} & " + " & ".join(B[k]) + " \\\\")
    body = ("\\begin{tabular}{lcccccccc}\n\\toprule\ntraining set & harmful & demos & cert. & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 & XSTest-unsafe \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/strat.tex", "w").write(body)


def pool2_table():
    """E8: the 0.5B study on a second pool (PKU-SafeRLHF), same arms as Table 2 where they apply."""
    J = _judges(); sel = json.load(open(f"{W}/results/step2_selections.json")); u = json.load(open(f"{W}/results/e6_utility.json"))
    arms = [("p2_full", "Full pool", "--", "0"), ("p2_random", "Random, size of certified", "--", "0"),
            ("p2_cert400", "Certified, $\\alpha{=}.10$, $n{=}400$", "Qwen-7B", "400"), ("p2_certlg400", "Certified, $\\alpha{=}.10$, $n{=}400$", "Llama Guard 3", "400"),
            ("p2_certstrat2", "Stratified certified, $n{=}800$", "Llama Guard 3", "800"), ("p2_oracle", "Oracle, all safe", "--", "4000")]
    models = json.load(open(f"{W}/results/gen_scores/wildguard.json")).keys(); rows, recs = [], []
    for key, name, scorer, labels in arms:
        ms = [m for m in models if _seeds(m, key)]; m_ = sel[key]
        c = f"{sum(bool(m_[x].get('certified')) for x in m_)}/{len(m_)}" if any("certified" in m_[x] for x in m_) else "--"
        comp = np.mean([m_[x]["unsafe_frac"] for x in m_]); demo = np.mean([m_[x]["safety_demo_share"] for x in m_])
        cells = [_cell([J[j](m) for m in ms]) for j in J] + [_cell([u[m]["xstest_unsafe_harmful"] for m in ms])]
        vals = [np.mean([J[j](m) for m in ms]) for j in J] + [np.mean([u[m]["xstest_unsafe_harmful"] for m in ms])]
        recs.append((key, cells, vals, f"{name} & {scorer} & {labels} & {comp:.3f} & {demo:.2f} & {c}"))
    lg = _lg_col(list(J)); B = _bold_min([(k, c, v) for k, c, v, _ in recs], None, list(J) + ["xs"], exclude=("p2_oracle",), self_scored={k: lg for k, *_ in recs if k in LG_SCORED})
    for k, c, v, head in recs:
        rows.append(f"{head} & " + " & ".join(B[k]) + " \\\\")
    body = ("\\begin{tabular}{llrcccccccc}\n\\toprule\ntraining set & scorer & labels & harmful & demos & cert. & beaver-dam & MD-Judge & WildGuard & Llama Guard 3 & XSTest-unsafe \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/pool2.tex", "w").write(body)



def quality_table():
    """E7: the certificate on response quality. Rendered from results/e7_quality.json, so the table
    rebuilds without the Alpaca data the experiment itself needs."""
    out = json.load(open(f"{W}/results/e7_quality.json"))
    NS = out[next(iter(out))]["n"]
    rows_t = []
    for key, rec in out.items():
        ptag, jd, a = key.split("|"); lab = {"q0.25": "0.25", "qnat": "0.82"}[ptag]
        rows_t.append(f"{lab} & {jd} & {rec['auc']:.2f} & {a[1:]} & {rec['margin']:+.3f} & "
                      + " & ".join(f"{r:.2f}/{p:.2f}" for r, p in zip(rec["rate"], rec["pred"]))
                      + f" & {max(rec['false']):.3f} \\\\")
    body = ("\\begin{tabular}{llrrr" + "r" * len(NS) + "r}\n\\toprule\npool low-quality & judge & AUC & $\\alpha$ & margin & "
            + " & ".join(f"$n{{=}}{n}$" for n in NS) + " & worst false cert. \\\\\n\\midrule\n"
            + "\n".join(rows_t) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{T}/e7_quality.tex", "w").write(body)

def harmsets_table():
    """E15/E15-b/E16: the field's harm prompt sets (DirectHarm4, HarmBench standard, HEx-PHI public
    release) for every arm of Tables 1 and 3: beaver-dam flag rate per set and over all 900 prompts,
    and the other three judges over all 900. results/harmsets.json (per model, per judge, per set)."""
    H = json.load(open(f"{W}/results/harmsets.json")); sets = ("DirectHarm4", "HarmBench", "HEx-PHI", "all"); JJ = ("MD-Judge", "WildGuard", "Llama Guard 3")
    plan = {"05b": ("harmsets_05b", [("base", "No fine-tuning"), ("full", "Full pool"), ("random1400", "Random, size of certified"), ("heur", "Fixed cutoff, judge"),
                                     ("anti_qwen", "Judge's rejects"), ("cert400", "Certified, judge"), ("lgfilter", "Fixed cutoff, Llama Guard 3"), ("certlg400", "Certified, Llama Guard 3"),
                                     ("certstrat2_lg800", "Stratified certified, Llama Guard 3"), ("certstrat2_qwen800", "Stratified certified, judge"), ("saft05", "SAFT"), ("certsaft05", "Certified, SAFT score"),
                                     ("certstrat_saft05", "Stratified certified, SAFT score"), ("labels", "Labeled-safe only"), ("oracle", "Oracle, all safe")]),
            "8b": ("harmsets_8b", [("base", "No fine-tuning"), ("full", "No filter (SFT)"), ("prompting", "Prompting filter"), ("lgfilter", "Fixed cutoff, Llama Guard 3"), ("saft", "SAFT (validated layer)"),
                                   ("saft15", "SAFT (layer 15)"), ("seal", "SEAL, 3 epochs"), ("seal_long", "SEAL, 30 epochs"), ("anti", "Prompting judge's rejects"), ("random", "Random, size of certified"),
                                   ("certprompt", "Certified, Prompting judge score"), ("certsaft", "Certified, SAFT score"), ("certseal", "Certified, SEAL score, 3 epochs"), ("certseal_long", "Certified, SEAL score, 30 epochs"),
                                   ("certlg", "Certified, Llama Guard 3"), ("certstrat_prompt", "Stratified certified, Prompting judge score"), ("certstrat_saft", "Stratified certified, SAFT score"), ("certstrat_seal", "Stratified certified, SEAL score"),
                                   ("certstrat2", "Stratified certified, Llama Guard 3"), ("labels", "Labeled-safe only"), ("oracle", "Oracle, all safe")])}
    for sub, (fname, arms) in plan.items():
        rows, recs = [], []
        for key, name in arms:
            ms = [m for m in H if m.startswith(sub + "/") and _seeds(m.split("/")[1], key)]
            if not ms: continue
            cols = [[H[m]["beaver-dam"][st] for m in ms] for st in sets] + [[H[m][j]["all"] for m in ms] for j in JJ]
            recs.append((key, [_cell(c) for c in cols], [np.mean(c) for c in cols], name))
        lg = 4 + JJ.index("Llama Guard 3")
        B = _bold_min([(k, c, v) for k, c, v, _ in recs], None, list(range(7)), exclude=("oracle", "base", "saft", "saft15"),
                      self_scored={k: lg for k, *_ in recs if k in LG_SCORED})
        for k, c, v, name in recs: rows.append(f"{name} & " + " & ".join(B[k]) + " \\\\")
        body = ("\\begin{tabular}{lccccccc}\n\\toprule\n & \\multicolumn{4}{c}{beaver-dam} & \\multicolumn{3}{c}{all three sets} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\n"
                "training set & DirectHarm4 & HarmBench & HEx-PHI & all three & MD-Judge & WildGuard & Llama Guard 3 \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        open(f"{T}/{fname}.tex", "w").write(body)


gate_table(); step2_table(); dose_table(); within05_table(); dose8_table(); utility_table()
_ARMS = [("full", "Full pool", "--", "0"), ("random1400", "Random, size of certified", "--", "0"), ("anti_qwen", "Bottom 35\\% by judge (its rejects)", "Qwen-7B", "0"),
         ("cert400", "Certified, $\\alpha{=}.10$, $n{=}400$", "Qwen-7B", "400"), ("lgfilter", "Fixed cutoff", "Llama Guard 3", "0"), ("certlg400", "Certified, $\\alpha{=}.10$, $n{=}400$", "Llama Guard 3", "400"),
         ("certstrat2_lg800", "Stratified certified, $n{=}800$", "Llama Guard 3", "800"), ("safedemo0.1", "Stratified random, share $0.10$", "--", "0"), ("safedemo0.25", "Stratified random, share $0.25$", "--", "0"), ("safedemo0.54", "Stratified random, share $0.54$", "--", "0")]
basemodel_table("tl_", "tinyllama", _ARMS); basemodel_table("qi_", "qwen_instruct", _ARMS); e3_table(); lambda_table(); strat_table(); pool2_table(); harmsets_table(); quality_table()
e3_table(arms=[("full", "No filter (SFT)", "0"), ("seal", "SEAL selector, top 80\\%, 3 epochs", "0$^\\dagger$"), ("seal_long", "SEAL selector, top 80\\%, 30 epochs", "0$^\\dagger$"),
               ("certseal", "Certified, SEAL score, 3 epochs", "400"), ("certseal_long", "Certified, SEAL score, 30 epochs", "400"), ("certstrat_seal", "Stratified certified, SEAL score, 30 epochs", "800"), ("oracle", "Oracle, all safe", "3000")],
         fname="seal_budgets", exclude=("oracle", "base", "full"))
print("tables written to", T)
