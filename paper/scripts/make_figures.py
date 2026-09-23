"""Figures for the LLM paper from results/*.json. Decile figure: harm of the fine-tuned model
against the judge's score decile of its training set (400 examples each, 3 seeds), with the
human harmful fraction and the safety-demonstration share of each decile."""
import os
import json, os, sys
import numpy as np, matplotlib
import re


def _seeds(m, key):
    """m is a seed model of arm key: <key>_s<digits>. A prefix test would let seal_ match seal_long_."""
    return re.match(re.escape(key) + r"_s\d+$", m) is not None

matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); F = f"{W}/paper/figures"; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
PAL = ["#1f4e79", "#c0504d", "#4f8f3a", "#7f7f7f"]          # judges: beaver-dam, MD-Judge, WildGuard, Llama Guard 3
GRP = {"benign": "#0072B2", "demo": "#009E73", "refusal": "#8064a2", "harmful": "#D55E00"}   # example groups, as in Figure 1


def legend_below(fig, handles=None, labels=None, ncol=4, pad=0.16, **kw):
    """One legend under the plot area, never over it; the figure keeps room for it."""
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
        for a in fig.axes[1:]:
            h2, l2 = a.get_legend_handles_labels()
            for h, l in zip(h2, l2):
                if l not in labels:
                    handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False, fontsize=7, **kw)
    fig.tight_layout(rect=(0, pad, 1, 1))


def decile_figure():
    ev = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md = json.load(open(f"{W}/results/gen_scores/mdjudge.json"))
    wg = json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg = json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
    sel = json.load(open(f"{W}/results/step2_selections.json"))
    J = {"beaver-dam": lambda m: ev[m]["beaverdam_flag"], "MD-Judge": lambda m: float(np.mean(md[m]["flag"])),
         "WildGuard": lambda m: float(np.mean(wg[m]["harmful_response"])), "Llama Guard 3": lambda m: float(np.mean(lg[m]["flag"]))}
    d = np.arange(10); unsafe = [sel[f"decile{i}"]["0"]["unsafe_frac"] for i in d]; demos = [sel[f"decile{i}"]["0"]["safety_demo_share"] for i in d]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.0))
    for (name, f), c in zip(J.items(), PAL):
        ys = [[f(f"decile{i}_s{s}") for s in range(3)] for i in d]
        ax[0].errorbar(d, [np.mean(y) for y in ys], yerr=[np.std(y, ddof=1) / np.sqrt(3) for y in ys], color=c, marker="o", ms=3, lw=1.2, label=name)
    ax[0].set_xlabel("judge-score decile (0 = judged safest)"); ax[0].set_ylabel("harmful\ngenerations")
    ax[1].plot(d, unsafe, color=GRP["harmful"], marker="s", ms=3, lw=1.2, label="harmful fraction (human label)")
    ax[1].plot(d, demos, color=GRP["demo"], marker="^", ms=3, lw=1.2, label="safety-demonstration share")
    ax[1].set_xlabel("judge-score decile"); ax[1].set_ylabel("share of the training set")
    for a in ax: a.set_xticks(d)
    legend_below(fig, ncol=6, pad=0.12); fig.savefig(f"{F}/decile.pdf", bbox_inches="tight"); fig.savefig(f"{F}/decile.png", dpi=200, bbox_inches="tight")
    print("saved figures/decile.{pdf,png}")




def _judges():
    ev = json.load(open(f"{W}/results/step2_eval2.json"))["per_model"]; md = json.load(open(f"{W}/results/gen_scores/mdjudge.json"))
    wg = json.load(open(f"{W}/results/gen_scores/wildguard.json")); lg = json.load(open(f"{W}/results/gen_scores/llamaguard.json"))
    return {"beaver-dam": lambda m: ev[m]["beaverdam_flag"], "MD-Judge": lambda m: float(np.mean(md[m]["flag"])),
            "WildGuard": lambda m: float(np.mean(wg[m]["harmful_response"])), "Llama Guard 3": lambda m: float(np.mean(lg[m]["flag"]))}


def _pool():
    sys.path.insert(0, f"{W}/scripts"); from common import main_pool
    pool = main_pool(); unsafe = np.array([not r["is_safe"] for r in pool])
    wg = json.load(open(f"{W}/results/scores_wildguard_pool.json"))
    return pool, unsafe, np.array(wg["harmful_request"]), np.array(wg["refusal"])


def mechanism_figure():
    """Judge score (log-odds of 'not harmful') among human-safe examples, split by prompt type, against human-harmful ones."""
    pool, unsafe, req, ref = _pool(); g = np.array(json.load(open(f"{W}/results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json")))
    groups = [("human-harmful", unsafe, GRP["harmful"]), ("safe, benign prompt", ~unsafe & ~req, GRP["benign"]),
              ("safe, harmful prompt, engaged", ~unsafe & req & ~ref, GRP["demo"]), ("safe, harmful prompt, refused", ~unsafe & req & ref, GRP["refusal"])]
    fig, ax = plt.subplots(figsize=(4.6, 1.7)); bins = np.linspace(g.min(), g.max(), 40)
    for name, m, c in groups:
        ax.hist(g[m], bins=bins, density=True, histtype="step", lw=1.3, color=c, label=f"{name} (n={int(m.sum())})")
    ax.set_xlabel("judge score (log-odds the response is not harmful)"); ax.set_ylabel("density")
    legend_below(fig, ncol=2, pad=0.22); fig.savefig(f"{F}/mechanism.pdf", bbox_inches="tight"); fig.savefig(f"{F}/mechanism.png", dpi=200, bbox_inches="tight")
    print("saved figures/mechanism.{pdf,png}")


def dose_figure():
    """E4b: harm against the safety-demonstration share at fixed size and composition, four judges."""
    J = _judges(); qs = ("0.1", "0.17", "0.25", "0.33", "0.4", "0.47", "0.54")
    fig, ax = plt.subplots(figsize=(3.2, 1.7))
    for (name, f), c in zip(J.items(), PAL):
        ys = [[f(f"safedemo{q}_s{s}") for s in range(5)] for q in qs]
        ax.errorbar([float(q) for q in qs], [np.mean(y) for y in ys], yerr=[np.std(y, ddof=1) / np.sqrt(5) for y in ys], color=c, marker="o", ms=3, lw=1.2, label=name)
    ax.set_xlabel("safety-demonstration share\namong safe examples"); ax.set_ylabel("harmful\ngenerations")
    legend_below(fig, ncol=2, pad=0.22); fig.savefig(f"{F}/dose.pdf", bbox_inches="tight"); fig.savefig(f"{F}/dose.png", dpi=200, bbox_inches="tight")
    print("saved figures/dose.{pdf,png}")


def lambda_figure():
    """Harm (beaver-dam) and helpfulness (ROUGE-L) against contamination at 8B."""
    lams = ("0.10", "0.25", "0.30"); R = {l: json.load(open(f"{W}/results/e3_eval_lambda{l}.json")) for l in lams}
    arms = [("full", "no filter", PAL[0], "o"), ("prompting", "Prompting filter", PAL[1], "s"), ("random", "random, matched", "#56B4E9", "v"),
            ("certlg", "certified, Llama Guard 3", PAL[2], "^"), ("certstrat2", "stratified certified", PAL[3], "D"), ("oracle", "oracle", "k", "x")]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.2))
    for key, name, c, mk in arms:
        for a, metric in zip(ax, ("beaverdam", "rouge_l")):
            ys = [[R[l][m][metric] for m in R[l] if _seeds(m, key)] for l in lams]
            a.errorbar([float(l) for l in lams], [np.mean(y) for y in ys], yerr=[np.std(y, ddof=1) / np.sqrt(len(y)) for y in ys], color=c, marker=mk, ms=3.5, lw=1.2, label=name)
    ax[0].set_ylabel("harmful (beaver-dam)"); ax[1].set_ylabel("ROUGE-L"); ax[1].set_ylim(0.2, 0.27)
    for a in ax: a.set_xlabel(r"pool contamination $\rho$"); a.set_xticks([0.10, 0.25, 0.30])
    legend_below(fig, ncol=6, pad=0.12); fig.savefig(f"{F}/lambda.pdf", bbox_inches="tight"); fig.savefig(f"{F}/lambda.png", dpi=200, bbox_inches="tight")
    print("saved figures/lambda.{pdf,png}")


def scatter_figure():
    """Balance: the eleven fine-tuned conditions of Table 1 other than SAFT's three, marker by family
    (five shapes), color by harmful-fraction bin, short text labels on the seven the text names."""
    J = _judges(); sel = json.load(open(f"{W}/results/step2_selections.json")); pool, unsafe, req, ref = _pool()
    fam = {"full": "pool, random", "random1400": "pool, random", "heur": "judge-scored", "anti_qwen": "judge-scored", "cert400": "judge-scored", "certstrat2_qwen800": "judge-scored",
           "lgfilter": "Llama Guard 3-scored", "certlg400": "Llama Guard 3-scored", "certstrat2_lg800": "Llama Guard 3-scored", "labels": "labels only", "oracle": "oracle"}
    mk = {"pool, random": "o", "judge-scored": "s", "Llama Guard 3-scored": "^", "labels only": "P", "oracle": "*"}
    text = {"cert400": "certified, judge", "anti_qwen": "judge's rejects", "full": "full pool", "certlg400": "certified, LG 3", "certstrat2_lg800": "stratified, LG 3", "certstrat2_qwen800": "stratified, judge", "oracle": "oracle"}
    R = (6, -1)   # every label sits directly to the right of its marker, a hair below center; two get a small extra shift to clear a neighbor
    off = {0: {"cert400": (6, 3), "anti_qwen": R, "full": R, "certlg400": (-6, 2), "certstrat2_lg800": (6, -5), "certstrat2_qwen800": (6, -4), "oracle": R},
           1: {"cert400": (6, 3), "anti_qwen": R, "full": R, "certlg400": (-6, -2), "certstrat2_lg800": (6, 3), "certstrat2_qwen800": (-6, -4), "oracle": R}}
    ha = {0: {"certlg400": "right"}, 1: {"certlg400": "right", "certstrat2_qwen800": "right"}}
    bins = [(-1, 0.10, "harmful fraction $\\leq$ .10", PAL[2]), (0.10, 0.25, ".10 to .25", PAL[0]), (0.25, 1.1, "> .25", PAL[1])]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True); fig.subplots_adjust(wspace=0.08)
    for a, f in fam.items():
        comp = np.mean([sel[a][x]["unsafe_frac"] for x in sel[a]])
        demo = np.mean([sel[a][x]["safety_demo_share"] if "safety_demo_share" in sel[a][x] else
                        (lambda idx: req[idx][~unsafe[idx]].sum() / len(idx))(np.array(json.load(open(f"{W}/selections/{a}_s{x}.json"))["idx"])) for x in sel[a]])
        harm = np.mean([J["beaver-dam"](f"{a}_s{x}") for x in sel[a]]); c = next(cc for lo, hi, _, cc in bins if lo < comp <= hi)
        for k, (axx, xv) in enumerate(zip(ax, (comp, demo))):
            axx.scatter([xv], [harm], s=95 if f == "oracle" else 34, marker=mk[f], color=c, edgecolor="k", linewidth=0.4, zorder=3)
            if a in text: axx.annotate(text[a], (xv, harm), fontsize=5.6, xytext=off[k][a], textcoords="offset points", va="center", ha=ha.get(k, {}).get(a, "left"))
    ax[0].set_xlabel("harmful fraction (human label)"); ax[1].set_xlabel("safety-demonstration share"); ax[0].set_ylabel("harmful generations\n(beaver-dam)")
    ax[0].set_ylim(0.33, 0.505); ax[0].set_xlim(-0.2, 0.66); ax[1].set_xlim(0.18, 0.70)
    ax[0].set_xticks([0.0, 0.2, 0.4, 0.6])   # a harmful fraction is non-negative; the left margin is label room, not data range
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cc, marker="o", ls="", ms=5.5, label=n) for _, _, n, cc in bins] + [Line2D([], [], color="0.35", marker=m, ls="", ms=7 if m == "*" else 5.5, label=f) for f, m in mk.items()]
    legend_below(fig, h, [x.get_label() for x in h], ncol=4, pad=0.22)
    fig.savefig(f"{F}/scatter.pdf", bbox_inches="tight"); fig.savefig(f"{F}/scatter.png", dpi=200, bbox_inches="tight"); print("saved figures/scatter.{pdf,png}")



def label_complexity_figure():
    """E5: measured certification rate (markers) against the exact closed form (lines), one panel per alpha."""
    from matplotlib.lines import Line2D
    out = json.load(open(f"{W}/results/e5_label_complexity.json"))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.75), sharey=True)
    colors = {"nat": PAL[1], "u0.25": PAL[0], "u0.10": PAL[2]}; styles = {"1.5B": "--", "7B": "-", "LG": ":"}; marks = {"1.5B": "s", "7B": "o", "LG": "^"}
    for ax, alpha in zip(axes, (0.10, 0.25)):
        for key, rec in out.items():
            ptag, jd, a = key.split("|")
            if a != f"a{alpha}": continue
            ax.plot(rec["n"], rec["pred"], styles[jd], color=colors[ptag], lw=1.2)
            ax.plot(rec["n"], rec["rate"], marks[jd], color=colors[ptag], ms=4, mfc="none" if jd == "1.5B" else None)
        ax.set_title(f"$\\alpha = {alpha:.2f}$", fontsize=8); ax.set_xlabel("calibration labels $n$"); ax.set_xscale("log"); ax.grid(alpha=.3)
        ax.set_xticks([50, 100, 200, 400, 800]); ax.set_xticklabels(["50", "100", "200", "400", "800"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("certification rate")
    h = [Line2D([], [], color=c, lw=1.2, label=f"pool harmful {l}") for l, c in (("0.57", PAL[1]), ("0.25", PAL[0]), ("0.10", PAL[2]))]
    h += [Line2D([], [], color="k", ls="-", marker="o", ms=4, label="7B judge"), Line2D([], [], color="k", ls="--", marker="s", ms=4, mfc="none", label="1.5B judge"),
          Line2D([], [], color="k", ls=":", marker="^", ms=4, label="Llama Guard 3"),
          Line2D([], [], color="0.4", ls="", marker="o", ms=4, label="markers: measured rate"), Line2D([], [], color="0.4", ls="-", lw=1.2, label="lines: closed form")]
    legend_below(fig, h, [x.get_label() for x in h], ncol=4, pad=0.24)
    fig.savefig(f"{F}/label_complexity.pdf", bbox_inches="tight"); fig.savefig(f"{F}/label_complexity.png", dpi=200, bbox_inches="tight")
    print("saved figures/label_complexity.{pdf,png}")




def mechanism8_figure():
    """The 8B analogue of Fig. 4 (left): the Prompting judge's score (log-odds of 'not harmful to fine-tune on')
    on the 3000-example pool by human label and prompt type (no refusal label at 8B: WildGuard's refusal
    label was computed for the 0.5B pool only)."""
    D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); S = json.load(open(f"{D}/scores.json"))
    unsafe = np.array([not r["is_safe"] for r in pool]); req = np.array(S["harmful_request"]); g = -np.array(S["prompting_yes_logodds"])
    groups = [("human-harmful", unsafe, GRP["harmful"]), ("safe, benign prompt", ~unsafe & ~req, GRP["benign"]), ("safe, harmful prompt", ~unsafe & req, GRP["demo"])]
    fig, ax = plt.subplots(figsize=(4.6, 1.85)); bins = np.linspace(g.min(), g.max(), 40)
    for name, m, c in groups: ax.hist(g[m], bins=bins, density=True, histtype="step", lw=1.3, color=c, label=f"{name} (n={int(m.sum())})")
    ax.set_xlabel("Prompting judge score (log-odds the sample is not harmful to fine-tune on)"); ax.set_ylabel("density")
    legend_below(fig, ncol=3, pad=0.22); fig.savefig(f"{F}/mechanism8.pdf", bbox_inches="tight"); fig.savefig(f"{F}/mechanism8.png", dpi=200, bbox_inches="tight")
    print("saved figures/mechanism8.{pdf,png}")


def saft_kde_figure():
    """SAFT's own diagnostic (their App. B): density of the spectral score by human label, on our 8B pool, layers 15 and 32, k = 1."""
    D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); unsafe = np.array([not r["is_safe"] for r in pool])
    req = np.array(json.load(open(f"{D}/scores.json"))["harmful_request"]); pool05, unsafe05, req05, _ = _pool()
    # The panels need only the first principal component's squared projection, one number per example.
    # The embedding matrices it comes from are hundreds of megabytes and are not distributed, so the
    # scores are cached here: computed from the embeddings when they are present, read back otherwise.
    cache = f"{W}/results/saft_kde_scores.json"
    emb = [f"{D}/saft_embeddings.npz", f"{W}/results/saft05_embeddings.npz"]
    if all(os.path.exists(e) for e in emb):
        Z = np.load(emb[0]); Z05 = np.load(emb[1])["layer"]
        def _score(Zl):
            Zc = Zl - Zl.mean(0, keepdims=True); _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
            return ((Zc @ Vt[0]) ** 2)
        scores = [_score(Z["layer15"]), _score(Z["layer32"]), _score(Z05)]
        json.dump([s.tolist() for s in scores], open(cache, "w"))
    else:
        scores = [np.array(s) for s in json.load(open(cache))]
    panels = [("8B, layer 15", scores[0], unsafe, req), ("8B, layer 32", scores[1], unsafe, req), ("0.5B, layer 12", scores[2], unsafe05, req05)]
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.1), sharey=False)
    for a, (title, sc, uns, rq) in zip(ax, panels):
        xs = np.linspace(0, np.quantile(sc, 0.99), 200); bw = 0.06 * xs[-1]
        for name, m, c in (("human-harmful", uns, GRP["harmful"]), ("safe, benign prompt", ~uns & ~rq, GRP["benign"]), ("safe, harmful prompt", ~uns & rq, GRP["demo"])):
            d = np.exp(-0.5 * ((xs[:, None] - sc[m][None]) / bw) ** 2).sum(1) / (m.sum() * bw); a.plot(xs, d, color=c, lw=1.3, label=name)
        a.set_title(f"{title}, $d = 1$", fontsize=8); a.set_xlabel("SAFT score $s_i$"); a.set_yticks([])
    ax[0].set_ylabel("density")
    legend_below(fig, ncol=3, pad=0.2); fig.savefig(f"{F}/saft_kde.pdf", bbox_inches="tight"); fig.savefig(f"{F}/saft_kde.png", dpi=200, bbox_inches="tight")
    print("saved figures/saft_kde.{pdf,png}")


def category_rates():
    """By-prompt-category flag rates for every arm of the category figure (results/categories_by_prompt.json), the values the heatmap prints."""
    C = json.load(open(f"{W}/results/eval_prompt_categories.json")); cats = C["categories"]; pc = C["prompt_categories"]
    mask = {c: np.array([c in x for x in pc]) for c in cats}; keep = [c for c in cats if mask[c].sum() >= 10]
    bd05 = json.load(open(f"{W}/results/gen_scores/beaverdam.json")); bd8 = json.load(open(f"{W}/results/e3_scores_lambda0.25.json"))["beaverdam"]
    vals = {}
    for title, src, fl, arms in (("Qwen2.5-0.5B", bd05, lambda m: np.array(bd05[m]) > 0.5, ("full", "cert400", "anti_qwen", "certlg400", "certstrat2_lg800", "oracle")),
                                 ("Llama-3.1-8B-Instruct", bd8, lambda m: np.array(bd8[m]["score"]) > 0.5, ("full", "prompting", "anti", "certlg", "certstrat2", "oracle"))):
        for key in arms:
            ms = [m for m in src if _seeds(m, key)]
            vals[f"{title}/{key}"] = {c: float(np.mean([fl(m)[mask[c]].mean() for m in ms])) for c in keep}
    json.dump({"n_prompts": {c: int(mask[c].sum()) for c in cats}, "flag_rate_by_prompt_category": vals}, open(f"{W}/results/categories_by_prompt.json", "w"), indent=1)


def category_figure():
    C = json.load(open(f"{W}/results/categories_by_prompt.json")); V = C["flag_rate_by_prompt_category"]; n = C["n_prompts"]
    short = {"animal_abuse": "animal abuse", "controversial_topics,politics": "politics", "discrimination,stereotype,injustice": "discrimination",
             "drug_abuse,weapons,banned_substance": "drugs, weapons", "financial_crime,property_crime,theft": "financial crime", "hate_speech,offensive_language": "hate speech",
             "misinformation_regarding_ethics,laws_and_safety": "misinformation", "non_violent_unethical_behavior": "unethical, non-violent", "privacy_violation": "privacy",
             "sexually_explicit,adult_content": "sexual content", "violence,aiding_and_abetting,incitement": "violence"}
    scales = [("Qwen2.5-0.5B", [("full", "full\npool"), ("cert400", "certified,\njudge"), ("anti_qwen", "judge's\nrejects"), ("certlg400", "certified,\nLlama Guard"), ("certstrat2_lg800", "stratified\ncertified"), ("oracle", "oracle")]),
              ("Llama-3.1-8B-Instruct", [("full", "full\npool"), ("prompting", "Prompting\nfilter"), ("anti", "judge's\nrejects"), ("certlg", "certified,\nLlama Guard"), ("certstrat2", "stratified\ncertified"), ("oracle", "oracle")])]
    cats = list(next(iter(V.values())).keys()); cats = sorted(cats, key=lambda c: -n[c])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4), sharey=True); fig.subplots_adjust(wspace=0.06)
    for ax, (title, arms) in zip(axes, scales):
        M = np.array([[V[f"{title}/{k}"][c] for k, _ in arms] for c in cats])
        im = ax.imshow(M, cmap="Reds", vmin=0.3, vmax=0.95, aspect="auto")   # scale spans the observed range (.31 to .92); darker = more harmful
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if M[i, j] > 0.68 else "black")
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([l.replace("\\n", " ") for _, l in arms], fontsize=6, rotation=28, ha="right", rotation_mode="anchor"); ax.set_title(title, fontsize=8)
        ax.set_yticks(range(len(cats))); ax.set_yticklabels([f"{short.get(c, c)} ({n[c]})" for c in cats], fontsize=6.5)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label("flagged (beaver-dam)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    fig.savefig(f"{F}/categories.pdf", bbox_inches="tight"); fig.savefig(f"{F}/categories.png", dpi=200, bbox_inches="tight"); print("saved figures/categories.{pdf,png}")


def hexphi_figure():
    """Harm by HEx-PHI category (10 x 30 prompts, balanced by construction): the field's radar
    (Qi et al. Fig. 1), beaver-dam flag rate per category, five arms per scale."""
    H = json.load(open(f"{W}/results/harmsets.json"))
    cats = [("category_1", "illegal activity"), ("category_3", "hate, harassment,\nviolence"), ("category_4", "malware"), ("category_5", "physical harm"), ("category_6", "economic harm"),
            ("category_7", "fraud,\ndeception"), ("category_8", "adult content"), ("category_9", "political\ncampaigning"), ("category_10", "privacy\nviolation"), ("category_11", "financial\nadvice")]
    scales = [("Qwen2.5-0.5B", "05b", [("full", "full pool", PAL[0], "-"), ("cert400", "judge-filtered", PAL[1], "-"), ("certlg400", "certified, Llama Guard 3", PAL[2], "-"),
                                       ("certstrat2_lg800", "stratified certified", PAL[3], "-"), ("oracle", "oracle", "k", ":")]),
              ("Llama-3.1-8B-Instruct", "8b", [("full", "full pool", PAL[0], "-"), ("prompting", "judge-filtered", PAL[1], "-"), ("certlg", "certified, Llama Guard 3", PAL[2], "-"),
                                               ("certstrat2", "stratified certified", PAL[3], "-"), ("oracle", "oracle", "k", ":")])]
    ang = np.linspace(0, 2 * np.pi, len(cats), endpoint=False); fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), subplot_kw=dict(polar=True)); fig.subplots_adjust(wspace=0.55)
    for ax, (title, sub, arms) in zip(axes, scales):
        for key, name, c, ls in arms:
            ms = [m for m in H if m.startswith(sub + "/") and _seeds(m.split("/")[1], key)]
            v = np.array([np.mean([H[m]["beaver-dam"][f"HEx-PHI/{k}"] for m in ms]) for k, _ in cats]); vv = np.r_[v, v[:1]]
            ax.plot(np.r_[ang, ang[:1]], vv, color=c, ls=ls, lw=1.2, label=name if ax is axes[0] else None)
        ax.set_xticks(ang); ax.set_xticklabels([n for _, n in cats], fontsize=5.5); ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["", "0.5", "", "1.0"]); ax.tick_params(axis="y", labelsize=5.5); ax.tick_params(axis="x", pad=13); ax.set_rlabel_position(200)
        ax.set_title(title, fontsize=8, pad=12)
    h, l = axes[0].get_legend_handles_labels(); legend_below(fig, h, l, ncol=5, pad=0.12)
    fig.savefig(f"{F}/hexphi.pdf", bbox_inches="tight"); fig.savefig(f"{F}/hexphi.png", dpi=200, bbox_inches="tight"); print("saved figures/hexphi.{pdf,png}")


if __name__ == "__main__":
    decile_figure(); mechanism_figure(); dose_figure(); lambda_figure(); scatter_figure(); label_complexity_figure(); saft_kde_figure(); mechanism8_figure(); category_rates(); category_figure(); hexphi_figure()
