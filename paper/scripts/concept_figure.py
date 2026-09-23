"""Figure 1, built from the data rather than drawn: four real pool examples with the judge's score,
the pooled and the stratified cut on the measured score distribution, and the harm of the models
the selections train. Follows the concept-figure convention of the adjacent literature (SEAL Fig. 1,
Bianchi et al. Fig. 1: concrete prompt/response cards) rather than the parent paper's schematic.

Example selection rule (fixed, so the figure is reproducible): among pool examples whose prompt is
at most 90 characters and whose response is at most 170 characters, the example of each of the
four groups (human-safe with a benign prompt; human-safe with a harmful prompt, engaged; human-safe
with a harmful prompt, refused; human-harmful) whose judge score is closest to the group's median,
skipping examples with sexual content, which cannot be printed.
"""
import os
import json, os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); BASE = f"{W}/paper"
sys.path.insert(0, f"{W}/scripts"); from common import build_pool, load_split  # noqa: E402
BLUE, ORANGE, GREEN, PURPLE, GREY, INK = "#0072B2", "#D55E00", "#009E73", "#8064a2", "#5f5f5f", "#2b2b2b"
F = 1.3   # uniform text scale
SKIP = ("lick", "cock", "explicit")   # printable-content filter for the example cards


def main():
    pool = build_pool(load_split("train"), 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
    wg = json.load(open(f"{W}/results/scores_wildguard_pool.json")); req = np.array(wg["harmful_request"]); ref = np.array(wg["refusal"])
    g = np.array(json.load(open(f"{W}/results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json")))
    short = np.array([len(r["prompt"]) <= 90 and len(r["response"]) <= 170 and not any(k in (r["prompt"] + r["response"]).lower() for k in SKIP) for r in pool])
    groups = [("safe, benign prompt", ~unsafe & ~req, BLUE), ("safe, harmful prompt, engaged", ~unsafe & req & ~ref, GREEN),
              ("safe, harmful prompt, refused", ~unsafe & req & ref, PURPLE), ("harmful response", unsafe, ORANGE)]
    picks = []
    for name, m, c in groups:
        cand = np.where(m & short)[0]; i = cand[np.argmin(np.abs(g[cand] - np.median(g[m])))]; picks.append((name, i, c))
    # right panel: Llama Guard 3 as scorer (the deployed stratified method, Table 2), with its two CERTIFIED seed-0
    # selections: pooled alpha .10 n 400 (certlg400_s0) and stratified alpha_h .20 / alpha_b .15 n 800 (certstrat2_lg800_s0)
    gl = -np.array(json.load(open(f"{W}/results/scores_llamaguard_pool.json"))["score_unsafe_minus_safe"])
    tau_pool = gl[np.array(json.load(open(f"{W}/selections/certlg400_s0.json"))["idx"])].min()
    ks = json.load(open(f"{W}/selections/certstrat2_lg800_s0.json"))["idx"]; k = np.zeros(len(pool), bool); k[ks] = True
    tau_h, tau_b = gl[k & req].min(), gl[k & ~req].min()

    fig = plt.figure(figsize=(12.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.12, left=0.01, right=0.99, top=0.74, bottom=0.16)
    # ---- (1) four examples with the judge's score
    ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    for j, (name, i, c) in enumerate(picks):
        y = 9.2 - 2.3 * j; r = pool[i]
        pl = textwrap.wrap("User: " + r["prompt"], 50)[:1]; rl = textwrap.wrap("Assistant: " + r["response"], 60)[:2]
        ax.add_patch(FancyBboxPatch((0.2, y - 1.95), 6.9, 2.05, boxstyle="round,pad=0.0,rounding_size=0.15", facecolor="white", edgecolor=c, lw=1.4))
        ax.text(0.4, y - 0.12, pl[0] + ("" if len("User: " + r["prompt"]) <= 50 else " ..."), fontsize=6.9 * F, color=INK, va="top", fontweight="bold")
        ax.text(0.4, y - 0.78, "\n".join(rl) + ("" if len("Assistant: " + r["response"]) <= 120 else " ..."), fontsize=6.4 * F, color=INK, va="top", linespacing=1.3)
        ax.text(8.6, y - 0.62, f"{g[i]:+.1f}", fontsize=10 * F, color=c, ha="center", va="center", fontweight="bold")
        ax.text(8.6, y - 1.5, name.replace(", ", ",\n"), fontsize=5.6 * F, color=c, ha="center", va="center", linespacing=1.15)
    ax.set_title("pool examples and the judge's score, one per group", fontsize=11.5 * F, color=INK, pad=40)
    ax.text(0.5, 1.01, "the judge scores a response by its prompt: a safe answer to a\nharmful prompt scores with the harmful examples", ha="center", va="bottom", fontsize=7.8 * F, color=GREY, transform=ax.transAxes, linespacing=1.3)

    # ---- (2) measured score distributions with the pooled and stratified cuts
    ax = fig.add_subplot(gs[0, 1]); ax.set_ylim(-1.5, 10); ax.axis("off")
    g2 = gl; xs = np.linspace(g2.min(), g2.max(), 300)
    def dens(v): return np.exp(-0.5 * ((xs[:, None] - v[None]) / 0.5) ** 2).sum(1)
    rows = [("benign prompts", ~req, 5.6), ("harmful prompts", req, 0.6)]
    for name, m, y0 in rows:
        parts = [(m & ~unsafe & ~req, BLUE), (m & ~unsafe & req & ~ref, GREEN), (m & ~unsafe & req & ref, PURPLE), (m & unsafe, ORANGE)]
        ds = [(dens(g2[mm]), c) for mm, c in parts if mm.sum()]; top = max(d.max() for d, _ in ds)
        for d, c in ds:
            ax.fill_between(xs, y0, y0 + 3.0 * d / top, color=c, alpha=0.45, lw=0)
        ax.annotate("", xy=(g2.max() + 0.5, y0), xytext=(g2.min() - 0.5, y0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0))
        for tv in (-10, -5, 0, 5, 10):
            ax.plot([tv, tv], [y0 - 0.15, y0], color=INK, lw=0.8); ax.text(tv, y0 - 0.25, f"{tv:+d}" if tv else "0", fontsize=5.8 * F, color=INK, ha="center", va="top")
        ax.text(g2.min() - 0.3, y0 + 3.15, name, fontsize=8.5 * F, color=INK, va="bottom")
    ax.set_xlim(g2.min() - 0.8, g2.max() + 1)
    # pooled cut: one threshold across both rows
    ax.plot([tau_pool, tau_pool], [0.6, 9.2], color=INK, lw=1.6, ls="--")
    ax.text(tau_pool + 0.2, 4.25, "pooled threshold:\none cut for both strata,\nkeeps mostly the blue", fontsize=6.4 * F, color=INK, va="top", ha="left", linespacing=1.25)
    # stratified cut: one threshold per row
    for (name, m, y0), tau, c in zip(rows, (tau_b, tau_h), (GREEN, GREEN)):
        ax.plot([tau, tau], [y0, y0 + 3.6], color=INK, lw=1.6)
    ax.text(tau_h - 0.2, 4.25, "stratified thresholds:\nlooser on harmful prompts,\ntighter on benign ones,\nkeep the green and purple", fontsize=6.4 * F, color=INK, va="top", ha="right", linespacing=1.25)
    ax.text(g2.max() + 0.8, -0.7, "scorer's log-odds that the response is safe", fontsize=7 * F, color=INK, ha="right", va="top")
    ax.set_title("where the certified thresholds fall", fontsize=11.5 * F, color=INK, pad=40)
    ax.text(0.5, 1.01, "pooled (dashed): one threshold certified on the whole pool; stratified (solid): one per prompt stratum.\nEither way the kept set is at most $\\alpha$ harmful with probability $1-\\delta$, or the certificate refuses.", ha="center", va="bottom", fontsize=7.4 * F, color=GREY, transform=ax.transAxes, linespacing=1.3)

    fig.canvas.draw(); axl = fig.get_axes()
    for a, b in ((axl[0], axl[1]),):
        x0, x1 = a.get_position().x1, b.get_position().x0; mid = 0.5 * (x0 + x1)
        fig.add_artist(FancyArrowPatch((mid - 0.045, 0.50), (mid - 0.009, 0.50), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20, lw=1.8, color=INK))
    fig.savefig(f"{BASE}/figures/concept.pdf", bbox_inches="tight"); fig.savefig(f"{BASE}/figures/concept.png", dpi=200, bbox_inches="tight")
    json.dump({"examples": [{"group": n, "index": int(i), "score": float(g[i]), "prompt": pool[i]["prompt"], "response": pool[i]["response"]} for n, i, _ in picks],
               "scorer_right_panel": "Llama Guard 3", "tau_pool": float(tau_pool), "tau_benign": float(tau_b), "tau_harmful": float(tau_h)}, open(f"{W}/results/concept_examples.json", "w"), indent=1)
    print("saved figures/concept.{pdf,png}")


if __name__ == "__main__":
    main()
