"""E5: label complexity on text, with the exact Prop 4 prediction overlaid.

For each pool (contamination 0.10 / 0.25 / natural 0.57), scorer (Qwen 1.5B / 7B judge),
and alpha in {0.10, 0.25}: certification rate over 200 calibration draws at
n in {50, 100, 200, 300, 400, 600, 800}, next to the exact closed-form Pr[certify]
of the source certificate paper (its Prop 4, cert_rate_theory.pr_certify) computed from
(N, N1, K1) of the first grid selection. Also the false-certification rate per cell.
Writes results/e5_label_complexity.json and paper/figures/label_complexity.pdf.
"""
import os
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from cert_rate_theory import pr_certify
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = load_split("train")
NS = [50, 100, 200, 300, 400, 600, 800]; DRAWS = 200
POOLS = [("nat", None), ("u0.25", 0.25), ("u0.10", 0.10)]
JUDGES = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct"]
out = {}
for ptag, pu in POOLS:
    rng = np.random.default_rng(0)
    pool = build_pool(rows, 4000, pu, 0) if pu is not None else [rows[i] for i in rng.choice(len(rows), 4000, replace=False)]
    unsafe = np.array([not r["is_safe"] for r in pool]); N = len(pool)
    for judge in JUDGES:
        g = np.array(json.load(open(f"{W}/results/scores_{judge}_N4000_{ptag}.json")))
        mask1 = g >= np.quantile(g, QS[0]); N1 = int(mask1.sum()); K1 = int(unsafe[mask1].sum())
        for alpha in (0.10, 0.25):
            key = f"{ptag}|{judge.split('-')[1]}|a{alpha}"
            rec = {"N": N, "N1": N1, "K1": K1, "u1": K1 / N1, "margin": alpha - K1 / N1, "n": NS, "rate": [], "false": [], "pred": []}
            for n in NS:
                r2 = np.random.default_rng(1); nc = nf = 0
                for _ in range(DRAWS):
                    ok, m, q, _ = ltt_walk(g, unsafe, r2.choice(N, n, replace=False), alpha, 0.10)
                    if ok:
                        nc += 1; nf += unsafe[m].mean() > alpha
                rec["rate"].append(nc / DRAWS); rec["false"].append(nf / DRAWS)
                rec["pred"].append(pr_certify(N, N1, K1, n, alpha=alpha, delta=0.10))
            out[key] = rec
            print(f"{key:28s} u1={rec['u1']:.3f} margin={rec['margin']:+.3f} | " +
                  " ".join(f"n{n}:{r:.2f}/{p:.2f}" for n, r, p in zip(NS, rec["rate"], rec["pred"])) +
                  f" | worst false {max(rec['false']):.3f}", flush=True)
json.dump(out, open(f"{W}/results/e5_label_complexity.json", "w"), indent=1)

# figure: measured (markers) vs exact prediction (lines), one panel per alpha
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
colors = {"nat": "C3", "u0.25": "C1", "u0.10": "C0"}; styles = {"1.5B": "--", "7B": "-"}
for ax, alpha in zip(axes, (0.10, 0.25)):
    for key, rec in out.items():
        ptag, jd, a = key.split("|")
        if a != f"a{alpha}": continue
        ax.plot(rec["n"], rec["pred"], styles[jd], color=colors[ptag], lw=1.2)
        ax.plot(rec["n"], rec["rate"], "o" if jd == "7B" else "s", color=colors[ptag], ms=4, mfc="none" if jd == "1.5B" else None)
    ax.set_title(f"$\\alpha = {alpha}$"); ax.set_xlabel("calibration labels $n$"); ax.set_xscale("log"); ax.grid(alpha=.3)
axes[0].set_ylabel("certification rate")
from matplotlib.lines import Line2D
h = [Line2D([], [], color=c, lw=1.2, label=f"pool unsafe {l}") for l, c in (("0.57", "C3"), ("0.25", "C1"), ("0.10", "C0"))]
h += [Line2D([], [], color="k", ls="-", marker="o", ms=4, label="7B judge"), Line2D([], [], color="k", ls="--", marker="s", ms=4, mfc="none", label="1.5B judge")]
axes[1].legend(handles=h, fontsize=7, loc="lower right", frameon=False)
fig.tight_layout(); os.makedirs(f"{W}/paper/figures", exist_ok=True)
fig.savefig(f"{W}/paper/figures/label_complexity.pdf"); fig.savefig(f"{W}/paper/figures/label_complexity.png", dpi=160)
# one-number summary of prediction quality
meas = np.array([r for rec in out.values() for r in rec["rate"]]); pred = np.array([p for rec in out.values() for p in rec["pred"]])
from scipy.stats import spearmanr
print(f"\nProp 4 prediction vs measured over {len(meas)} cells: Spearman {spearmanr(pred, meas).statistic:+.3f}, MAE {np.abs(pred - meas).mean():.3f}, "
      f"worst false-cert anywhere {max(max(r['false']) for r in out.values()):.3f}")
