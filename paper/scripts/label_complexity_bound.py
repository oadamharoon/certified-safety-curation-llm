"""Proposition (label complexity): the exact smallest n reaching a target certification rate,
its closed-form sufficient budget, and the measured first budget on the E5 grid, per cell.
Cells with a nonpositive margin have no finite n* and are reported as such.
Output: paper/data/tables/label_bound.tex and results/label_bound.json."""
import os
import json, os, sys, math
import numpy as np
W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")); from cert_rate_theory import pr_certify
DELTA, TAU = 0.10, 0.5
d = json.load(open(f"{W}/results/e5_label_complexity.json"))


def n_star(N, N1, K1, alpha, tau, cap=4000):
    for n in range(10, cap + 1, 10):
        if pr_certify(N, N1, K1, n, alpha=alpha, delta=DELTA) >= tau:
            return n
    return None


def bound(N, N1, eps, tau):
    p = N1 / N; L = math.log(2 / (1 - tau)); M = 2 * max(math.log(1 / DELTA), L) / eps ** 2
    # second term 2L/phi^2 (not L/(2 phi^2)): the unconditional form, see the proof in App. A (fixed 2026-09-18 audit)
    return math.ceil(2 * M / p + 2 * L / (p * p))


rows, out = [], {}
for key, rec in d.items():
    ptag, jd, a = key.split("|"); alpha = float(a[1:]); eps = rec["margin"]
    meas = next((n for n, r in zip(rec["n"], rec["rate"]) if r >= TAU), None)
    if eps <= 0:
        ns, bd = None, None
    else:
        ns, bd = n_star(rec["N"], rec["N1"], rec["K1"], alpha, TAU), bound(rec["N"], rec["N1"], eps, TAU)
    out[key] = {"eps": eps, "n_star": ns, "bound": bd, "measured_first": meas}
    lab = {"nat": "0.57", "u0.25": "0.25", "u0.10": "0.10"}[ptag]
    f = lambda v: "--" if v is None else (f"{v}" if v < 100000 else "$>10^5$")
    rows.append(f"{lab} & {'Llama Guard 3' if jd == 'LG' else jd} & {alpha:.2f} & {eps:+.3f} & {f(ns)} & {f(bd)} & {f(meas)} \\\\")
    print(key, out[key])
json.dump(out, open(f"{W}/results/label_bound.json", "w"), indent=1)
body = ("\\begin{tabular}{llrrrrr}\n\\toprule\npool harmful & scorer & $\\alpha$ & margin $\\epsilon$ & $n^{*}(0.5)$ exact & sufficient budget & first grid $n$ with rate $\\ge 0.5$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
open(f"{W}/paper/data/tables/label_bound.tex", "w").write(body); print("wrote label_bound.tex")
