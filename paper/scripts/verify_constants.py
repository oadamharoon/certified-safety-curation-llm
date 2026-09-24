"""Source trace for every number in the paper's prose that is NOT a cell of a generated table.

audit_prose.py checks that a value exists somewhere in results/; that is existence-matching, and it
cannot catch a stale derived statistic (it passed a Spearman range that had not been recomputed since
the condition set changed). This file binds each such number to the one expression that produces it,
so a number that drifts from its source fails here. Every check names the sentence it guards.
Usage: python paper/scripts/verify_constants.py   (exit 1 on any mismatch)
"""
import os
import json, re, sys
import numpy as np

W = (os.environ.get("CDC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
L = lambda p: json.load(open(f"{W}/{p}"))
sel = L("results/step2_selections.json")
e3s = {l: L(f"selections_e3/lambda{l}/summary.json") for l in ("0.10", "0.25", "0.30")}
ten, ten8 = L("results/tension.json"), L("results/tension_8b.json")
cats = L("results/categories_by_prompt.json")
ja, e5, tr = L("results/judge_agreement.json"), L("results/e5_summary.json"), L("results/harmsets_transfer.json")
gate7n = L("results/llm_certify_Qwen2.5-7B-Instruct_nat.json")

sys.path.insert(0, f"{W}/scripts")
from common import main_pool                      # the study's pool, dataset or cache
_pool = main_pool()
_unsafe = np.array([not r["is_safe"] for r in _pool])
_wg = L("results/scores_wildguard_pool.json"); _req = np.array(_wg["harmful_request"]); _ref = np.array(_wg["refusal"])
_g = np.array(L("results/scores_Qwen2.5-7B-Instruct_N4000_u0.25.json"))          # judge log-odds on the pool
_lg = -np.array(L("results/scores_llamaguard_pool.json")["score_unsafe_minus_safe"])
_GRID = (0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)


def _stratum_floor():
    """lowest harmful fraction any grid cutoff of Llama Guard 3 reaches inside the harmful-prompt stratum"""
    return min(float(_unsafe[_req][_lg[_req] >= np.quantile(_lg[_req], q)].mean()) for q in _GRID)


def _facts():
    """Dataset-level counts, computed from the sources when present and cached beside the results
    so this audit runs from the archive alone."""
    cache = f"{W}/results/dataset_facts.json"
    import glob
    if not glob.glob(f"{W}/data/hf/alpaca/data/*.parquet"):
        return json.load(open(cache))
    import pyarrow.parquet as pq
    from common import load_split, split_counts
    keep = {(r["instruction"], r["input"], r["output"]) for r in
            pq.read_table(glob.glob(f"{W}/data/hf/alpagasus/data/*.parquet")[0]).to_pylist()}
    alp = pq.read_table(glob.glob(f"{W}/data/hf/alpaca/data/*.parquet")[0]).to_pylist()
    low = sum(1 for r in alp if (r["instruction"], r["input"], r["output"]) not in keep)
    refs = {r["prompt"] for r in load_split("test") if r["is_safe"]}
    ev = L("results/eval_sets.json")["gen_prompts"]
    pku = [json.loads(l) for l in open(f"{W}/data/hf/PKU-SafeRLHF/data/Alpaca-7B/train.jsonl")]
    sc = split_counts()
    out = {"train_n": sc["train"]["n"], "train_harmful_frac": sc["train"]["harmful_frac"],
           "with_reference": sum(1 for p in ev if p in refs),
           "pku_shared": len({r["prompt"] for r in pku if r["prompt"] in set(ev)}),
           "alpagasus_kept": len(keep), "alpaca_n": len(alp), "alpaca_low_frac": low / len(alp)}
    json.dump(out, open(cache, "w"), indent=1)
    return out


_F = _facts()


def _flagged(kind):
    """share of the 8B pool's safe / harmful examples the Prompting filter removes"""
    r = e3s["0.25"]["prompting"]["0"]; n, u = r["n"], r["unsafe_frac"]
    H, S = 3000 * 0.25, 3000 * 0.75
    return 1 - (n - u * n) / S if kind == "safe" else 1 - (u * n) / H

def seeds(d, k, f):   # per-seed field of a selection, in seed order
    return [d[k][s][f] for s in sorted(d[k], key=int)]

CHECKS = [
    # (paper sentence, claimed, actual, tolerance)
    ("Sec 4: BeaverTails train is 27k pairs", 27186, _F["train_n"], 0),
    ("Sec 4: BeaverTails train is 57 percent harmful", 0.57,
     _F["train_harmful_frac"], 0.005),
    ("Sec 5: the 7B judge's top 15 percent of the natural pool is 18 percent harmful", 0.18, gate7n["alpha0.25"]["mean_contam"], 0.005),
    ("Sec 5: the certificate succeeds on a quarter of the draws", 0.25, gate7n["alpha0.25"]["cert_rate"], 0.01),
    ("Sec 5: all of them correct", 0.0, gate7n["alpha0.25"]["false_cert"], 1e-9),
    ("Sec 5: closed form within 0.010 mean absolute error", 0.010, e5["mae"], 0.0005),
    ("Sec 5: over 126 settings", 126, e5["cells"], 0),
    ("Sec 5: Spearman 0.989", 0.989, e5["spearman"], 0.0005),
    ("Sec 5: false certification at or below 0.085", 0.085, e5["max_false"], 1e-9),
    ("Sec 6: certified set's first seed is 1400 examples", 1400, seeds(sel, "cert400", "n")[0], 0),
    ("Sec 6: first seed 6.6 percent harmful", 0.066, seeds(sel, "cert400", "unsafe_frac")[0], 0.0005),
    ("Sec 5: four of five certified selections span 5.7 to 6.6 percent", (0.057, 0.066),
     (min(v for v in seeds(sel, "cert400", "unsafe_frac") if v < 0.1), max(v for v in seeds(sel, "cert400", "unsafe_frac") if v < 0.1)), 0.0005),
    ("Sec 5: the fifth is 10.4 percent", 0.104, max(seeds(sel, "cert400", "unsafe_frac")), 0.0005),
    ("Sec 6: Llama Guard 3 selections are 9.8 percent harmful", 0.098, float(np.mean(seeds(sel, "certlg400", "unsafe_frac"))), 0.0005),
    ("Sec 6: the fixed cutoff kept a set 13.9 percent harmful", 0.139, float(np.mean(seeds(sel, "lgfilter", "unsafe_frac"))), 0.0005),
    ("Sec 7: the stratified selection keeps 2584 to 2817 examples", (2584, 2817),
     (min(seeds(sel, "certstrat2_lg800", "n")), max(seeds(sel, "certstrat2_lg800", "n"))), 0),
    ("App K: SAFT at 0.5B keeps 511, 1104 and 2345 examples", [511, 1104, 2345], seeds(sel, "saft05", "n"), 0),
    ("App K: SAFT at 0.5B is 22 percent harmful", 0.22, float(np.mean(seeds(sel, "saft05", "unsafe_frac"))), 0.005),
    ("Table 9: the within-stratum test has 92 harmful examples", 92,
     round(seeds(sel, "judgetop05", "n")[0] * seeds(sel, "judgetop05", "unsafe_frac")[0]), 0),
    ("Sec 9: SAFT's seeds keep 1004, 31 and 2519 of 3000", [1004, 31, 2519], seeds(e3s["0.25"], "saft", "n"), 0),
    ("App J: two published-layer seeds keep 24 and 25 examples at rho 0.30", [24, 25],
     sorted(seeds(e3s["0.30"], "saft15", "n"))[:2], 0),
    ("App J: SAFT keeps 1669 to 2773 of 3000 at rho 0.10", (1669, 2773),
     (min(seeds(e3s["0.10"], "saft", "n") + seeds(e3s["0.10"], "saft15", "n")),
      max(seeds(e3s["0.10"], "saft", "n") + seeds(e3s["0.10"], "saft15", "n"))), 0),
    ("Sec 9: labeled-only is 302 to 310 human-safe examples", (302, 310),
     (min(seeds(e3s["0.25"], "labels", "n")), max(seeds(e3s["0.25"], "labels", "n"))), 0),
    ("Table 8: the certified Llama Guard 3 set is 1584 examples, 8.3 percent", (1584, 0.083),
     (round(float(np.mean(seeds(e3s["0.25"], "certlg", "n")))), float(np.mean(seeds(e3s["0.25"], "certlg", "unsafe_frac")))), 0.0005),
    ("Table 8: 132 harmful examples in the within-stratum rows", 132,
     round(seeds(e3s["0.25"], "harmtop8", "n")[0] * seeds(e3s["0.25"], "harmtop8", "unsafe_frac")[0]), 0),
    ("App A/Sec 6: c is between 2.0 and 2.1 on the four certified seeds meeting alpha", (2.0, 2.1),
     (min(v["c"] for v in ten["seeds"].values() if v["ahat"] <= 0.10), max(v["c"] for v in ten["seeds"].values() if v["ahat"] <= 0.10)), 0.05),
    ("Sec 6: the bound gives a share of at most 0.37", 0.37,
     max(v["bound"] for v in ten["seeds"].values() if v["ahat"] <= 0.10), 0.005),
    ("Sec 6: against a pool share of 0.54", 0.54, ten["pi"], 0.005),
    ("App K: the 8B judge retains demonstrations 0.58 to 0.60 times as often", (0.58, 0.60),
     (min(v["c"] for v in ten8["seeds"].values()), max(v["c"] for v in ten8["seeds"].values())), 0.005),
    ("App K: any qualifying selection keeps a demonstration share below 0.11", 0.11,
     max(v["bound"] for v in ten8["seeds"].values()), 0.005),
    ("App K: against the pool's 0.55", 0.55, ten8["pi"], 0.005),
    ("App C: 115,000 generations at 0.5B", 115000, ja["0.5B"]["n_generations"], 500),
    ("App C: 75,000 generations at 8B", 75000, ja["8B"]["n_generations"], 500),
    ("App C: 230 models at 0.5B", 230, ja["0.5B"]["n_models"], 0),
    ("App C: 150 models at 8B", 150, ja["8B"]["n_models"], 0),
    ("App D: BeaverTails category mix of the 500 evaluation prompts", [121, 115, 62, 51],
     [cats["n_prompts"]["violence,aiding_and_abetting,incitement"], cats["n_prompts"]["non_violent_unethical_behavior"],
      cats["n_prompts"]["discrimination,stereotype,injustice"], cats["n_prompts"]["hate_speech,offensive_language"]], 0),
    # the clause that justifies drawing HEx-PHI as a radar and BeaverTails as a heatmap:
    # equal angular sectors are honest only where the categories are equally sized
    ("App D: the plotted BeaverTails categories run 10 to 121 prompts", (10, 121),
     (min(v for v in cats["n_prompts"].values() if v >= 10),
      max(cats["n_prompts"].values())), 0),
    ("Sec 9/App I: benchmark transfer Spearman 0.97 to 0.99 per judge", (0.97, 0.99),
     (tr["8b"]["pooled_min"], tr["8b"]["pooled_max"]), 0.006),
    ("App I: two of twelve (judge, set) pairs fall below 0.9, at 0.886 and 0.891", 2, tr["8b"]["clean_judges_below_0.9"], 0),
    ("App I: 0.5B transfer spans 0.053 to 0.753", (0.053, 0.753),
     (tr["05b"]["clean_judges_min"], tr["05b"]["clean_judges_max"]), 0.0005),
    # --- quantities computed from the pool and its scores, not stored in any results file ---
    ("Sec 6: safe answers to benign prompts score 16.9 on the judge's log-odds scale", 16.9,
     float(_g[~_unsafe & ~_req].mean()), 0.05),
    ("Sec 6: safe answers to harmful prompts score 5.5 when they engage", 5.5,
     float(_g[~_unsafe & _req & ~_ref].mean()), 0.05),
    ("Sec 6: and -3.1 when they refuse", -3.1, float(_g[~_unsafe & _req & _ref].mean()), 0.05),
    ("Sec 6: the human-harmful examples score -2.7", -2.7, float(_g[_unsafe].mean()), 0.05),
    ("Sec 7: the harmful-prompt stratum is 35 percent harmful", 0.35, float(_unsafe[_req].mean()), 0.005),
    ("Sec 7: no cutoff brings that stratum below 10.6 percent", 0.106, _stratum_floor(), 0.0005),
    ("Sec 9: the Prompting judge flags 17 percent of the human-safe examples", 0.17, _flagged("safe"), 0.005),
    ("Sec 9: and 4 percent of the harmful ones", 0.04, _flagged("harmful"), 0.005),
    ("Sec 9: Prompting keeps a set 27.6 percent harmful", 0.276,
     float(np.mean(seeds(e3s["0.25"], "prompting", "unsafe_frac"))), 0.0005),
    ("App K: the 0.5B SAFT fallback is 26 percent harmful", 0.26,
     float(np.mean(seeds(sel, "certsaft05", "unsafe_frac"))), 0.005),
    ("App C: all four judges agree on 71 percent of 0.5B generations", 0.71, ja["0.5B"]["all_four_agree"], 0.005),
    ("App C: and 75 percent at 8B", 0.75, ja["8B"]["all_four_agree"], 0.005),
    ("Sec 4: 500 fixed held-out BeaverTails prompts", 500, len(L("results/eval_sets.json")["gen_prompts"]), 0),
    ("App E: 229 held-out prompts have at least one human-safe reference", 229, _F["with_reference"], 0),
    ("App G: 152 PKU-SafeRLHF prompts are shared with the evaluation prompts and excluded", 152, _F["pku_shared"], 0),
    ("App H: the AlpaGasus reimplementation keeps 9,229 examples, making 82 percent of Alpaca low quality",
     (9229, 0.82), (_F["alpagasus_kept"], _F["alpaca_low_frac"]), 0.005),
]

fails = 0
for name, claimed, actual, tol in CHECKS:
    if isinstance(claimed, (tuple, list)):
        ok = len(claimed) == len(actual) and all(abs(c - a) <= tol for c, a in zip(claimed, actual))
    else:
        ok = abs(claimed - actual) <= tol
    if not ok: fails += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n        paper says {claimed}, source gives {actual}"))
print(f"\nCONSTANTS: {'PASS' if not fails else f'{fails} MISMATCH'} ({len(CHECKS)} checks)")
sys.exit(1 if fails else 0)
