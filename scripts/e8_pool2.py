"""E8-lite: a second contaminated pool, PKU-SafeRLHF (Alpaca-7B responses, human safety labels),
to replicate the diagnosis and the stratified method at 0.5B. Pool of 4000 at 25 percent harmful
(one response per prompt; prompts that appear among the 500 BeaverTails evaluation prompts are
excluded, since PKU-SafeRLHF and BeaverTails share prompt sources). Scorers: Qwen2.5-7B-Instruct
judge (log-odds), Llama Guard 3, WildGuard prompt label for strata. Arms per seed s in {0,1,2}:
p2_full, p2_random (size of p2_cert400), p2_cert400 (judge, alpha .10, n 400), p2_certlg400,
p2_certstrat2 (LG, .20/.15, n 800), p2_oracle. Selections in selections/p2_*.json (indices into
results/pool2.json); metadata appended to results/step2_selections.json. Runs under PY."""
import json, os, sys, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
rows = [json.loads(l) for l in open(f"{W}/data/hf/PKU-SafeRLHF/data/Alpaca-7B/train.jsonl")]
ev = set(json.load(open(f"{W}/results/eval_sets.json"))["gen_prompts"])
seen, items = set(), []
for r in rows:
    if r["prompt"] in ev or r["prompt"] in seen: continue
    seen.add(r["prompt"]); k = int(hashlib.md5(r["prompt"].encode()).hexdigest(), 16) % 2   # deterministic pick of one response
    items.append({"prompt": r["prompt"], "response": r[f"response_{k}"], "is_safe": r[f"is_response_{k}_safe"] in (True, "True")})
print(f"PKU-SafeRLHF: {len(rows)} rows -> {len(items)} unique prompts not in the eval set; safe share {np.mean([x['is_safe'] for x in items]):.3f}", flush=True)
pool = build_pool(items, 4000, 0.25, 0); unsafe = np.array([not r["is_safe"] for r in pool])
os.makedirs(f"{W}/results", exist_ok=True); json.dump(pool, open(f"{W}/results/pool2.json", "w"))
sp = f"{W}/results/scores_pool2.json"
if os.path.exists(sp):
    S = json.load(open(sp))
else:
    g = judge_scores("Qwen/Qwen2.5-7B-Instruct", pool); lg_sc, lg_fl = llama_guard_scores(pool); req, ref, harm, raw = wildguard_labels(pool)
    S = {"qwen7b": g.tolist(), "lg_safe_score": (-lg_sc).tolist(), "lg_flag": lg_fl.tolist(), "harmful_request": req.tolist(), "refusal": ref.tolist()}
    json.dump(S, open(sp, "w"))
from sklearn.metrics import roc_auc_score
gq = np.array(S["qwen7b"]); gl = np.array(S["lg_safe_score"]); req = np.array(S["harmful_request"]); demo = req & ~unsafe
print(f"pool2: unsafe {unsafe.mean():.3f}, harmful prompts {req.mean():.3f}, demos {demo.mean():.3f}; AUC judge {roc_auc_score(~unsafe, gq):.3f} LG {roc_auc_score(~unsafe, gl):.3f}", flush=True)
summ = json.load(open(f"{W}/results/step2_selections.json"))
def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()), pool="pool2")
    json.dump({"idx": idx, "meta": meta}, open(f"{W}/selections/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:14s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} {meta.get('certified','')}", flush=True)
N = len(pool)
for s in range(3):
    save("p2_full", s, np.arange(N), {}); save("p2_oracle", s, np.where(~unsafe)[0], {})
    for arm, g in (("p2_cert400", gq), ("p2_certlg400", gl)):
        cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False); ok, mask, q, _ = ltt_walk(g, unsafe, cal, 0.10, 0.10)
        if ok: save(arm, s, np.where(mask)[0], {"certified": True, "q": q})
        else: fm, lb = cp_fallback(g, unsafe, cal, 0.10); save(arm, s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb})
    save("p2_random", s, np.random.default_rng(7000 + s).choice(N, summ["p2_cert400"][str(s)]["n"], replace=False), {"rule": "uniform at p2_cert400 size"})
    cal = np.random.default_rng(1000 + s).choice(N, 800, replace=False); keep = np.zeros(N, bool); ok_all = True
    for key, stratum, a in (("h", req, 0.20), ("b", ~req, 0.15)):
        ids = np.where(stratum)[0]; pos = {i: j for j, i in enumerate(ids)}; cl = np.array([pos[c] for c in cal if stratum[c]])
        ok, mask, q, _ = ltt_walk(gl[ids], unsafe[ids], cl, a, 0.05)
        if ok: keep[ids[mask]] = True
        else: ok_all = False; fm, lb = cp_fallback(gl[ids], unsafe[ids], cl, 0.05); keep[ids[fm]] = True
    sh = req[keep].mean(); save("p2_certstrat2", s, np.where(keep)[0], {"certified": ok_all, "overall_bound": float(0.20 * sh + 0.15 * (1 - sh))})
json.dump(summ, open(f"{W}/results/step2_selections.json", "w"), indent=1); print("E8 SELECT DONE")
