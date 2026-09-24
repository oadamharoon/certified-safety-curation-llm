"""E13: arm-set completeness across the two scales (pre-registered 2026-09-16).
8B (selections_e3/lambda0.25): certprompt = our certificate on SAFT's Prompting judge score
(alpha .10, n 400; the 8B analogue of the 0.5B judge-certified arm); certstrat_saft and
certstrat_seal = the stratified certificate (alpha_h .20 / alpha_b .15, n 800) on the SAFT score
(validated layer/k of seed s) and on the SEAL selector logit (budget-matched selector when present).
0.5B (selections/): saft05 = SAFT reimplemented with Qwen2.5-0.5B embeddings (layer 12 of 24, k and
tau by F1 on 100 labels, as at 8B) on the step-2 pool; certsaft05 = its score under our certificate.
Run in the PY_LLM interpreter (0.5B embeddings on GPU, ~2 GB)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import W, ltt_walk, cp_fallback, build_pool, load_split
from sklearn.metrics import f1_score, roc_auc_score


def strat(g, unsafe, req, s, N):
    cal = np.random.default_rng(1000 + s).choice(N, 800, replace=False); keep = np.zeros(N, bool); ok_all = True
    for stratum, a in ((req, 0.20), (~req, 0.15)):
        ids = np.where(stratum)[0]; pos = {i: j for j, i in enumerate(ids)}; cl = np.array([pos[c] for c in cal if stratum[c]])
        ok, mask, q, _ = ltt_walk(g[ids], unsafe[ids], cl, a, 0.05)
        if ok: keep[ids[mask]] = True
        else: ok_all = False; fm, lb = cp_fallback(g[ids], unsafe[ids], cl, 0.05); keep[ids[fm]] = True
    sh = req[keep].mean(); return keep, {"certified": ok_all, "overall_bound": float(0.20 * sh + 0.15 * (1 - sh))}


def saver(D, summ, unsafe, demo):
    def save(arm, s, idx, meta):
        idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
        json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
        print(f"  {arm:15s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} {meta.get('certified','')}", flush=True)
    return save


# ---------------- 8B
D = f"{W}/selections_e3/lambda0.25"; pool = json.load(open(f"{D}/pool.json")); N = len(pool)
unsafe = np.array([not r["is_safe"] for r in pool]); S = json.load(open(f"{D}/scores.json")); req = np.array(S["harmful_request"]); demo = req & ~unsafe
summ = json.load(open(f"{D}/summary.json")); save = saver(D, summ, unsafe, demo)
pr = np.array(S["prompting_yes_logodds"]); gp = -pr                      # Prompting judge: higher = safer
Zs = np.load(f"{D}/saft_embeddings.npz")
def saft_scores(Z, k):
    Zc = Z - Z.mean(0, keepdims=True); _, _, Vt = np.linalg.svd(Zc, full_matrices=False); return (Zc @ Vt[:k].T) ** 2 @ np.ones(k) / k
seal_file = "seal_scores_long.json" if os.path.exists(f"{D}/seal_scores_long.json") else "seal_scores.json"
gseal = np.array(json.load(open(f"{D}/{seal_file}"))["logits"]); print("SEAL scores from", seal_file, flush=True)
for s in range(3):
    cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False); ok, mask, q, _ = ltt_walk(gp, unsafe, cal, 0.10, 0.10)
    if ok: save("certprompt", s, np.where(mask)[0], {"certified": True, "q": q, "scorer": "Prompting judge"})
    else: fm, lb = cp_fallback(gp, unsafe, cal, 0.10); save("certprompt", s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb, "scorer": "Prompting judge"})
    m = summ["saft"][str(s)]; gs_ = -saft_scores(Zs[f"layer{m['layer']}"], m["k"])
    keep, meta = strat(gs_, unsafe, req, s, N); save("certstrat_saft", s, np.where(keep)[0], dict(meta, layer=m["layer"], k=m["k"]))
    keep, meta = strat(gseal, unsafe, req, s, N); save("certstrat_seal", s, np.where(keep)[0], dict(meta, selector=seal_file))
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1)

# ---------------- 0.5B: SAFT with the 0.5B model's embeddings on the step-2 pool
pool = build_pool(load_split("train"), 4000, 0.25, 0); N = len(pool); unsafe = np.array([not r["is_safe"] for r in pool])
wg = json.load(open(f"{W}/results/scores_wildguard_pool.json")); req = np.array(wg["harmful_request"]); demo = req & ~unsafe
ep = f"{W}/results/saft05_embeddings.npz"
if not os.path.exists(ep):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    BASE = "Qwen/Qwen2.5-0.5B"; tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "right"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval(); L = model.config.num_hidden_layers // 2
    Z = np.zeros((N, model.config.hidden_size), np.float32)
    with torch.no_grad():
        for i in range(0, N, 16):
            rs = pool[i:i + 16]; heads = [f"User: {r['prompt']}\nAssistant:" for r in rs]     # the base model has no chat template; the step-2 SFT format
            enc = tok([h + " " + r["response"] for h, r in zip(heads, rs)], return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            pos = [min(len(tok(h)["input_ids"]), 512) - 1 for h in heads]
            hs = model(**enc, output_hidden_states=True).hidden_states[L]
            Z[i:i + 16] = hs[torch.arange(len(rs)), torch.tensor(pos)].float().cpu().numpy()
    del model; torch.cuda.empty_cache(); np.savez(ep, layer=Z, L=L)
Z = np.load(ep)["layer"]; scores = {k: saft_scores(Z, k) for k in (1, 2, 3, 4)}
print("SAFT at 0.5B, AUROC(harmful) by k:", {k: round(float(roc_auc_score(unsafe, v)), 3) for k, v in scores.items()}, flush=True)
summ2 = json.load(open(f"{W}/results/step2_selections.json")); save2 = saver(f"{W}/selections", summ2, unsafe, demo)
for s in range(3):
    val = np.random.default_rng(500 + s).choice(N, 100, replace=False); best = None
    for k, sc in scores.items():
        a, b = sc[val].min(), sc[val].max()
        for tau in a + (b - a) / 100 * np.arange(100):
            f1 = f1_score(unsafe[val], sc[val] > tau, zero_division=0)
            if best is None or f1 > best[0]: best = (f1, k, float(tau))
    f1, k, tau = best; sc = scores[k]
    save2("saft05", s, np.where(sc <= tau)[0], {"rule": "SAFT: keep s_i <= tau", "k": k, "tau": tau, "val_f1": float(f1), "flagged_frac": float(np.mean(sc > tau))})
    cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False); ok, mask, q, _ = ltt_walk(-sc, unsafe, cal, 0.10, 0.10)
    if ok: save2("certsaft05", s, np.where(mask)[0], {"certified": True, "q": q, "k": k})
    else: fm, lb = cp_fallback(-sc, unsafe, cal, 0.10); save2("certsaft05", s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb, "k": k})
json.dump(summ2, open(f"{W}/results/step2_selections.json", "w"), indent=1); print("E13 SELECT DONE")
