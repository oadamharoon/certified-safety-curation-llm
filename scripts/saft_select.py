"""SAFT (Choi, Du, Li 2024, arXiv 2410.10014) reimplemented on our E3 pool, plus the same score
under our certificate.

SAFT as published (Sec. 4, App. A): embed every pool example with the model to be fine-tuned,
taking the hidden state at the START of the response section (their stated choice; we use
the last token of the assistant header under Llama-3.1-8B-Instruct's chat template); center
the embedding matrix; SVD; filtering score s_i = (1/k) sum_{j<=k} <z_i, v_j>^2; flag as
harmful when s_i > tau. k in {1,2,3,4} and tau (100 candidate thresholds from min to max
score) are validated on 100 labeled held-out samples by F1 against the human labels. Layer:
their Fig. 4 finds median layers best (layer 15 of 32 on Llama-2); we use layer 15 of 32 and
record the last layer as an ablation. Not specified by SAFT and fixed here: the 100 labeled
samples are drawn from the pool and stay in the fine-tuning set (their labels set tau only),
so the SAFT arm and our certified arms spend labels on the same pool; F1 is the validation
criterion. Arms written to selections_e3/lambda{L}/:
  saft15_s{s}    keep s_i <= tau at layer 15 (as published)
  saft_s{s}      same, with the layer in {15, 32} validated on the same 100 labels
  certsaft_s{s}  our certificate on the SAFT score (-s_i as the safety score), alpha .10,
                 n 400, the same calibration draws as certlg
Runs in the certcurate-llm env. Usage: LAMBDA=0.25 python saft_select.py
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import W, ltt_walk, cp_fallback
from sklearn.metrics import roc_auc_score, f1_score

LAMS = os.environ.get("LAMBDA", "0.25"); D = f"{W}/selections_e3/lambda{LAMS}"
BASE = os.environ.get("E3_BASE", "meta-llama/Llama-3.1-8B-Instruct"); LAYERS = (15, 32)
pool = json.load(open(f"{D}/pool.json")); N = len(pool)
unsafe = np.array([not r["is_safe"] for r in pool]); S = json.load(open(f"{D}/scores.json"))
req = np.array(S["harmful_request"]); demo = req & ~unsafe
ep = f"{D}/saft_embeddings.npz"
if not os.path.exists(ep):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE); tok.padding_side = "right"; tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    Z = {L: np.zeros((N, model.config.hidden_size), np.float32) for L in LAYERS}
    with torch.no_grad():
        for i in range(0, N, 8):
            rs = pool[i:i + 8]
            heads = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in rs]
            full = [h + r["response"] for h, r in zip(heads, rs)]
            enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False).to("cuda")
            pos = [min(len(tok(h, add_special_tokens=False)["input_ids"]), 512) - 1 for h in heads]   # last token of the assistant header
            hs = model(**enc, output_hidden_states=True).hidden_states
            for L in LAYERS:
                Z[L][i:i + 8] = hs[L][torch.arange(len(rs)), torch.tensor(pos)].float().cpu().numpy()
            if i % 400 == 0:
                print(f"  embedded {i}/{N}", flush=True)
    del model; torch.cuda.empty_cache()
    np.savez(ep, **{f"layer{L}": Z[L] for L in LAYERS})
Zs = np.load(ep)


def saft_scores(Z, k):
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return (Zc @ Vt[:k].T) ** 2 @ np.ones(k) / k


summ = json.load(open(f"{D}/summary.json")) if os.path.exists(f"{D}/summary.json") else {}
out = {}
for L in LAYERS:
    Z = Zs[f"layer{L}"]
    for k in (1, 2, 3, 4):
        s = saft_scores(Z, k); out[f"layer{L}_k{k}"] = {"auc_harmful": float(roc_auc_score(unsafe, s)), "scores": s.tolist()}
        print(f"layer {L} k={k}: AUROC(harmful) {out[f'layer{L}_k{k}']['auc_harmful']:.3f}", flush=True)
json.dump({k: v["auc_harmful"] for k, v in out.items()}, open(f"{D}/saft_auc.json", "w"), indent=1)


def save(arm, s, idx, meta):
    idx = [int(i) for i in idx]; meta = dict(meta, n=len(idx), unsafe_frac=float(unsafe[idx].mean()), safety_demo_share=float(demo[idx].mean()))
    json.dump({"idx": idx, "meta": meta}, open(f"{D}/{arm}_s{s}.json", "w")); summ.setdefault(arm, {})[str(s)] = meta
    print(f"  {arm:9s} s{s}: n={len(idx):4d} unsafe={meta['unsafe_frac']:.3f} demos={meta['safety_demo_share']:.3f} {meta}", flush=True)


def validate(s, layers):
    """SAFT's validation on 100 labeled samples: k in {1,2,3,4} and tau (100 thresholds) by F1;
    the layer is validated the same way when more than one is offered."""
    val = np.random.default_rng(500 + s).choice(N, 100, replace=False)
    best = None
    for L in layers:
        for k in (1, 2, 3, 4):
            sc = np.array(out[f"layer{L}_k{k}"]["scores"]); a, b = sc[val].min(), sc[val].max()
            for tau in a + (b - a) / 100 * np.arange(100):
                f1 = f1_score(unsafe[val], sc[val] > tau, zero_division=0)
                if best is None or f1 > best[0]:
                    best = (f1, L, k, float(tau))
    return best


for s in range(3):
    # saft15: as published (layer 15, their best layer on Llama-2); saft: the layer validated on
    # the same 100 labels as k and tau (more favorable to SAFT on Llama-3.1, where the last
    # layer separates better: AUROC .61 vs .56). certsaft uses the validated (layer, k).
    for arm, layers in (("saft15", (15,)), ("saft", LAYERS)):
        f1, L, k, tau = validate(s, layers); sc = np.array(out[f"layer{L}_k{k}"]["scores"])
        save(arm, s, np.where(sc <= tau)[0], {"rule": "SAFT: keep s_i <= tau", "layer": L, "k": k, "tau": tau, "val_f1": float(f1),
             "flagged_frac": float(np.mean(sc > tau)), "flag_recall_unsafe": float(np.mean(sc[unsafe] > tau)), "flag_rate_safe": float(np.mean(sc[~unsafe] > tau))})
    g = -np.array(out[f"layer{L}_k{k}"]["scores"])                             # safety score for the certificate: low SAFT score = safe
    cal = np.random.default_rng(1000 + s).choice(N, 400, replace=False)
    ok, mask, q, _ = ltt_walk(g, unsafe, cal, 0.10, 0.10)
    if ok: save("certsaft", s, np.where(mask)[0], {"certified": True, "q": q, "layer": L, "k": k})
    else: fm, lb = cp_fallback(g, unsafe, cal, 0.10); save("certsaft", s, np.where(fm)[0], {"certified": False, "cp_lower_bound": lb, "layer": L, "k": k})
json.dump(summ, open(f"{D}/summary.json", "w"), indent=1)
print("SAFT SELECT DONE")
