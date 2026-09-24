"""E7: a second property at the certificate level. Property u(x) = 1 when an Alpaca example is
LOW QUALITY, defined as absent from the AlpaGasus subset (GPT-4-rated 4.5 or above,
gpt4life's release via mlabonne/alpagasus: 9,229 of the 52,002 Alpaca examples kept). Pools of
4000 at low-quality fraction rho in {0.25, natural (0.82)}; scorer = Qwen2.5-7B-Instruct and
1.5B-Instruct asked whether the response is accurate, complete and helpful (log-odds of Yes);
certificate at alpha in {0.10, 0.25}, delta 0.10, the same grid, n in {50,...,800}, 200 draws
per cell, with the exact closed-form rate beside the measured one (as in E5). No fine-tuning:
this establishes property-agnosticity of the certificate, not a downstream claim.
Writes results/e7_quality.json and paper/data/tables/e7_quality.tex. Runs under PY."""
import os
import json, os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import W, build_pool, ltt_walk, hyp_p
from cert_rate_theory import pr_certify
import pyarrow.parquet as pq
al = pq.read_table(glob.glob(f"{W}/data/hf/alpaca/data/*.parquet")[0]).to_pylist()
keep = set((r["instruction"], r["input"], r["output"]) for r in pq.read_table(glob.glob(f"{W}/data/hf/alpagasus/data/*.parquet")[0]).to_pylist())
rows = [{"prompt": (r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")), "response": r["output"],
         "is_safe": (r["instruction"], r["input"], r["output"]) in keep} for r in al]   # is_safe reused as "high quality" so build_pool/ltt_walk apply unchanged
print(f"Alpaca {len(rows)}; high-quality share {np.mean([r['is_safe'] for r in rows]):.3f}", flush=True)
NS = [50, 100, 200, 300, 400, 600, 800]; DRAWS = 200; QS = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
POOLS = [("q0.25", 0.25), ("qnat", None)]; JUDGES = ["Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct"]
SYS = "You are a strict reviewer of instruction-following data."
def quality_scores(name, items, batch=8):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(f"Qwen/{name}"); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(f"Qwen/{name}", dtype=torch.float16).cuda().eval()
    yes = tok.encode("Yes", add_special_tokens=False)[0]; no = tok.encode("No", add_special_tokens=False)[0]; g = np.zeros(len(items))
    with torch.no_grad():
        for i in range(0, len(items), batch):
            ps = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": f"Instruction: {r['prompt'][:1500]}\n\nResponse: {r['response'][:2500]}\n\nIs the response accurate, complete and helpful for the instruction? Reply with exactly one word: Yes or No."}], tokenize=False, add_generation_prompt=True) for r in items[i:i + batch]]
            enc = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=1536).to("cuda")
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float(); g[i:i + batch] = (lg[:, yes] - lg[:, no]).cpu().numpy()   # last-position logits only: the full vocab x sequence tensor OOMs beside other jobs
    del model; torch.cuda.empty_cache(); return g
out = {}
for ptag, pu in POOLS:
    rng = np.random.default_rng(0)
    pool = build_pool(rows, 4000, pu, 0) if pu is not None else [rows[i] for i in rng.choice(len(rows), 4000, replace=False)]
    low = np.array([not r["is_safe"] for r in pool]); N = len(pool)
    for jd in JUDGES:
        sp = f"{W}/results/scores_quality_{jd}_{ptag}.json"
        if os.environ.get("E7_SMOKE"):
            g = np.random.default_rng(0).normal(size=N) + 2 * np.array([r["is_safe"] for r in pool]); NS = [50, 100]; DRAWS = 5
        else:
            g = np.array(json.load(open(sp))) if os.path.exists(sp) else quality_scores(jd, pool)
            json.dump(g.tolist(), open(sp, "w"))
        from sklearn.metrics import roc_auc_score
        tau1 = np.quantile(g, QS[0]); m1 = g >= tau1; N1, K1 = int(m1.sum()), int(low[m1].sum())
        for alpha in (0.10, 0.25):
            key = f"{ptag}|{jd.split('-')[1]}|a{alpha}"; rec = {"N": N, "N1": N1, "K1": K1, "u1": K1 / N1, "margin": alpha - K1 / N1, "auc": float(roc_auc_score(~low, g)), "n": NS, "rate": [], "false": [], "pred": []}
            for n in NS:
                r = np.random.default_rng(100 + n); cert = 0; false = 0
                for _ in range(DRAWS):
                    cal = r.choice(N, n, replace=False); ok, mask, q, _ = ltt_walk(g, low, cal, alpha, 0.10)
                    cert += ok; false += ok and low[mask].mean() > alpha
                rec["rate"].append(cert / DRAWS); rec["false"].append(false / DRAWS); rec["pred"].append(pr_certify(N, N1, K1, n, alpha=alpha, delta=0.10))
            out[key] = rec
            print(f"{key:26s} auc={rec['auc']:.3f} u1={rec['u1']:.3f} margin={rec['margin']:+.3f} | " + " ".join(f"n{n}:{a:.2f}/{p:.2f}" for n, a, p in zip(NS, rec["rate"], rec["pred"])) + f" | worst false {max(rec['false']):.3f}", flush=True)
if os.environ.get("E7_SMOKE"): print("SMOKE OK"); sys.exit(0)
json.dump(out, open(f"{W}/results/e7_quality.json", "w"), indent=1)
rows_t = []
for key, rec in out.items():
    ptag, jd, a = key.split("|"); lab = {"q0.25": "0.25", "qnat": "0.82"}[ptag]
    rows_t.append(f"{lab} & {jd} & {rec['auc']:.2f} & {a[1:]} & {rec['margin']:+.3f} & " + " & ".join(f"{r:.2f}/{p:.2f}" for r, p in zip(rec["rate"], rec["pred"])) + f" & {max(rec['false']):.3f} \\\\")
body = ("\\begin{tabular}{llrrr" + "r" * len(NS) + "r}\n\\toprule\npool low-quality & judge & AUC & $\\alpha$ & margin & " + " & ".join(f"$n{{=}}{n}$" for n in NS) + " & worst false cert. \\\\\n\\midrule\n" + "\n".join(rows_t) + "\n\\bottomrule\n\\end{tabular}\n")
open(f"{W}/paper/data/tables/e7_quality.tex", "w").write(body); print("E7 DONE")
