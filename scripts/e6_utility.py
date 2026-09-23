"""E6: helpfulness and over-refusal for every fine-tuned model.

(1) Helpfulness as SAFT measures it: ROUGE-L (LCS-based F1, implemented here, no package)
    between each model's generation and the human-labeled SAFE reference response(s) for the
    same held-out prompt, over the evaluation prompts that have at least one safe reference.
    Judge-free.
(2) XSTest (Rottger et al. 2024): 250 safe prompts that superficially resemble unsafe ones
    (over-refusal) and 200 contrast unsafe prompts. Each model generates on all 450; WildGuard
    labels refusal and harmful response. Reported: refusal rate on safe prompts (lower is
    better), harmful-response rate on unsafe prompts (lower is better).
Caches: results/gen_xstest/{model}.json, results/gen_scores/xstest_wildguard.json,
results/e6_utility.json.
"""
import csv, glob, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM

names = sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(f"{W}/results/models/*/config.json"))
ev = json.load(open(f"{W}/results/eval_sets.json"))
test = load_split("test")
refs = {}
for r in test:
    if r["is_safe"]:
        refs.setdefault(r["prompt"], []).append(r["response"])
gen_prompts = ev["gen_prompts"]; have = [i for i, p in enumerate(gen_prompts) if p in refs]
print(f"{len(have)} of {len(gen_prompts)} evaluation prompts have a safe reference", flush=True)


def lcs(a, b):
    m, n = len(a), len(b); prev = [0] * (n + 1)
    for i in range(m):
        cur = [0] * (n + 1)
        for j in range(n):
            cur[j + 1] = prev[j] + 1 if a[i] == b[j] else max(prev[j + 1], cur[j])
        prev = cur
    return prev[n]


def rouge_l(hyp, ref):
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r: return 0.0
    l = lcs(h, r); p, rc = l / len(h), l / len(r)
    return 0.0 if l == 0 else 2 * p * rc / (p + rc)


xs = list(csv.DictReader(open(f"{W}/data/xstest/xstest_prompts.csv")))
os.makedirs(f"{W}/results/gen_xstest", exist_ok=True)
out = json.load(open(f"{W}/results/e6_utility.json")) if os.path.exists(f"{W}/results/e6_utility.json") else {}
for n in names:
    gp = f"{W}/results/gen_xstest/{n}.json"
    if n in out and os.path.exists(gp): continue
    gens = json.load(open(f"{W}/results/gen/{n}.json"))
    rl = [max(rouge_l(gens[i]["response"], ref) for ref in refs[gen_prompts[i]]) for i in have]
    out[n] = {"rouge_l": float(np.mean(rl))}
    if not os.path.exists(gp):
        tok = AutoTokenizer.from_pretrained(f"{W}/results/models/{n}"); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token   # each model's own tokenizer (E24 adds a second base model)
        model = AutoModelForCausalLM.from_pretrained(f"{W}/results/models/{n}", dtype=torch.bfloat16).cuda().eval(); xg = []
        with torch.no_grad():
            for i in range(0, len(xs), 50):
                ps = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True) for r in xs[i:i + 50]]
                enc = tok(ps, return_tensors="pt", padding=True).to("cuda")
                o = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
                for r, oo in zip(xs[i:i + 50], o):
                    xg.append({"prompt": r["prompt"], "label": r["label"], "type": r["type"], "response": tok.decode(oo[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        json.dump(xg, open(gp, "w")); del model; torch.cuda.empty_cache()
    json.dump(out, open(f"{W}/results/e6_utility.json", "w"), indent=1)
    print(f"  {n}: ROUGE-L {out[n]['rouge_l']:.3f}, xstest generated", flush=True)

# WildGuard on all XSTest generations, cached per model
path = f"{W}/results/gen_scores/xstest_wildguard.json"
d = json.load(open(path)) if os.path.exists(path) else {}
todo = [n for n in names if n not in d]
if todo:
    allg = [g for n in todo for g in json.load(open(f"{W}/results/gen_xstest/{n}.json"))]
    req, ref, harm, raw = wildguard_labels(allg)
    for i, n in enumerate(todo):
        d[n] = {"refusal": ref[i*450:(i+1)*450].tolist(), "harmful_response": harm[i*450:(i+1)*450].tolist()}
    json.dump(d, open(path, "w"))
labels = np.array([r["label"] for r in xs])
for n in names:
    ref = np.array(d[n]["refusal"]); harm = np.array(d[n]["harmful_response"])
    out[n].update({"xstest_safe_refusal": float(ref[labels == "safe"].mean()), "xstest_unsafe_harmful": float(harm[labels == "unsafe"].mean()),
                   "xstest_unsafe_refusal": float(ref[labels == "unsafe"].mean())})
json.dump(out, open(f"{W}/results/e6_utility.json", "w"), indent=1)
sel = json.load(open(f"{W}/results/step2_selections.json"))
arms = ["base", "full", "random1400", "heur", "cert200", "cert400", "certbd400", "certlg400", "certlg200", "lgfilter", "labels", "oracle"] + sorted(k for k in sel if k.startswith("safedemo"))
def cell(v): return f"{np.mean(v):.3f}±{np.std(v, ddof=1)/np.sqrt(len(v)):.3f}" if len(v) > 1 else f"{np.mean(v):.3f}"
print(f"\n{'arm':12s} {'ROUGE-L':>12s} {'XSTest safe: refusal':>21s} {'XSTest unsafe: harmful':>23s} {'XSTest unsafe: refusal':>23s}")
for a in arms:
    ms = [n for n in names if n.startswith(a + "_")]
    if not ms: continue
    print(f"{a:12s} {cell([out[n]['rouge_l'] for n in ms]):>12s} {cell([out[n]['xstest_safe_refusal'] for n in ms]):>21s} {cell([out[n]['xstest_unsafe_harmful'] for n in ms]):>23s} {cell([out[n]['xstest_unsafe_refusal'] for n in ms]):>23s}")
