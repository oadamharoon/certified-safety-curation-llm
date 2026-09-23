"""E3: SAFT's protocol on an aligned chat model.

Model: meta-llama/Llama-3.1-8B-Instruct (aligned). LoRA r=8, alpha=16, dropout 0.05, on
q/k/v/o/gate/up/down (DataShield's target set; SAFT's appendix config not available to us,
stated as an assumption). 4 epochs, lr 2e-5, effective batch 16, max length 512, bf16,
loss on assistant tokens under the model's own chat template. Identical for every arm.
Runs in the certcurate-llm env (peft). Usage: ARM=<arm> SEED=<s> LAMBDA=0.25 python e3_lora_sft.py
Pools live in selections_e3/lambda{LAMBDA}/{ARM}_s{SEED}.json (built by e3_select.py); models
save to results/models_e3/lambda{LAMBDA}/{ARM}_s{SEED}/ (adapter only).
"""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import W, load_split
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ARM, SEED, LAM = os.environ["ARM"], int(os.environ["SEED"]), os.environ.get("LAMBDA", "0.25")
BASE = os.environ.get("E3_BASE", "meta-llama/Llama-3.1-8B-Instruct")
EPOCHS, LR, MAXLEN, MICRO, ACCUM = 4, 2e-5, 512, 2, 8   # same effective batch 16; micro 2 to share the GPU
OUT = f"{W}/results/models_e3/lambda{LAM}/{ARM}_s{SEED}"
if os.path.exists(f"{OUT}/adapter_config.json"):
    print(f"{OUT} exists, skipping"); sys.exit(0)
torch.manual_seed(SEED); np.random.seed(SEED)
pool = json.load(open(f"{W}/selections_e3/lambda{LAM}/pool.json"))
sel = json.load(open(f"{W}/selections_e3/lambda{LAM}/{ARM}_s{SEED}.json"))
items = [pool[i] for i in sel["idx"]]
if os.environ.get("SFT_SMOKE"):
    items, EPOCHS, OUT = items[:16], 1, os.environ["SFT_SMOKE"]
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.pad_token or tok.eos_token


def encode(r):
    p = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
    full = p + r["response"] + tok.eos_token
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"]
    return ids, [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]


enc = [encode(r) for r in items]
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable(); model.config.use_cache = False; model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
steps_per_epoch = math.ceil(len(enc) / (MICRO * ACCUM)); total = steps_per_epoch * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
rng = np.random.default_rng(SEED); log = {"arm": ARM, "seed": SEED, "lambda": LAM, "n_train": len(enc), "steps": total, "loss": []}
print(f"{ARM} s{SEED} lambda {LAM}: {len(enc)} examples, {total} steps", flush=True)
t0 = time.time(); step = 0; model.train()
for ep in range(EPOCHS):
    order = rng.permutation(len(enc))
    for b in range(0, len(order), MICRO * ACCUM):
        opt.zero_grad(set_to_none=True); acc = 0.0; chunk = order[b:b + MICRO * ACCUM]
        for mb in range(0, len(chunk), MICRO):
            idx = chunk[mb:mb + MICRO]; L = max(len(enc[i][0]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id); lab = torch.full((len(idx), L), -100); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r_, i in enumerate(idx):
                a, l = enc[i]; ids[r_, :len(a)] = torch.tensor(a); lab[r_, :len(l)] = torch.tensor(l); att[r_, :len(a)] = 1
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            (out.loss * len(idx) / len(chunk)).backward(); acc += out.loss.item() * len(idx) / len(chunk)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); step += 1; log["loss"].append(acc)
        if step % 25 == 0 or step == total:
            print(f"  step {step}/{total} loss {np.mean(log['loss'][-25:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
os.makedirs(OUT, exist_ok=True); model.save_pretrained(OUT); log["seconds"] = time.time() - t0
json.dump(log, open(f"{OUT}/train_log.json", "w")); print(f"saved {OUT}", flush=True)
