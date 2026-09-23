"""Step 2b: supervised fine-tuning of Qwen2.5-0.5B (base) on one arm's selection.

Usage: ARM=cert400 SEED=0 python scripts/step2_sft.py
Identical settings for every arm (see README): full fine-tuning, bf16, loss on assistant
tokens only, AdamW lr 1e-5, 3 percent warmup then cosine, 3 epochs, effective batch 16,
max length 512, grad clip 1.0. Saves results/models/{ARM}_s{SEED}/ and a train log json.
"""
import json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import W, load_split, build_pool
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

ARM, SEED = os.environ["ARM"], int(os.environ["SEED"])
BASE = os.environ.get("SFT_BASE", "Qwen/Qwen2.5-0.5B")
EPOCHS, LR, MAXLEN, MICRO, ACCUM = 3, 1e-5, 512, 8, 2
TAG = os.environ.get("SFT_TAG", "")   # E24: a prefix for models of a second base model (selection files are shared)
OUT = f"{W}/results/models/{TAG}{ARM}_s{SEED}"
if os.path.exists(f"{OUT}/config.json"):
    print(f"{OUT} exists, skipping"); sys.exit(0)
torch.manual_seed(SEED); np.random.seed(SEED)

pool = json.load(open(os.environ["POOL_JSON"])) if os.environ.get("POOL_JSON") else build_pool(load_split("train"), 4000, 0.25, seed=0)   # E8: a second pool
sel = json.load(open(f"{W}/selections/{ARM}_s{SEED}.json"))
items = [pool[i] for i in sel["idx"]]
if os.environ.get("SFT_SMOKE"):
    items, EPOCHS, OUT = items[:24], 1, os.environ["SFT_SMOKE"]
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
if tok.chat_template is None:   # E24: a base model without a chat template (TinyLlama) gets Qwen2.5-0.5B's ChatML template, so the
    tok.chat_template = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").chat_template   # prompt format is identical across base models; saved with the model


def encode(r):
    msgs = [{"role": "user", "content": r["prompt"]}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = p + r["response"] + tok.eos_token
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels


enc = [encode(r) for r in items]
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable(); model.config.use_cache = False
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
steps_per_epoch = math.ceil(len(enc) / (MICRO * ACCUM))
total = steps_per_epoch * EPOCHS
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
rng = np.random.default_rng(SEED)
log = {"arm": ARM, "seed": SEED, "n_train": len(enc), "steps": total, "loss": []}
print(f"{ARM} s{SEED}: {len(enc)} examples, {total} optimizer steps", flush=True)
t0 = time.time(); step = 0; model.train()
for ep in range(EPOCHS):
    order = rng.permutation(len(enc))
    for b in range(0, len(order), MICRO * ACCUM):
        opt.zero_grad(set_to_none=True); acc = 0.0
        chunk = order[b:b + MICRO * ACCUM]
        for mb in range(0, len(chunk), MICRO):
            idx = chunk[mb:mb + MICRO]
            L = max(len(enc[i][0]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id); lab = torch.full((len(idx), L), -100); att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                a, l = enc[i]; ids[r, :len(a)] = torch.tensor(a); lab[r, :len(l)] = torch.tensor(l); att[r, :len(a)] = 1
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            (out.loss * len(idx) / len(chunk)).backward(); acc += out.loss.item() * len(idx) / len(chunk)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        log["loss"].append(acc)
        if step % 25 == 0 or step == total:
            print(f"  step {step}/{total} loss {np.mean(log['loss'][-25:]):.4f} ({time.time()-t0:.0f}s)", flush=True)
model.config.use_cache = True
model.save_pretrained(OUT); tok.save_pretrained(OUT)
log["seconds"] = time.time() - t0
json.dump(log, open(f"{OUT}/train_log.json", "w"))
print(f"saved {OUT}", flush=True)
