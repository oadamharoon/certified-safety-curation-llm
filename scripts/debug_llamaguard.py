"""Why does the Llama Guard 3 wrapper flag nothing? Inspect tokenization of safe/unsafe,
the chat-template output, and the model's actual greedy generation on labeled pool items."""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); from common import *
from transformers import AutoTokenizer, AutoModelForCausalLM
name = "meta-llama/Llama-Guard-3-8B"
tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
print("tok('safe')", tok.encode("safe", add_special_tokens=False), "tok('unsafe')", tok.encode("unsafe", add_special_tokens=False),
      "tok('\\n\\nunsafe')", tok.encode("\n\nunsafe", add_special_tokens=False))
pool = build_pool(load_split("train"), 4000, 0.25, 0)
ex = [r for r in pool if not r["is_safe"]][:3] + [r for r in pool if r["is_safe"]][:3]
p = tok.apply_chat_template([{"role": "user", "content": ex[0]["prompt"]}, {"role": "assistant", "content": ex[0]["response"]}], tokenize=False)
print("template tail:", repr(p[-160:]))
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
ps = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}], tokenize=False) for r in ex]
enc = tok(ps, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
with torch.no_grad():
    g = m.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    lg = m(**enc).logits[:, -1, :].float()
top = lg.topk(3, dim=-1)
for i, r in enumerate(ex):
    print(f"label unsafe={not r['is_safe']} | gen={tok.decode(g[i][enc['input_ids'].shape[1]:], skip_special_tokens=True)!r} | top3 first-token ids {top.indices[i].tolist()} = {[tok.decode([t]) for t in top.indices[i].tolist()]}")
