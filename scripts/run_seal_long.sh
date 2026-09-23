#!/bin/bash
# SEAL selector, per-example update budget matched to the original rather than the epoch count:
# SEAL's loop zips the upper-level and new-dataset loaders, so an epoch is one pass over the
# NEW dataset (3000 here vs their 112k RedOrca), giving 47 selector updates per epoch at batch 64
# against their 1750. 30 epochs x 3000 = 90k paired steps (~1400 updates, each example weighted
# 30 times) and the full 112k BlueOrca as the upper-level pool. Everything else as run_seal.sh.
# E11: SEAL selector training on our lambda-.25 pool with SEAL's own code and recipe, then
# (a) SEAL's top-p .8 selection as a baseline arm and (b) SEAL's logits as a scorer under
# our certificate. Waits for the decile job (last of our GPU queue). Deviations from SEAL's
# script, all stated in the paper: Llama-3.1-8B-Instruct (theirs 3-8B), single GPU ZeRO-2
# (theirs 4 GPUs ZeRO-3), no flash-attn, upper-level BlueOrca truncated to 6000 (theirs 112k),
# new_dataset = our 3000-example pool, max_len 512 (theirs 1024; our pool examples are all
# within 512 under the E3 recipe, so only BlueOrca upper-level examples are truncated),
# ZeRO stage 0 (single GPU: stage 2 only adds a 1 GB reduce buffer), torch AdamW in place of the JIT-compiled FusedAdam (no nvcc on this machine; same update).
PY="${PY:-python}"
PY_LLM="${PY_LLM:-python}"
PY_SEAL="${PY_SEAL:-python}"
until [ -z "$(ps -eo args | grep "[e]3_lora_sft\\|[e]3_eval\\|[s]tep2_sft")" ]; do sleep 120; done
set -u; W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"; S=$W/third_party/SEAL/examples; cd $W
PYS=${PY_SEAL}; PY1=${PY}; PY2=${PY_LLM}
mkdir -p results/seal logs/seal
env PYTHONNOUSERSITE=1 $PY1 - <<'PY'
import json
W="${CDC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
pool=json.load(open(f"{W}/selections_e3/lambda0.25/pool.json"))
with open(f"{W}/results/seal/pool_lambda0.25.jsonl","w") as f:
    for r in pool: f.write(json.dumps({"system_prompt":"You are a helpful assistant.","question":r["prompt"],"response":r["response"]})+"\n")
import itertools
with open(f"{W}/third_party/SEAL/examples/scripts/datasets/BlueOrca/train.jsonl") as fi, open(f"{W}/results/seal/blueorca_full.jsonl","w") as fo:
    for line in fi: fo.write(line)
print("pool and upper-level set written")
PY
cd $S/scripts && mkdir -p ckpt && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=0 PYTHONPATH=$W/third_party/SEAL PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PYS -m deepspeed.launcher.runner --num_gpus 1 ../train_sft_selector.py \
    --seed 42 --max_len 512 \
    --dataset $W/results/seal/blueorca_full.jsonl --dataset_probs 1. \
    --new_dataset $W/results/seal/pool_lambda0.25.jsonl \
    --upperlevel_weight 1. --upperlevel_weight_decay 0.03 \
    --train_batch_size 64 --micro_train_batch_size 1 --max_samples 112000 \
    --pretrain meta-llama/Llama-3.1-8B-Instruct --ref_constant 0. \
    --selector_activation softmax --selector_name ours_lambda025_long \
    --save_steps -1 --logging_steps 1 --eval_steps -1 --zero_stage 0 --max_epochs 30 --bf16 \
    --learning_rate 1e-5 --selector_learning_rate 5e-3 --selector_lr_scheduler constant --lr_scheduler constant \
    --gradient_checkpointing --lora_rank 16 --lora_alpha 16 --target_modules q_proj v_proj \
    > $W/logs/seal/selector_train_long.log 2>&1 && echo "SEAL SELECTOR TRAINED" || { echo "SEAL SELECTOR FAILED"; tail -n 20 $W/logs/seal/selector_train.log; exit 1; }
ls -la ckpt/ | tail -n 3; cp ckpt/ours_lambda025_long_softmax.pt ckpt/ours_lambda025_long_softmax_ep*.pt $W/results/seal/
echo SEAL LONG DONE
