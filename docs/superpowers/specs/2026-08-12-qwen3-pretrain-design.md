# Qwen3-0.6B From-Scratch Pretrain on Ascend NPUs with ZeRO-Infinity Offload

Date: 2026-08-12

## Goal

Write and run scripts that pretrain (not finetune) a Qwen3-0.6B model from scratch on Ascend 910B4 NPUs, using DeepSpeed ZeRO-3 with parameter offload to CPU and optimizer offload to NVMe, on the alpaca_zh dataset (already present locally at `alpaca_zh/alpaca_data_zh_51k.json`).

## Environment facts (verified)

- `uv` venv at `.venv`, Python 3.12, torch 2.10.0+cpu, torch_npu 2.10.0, DeepSpeed 0.19.5 editable (`DeepSpeed/` submodule, NPU accelerator support built in).
- CANN 8.5.0 at `/usr/local/Ascend/cann-8.5.0`; `npu-smi` shows 8x 910B4-1 cards.
- NPUs 0–3 are free; NPUs 4–7 run another user's (`open`) Megatron job → use 0–3, with NPU set configurable in the launch script.
- huggingface.co unreachable; ModelScope reachable → model/tokenizer downloaded from ModelScope (`Qwen/Qwen3-0.6B`).
- No dedicated NVMe device; `/home` is SSD-backed LVM → NVMe offload path lives under `/home` (configurable).
- `torch_npu` import was broken only due to missing `pyyaml` (fixed); `deepspeed` and `deepspeed.ops.aio` import cleanly.
- Dataset: 51,155 Chinese instruction triples (`instruction`, `input`, `output`).

## Design decisions

- **Approach:** direct `deepspeed.initialize` engine + lightweight custom `train.py`. No HF Trainer / accelerate (avoids extra deps and NPU plugin friction).
- **From scratch:** instantiate `Qwen3ForCausalLM` from the ModelScope config (random weights), no checkpoint load.
- **Pretrain semantics:** full causal-LM loss on all tokens; no output masking. Data is formatted as text (`instruction + input + output`) since alpaca_zh is an instruction set.
- **Data pipeline:** tokenize each example with Qwen3 tokenizer, concatenate, separate with `<|endoftext|>`, pack into `--seq-len` (default 2048) sequences. Distributed sampler shards across ranks.
- **ZeRO-3 config (ds_config.json):**
  - `offload_param: {device: cpu, pin_memory: true}`
  - `offload_optimizer: {device: nvme, nvme_path: ./nvme_offload, buffer_count: 4, fast_read: true}`
  - `bf16: true`, `overlap_comm: true`, `contiguous_gradients: true`, gradient clipping 1.0
  - Global batch = micro-batch × 4 NPUs × grad-accum; all knobs configurable.
- **Knobs (argparse):** `--seq-len, --steps, --epochs, --micro-batch, --grad-accum, --lr, --save-interval, --eval-interval, --smoke-test, --nvme-path, --output-dir, --ckpt-dir`.
- **Launch script:** `run_train.sh` sets `ASCEND_RT_VISIBLE_DEVICES` from `NPU_IDS` env (default `0,1,2,3`), runs `deepspeed --num_gpus`.
- **Checkpointing:** DeepSpeed sharded checkpoints every `--save-interval`; final merge to single fp16 weights via `zero_to_fp32.py`.
- **Packages to install:** `transformers`, `modelscope`, `safetensors` (pyyaml already installed).

## Deliverables (in `qwen3_pretrain/`)

1. `train.py` — model build, data prep/packing, training loop, eval loop, checkpointing
2. `ds_config.json` — ZeRO-3 + CPU/NVMe offload config
3. `run_train.sh` — NPU-configurable launcher
4. `requirements.txt` — deps

## Verification

1. Smoke test: `--smoke-test` (few steps, tiny batch, 2 NPUs) — confirm loss decreases, offload dir populated, checkpoints written.
2. Real run on NPUs 0–3 with default 500 steps.
3. Verify NPU 4–7 untouched.
