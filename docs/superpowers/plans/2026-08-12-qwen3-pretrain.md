# Qwen3-0.6B Pretrain (from scratch) on Ascend NPU with ZeRO-Infinity Offload — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write and run scripts that pretrain a Qwen3-0.6B model from scratch on 4 free Ascend 910B4 NPUs (configurable set), using DeepSpeed ZeRO-3 with parameter offload to CPU and optimizer offload to NVMe, on the alpaca_zh dataset.

**Architecture:** A single `train.py` builds `Qwen3ForCausalLM` with random weights from the ModelScope config, packs alpaca_zh text into fixed-length sequences, and trains through a direct `deepspeed.initialize` engine (no HF Trainer). `ds_config.json` carries the static ZeRO-3/offload settings; `run_train.sh` launches via the DeepSpeed launcher with a configurable `NPU_IDS`.

**Tech Stack:** Python 3.12 (uv venv at `.venv`), torch 2.10.0+cpu, torch_npu 2.10.0 (CANN 8.5.0), DeepSpeed 0.19.5 (editable install at `DeepSpeed/`), transformers, modelscope, pytest.

## Global Constraints

- All new files live in `/home/yifan/deepspeed/qwen3_pretrain/`.
- Use the existing uv venv: `source /home/yifan/deepspeed/.venv/bin/activate` (or `/home/yifan/deepspeed/.venv/bin/python` directly).
- Install packages with `uv pip install ...` into that venv.
- Model/tokenizer must come from **ModelScope** (`Qwen/Qwen3-0.6B`) — huggingface.co is unreachable.
- NPUs: default `NPU_IDS=0,1,2,3` (cards 4–7 run another user's job — never touch them).
- DeepSpeed is a **from-scratch** pretrain: random init, full causal-LM loss on all tokens (no masking).
- NVMe offload config keys in this DeepSpeed fork: `offload_optimizer: {device: nvme, nvme_path, buffer_count, pin_memory, pipeline_read, pipeline_write, fast_init}`. There is NO `fast_read` key (it was replaced by `pipeline_read`/`pipeline_write` — see `DeepSpeed/deepspeed/runtime/zero/offload_config.py`).
- No optimizer scheduler complexity: WarmupLR (warmup then constant).
- Git commits: the root repo has **no git identity configured** (`git config user.name/email` unset) — commit steps in this plan are **optional**; if they fail with "empty ident name", leave changes uncommitted and tell the user.
- DeepSpeed repo AGENTS.md (license headers, signoff, yapf/flake8) applies only to files inside `DeepSpeed/` — our files are outside it.

---

### Task 1: Install dependencies and verify model download

**Files:**
- Create: `qwen3_pretrain/requirements.txt`

**Interfaces:**
- Produces: working venv with `transformers`, `modelscope`, `safetensors`, `pytest`; Qwen3-0.6B cached under `~/.cache/modelscope/hub/Qwen/Qwen3-0.6B`.

- [ ] **Step 1: Create requirements.txt**

```text
transformers>=4.52
modelscope
safetensors
pytest
pyyaml
```

- [ ] **Step 2: Install packages**

Run: `source /home/yifan/deepspeed/.venv/bin/activate && cd /home/yifan/deepspeed && uv pip install -r qwen3_pretrain/requirements.txt`
Expected: "Installed N packages" with no error. Verify: `/home/yifan/deepspeed/.venv/bin/python -c "import transformers, modelscope, yaml, pytest; print(transformers.__version__)"` prints a version ≥ 4.52.

- [ ] **Step 3: Verify torch_npu + deepspeed still import**

Run: `/home/yifan/deepspeed/.venv/bin/python -c "import torch, torch_npu, deepspeed; print('ok', torch.__version__, deepspeed.__version__)" 2>&1 | grep -v Warning`
Expected: `ok 2.10.0+cpu 0.19.5+ab2d1a272`

- [ ] **Step 4: Download Qwen3-0.6B from ModelScope**

Run:
```bash
source /home/yifan/deepspeed/.venv/bin/activate && cd /home/yifan/deepspeed && python -c "
from modelscope import snapshot_download
p = snapshot_download('Qwen/Qwen3-0.6B')
print('MODEL_DIR:', p)
"
```
Expected: `MODEL_DIR: /home/yifan/.cache/modelscope/hub/Qwen/Qwen3-0.6B` (or similar) and the dir contains `config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors` etc.

- [ ] **Step 5: Verify tokenizer + config load from cache (offline)**

Run:
```bash
source /home/yifan/deepspeed/.venv/bin/activate && python -c "
from transformers import AutoConfig, AutoTokenizer
p = '/home/yifan/.cache/modelscope/hub/Qwen/Qwen3-0.6B'
c = AutoConfig.from_pretrained(p)
t = AutoTokenizer.from_pretrained(p)
print(c.model_type, c.hidden_size, c.num_layers if hasattr(c,'num_layers') else c.num_hidden_layers, '| eos:', t.eos_token, t.eos_token_id)
"
```
Expected: `qwen3 1536 28 | eos: <|endoftext|> 151643` (eos id may differ slightly — record the printed value; `tokenizer.eos_token_id` is what code must use).

- [ ] **Step 6: Commit (optional — see Global Constraints)**

```bash
cd /home/yifan/deepspeed && git add qwen3_pretrain/requirements.txt && git commit -m "chore: add qwen3 pretrain requirements"
```

---

### Task 2: data_utils.py — text formatting + sequence packing (TDD)

**Files:**
- Create: `qwen3_pretrain/data_utils.py`
- Test: `qwen3_pretrain/tests/test_data_utils.py`

**Interfaces:**
- Produces (used by train.py Task 3):
  - `format_example(example: dict) -> str` — renders one alpaca_zh record as plain text.
  - `pack_sequences(tokenizer, texts: list[str], seq_len: int, eos_token_id: int) -> list[list[int]]` — concatenates tokenized texts (each followed by eos), then cuts into `seq_len`-token windows; drops the final partial window.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from data_utils import format_example, pack_sequences


class StubTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def test_format_example_without_input():
    ex = {"instruction": "指令", "input": "", "output": "回答"}
    assert format_example(ex) == "指令\n\n回答"


def test_format_example_with_input():
    ex = {"instruction": "指令", "input": "输入", "output": "回答"}
    assert format_example(ex) == "指令\n\n输入\n\n回答"


def test_pack_sequences_single_window():
    tok = StubTokenizer()
    texts = ["ab", "cd"]
    packed = pack_sequences(tok, texts, seq_len=4, eos_token_id=99)
    assert packed == [[97, 98, 99, 99]]


def test_pack_sequences_multiple_windows_and_tail_drop():
    tok = StubTokenizer()
    texts = ["abcdefgh", "ij"]
    packed = pack_sequences(tok, texts, seq_len=4, eos_token_id=99)
    assert packed == [[97, 98, 99, 100], [101, 102, 103, 104]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/yifan/deepspeed/qwen3_pretrain && /home/yifan/deepspeed/.venv/bin/python -m pytest tests/test_data_utils.py -v 2>&1 | tail -5`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_utils'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Data utilities for pretraining Qwen3-0.6B from scratch on alpaca_zh."""

# SPDX-License-Identifier: Apache-2.0


def format_example(example):
    """Render one alpaca_zh record as plain text for causal-LM pretraining."""
    instruction = example["instruction"]
    input_text = example.get("input") or ""
    output = example["output"]
    parts = [instruction]
    if input_text:
        parts.append(input_text)
    parts.append(output)
    return "\n\n".join(parts)


def pack_sequences(tokenizer, texts, seq_len, eos_token_id):
    """Tokenize all texts, concatenate with eos separators, cut into windows."""
    tokens = []
    for text in texts:
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(eos_token_id)
    return [tokens[i:i + seq_len] for i in range(0, len(tokens) - seq_len + 1, seq_len)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/yifan/deepspeed/qwen3_pretrain && /home/yifan/deepspeed/.venv/bin/python -m pytest tests/test_data_utils.py -v 2>&1 | tail -3`
Expected: `4 passed`

- [ ] **Step 5: Commit (optional)**

```bash
cd /home/yifan/deepspeed && git add qwen3_pretrain/data_utils.py qwen3_pretrain/tests/test_data_utils.py && git commit -m "feat: add alpaca_zh formatting and sequence packing"
```

---

### Task 3: ds_config.json + train.py — main training script

**Files:**
- Create: `qwen3_pretrain/ds_config.json`
- Create: `qwen3_pretrain/train.py`

**Interfaces:**
- Consumes: `data_utils.format_example`, `data_utils.pack_sequences` (Task 2); ModelScope cache dir from Task 1.
- Produces: `train.py` accepting argparse flags: `--model-name --data-path --seq-len --micro-batch --grad-accum --steps --lr --warmup-steps --save-interval --eval-interval --eval-seqs --output-dir --nvme-path --ds-config --seed --log-interval --smoke-test --local_rank`. `ds_config.json` static ZeRO-3 config.

- [ ] **Step 1: Write ds_config.json**

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "nvme_offload",
      "buffer_count": 4,
      "pin_memory": true,
      "pipeline_read": true,
      "pipeline_write": true,
      "fast_init": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 50000000,
    "stage3_prefetch_bucket_size": 50000000,
    "stage3_param_persistence_threshold": 1000000,
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "sub_group_size": 1000000000
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "wall_clock_breakdown": false
}
```

- [ ] **Step 2: Write train.py**

```python
"""From-scratch pretrain of Qwen3-0.6B on Ascend NPU with DeepSpeed ZeRO-3
(parameters offloaded to CPU, optimizer states to NVMe)."""

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import random
import time

import torch
from modelscope import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import deepspeed
import deepspeed.comm as dist
from deepspeed.utils import logger as ds_logger

from data_utils import format_example, pack_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain Qwen3-0.6B from scratch with DeepSpeed ZeRO-3 offload")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", default="../alpaca_zh/alpaca_data_zh_51k.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-seqs", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--nvme-path", default="nvme_offload")
    parser.add_argument("--ds-config", default="ds_config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny run: 3 steps, seq-len 128, save every step")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_ds_config(args, world_size):
    with open(args.ds_config, encoding="utf-8") as f:
        config = json.load(f)
    config["train_micro_batch_size_per_gpu"] = args.micro_batch
    config["train_batch_size"] = args.micro_batch * world_size * args.grad_accum
    config["optimizer"] = {"type": "Adam", "params": {"lr": args.lr}}
    config["lr_scheduler"] = {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0.0,
            "warmup_max_lr": args.lr,
            "warmup_num_steps": args.warmup_steps,
        },
    }
    config["zero_optimization"]["offload_optimizer"]["nvme_path"] = os.path.abspath(args.nvme_path)
    return config


def evaluate(model_engine, eval_ids, device, micro_batch):
    """Mean loss over this rank's slice of eval sequences."""
    model_engine.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(eval_ids), micro_batch):
            batch = torch.tensor(eval_ids[start:start + micro_batch], device=device)
            loss = model_engine(input_ids=batch, labels=batch).loss
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    mean = torch.tensor(total / max(count, 1), device=device)
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    return mean.item() / dist.get_world_size()


def main():
    args = parse_args()
    if args.smoke_test:
        args.steps = 3
        args.micro_batch = 1
        args.grad_accum = 1
        args.seq_len = 128
        args.eval_seqs = 8
        args.save_interval = 1
        args.eval_interval = 1
        args.warmup_steps = 1
        args.log_interval = 1

    deepspeed.init_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    torch.npu.set_device(local_rank)
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.nvme_path, exist_ok=True)

    model_dir = snapshot_download(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_config(config)

    with open(args.data_path, encoding="utf-8") as f:
        examples = json.load(f)
    texts = [format_example(ex) for ex in examples]
    packed = pack_sequences(tokenizer, texts, args.seq_len, tokenizer.eos_token_id)

    train_ids = packed[:-args.eval_seqs] if args.eval_seqs < len(packed) else []
    eval_ids = packed[-args.eval_seqs:]

    def make_loader(seqs):
        dataset = torch.utils.data.TensorDataset(torch.tensor(seqs, dtype=torch.long))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.micro_batch, sampler=sampler, num_workers=0)

    if len(train_ids) > 0:
        train_loader = make_loader(train_ids)
    else:
        raise SystemExit("packed dataset too small to train")

    ds_config = build_ds_config(args, world_size)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config)

    eval_ids_rank = eval_ids[rank::world_size]
    step = 0
    epoch = 0
    t_start = time.time()
    while step < args.steps:
        # Reshuffle the sampler at the start of every epoch.
        epoch += 1
        train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = batch[0].to(torch.npu.current_device())
            for _ in range(args.grad_accum):
                # engine.backward auto-scales loss by gradient accumulation steps.
                outputs = model_engine(input_ids=batch, labels=batch)
                model_engine.backward(outputs.loss)
            model_engine.step()
            step += 1
            if rank == 0 and step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"] if optimizer else args.lr
                elapsed = time.time() - t_start
                tok_per_s = step * args.micro_batch * world_size * args.grad_accum * args.seq_len / max(elapsed, 1e-9)
                ds_logger.info(
                    f"[step {step}/{args.steps}] loss={outputs.loss.item():.4f} lr={lr:.2e} "
                    f"tok/s={tok_per_s:.0f} epoch={epoch}")
            if step % args.save_interval == 0:
                model_engine.save_checkpoint(args.output_dir, tag=f"step-{step}")
            if step % args.eval_interval == 0:
                val_loss = evaluate(model_engine, eval_ids_rank,
                                    torch.npu.current_device(), args.micro_batch)
                if rank == 0:
                    ds_logger.info(f"[eval step {step}] val_loss={val_loss:.4f}")
            if step >= args.steps:
                break

    if rank == 0:
        model_engine.save_16bit_model(args.output_dir, save_filename="qwen3-0.6b-pretrained.bin")
    ds_logger.info("Pretraining finished.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Sanity-check imports and CLI on CPU**

Run: `cd /home/yifan/deepspeed/qwen3_pretrain && /home/yifan/deepspeed/.venv/bin/python train.py --help 2>&1 | grep -c "\-\-smoke-test"`
Expected: prints `1` (argparse parses; no NPU needed).

- [ ] **Step 4: Smoke test on 2 NPUs (0 and 1)**

Run:
```bash
source /home/yifan/deepspeed/.venv/bin/activate && cd /home/yifan/deepspeed/qwen3_pretrain && \
ASCEND_RT_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 train.py --smoke-test --output-dir output_smoke --nvme-path nvme_offload_smoke
```
Expected (allow 2–10 min): ranks initialize via HCCL, 3 steps train with falling loss, eval prints `val_loss`, `save_checkpoint` creates `output_smoke/step-1/...`, and `nvme_offload_smoke/zero_stage_3/` contains swap files. If bf16 raises an error on this stack, fall back by editing `ds_config.json`: replace the `"bf16"` block with `"fp16": {"enabled": true, "loss_scale_window": 1000}` and re-run.

- [ ] **Step 5: Verify NVMe offload files exist**

Run: `ls -la /home/yifan/deepspeed/qwen3_pretrain/nvme_offload_smoke/zero_stage_3/ && ls /home/yifan/deepspeed/qwen3_pretrain/output_smoke/`
Expected: non-empty `zero_stage_3` dir (swap files `*.swap` or `bias*.pt` etc.) and `step-1/` checkpoint dirs.

- [ ] **Step 6: Commit (optional)**

```bash
cd /home/yifan/deepspeed && git add qwen3_pretrain/train.py qwen3_pretrain/ds_config.json && git commit -m "feat: add Qwen3-0.6B pretrain script with ZeRO-3 CPU/NVMe offload"
```

---

### Task 4: run_train.sh — configurable NPU launcher

**Files:**
- Create: `qwen3_pretrain/run_train.sh`

**Interfaces:**
- Consumes: `train.py` (Task 3).
- Produces: `./run_train.sh [train.py args...]` — launches on `NPU_IDS` env var (default `0,1,2,3`), passing through all extra args.

- [ ] **Step 1: Write run_train.sh**

```bash
#!/bin/bash
# Launch Qwen3-0.6B pretraining on a configurable set of NPUs.
# Usage: NPU_IDS=0,1,2,3 ./run_train.sh [train.py args...]
set -euo pipefail

NPU_IDS="${NPU_IDS:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS"

NUM_NPUS=$(echo "$NPU_IDS" | tr ',' '\n' | grep -c '[0-9]')
echo "[run_train] using NPU_IDS=$NPU_IDS (${NUM_NPUS} cards)"

cd "$(dirname "$0")"
exec deepspeed --num_gpus "$NUM_NPUS" train.py "$@"
```

- [ ] **Step 2: Make executable and syntax-check**

Run: `chmod +x /home/yifan/deepspeed/qwen3_pretrain/run_train.sh && bash -n /home/yifan/deepspeed/qwen3_pretrain/run_train.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

- [ ] **Step 3: Dry-run smoke test through the launcher**

Run: `NPU_IDS=0,1 /home/yifan/deepspeed/qwen3_pretrain/run_train.sh --smoke-test --output-dir output_smoke2 --nvme-path nvme_offload_smoke2`
Expected: log line `[run_train] using NPU_IDS=0,1 (2 cards)` followed by the same 3-step smoke run as Task 3 Step 4.

- [ ] **Step 4: Commit (optional)**

```bash
cd /home/yifan/deepspeed && git add qwen3_pretrain/run_train.sh && git commit -m "feat: add NPU-configurable launcher script"
```

---

### Task 5: Full training run on NPUs 0–3 + verification

**Files:**
- No new files. Uses `qwen3_pretrain/run_train.sh`, `train.py`, `ds_config.json`.

- [ ] **Step 1: Confirm NPUs 0–3 free, 4–7 untouched**

Run: `npu-smi info | grep -E "^\| [0-9]"` and `ps -o user,pid,cmd -p 823792,823793,823794,823795 | tail -2`
Expected: cards 0–3 idle; cards 4–7 still owned by user `open`'s PIDs (do not kill).

- [ ] **Step 2: Launch full run (500 steps, defaults) in background**

Run:
```bash
cd /home/yifan/deepspeed/qwen3_pretrain && nohup ./run_train.sh \
  --output-dir output \
  --nvme-path nvme_offload \
  > train.log 2>&1 &
echo "PID: $!"
```
Expected: PID printed; `run_train.sh` says `using NPU_IDS=0,1,2,3 (4 cards)`.

- [ ] **Step 3: Verify startup on all 4 ranks**

Run: `sleep 90 && grep -E "Initializing TorchBackend|Zero Stage 3|offload|step 1/500" /home/yifan/deepspeed/qwen3_pretrain/train.log | head -20 && npu-smi info | grep -E "^\| [0-9]" | head -8`
Expected: 4 ranks initialized (HCCL), stage-3 + offload messages, first `[step ...]` loss line; `npu-smi` shows python3 processes only on cards 0–3.

- [ ] **Step 4: Verify offload is happening on disk**

Run: `ls /home/yifan/deepspeed/qwen3_pretrain/nvme_offload/zero_stage_3/ | head -5 && du -sh /home/yifan/deepspeed/qwen3_pretrain/nvme_offload/`
Expected: swap/state files present and growing (optimizer states → NVMe), while `npu-smi` HBM usage on cards 0–3 stays well under 64 GB (parameter/activation memory only).

- [ ] **Step 5: Monitor to completion**

Run: `tail -5 /home/yifan/deepspeed/qwen3_pretrain/train.log` and check process with `ps -p $PID`
Expected: loss trending down (e.g. from ~11 toward <8), val_loss logged every 50 steps, checkpoints in `output/step-100/`, ..., `step-500/`.

- [ ] **Step 6: Verify final merged weights**

Run: `ls -la /home/yifan/deepspeed/qwen3_pretrain/output/qwen3-0.6b-pretrained.bin`
Expected: file exists, ~1.2 GB (fp16 0.6B params).

- [ ] **Step 7: Confirm NPUs 4–7 untouched after run**

Run: `npu-smi info | grep -E "^\| [0-9]"`
Expected: cards 4–7 show the same other-user processes as before (PIDs 823792–823795).

- [ ] **Step 8: Report**

Report to the user: final loss curve summary (first/last train loss, final val_loss), wall time, tok/s, checkpoint list, merged weights path, and NVMe offload size. Leave the training artifacts in place.
