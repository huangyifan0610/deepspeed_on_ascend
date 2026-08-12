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


def enable_info_logging():
    """DeepSpeed's logger defaults to WARNING, which would hide the per-step
    loss and eval lines; surface them for observability."""
    import logging

    ds_logger.setLevel(logging.INFO)
    for handler in ds_logger.handlers:
        handler.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain Qwen3-0.6B from scratch with DeepSpeed ZeRO-3 offload")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-path", default="../alpaca_zh/alpaca_data_zh_51k.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-seqs", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--nvme-path", default="nvme_offload")
    parser.add_argument("--ds-config", default="ds_config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="tiny run: 3 steps, seq-len 128, eval every step")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_ds_config(args):
    with open(args.ds_config, encoding="utf-8") as f:
        config = json.load(f)
    config["train_micro_batch_size_per_gpu"] = args.micro_batch
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
    """Mean loss over this rank's slice of eval sequences (all-reduced over ranks)."""
    model_engine.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(eval_ids), micro_batch):
            batch = torch.tensor(eval_ids[start:start + micro_batch], device=device)
            loss = model_engine(input_ids=batch, labels=batch).loss
            total += loss.item() * batch.size(0)
            count += batch.size(0)
    total_t = torch.tensor(total, device=device)
    count_t = torch.tensor(count, device=device)
    dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    return (total_t / count_t).item() if count_t.item() > 0 else float("nan")


def main():
    args = parse_args()
    if args.smoke_test:
        args.steps = 3
        args.micro_batch = 1
        args.seq_len = 128
        args.eval_seqs = 8
        args.eval_interval = 1
        args.warmup_steps = 1
        args.log_interval = 1

    if args.eval_seqs < 1:
        raise SystemExit("--eval-seqs must be >= 1")

    enable_info_logging()
    deepspeed.init_distributed()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    torch.npu.set_device(local_rank)
    seed_everything(args.seed)

    os.makedirs(args.nvme_path, exist_ok=True)

    # Download once on rank 0, then all ranks resolve the cached path.
    if rank == 0:
        model_dir = snapshot_download(args.model_name)
    dist.barrier()
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
    if len(train_loader) == 0:
        raise SystemExit("no per-rank micro-batches after drop_last; lower world size or use more data")

    ds_config = build_ds_config(args)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config)

    # Gradient accumulation is fully managed by DeepSpeed: without
    # train_batch_size in ds_config.json it defaults to 1 (one optimizer step
    # per micro-batch); set train_batch_size there to enable accumulation.
    gas = model_engine.gradient_accumulation_steps()
    ds_logger.info(f"DeepSpeed gradient_accumulation_steps={gas}")

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
            # Managed gradient accumulation: engine.backward() auto-scales the
            # loss by the accumulation steps, and engine.step() applies the
            # optimizer update only at the internal accumulation boundary.
            outputs = model_engine(input_ids=batch, labels=batch)
            model_engine.backward(outputs.loss)
            at_boundary = model_engine.is_gradient_accumulation_boundary()
            model_engine.step()
            if not at_boundary:
                continue
            step += 1
            if rank == 0 and step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"] if optimizer else args.lr
                elapsed = time.time() - t_start
                tok_per_s = step * args.micro_batch * world_size * gas * args.seq_len / max(elapsed, 1e-9)
                ds_logger.info(
                    f"[step {step}/{args.steps}] loss={outputs.loss.item():.4f} lr={lr:.2e} "
                    f"tok/s={tok_per_s:.0f} epoch={epoch}")
            if step % args.eval_interval == 0:
                val_loss = evaluate(model_engine, eval_ids_rank,
                                    torch.npu.current_device(), args.micro_batch)
                model_engine.train()
                if rank == 0:
                    ds_logger.info(f"[eval step {step}] val_loss={val_loss:.4f}")
            if step >= args.steps:
                break

    ds_logger.info("Pretraining finished.")


if __name__ == "__main__":
    main()
