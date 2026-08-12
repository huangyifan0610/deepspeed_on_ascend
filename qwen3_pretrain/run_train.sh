#!/bin/bash
# =============================================================================
# Qwen3-0.6B from-scratch pretrain — DeepSpeed ZeRO-3 launcher (Ascend NPU)
# =============================================================================
# Usage:
#   ./run_train.sh [train.py args...]          # defaults (4 NPUs: 0,1,2,3)
#   NPU_IDS=0,1,2 ./run_train.sh [args...]     # custom NPU set
#
# Customizable via environment variables (before running):
#   NPU_IDS       comma-separated NPU ids to use (default: 0,1,2,3)
#   MASTER_PORT   port for the distributed job (default: 29500)
#   LD_PRELOAD    already wired to the bundled memlock shim; override if needed
#
# All training hyperparameters are passed through to train.py. The complete
# argument list (name, default, meaning):
#
#   --model-name     Qwen/Qwen3-0.6B   ModelScope id (weights are NOT loaded;
#                                      the model is built from this config with
#                                      random init — from-scratch pretraining)
#   --data-path      ../alpaca_zh/alpaca_data_zh_51k.json
#                                      alpaca_zh json (instruction/input/output)
#   --seq-len        2048              sequence length after packing
#   --micro-batch    2                 micro-batch size per NPU per step
#   --grad-accum     8                 gradient accumulation steps; feeds
#                                      train_batch_size in ds_config — DeepSpeed
#                                      derives its internal GAS from the batch
#                                      sizes (global batch = micro-batch x NPUs
#                                      x grad-accum)
#   --steps          500               total optimizer steps
#   --lr             3e-4              peak learning rate
#   --warmup-steps   50                linear warmup steps
#   --eval-interval  50                run eval every N steps
#   --eval-seqs      64                held-out packed sequences for eval
#   --log-interval   10                print loss every N steps
#   --nvme-path      nvme_offload      dir for optimizer NVMe swap files
#   --ds-config      ds_config.json    DeepSpeed config (ZeRO-3 + offloads)
#   --seed           42                random seed
#   --smoke-test     (flag)            3 steps, seq-len 128 — sanity check
#   --local_rank     (auto)            set by the DeepSpeed launcher
#
# Environment notes:
#   * huggingface.co is unreachable here — the model/tokenizer come from
#     ModelScope (snapshot_download), cached under ~/.cache/modelscope.
#   * This host caps the memlock rlimit (ulimit -l = 64 MB), while the NVMe
#     swap pool page-locks ~1.2 GB/rank. The bundled mlock_shim.so turns
#     mlock() into a no-op (memory just isn't page-locked). It is harmless on
#     hosts with a normal rlimit; unset LD_PRELOAD to disable.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- NPU selection (customize here or via NPU_IDS env) -----------------------
NPU_IDS="${NPU_IDS:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_IDS"
NUM_NPUS=$(echo "$NPU_IDS" | tr ',' '\n' | grep -c '[0-9]')
echo "[run_train] NPUs: $NPU_IDS (${NUM_NPUS} cards)"
# if echo "$NPU_IDS" | grep -qE '(^|,)[4-7](,|$)'; then
#   echo "[run_train] WARNING: NPU_IDS includes cards 4-7; confirm with 'npu-smi info' that no other job is using them." >&2
# fi

# # --- memlock shim (see header); prepend so an existing LD_PRELOAD is kept ---
export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$SCRIPT_DIR/mlock_shim.so"

# --- distributed settings ----------------------------------------------------
export MASTER_PORT="${MASTER_PORT:-29500}"

# --- launch ------------------------------------------------------------------
exec uv run deepspeed \
  --num_gpus "$NUM_NPUS" \
  train.py \
  --steps 10 \
  --grad-accum 1 \
  --log-interval 1
