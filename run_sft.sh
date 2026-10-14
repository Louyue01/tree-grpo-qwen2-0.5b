#!/bin/bash
# SFT format-warmup: teach Qwen2.5-0.5B-Base the ReAct `<think>/<search>/<answer>` format
# before GRPO / Tree-GRPO. Runs the repo's single-file FSDP SFT trainer.
#
# Requires scripts/make_sft_data.py to have produced $DATA_DIR/sft.parquet first.
#
# Env vars (with defaults):
#   CUDA_VISIBLE_DEVICES  (default 0)
#   BASE_MODEL            base model to warm up (default ./models/Qwen2.5-0.5B)
#   DATA_DIR              dir holding sft.parquet (default ./data)
#   OUTPUT_DIR            HF checkpoint output dir (default ./sft_checkpoint)
#   TOTAL_EPOCHS          (default 3)

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export BASE_MODEL="${BASE_MODEL:-$BASE_DIR/models/Qwen2.5-0.5B}"
export DATA_DIR="${DATA_DIR:-$BASE_DIR/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/sft_checkpoint}"
total_epochs="${TOTAL_EPOCHS:-3}"

cd "$BASE_DIR"
mkdir -p "$OUTPUT_DIR"

torchrun --standalone --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$DATA_DIR/sft.parquet" \
    data.val_files="$DATA_DIR/sft.parquet" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    data.truncation=left \
    data.train_batch_size=8 \
    data.micro_batch_size=2 \
    model.partial_pretrain="$BASE_MODEL" \
    model.enable_gradient_checkpointing=true \
    model.trust_remote_code=false \
    optim.lr=1e-5 \
    trainer.project_name="sft-warmup" \
    trainer.experiment_name="qwen2.5-0.5b-multihop" \
    trainer.total_epochs="$total_epochs" \
    trainer.logger="['console']" \
    trainer.default_local_dir="$OUTPUT_DIR"

echo ""
echo "SFT finished. Checkpoints are under: $OUTPUT_DIR"
echo "Point the RL scripts at the final checkpoint, e.g.:"
echo "  BASE_MODEL=$OUTPUT_DIR/global_step_<N> bash train_multihopqa_tree_search.sh"
