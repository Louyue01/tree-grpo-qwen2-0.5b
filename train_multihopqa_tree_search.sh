#!/bin/bash
# Tree-GRPO (tree search) multi-hop QA training — single-node, direct launch.
#
# Adapted from the original 8-GPU `ray job submit` version for a single 24GB GPU
# and Qwen2.5-0.5B-Instruct. Changes vs. the original:
#   * direct `python3 -m verl.trainer.main_ppo_format_ts` (the module calls ray.init()
#     itself), so no `ray job submit` / running ray head is required
#   * `trainer.logger` -> console only (drops the swanlab/cloud dependency)
#   * `lr` is now defined (was an undefined variable in the original script)
#   * BASE_MODEL / DATA_DIR / retriever.url / CUDA_VISIBLE_DEVICES / batch sizes are
#     parameterized for a single GPU
#
# Override any of the env vars below before running, e.g.:
#   BASE_MODEL=./sft_checkpoint/global_step_100 DATA_DIR=./data bash train_multihopqa_tree_search.sh

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DATA_DIR="${DATA_DIR:-$BASE_DIR/data}"
export BASE_MODEL="${BASE_MODEL:-$BASE_DIR/models/Qwen2.5-0.5B-Instruct}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-multihopqa-treesearch-qwen2.5-0.5b}"
export RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"

# RL learning rate for a 0.5B base model (the original script referenced an undefined `$lr`).
lr="${LR:-1e-5}"
total_training_steps="${TOTAL_TRAINING_STEPS:-180}"

export WG_BACKEND="${WG_BACKEND:-ray}"
# flash-attn is the dependency actually installed by the upstream README (xformers is not);
# override to XFORMERS only if you also install xformers.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_gsc_rpc_server_reconnect_timeout_s=100
# Reclaim CUDA allocator fragmentation. The actor backward pass OOM'd with ~2.9 GiB
# reserved-but-unallocated; expandable segments makes that memory usable again.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- single-GPU batch sizes (tunable; scaled down from the original 8-GPU values) ----
n_gpus_per_node=1
train_batch_size=16          # prompts per training step
val_batch_size=8
actor_ppo_mini_batch_size=8
actor_ppo_micro_batch_size=1
log_prob_micro_batch_size=1

# Tree search hyper-parameters: m initial chains, expand n non-leaf nodes for l levels,
# then sample k leaves per tree.
tree_search_m=2
tree_search_n=2
tree_search_l=1
tree_search_k=3

mkdir -p "$BASE_DIR/verl_log" "$BASE_DIR/verl_checkpoints"
ulimit -n 65535

python3 -m verl.trainer.main_ppo_format_ts \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=tree \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tree_search=true \
    actor_rollout_ref.rollout.ts_m=$tree_search_m \
    actor_rollout_ref.rollout.ts_n=$tree_search_n \
    actor_rollout_ref.rollout.ts_l=$tree_search_l \
    actor_rollout_ref.rollout.ts_k=$tree_search_k \
    actor_rollout_ref.rollout.reward_mode=base \
    actor_rollout_ref.rollout.expand_mode=random \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="['console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=60 \
    trainer.project_name="Tree-GRPO" \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=$total_training_steps \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$BASE_DIR/verl_checkpoints/$EXPERIMENT_NAME" \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    max_turns=3 \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk=3 \
    2>&1 | tee "verl_log/$EXPERIMENT_NAME.log"
