#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/third_party/CapRL_Training:$PWD:${PYTHONPATH:-}"
ARM="${ARM:?set ARM}"
REWARD_URL="${REWARD_URL:?set REWARD_URL (…/get_reward)}"
POLICY="${POLICY:-models/Qwen2.5-VL-3B-Instruct}"     # local directory with the HF weights
NGPU="${NGPU:-$(python -c "import os;print(len([g for g in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if g]))")}"
[ "$NGPU" -ge 1 ] || { echo "set CUDA_VISIBLE_DEVICES"; exit 1; }
OUT="${OUT:-outputs/$ARM}"; mkdir -p "$OUT"

case "$ARM" in
  lineuprl|rm7b|rm3b|negrand|judge) DATASET=out_10k_xdomain/rl_decod ;;
  neg6|neg8|neg10)                 DATASET=out_10k_xdomain/rl_decod_k10 ;;
  answerability)                   DATASET=out_10k_xdomain/rl ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
[ "$ARM" = negrand ] && DATASET=out_10k_xdomain/rl_decod_negrand
DATASET="${DATASET_OVERRIDE:-$DATASET}"

NUM_EPISODES=4; SAVE_STEPS=50
PROMPT_FLAGS=(--cap_prompt "$(python -m decodability_rl.rl.cap_prompt)")
case "$ARM" in
  judge)         NUM_EPISODES=2 ;;                       # two passes over the training set for the judge arm
  answerability) NUM_EPISODES=8; SAVE_STEPS=25; PROMPT_FLAGS=() ;;   # upstream default instruction
esac

TRAIN_BS=128; MICRO_TRAIN=4
(( TRAIN_BS % (NGPU * MICRO_TRAIN) == 0 )) || { echo "train_batch_size 128 must divide by NGPU*4"; exit 1; }

python -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 --ref_num_gpus_per_node "$NGPU" \
  --actor_num_nodes 1 --actor_num_gpus_per_node "$NGPU" \
  --vllm_num_engines "$NGPU" --vllm_tensor_parallel_size 1 \
  --colocate_all_models --vllm_enable_sleep --vllm_gpu_memory_utilization 0.4 --enable_prefix_caching \
  --remote_rm_url "$REWARD_URL" \
  --pretrain "$POLICY" \
  --save_path "$OUT" --ckpt_path "$OUT/ckpt" --save_hf_ckpt --save_steps "$SAVE_STEPS" --load_checkpoint \
  --remove_pi_old --reward_type chart --exp_mode cap_v2 --cap_v2_multi_qa --cap_v2_norm --cap_v2_only_cap --cap_remote_reward \
  --format_weight 0.0 \
  --micro_train_batch_size "$MICRO_TRAIN" --train_batch_size "$TRAIN_BS" \
  --micro_rollout_batch_size 8 --rollout_batch_size 32 \
  --temperature 1.0 --n_caps_per_prompt 8 --n_samples_per_prompt 8 "${PROMPT_FLAGS[@]}" \
  --max_epochs 1 --num_episodes "$NUM_EPISODES" \
  --prompt_max_len 2048 --max_samples 100000 --generate_max_len 1024 \
  --advantage_estimator rloo --zero_stage 2 --bf16 \
  --actor_learning_rate 1e-6 --lr_warmup_steps 20 --init_kl_coef 0.0 \
  --prompt_data "$DATASET/train_messages.jsonl" --input_key message \
  --normalize_reward --gradient_checkpointing --train_vlm --attn_implementation flash_attention_2 \
  ${WANDB_API_KEY:+--use_wandb "$WANDB_API_KEY" --wandb_project "${WANDB_PROJECT:-lineuprl}" --wandb_run_name "$ARM"}

