#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
DATA=out_10k_xdomain/rl_decod
TEACHER_DIR="${TEACHER_DIR:-out_10k_xdomain/sft_teacher}"
OUT="${OUT:-outputs/sft}"
NGPU="${NGPU:-$(python -c "import os;print(len([g for g in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if g]))")}"

python -m decodability_rl.sft.gen_teacher_captions \
    --dataset "$DATA/val_messages.jsonl,$DATA/train_messages.jsonl" \
    --model "${TEACHER:-Qwen/Qwen2.5-VL-72B-Instruct}" \
    --out "$TEACHER_DIR/captions_val.jsonl,$TEACHER_DIR/captions_train.jsonl" \
    --tp "${TP:-4}" --temperature 0.0 --max-tokens 1024 --max-model-len 4096 --gpu-mem-util 0.90 --resume

python -m decodability_rl.sft.build_sft_dataset --dir "$TEACHER_DIR"

PDBS=2; GRAD_ACCUM=$(( 64 / (PDBS * NGPU) )); [ "$GRAD_ACCUM" -ge 1 ] || GRAD_ACCUM=1
MULTI=--multi_gpu; [ "$NGPU" -eq 1 ] && MULTI=""
accelerate launch $MULTI --num_processes "$NGPU" --num_machines 1 \
  -m decodability_rl.sft.train_sft \
  --train "$TEACHER_DIR/sft_train.jsonl" --val "$TEACHER_DIR/sft_val.jsonl" \
  --base-model "${POLICY:-models/Qwen2.5-VL-3B-Instruct}" --output-dir "$OUT" \
  --epochs 5 --lr 1e-5 --per-device-batch-size "$PDBS" --grad-accum "$GRAD_ACCUM" \
  --max-length 2048 --warmup-ratio 0.05 --save-steps 50 --eval-steps 50 --seed 0
