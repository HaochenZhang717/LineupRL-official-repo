#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ARM="${ARM:?set ARM}"; CKPT="${CKPT:?set CKPT}"
READER="${READER:-Qwen/Qwen2.5-14B-Instruct}"
BENCH="${BENCH:-bench_data}"
OUT_ROOT="${OUT_ROOT:-results/eval_protocol}"
BT_DIR="$OUT_ROOT/bedtime/$ARM"; CATS_DIR="$OUT_ROOT/cats_mcq"; mkdir -p "$BT_DIR" "$CATS_DIR" "$OUT_ROOT/cats_gengt/nli_cats"

python -m ts_eval.run_eval --benchmark bedtime --captioner-ckpt "$CKPT" --answerer-ckpt "$READER" \
  --run-name "$ARM" --out-dir "$OUT_ROOT" --prompt-family decodability --temperature 0.0 \
  --max-tokens-caption 1024 --max-tokens-answer 16 --answer-style evidence \
  --batch-size 512 --gpu-memory-utilization 0.85 ${LIMIT:+--limit "$LIMIT"}
python -m ts_bedtime.mcq_metrics --arm "$ARM" --captions "$BT_DIR/captions.jsonl" \
  --out-dir "$OUT_ROOT/bedtime_mcq" --reader "$READER" --metrics AB --gpu-memory-utilization 0.90 ${LIMIT:+--limit "$LIMIT"}
python -m ts_eval.caption_step --benchmark bedtime --captioner-ckpt "$CKPT" --image-root "$BT_DIR/render" \
  --captions-out "$BT_DIR/captions_gen150.jsonl" --prompt-family bedtime_gen --temperature 0.0 \
  --max-tokens-caption 150 --batch-size 512 --gpu-memory-utilization 0.85 ${LIMIT:+--limit "$LIMIT"}
for V in deployed gen150; do
  CAPS="$BT_DIR/captions.jsonl"; [ "$V" = gen150 ] && CAPS="$BT_DIR/captions_gen150.jsonl"
  python -m ts_bedtime.nli_score --captions "$CAPS" --run-name "$ARM" --variant "$V" --controls \
    --batch-size 64 --out "$BT_DIR/generation_$V.json"
done

python -m ts_eval.caption_series_file --series "$BENCH/cats_bench/series.jsonl" --captioner-ckpt "$CKPT" \
  --image-root "$CATS_DIR/render" --captions-out "$CATS_DIR/captions_$ARM.jsonl" --prompt-family decodability \
  --temperature 0.0 --max-tokens-caption 1024 --batch-size 256 --gpu-memory-utilization 0.90 --tp "${TP:-1}" ${LIMIT:+--limit "$LIMIT"}
python -m ts_bedtime.mcq_metrics --arm "$ARM" --series "$BENCH/cats_bench/series.jsonl" \
  --captions "$CATS_DIR/captions_$ARM.jsonl" --exclude-datasets "" --pool-negatives-across-datasets \
  --out-dir "$CATS_DIR" --reader "$READER" --metrics AB --gpu-memory-utilization 0.90 ${LIMIT:+--limit "$LIMIT"}
python -m ts_bedtime.nli_score --captions "$CATS_DIR/captions_$ARM.jsonl" \
  --series data/cats/cats_qualitative_series.jsonl --run-name "$ARM" --variant deployed --controls \
  --batch-size 64 --out "$OUT_ROOT/cats_gengt/nli_cats/${ARM}_qualitative.json"

