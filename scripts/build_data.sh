#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
OUT="${OUT:-out_10k_xdomain}"

python -m ts_render.render_dataset \
    --source WinfredGe/TSFragment-600K --split train \
    --include-ids "$OUT/fragments.jsonl" \
    --out-dir "$OUT" --cache-dir "${HF_CACHE:-./hf_cache}" \
    --num-workers "${NUM_WORKERS:-8}"
test "$(wc -l < "$OUT/fragments.jsonl")" -eq 10000
test "$(ls "$OUT/render" | wc -l)" -eq 10000

python -m decodability_rl.rl.build_decodability_dataset \
    --input "$OUT/fragments.jsonl" --image-root "$OUT" --out-dir "$OUT/rl_decod" \
    --val-size 500 --pool-size 8 --n-distractors 3 --max-overlap-frac 0.2 --seed 0

python -m decodability_rl.rl.build_decodability_dataset \
    --input "$OUT/fragments.jsonl" --image-root "$OUT" --out-dir "$OUT/rl_decod_k10" \
    --val-size 500 --pool-size 9 --n-distractors 9 --max-overlap-frac 0.2 --seed 0

python -m decodability_rl.rl.build_decodability_dataset \
    --input "$OUT/fragments.jsonl" --image-root "$OUT" --out-dir "$OUT/rl_decod_negrand" \
    --val-size 500 --pool-size 9 --n-distractors 9 --max-overlap-frac 0.2 --seed 0 \
    --negatives-order random --neg-seed 1234

if [ -s data/answerability/qa_verified_keep.jsonl ]; then
  python -m ts_rl.build_rl_dataset \
      --input data/answerability/qa_verified_keep.jsonl --image-root "$OUT" \
      --out-dir "$OUT/rl" --min-qa 2 --val-size 500 --seed 0
else
  echo "skipping the answerability manifest: data/answerability/qa_verified_keep.jsonl not present"
fi
