#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
ARM="${ARM:?set ARM}"; CKPT="${CKPT:?set CKPT}"
TSF="${TSF:-bench_data/TSF}"
OUT_ROOT="${OUT_ROOT:-results/eval_protocol}"
SUFFIX="${TRAIN_SEED:+_seed$TRAIN_SEED}"
FC_ROOT="$OUT_ROOT/pipeline3_forecast"             # captions of the lookback windows (shared by both readouts)
FC_OUT="$OUT_ROOT/pipeline3_forecast_notrunc$SUFFIX"
RC_OUT="$OUT_ROOT/ts_recon$SUFFIX"

declare -A SPEC=(
  [ETTh2]="ett:$TSF/ETT-small/ETTh2.csv:OT"
  [ETTm2]="ett:$TSF/ETT-small/ETTm2.csv:OT"
  [saugeen]="ett:$TSF/saugeen/saugeen.csv:OT"
  [aus_elec]="ett:$TSF/aus_elec/aus_elec.csv:OT"
)
COMMON=(--encoder-model google/embeddinggemma-300m --lookback 128 --horizon 16 --stride 1
        --max-windows 5000 --seed 2020 --epochs 30 --lora-r 16 --lr 3e-4)

for DS in ETTh2 ETTm2 saugeen aus_elec; do
  CAPS="$FC_ROOT/$DS/$ARM/captions.jsonl"
  if [ ! -s "$CAPS" ]; then                        # 1) caption the windows once (also runs the seed-2020 forecasting readout)
    python -m ts_downstream.run_pipeline3 --task forecasting --dataset "${SPEC[$DS]}" \
      --captioner vlm --captioner-ckpt "$CKPT" --captioner-tp "${TP:-1}" --prompt-family decodability \
      --run-name "$ARM" --out-dir "$FC_ROOT/$DS/$ARM" "${COMMON[@]}" --device cuda
  fi
  for TASK in forecasting reconstruction; do       # 2) readouts on the SAME captions
    DEST="$FC_OUT"; [ "$TASK" = reconstruction ] && DEST="$RC_OUT"
    python -m ts_downstream.readout_step --task "$TASK" --dataset "${SPEC[$DS]}" --captions "$CAPS" \
      --caption-control "${CONTROL:-none}" --shuffle-key "$DS" "${COMMON[@]}" \
      --batch-size 32 --micro-batch-size 8 --max-text-len 0 --fail-on-truncation \
      ${TRAIN_SEED:+--train-seed "$TRAIN_SEED"} --run-name "$ARM" --out-dir "$DEST/$DS/$ARM"
  done
done
