#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ARM="${ARM:?set ARM}"
PORT="${PORT:-8890}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-8900}"
LOG_DIR="${LOG_DIR:-outputs/reward_$ARM}"; mkdir -p "$LOG_DIR"
NGPU=$(python -c "import os;print(len([g for g in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if g]))")
[ "$NGPU" -ge 1 ] || { echo "set CUDA_VISIBLE_DEVICES"; exit 1; }

case "$ARM" in
  lineuprl) MODEL=Qwen/Qwen2.5-14B-Instruct; K=4 ;;
  rm7b)    MODEL=Qwen/Qwen2.5-7B-Instruct;  K=4 ;;
  rm3b)    MODEL=Qwen/Qwen2.5-3B-Instruct;  K=4 ;;
  neg6)    MODEL=Qwen/Qwen2.5-14B-Instruct; K=6 ;;
  neg8)    MODEL=Qwen/Qwen2.5-14B-Instruct; K=8 ;;
  neg10)   MODEL=Qwen/Qwen2.5-14B-Instruct; K=10 ;;
  negrand) MODEL=Qwen/Qwen2.5-14B-Instruct; K=4 ;;
  judge)   MODEL=Qwen/Qwen2.5-14B-Instruct ;;
  answerability) MODEL="${REWARD_MODEL:-Qwen/Qwen2.5-3B-Instruct}" ;;
  *) echo "unknown ARM $ARM"; exit 1 ;;
esac
MODEL="${REWARD_MODEL:-$MODEL}"
PIDS=(); trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT INT TERM

if [ "$ARM" = answerability ]; then
  python ts_rl/reward_server_ts.py --role worker --reward_pretrain "$MODEL" --all_qa --tp 1 \
      --port "$PORT" --worker_base_port "$WORKER_BASE_PORT" >"$LOG_DIR/worker.log" 2>&1 &
  PIDS+=($!)
  sleep "${WARMUP:-120}"
  exec python ts_rl/reward_server_ts.py --role master --num_workers "$NGPU" --port "$PORT" \
      --worker_base_port "$WORKER_BASE_PORT" --worker_hosts 127.0.0.1
fi

if [ "$ARM" = judge ]; then
  MODULE=decodability_rl.rl.reward_server_judge
  COMMON=(--reward-pretrain "$MODEL" --batch-size 32 --max-new-tokens 8 --reward-scale 2.0
          --group-stats-jsonl "$LOG_DIR/group_stats.jsonl")
else
  MODULE=decodability_rl.rl.reward_server_decodability
  COMMON=(--reward_pretrain "$MODEL" --n-options "$K" --batch-size 16 --score-mode generate
          --max-new-tokens 8 --n-rotations 4 --reward-scale 2.0 --reward-transform raw
          --answer-format letter --resample-negatives -1 --number-mask-weight 0.0
          --mask-api-model "" --group-stats-jsonl "$LOG_DIR/group_stats.jsonl")
fi

for ((r = 0; r < NGPU; r++)); do
  python -m "$MODULE" --role worker --worker-rank "$r" --worker-base-port "$WORKER_BASE_PORT" \
    "${COMMON[@]}" >"$LOG_DIR/worker_$r.log" 2>&1 &
  PIDS+=($!)
done
sleep "${WARMUP:-240}"   # weights load before the master routes to the replicas
echo "REWARD_URL=http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/get_reward"
python -m "$MODULE" --role master --num-workers "$NGPU" --worker-base-port "$WORKER_BASE_PORT" \
  --port "$PORT" "${COMMON[@]}"
