# decodability_rl/sft — the SFT baseline

The control arm for LineupRL: same backbone (`Qwen2.5-VL-3B-Instruct`), same 9,500/500
charts and split (it reads the RL manifests directly), same instruction
(`decodability_rl/rl/cap_prompt.py:TS_CAP_PROMPT`, no system message), but the learning
signal is imitation of a Qwen2.5-VL-72B-Instruct teacher's greedy captions instead of the
lineup reward. `bash scripts/train_sft.sh` runs the whole pipeline; the steps are:

```bash
# 1. teacher captions on the RL train/val charts (vLLM, TP=4 for the 72B)
python -m decodability_rl.sft.gen_teacher_captions \
    --dataset out_10k_xdomain/rl_decod/val_messages.jsonl,out_10k_xdomain/rl_decod/train_messages.jsonl \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --out out_10k_xdomain/sft_teacher/captions_val.jsonl,out_10k_xdomain/sft_teacher/captions_train.jsonl \
    --tp 4 --resume

# 2. quality checks (length, truncation, template collapse, duplicates) -> sft_{train,val}.jsonl + teacher_qc.md
python -m decodability_rl.sft.build_sft_dataset --dir out_10k_xdomain/sft_teacher

# 3. optional: put the teacher on the same axis as the RL arms (needs a running reward server)
python -m decodability_rl.sft.score_captions \
    --captions out_10k_xdomain/sft_teacher/captions_val.jsonl \
    --dataset  out_10k_xdomain/rl_decod/val_messages.jsonl \
    --url "$REWARD_URL" --label teacher72b

# 4. full-parameter SFT (trl SFTTrainer, effective batch 64, 5 epochs, lr 1e-5, cosine)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
    -m decodability_rl.sft.train_sft \
    --train out_10k_xdomain/sft_teacher/sft_train.jsonl \
    --val   out_10k_xdomain/sft_teacher/sft_val.jsonl \
    --output-dir outputs/sft
```

Checkpoint selection mirrors the RL arm: checkpoints are saved and evaluated every 50
steps and the reported model is the lowest-validation-loss checkpoint
(`load_best_model_at_end`). `make_caption_comparison.py` writes a side-by-side markdown of
base / RL / SFT / teacher captions with code-computed ground truth for a few downstream
series.

Requirements beyond the RL arm: `trl==0.24`, `datasets`, `accelerate`, `vllm` and
`qwen_vl_utils` for the teacher step.
