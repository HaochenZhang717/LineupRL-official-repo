# decodability_rl/rl — the LineupRL reward and its tooling

The method is called LineupRL in the paper; in the code it is "decodability", and the
main arm is `decod550` (the step-550 checkpoint). The reward asks a frozen text reader to
pick the described series out of K candidate value lists given only the caption. Every
script below runs with the repository root as the working directory and
`PYTHONPATH=$PWD` (each script also inserts the root into `sys.path` itself).

## Scripts

| script | what it does | run |
|---|---|---|
| `cap_prompt.py` | The captioning instruction (`TS_CAP_PROMPT`) shared by rollout, validation and every evaluation; printed to stdout so shell scripts can read it. | `python -m decodability_rl.rl.cap_prompt` |
| `build_decodability_dataset.py` | Builds the RL manifests from `out_10k_xdomain/fragments.jsonl`: per item the true series, its nearest-neighbour negatives in standardised (mean, std, min, max) space, same length, same-source sliding windows filtered by `--max-overlap-frac`; val items carry a fixed set of negatives, train items a pool that the server resamples. `--negatives-order random` builds the random-negatives ablation with the identical split. Ends with a self-check that reproduces the trainer's parse. | `python -m decodability_rl.rl.build_decodability_dataset --val-size 500 --pool-size 8` (see `scripts/build_data.sh` for the K=10 and random-negative variants) |
| `reward_server_decodability.py` | The reward server (FastAPI, `POST /get_reward`, OpenRLHF's remote-reward contract). Each caption is asked K times with the true series rotated through every option; reward = accuracy x `--reward-scale`. `--role master/worker` runs one replica per GPU behind a master that resamples negatives and rotations per image. Optional `--number-mask-weight` scores a second, value-masked view of the caption (see `llm_mask.py`). | `bash scripts/reward_server.sh` with `ARM=lineuprl`, or directly `python -m decodability_rl.rl.reward_server_decodability --reward_pretrain Qwen/Qwen2.5-14B-Instruct --n-options 4 --port 8890` |
| `smoke_test_reward.py` | Pre-training gate: scores real val items under the dataset caption, a generic template and an empty caption and refuses if the gaps are not there. | `python -m decodability_rl.rl.smoke_test_reward --url http://127.0.0.1:8890/get_reward --n 20` |
| `llm_mask.py` | Value masking done by the reader model: the caption is cut into short chunks, the reader flags each number as a series value or not, code substitutes `<val>`. `main()` evaluates the masker on real captions. | `python -m decodability_rl.rl.llm_mask --model Qwen/Qwen2.5-14B-Instruct` |
| `negatives_v2.py` | Offline diagnostic: re-scores a checkpoint's val captions under alternative distractor rules and with numerals masked, to measure how much of the reward is paid for quoted values. | `caption` / `build` / `score` sub-commands, see the module docstring |
| `score_arm_views.py` | Scores caption files of several arms against a running server (full or masked view) and writes per-item and summary JSON. | `python -m decodability_rl.rl.score_arm_views --url ... --view full --arm decod550=data/ref_captions/captions_decod550.jsonl` |
| `judge_prompts.py` | Rubric and parser for the LLM-as-judge baseline reward. | imported |
| `reward_server_judge.py` | Judge reward server (same HTTP contract): the frozen reader grades one caption against the series values, 1-10, mapped to [0, 1] and scaled. Runs an ordering self-test before serving. | `bash scripts/reward_server.sh` with `ARM=judge`, or `python -m decodability_rl.rl.reward_server_judge --reward-pretrain Qwen/Qwen2.5-14B-Instruct --port 8890` |
| `judge_verify.py` | Offline check of the judge reward on real val items (described / generic / empty captions) before a server exists. | `python -m decodability_rl.rl.judge_verify --dataset out_10k_xdomain/rl_decod/val_messages.jsonl --n 16` |
| `build_reader_k_items.py` | Builds the items for the reader-K sweep: one file carrying all `--n-distractors` nested negatives per caption, for the RL val split (`--source tsfragment`) or an external benchmark (`--source jsonl`). | `python -m decodability_rl.rl.build_reader_k_items --n-distractors 11` |
| `reader_k_sweep.py` | Scores one items file against a server started with `--n-options K` and writes `summary_<arm>_k<K>.json` plus per-item accuracies. | `python -m decodability_rl.rl.reader_k_sweep --url ... --n-options 6 --items <items.jsonl> --out-dir results/decodability_rl/reader_k_sweep` |
| `reader_k_table.py` | Folds the per-K summaries of one domain into a table with paired confidence intervals. | `python -m decodability_rl.rl.reader_k_table --in-dir results/decodability_rl/reader_k_sweep` |
| `reader_k_multi.py` | The same across several domains. | `python -m decodability_rl.rl.reader_k_multi --domain train=<dir> --domain cats=<dir>` |
| `deepspeed_to_hf.py` | Rebuilds an HF checkpoint directory from a DeepSpeed ZeRO-2 `_actor` state when the HF export of that step is gone. | `python -m decodability_rl.rl.deepspeed_to_hf --state <ckpt_path>/_actor/global_step50/mp_rank_00_model_states.pt --config-src <run>/global_step100_hf --out <out>` |

`repro/indices_10k_xdomain.json` holds the exact ids of the 10,000 TSFragment series
used by the paper (the same ids are in `out_10k_xdomain/fragments.jsonl`).

## Order of operations

1. Render the charts and build the manifests: `bash scripts/build_data.sh`.
2. Start the reward server: `CUDA_VISIBLE_DEVICES=0,1,2,3 ARM=lineuprl bash scripts/reward_server.sh`
   (one replica of the 14B reader per visible GPU; replica count only changes throughput).
   Wait for `REWARD_URL=...` on stdout and `GET /health` to answer.
3. Gate: `python -m decodability_rl.rl.smoke_test_reward --url $REWARD_URL --n 20`.
   Expected on the val split (reward x2 / accuracy): dataset caption ~1.63 / 0.81, generic
   template ~0.65 / 0.32, empty caption ~0.65 / 0.32. The template must score what the empty
   caption scores; if it does not, do not train.
4. Train: `REWARD_URL=$REWARD_URL ARM=lineuprl bash scripts/train_rl.sh` (exact flags in
   `configs/lineuprl.yaml`). Validation every 50 steps with `ts_rl/eval_val_reward.py`
   (greedy, 500 val charts, fixed negatives); the reported checkpoint is step 550.

## Environment essentials

- Reward server: `torch`, `transformers>=4.57`, `fastapi`, `uvicorn`, `httpx`, `numpy`.
  It uses plain `transformers` generation (no vLLM), needs about 28 GB of GPU memory per
  replica for the bf16 14B reader, and answers with `max_new_tokens 8` -- the reply is one
  letter. Thinking models are not supported as readers (no first-token letter, ~800x slower).
- Training: OpenRLHF fork in `third_party/CapRL_Training` (`PYTHONPATH` must include it)
  with `vllm==0.11`, `ray`, `deepspeed`, `flash-attn` (or `--attn_implementation sdpa`).
  The policy `Qwen/Qwen2.5-VL-3B-Instruct` must be a local directory (`models/...`); the
  reader can be an HF id. Check the fork imports before launching:
  `PYTHONPATH=third_party/CapRL_Training python -c "from openrlhf.datasets import PromptDataset"`.
- Data building is CPU only (`numpy`, `matplotlib`, `datasets` for the HF download).
- The manifests store absolute image paths, so rebuild them after moving the data directory.
- Per rollout step the server answers `rollout_batch_size x n_samples_per_prompt x K`
  questions (32 x 8 x 4 = 1024 with the paper's settings); with four replicas of the 14B
  reader that is roughly 30 s per step, and the reward is the dominant term in step time.
- Rewards in the training log are sampled at temperature 1.0 and sit well below the greedy
  validation numbers; judge progress by the validation curve, not the training log.
