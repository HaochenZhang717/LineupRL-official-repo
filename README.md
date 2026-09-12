# LineupRL: Verifiable RL for Time Series Captioning via Caption-to-Series Identification

Code and configuration behind the paper. A 3B vision-language
model (Qwen2.5-VL-3B-Instruct) writes a caption for a bare line chart of a univariate
series; a frozen 14B text LLM reads **only the caption** and the raw values of K candidate
series and must identify the described one. The K-1 distractors are real series of the
same length that are nearest to the target in (mean, std, min, max). The verifier's
identification accuracy, averaged over the K placements of the target, is the reward
(RLOO, OpenRLHF).

In the code and in the result files the method arm is called `decod550` (step 550 of the
"decodability" reward run). The other arms are `sft` (SFT on the 72B teacher's captions),
`rl650` (answerability reward), `judge550` (LLM-as-judge reward), `base` (untuned
initialization), `teacher72b` (Qwen2.5-VL-72B), and the ablations `rm7b`, `rm3b`
(7B / 3B verifier), `neg6`, `neg8` (K = 6 / 8) and `negrand` (random distractors).

## Layout

| path | what |
|---|---|
| `configs/lineuprl.yaml` | the reported recipe: data, reward server, trainer, validation |
| `configs/ablations.yaml` | verifier size, candidate count and random-distractor arms; the verifier x K grid |
| `configs/baselines.yaml` | answerability RL, LLM-as-judge RL, SFT, off-the-shelf captioners |
| `configs/evaluation.yaml` | the three criteria (entailment, identification, caption-only readout) |
| `scripts/` | `build_data.sh`, `reward_server.sh`, `train_rl.sh`, `train_sft.sh`, `eval_bench.sh`, `eval_downstream.sh` |
| `ts_render/` | TSFragment loader and the chart renderer (no title, no value labels, few ticks) |
| `decodability_rl/rl/` | **the method**: dataset builder with nearest-neighbour distractors, identification reward server, caption instruction (`cap_prompt.py`), judge reward server, verifier x K tools |
| `decodability_rl/sft/` | SFT baseline: teacher captions, filtering, full-parameter fine-tuning |
| `decodability_rl/test_reward_design/` | the offline pilot that chose the verifier and the distractor construction |
| `ts_qa/`, `ts_rl/` | answerability baseline: VLM-written questions, code-verified by a tool-driving text LLM, CapRL-style answer-accuracy reward server |
| `ts_eval/` | benchmark adapters, captioning drivers (VLM / text LLM / ChatTS), CaTS reference rewriting (`cats_agnostic/`) |
| `ts_bedtime/` | BEDTime preparation, identification metric (four-rotation protocol), NLI entailment scorer, cross-verifier scripts, report generators |
| `ts_downstream/` | caption-only forecasting / reconstruction readout (EmbeddingGemma-300M + LoRA + MLP head), dataset fetcher |
| `ts_cap/` | local VLM captioning helpers used by the off-the-shelf captioners |
| `third_party/CapRL_Training/` | the OpenRLHF fork used for training (Apache 2.0, see `LICENSE.CapRL`) |
| `out_10k_xdomain/fragments.jsonl` | the exact 10,000 TSFragment-600K fragments (ids, raw values) behind every trained arm |
| `data/cats/` | the CaTS-Bench series file and the number-free rewrites of its references used by Criterion 1 |
| `tests/` | unit tests (`python -m pytest tests -q`) |

## Environment

Two environments were used because the vLLM versions conflict: one for rendering,
reward servers and evaluation (torch 2.8, transformers 4.57, vLLM 0.10.2) and one for
training only (vLLM 0.11.0, ray 2.49.1, deepspeed 0.16.3, flash-attn 2.8.3).
`requirements.txt` lists the CPU-side dependencies; `decodability_rl/rl/README.md` gives
the model downloads and the server-side setup. Training imports the fork with
`PYTHONPATH=third_party/CapRL_Training:.`.

## Reproducing LineupRL

```bash
bash scripts/build_data.sh                    # render the 10k charts, build the RL datasets
CUDA_VISIBLE_DEVICES=0,1,2,3 ARM=lineuprl bash scripts/reward_server.sh      # 14B verifier, K=4 -> REWARD_URL
python -m decodability_rl.rl.smoke_test_reward --url $REWARD_URL --n 20     # a generic caption must score like an empty one
CUDA_VISIBLE_DEVICES=4,5,6,7 ARM=lineuprl REWARD_URL=$REWARD_URL bash scripts/train_rl.sh
```

The reported checkpoint is rollout step 550 (validation every 50 steps on the 500
held-out charts with fixed distractors; `configs/lineuprl.yaml` lists the reference
values). Ablation arms use the same two commands with `ARM=rm7b|rm3b|neg6|neg8|negrand`,
the judge and answerability baselines with `ARM=judge|answerability`, SFT with
`scripts/train_sft.sh`.

## Evaluation

```bash
python -m ts_bedtime.prepare --strategy sbert          # BEDTime bank
python -m ts_eval.build_cats_series                    # CaTS-Bench series file
python -m ts_downstream.fetch_forecast_datasets --out-dir bench_data/TSF   # ETTh2, ETTm2, saugeen, aus_elec
ARM=lineuprl CKPT=<hf checkpoint dir> bash scripts/eval_bench.sh           # criteria 1 and 2
ARM=lineuprl CKPT=<hf checkpoint dir> bash scripts/eval_downstream.sh      # criterion 3
```

Identification is scored by frozen verifiers that took no part in training
(`ts_bedtime/mcq_metrics.py`; the paper reports the mean of GLM-4-9B and Phi-4-14B, and
`ts_bedtime/xverifier_*.py` collect the readings of five verifiers). CaTS-Bench entailment
is scored against number-free rewrites of the CaTS references
(`data/cats/cats_qualitative_refs.jsonl`; the rewriting pipeline and its prompts are in
`ts_eval/cats_agnostic/`). Every evaluation script writes its metric files under `results/`.

## License

MIT (see `LICENSE`). The training fork under `third_party/CapRL_Training/` keeps its own
Apache 2.0 license.
