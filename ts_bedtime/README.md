# ts_bedtime — BEDTime evaluation

Caption-quality evaluation on BEDTime (*BEDTime: A Unified Benchmark for Automatically
Describing Time Series*, arXiv:2509.05215). Data: the four public CSVs from
<https://huggingface.co/datasets/HartvigsenGroup/BEDTime> (MIT), placed under
`bench_data/bedtime/`. NICU-HR is not in the public release, so everything here covers
four datasets (truce_stock, truce_synthetic, sushi, taxosynth; 10,164 series).

## What the package does

- **Preparation** (`prepare.py`, `data.py`, `negatives.py`, `prompts.py`): subsample 500
  series per dataset (seed 2020) and build BEDTime's recognition (True/False) and
  differentiation (4-way) items. Distractors are ported from the upstream negative-sampling
  code (S-BERT primary, Euclidean as a robustness arm) with two extra guards: a distractor
  is never another annotation of the same series, and the cross-class constraint is
  always applied on sushi/taxosynth. Question text is built at load time from
  `prompts.py`, so items store only ids.
- **Identification MCQ metrics** (`mcq_metrics.py`): two 4-way tasks scored by a frozen
  text reader over the same items and the same distractor series for every arm.
  Metric A: caption -> pick the true series out of 4. Metric B: series -> pick the true
  caption out of 4 captions from the same captioner. Distractors are the three nearest
  neighbours in z-scored (mean, std, min, max) space within the same dataset and length,
  a pure function of the series ids. **Four-rotation protocol**: every item is asked four
  times with the gold option moved to A, B, C and D (distractors kept in order), so a
  position prior scores exactly 0.25 and an item scores 0, .25, .5, .75 or 1. Controls:
  `--controls empty` (placeholder caption; measures candidate-set leakage and the position
  prior), `--controls mismatch` (another series' caption from the same arm), and
  `--order random` (one seeded permutation per item, for comparison with full rotation).
  Sushi (2048-point series) is excluded because a 4-way option set does not fit the
  reader's context.
- **NLI entailment scorer** (`nli_score.py`): BEDTime task 3 (open generation). Scores
  each caption against the dataset's reference descriptions with an NLI model in both
  directions (`gen_entails_gt`, `gt_entails_gen`) plus the strict bidirectional case, with
  optional length/shuffle controls.
- **Cross-verifier scripts** (`xverifier_*.py`): re-score the identification metrics
  under several reader families and the controls above, then report whether the arm
  ranking depends on the reader. `mcq_metrics.py --reader ... --reader-tag <tag>` writes
  to `results/eval_protocol/xverifier/<bench>/<tag>/`; `xverifier_status.py` lists which
  (bench, reader, arm) cells exist; `xverifier_report.py` builds the per-bench report
  (bootstrap CIs, Spearman rank agreement between readers, control comparisons);
  `xverifier_csv.py` flattens every cell into CSVs.
- **Summary / report scripts**: `report.py` (dataset x task x arm cross-tab with the
  L0/L2 admission gate and information-recovery rate), `mcq_report.py` (the three caption
  metrics in one table), `all_models_report.py` (every arm and off-the-shelf VLM on
  BEDTime and CaTS), `full_table.py`, `final_analysis.py` (paired McNemar tests with
  Bonferroni correction), `criteria12_sig.py` (paired tests for the identification table),
  `diff_rotated.py` / `diff_rot_report.py` (differentiation under full gold-position
  rotation), `answer_examples.py` / `task_examples.py` (worked prompt/answer examples),
  `nli_length_probe.py` (does caption length explain the NLI gap), and
  `rl_arm_caption_cases.py` (case study of the three RL rewards' captions).

## Commands

```bash
# 0. build items (CPU); --synthetic gives an offline smoke test
python -m ts_bedtime.prepare --strategy sbert
python -m ts_bedtime.prepare --strategy euclid
python -m ts_bedtime.prepare --synthetic 20 --strategy euclid --out-dir /tmp/bt

# 1. caption pass + caption-mediated QA (Protocol A) and direct VQA (Protocol B)
#    run through ts_eval/run_eval.py and ts_eval/vlm_direct_step.py; see scripts/eval_bench.sh

# 2. identification MCQ metrics (four-rotation protocol, published configuration)
python -m ts_bedtime.mcq_metrics --arm decod550 --reader Qwen/Qwen2.5-14B-Instruct
#    cross-verifier cell: another reader plus controls, routed under --reader-tag
python -m ts_bedtime.mcq_metrics --arm decod550 --reader <hf-model> --reader-tag <tag> \
    --controls none,empty,mismatch --order rotation,random

# 3. NLI entailment (task 3)
python -m ts_bedtime.nli_score --run-name decod550 --variant deployed --controls \
    --captions results/eval_protocol/bedtime/decod550/captions.jsonl \
    --out results/eval_protocol/bedtime/decod550/generation_deployed.json

# 4. cross-verifier status / report / CSV
python -m ts_bedtime.xverifier_status
python -m ts_bedtime.xverifier_report --bench bedtime
python -m ts_bedtime.xverifier_report --bench cats
python -m ts_bedtime.xverifier_csv

# 5. summaries and reports
python -m ts_bedtime.report --arms base,decod550,sft
python -m ts_bedtime.mcq_report > results/eval_protocol/bedtime_three_metrics.md
python -m ts_bedtime.all_models_report --out results/eval_protocol/bedtime_cats_all_models.md
python -m ts_bedtime.full_table --out results/eval_protocol/caption_metrics_full.md
python -m ts_bedtime.final_analysis > results/eval_protocol/bedtime_tables.md
python -m ts_bedtime.criteria12_sig --out results/eval_protocol/criteria12_significance.md
python -m ts_bedtime.diff_rotated --arm decod550
python -m ts_bedtime.diff_rot_report --out results/eval_protocol/bedtime_differentiation.md
python -m ts_bedtime.answer_examples --out results/eval_protocol/bedtime_answer_examples.md
python -m ts_bedtime.task_examples --out results/eval_protocol/bedtime_task_examples.md
python -m ts_bedtime.nli_length_probe 250
python -m ts_bedtime.rl_arm_caption_cases --out results/eval_protocol/rl_reward_caption_cases.md
```

## Notes

- Protocol A (caption-mediated) is the target of the identification reward; Protocol B
  (the VLM answers directly from the chart) removes the caption step. The two have
  different floors: L0 = empty caption for Protocol A, `--no-image` for Protocol B.
  Check `unparsed_rate` in `report.json` before reading Protocol B accuracies.
- Admission gate for Protocol A cells: `acc_L2 - acc_L0 >= 0.10`, where L2 is the same
  answerer given the raw values. Cells below the gate are reported but flagged.
- Metric A reuses the identification training prompt verbatim, so `decod550` has
  home-field advantage there by construction; Metric B and the NLI scores do not.
- taxosynth: 42 series containing `nan` (all `outliers/temporal_disruption`) are dropped
  because the renderer rejects non-finite values.
