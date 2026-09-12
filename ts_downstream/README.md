# ts_downstream — Evaluation Pipeline 3 (downstream-task caption utility)

Feed **only the caption** into a small LoRA-tuned text encoder + MLP readout and measure how
well it performs a real time-series task. Better caption ⇒ better task performance. Complements
pipeline 1 (QA-answerability, `ts_eval/`) and pipeline 2 (decodability, `decodability_rl/`).

## Two stages (kept as separate processes, like pipeline 1)

```
Stage A  caption_step   dataset series --(render+VLM | mock)--> captions.jsonl {id, caption}
Stage B  readout_step   captions.jsonl + targets --EmbeddingGemma(+LoRA)+MLP--> metrics.json
```

- **Caption-only** input (no numeric series) — isolates caption information, avoids the
  text-enhanced-forecasting question.
- **One architecture for all tasks:** `google/embeddinggemma-300m` (LoRA) → mean-pool → MLP.
  classification → CE, macro-F1/accuracy; forecasting / reconstruction → MSE, MSE/MAE.
- **Per-captioner readout, fixed recipe** across captioners (fair measuring instrument).
- Univariate only.

## Smoke test (offline: no GPU, no vLLM, no download)

```bash
python -m pytest tests/test_pipeline3_smoke.py -q
# or the full CLI, all mocked:
python -m ts_downstream.run_pipeline3 --dataset synthetic --task classification \
    --captioner mock --encoder-model mock --run-name base --device cpu
```

`--captioner mock` emits a deterministic stat sentence; `--encoder-model mock` is a hashed
bag-of-tokens encoder. Both exist purely to prove the plumbing runs end to end.

## Real run (GPU)

```bash
# accept the EmbeddingGemma license on HF for the account behind HF_TOKEN (.env), then:
python -m ts_downstream.run_pipeline3 \
    --dataset ucr:GunPoint --task classification \
    --captioner vlm --captioner-ckpt checkpoints/decod550_hf --prompt-family decodability \
    --encoder-model google/embeddinggemma-300m --run-name decod550

python -m ts_downstream.run_pipeline3 \
    --dataset ett:bench_data/TSF/ETT-small/ETTh2.csv:OT --task forecasting --horizon 96 \
    --captioner vlm --captioner-ckpt checkpoints/decod550_hf --prompt-family decodability \
    --encoder-model google/embeddinggemma-300m --run-name decod550
```

Run once per captioner checkpoint (`base` / `sft` / `decod550` / ...) and compare the metrics.
Stage B can be run on its own against an existing `captions.jsonl` with
`python -m ts_downstream.readout_step --captions <path> ...`; `--caption-control null|shuffled`
gives the no-information controls and `--max-text-len 0` disables caption truncation.

## Datasets

- `synthetic` — built-in, offline (smoke).
- `ucr:<Name>` — UCR-128 univariate via `sktime` (e.g. `ucr:GunPoint`, `ucr:ECG200`).
- `ett:<csv>[:<target>]` — Autoformer-style CSV, univariate target column (default `OT`).
  `python -m ts_downstream.fetch_forecast_datasets --out-dir bench_data/TSF` downloads the
  four forecasting/reconstruction datasets (ETTh2, ETTm2, saugeen, aus_elec) in this format.

## Files

| file | role |
|---|---|
| `config.py` | `ReadoutConfig`, `TrainConfig`, `DataConfig` |
| `datasets.py` | `DownstreamItem`, synthetic/UCR/ETT loaders, windowing, scaling |
| `captioners.py` | `CaptionFn`, `MockStatCaptioner`, `VLMCaptioner` (vLLM), captions.jsonl IO |
| `metrics.py` | classification (acc/macro-F1), forecasting (MSE/MAE, naive skill) |
| `readout.py` | `MockEncoder`, `HFEncoder` (EmbeddingGemma+LoRA), `ReadoutModel` |
| `caption_step.py` | Stage A CLI |
| `caption_many.py` | Stage A for many datasets with one vLLM engine |
| `readout_step.py` | Stage B CLI (training loop, controls, report) |
| `run_pipeline3.py` | orchestrator (A → B) |
| `mcq_readout.py` | alternative readout: a small LLM answers a multiple-choice question about the caption |
| `fetch_forecast_datasets.py` | download the forecasting/reconstruction CSVs |
| `export_csv.py`, `seed_report.py`, `notrunc_report.py`, `collect_len_rerun.py` | result collection and reports |
