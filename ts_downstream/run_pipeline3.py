from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline 3: downstream-task caption utility (Stage A -> Stage B).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--task", choices=["classification", "forecasting"], required=True)
    p.add_argument("--captioner", choices=["mock", "vlm"], default="mock")
    p.add_argument("--captioner-ckpt", default=None)
    p.add_argument("--captioner-tp", type=int, default=1,
                   help="vLLM tensor_parallel_size for stage A (4 for the 72B teacher)")
    p.add_argument("--prompt-family", choices=["qa", "decodability"], default="qa")
    p.add_argument("--encoder-model", default="google/embeddinggemma-300m")
    p.add_argument("--run-name", default="base")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-text-len", type=int, default=None,
                   help="passthrough to readout_step; see its --help. Omit to keep that "
                        "step's own default (256).")
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--device", default="auto")
    return p


def _common(args) -> list[str]:
    ds_tag = args.dataset.replace(":", "_").replace("/", "_")
    out_dir = args.out_dir or str(Path("results/eval_protocol/pipeline3") / ds_tag / args.run_name)
    shared = ["--dataset", args.dataset, "--task", args.task, "--run-name", args.run_name,
              "--out-dir", out_dir, "--lookback", str(args.lookback),
              "--horizon", str(args.horizon), "--stride", str(args.stride),
              "--seed", str(args.seed)]
    if args.max_windows is not None:
        shared += ["--max-windows", str(args.max_windows)]
    if args.limit is not None:
        shared += ["--limit", str(args.limit)]
    return shared, out_dir


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    shared, out_dir = _common(args)
    py = sys.executable

    cap = [py, "-m", "ts_downstream.caption_step", "--captioner", args.captioner,
           "--prompt-family", args.prompt_family] + shared
    if args.captioner == "vlm":
        cap += ["--captioner-ckpt", args.captioner_ckpt or "",
                "--captioner-tp", str(args.captioner_tp)]
    subprocess.run(cap, check=True)

    read = [py, "-m", "ts_downstream.readout_step", "--encoder-model", args.encoder_model,
            "--captions", str(Path(out_dir) / "captions.jsonl"),
            "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
            "--lr", str(args.lr), "--lora-r", str(args.lora_r), "--device", args.device] + shared
    if args.max_text_len is not None:
        read += ["--max-text-len", str(args.max_text_len)]
    subprocess.run(read, check=True)


if __name__ == "__main__":
    main()
