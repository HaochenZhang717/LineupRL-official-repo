from __future__ import annotations

import argparse
from pathlib import Path

from .captioners import build_captioner, write_captions_jsonl
from .datasets import load_dataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline 3 Stage A: series -> captions.jsonl.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--task", choices=["classification", "forecasting"], required=True)
    p.add_argument("--captioner", choices=["mock", "vlm"], default="mock")
    p.add_argument("--captioner-ckpt", default=None, help="HF checkpoint (vlm only)")
    p.add_argument("--prompt-family", choices=["qa", "decodability"], default="qa")
    p.add_argument("--captioner-tp", type=int, default=1,
                   help="vLLM tensor_parallel_size; 4 for the 72B teacher arm, 1 for 3B")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-tokens-caption", type=int, default=1024)
    p.add_argument("--run-name", default="base")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=2020)
    return p


def main(argv=None) -> Path:
    args = build_argparser().parse_args(argv)
    items = load_dataset(args.dataset, args.task, lookback=args.lookback, horizon=args.horizon,
                         stride=args.stride, max_windows=args.max_windows, seed=args.seed,
                         limit=args.limit)
    ds_tag = args.dataset.replace(":", "_").replace("/", "_")
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path("results/eval_protocol/pipeline3") / ds_tag / args.run_name

    if args.captioner == "vlm":
        if not args.captioner_ckpt:
            raise SystemExit("--captioner vlm requires --captioner-ckpt")
        captioner = build_captioner("vlm", ckpt=args.captioner_ckpt,
                                    prompt_family=args.prompt_family,
                                    gpu_memory_utilization=args.gpu_memory_utilization,
                                    max_tokens=args.max_tokens_caption,
                                    image_root=out_dir / "render",
                                    tensor_parallel_size=args.captioner_tp)
    else:
        captioner = build_captioner("mock")

    captions = captioner(items)
    path = out_dir / "captions.jsonl"
    write_captions_jsonl(path, [it.id for it in items], captions)
    print(f"wrote {len(captions)} captions -> {path}")
    return path


if __name__ == "__main__":
    main()
