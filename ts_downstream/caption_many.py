from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .captioners import build_captioner, write_captions_jsonl
from .datasets import load_dataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", required=True, help="path to a jobs JSON file, or inline JSON")
    p.add_argument("--captioner-ckpt", required=True)
    p.add_argument("--captioner-tp", type=int, default=1)
    p.add_argument("--prompt-family", choices=["qa", "decodability"], default="decodability")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-tokens-caption", type=int, default=1024)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--overwrite", action="store_true",
                   help="recaption datasets whose captions.jsonl already exists")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    raw = args.jobs
    jobs = json.loads(Path(raw).read_text()) if Path(raw).exists() else json.loads(raw)

    todo = []
    for j in jobs:
        out = Path(j["out_dir"]) / "captions.jsonl"
        if out.exists() and not args.overwrite:
            print(f"skip (exists): {out}", flush=True)
            continue
        todo.append(j)
    if not todo:
        print("nothing to caption")
        return

    captioner = build_captioner(
        "vlm", ckpt=args.captioner_ckpt, prompt_family=args.prompt_family,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens_caption, image_root="render",
        tensor_parallel_size=args.captioner_tp)

    for j in todo:
        out_dir = Path(j["out_dir"])
        t0 = time.time()
        items = load_dataset(j["dataset"], j["task"],
                             lookback=j.get("lookback", 96), horizon=j.get("horizon", 96),
                             stride=j.get("stride", 1), max_windows=j.get("max_windows"),
                             seed=args.seed, limit=j.get("limit"))
        captioner.image_root = out_dir / "render"
        captions = captioner(items)
        path = out_dir / "captions.jsonl"
        write_captions_jsonl(path, [it.id for it in items], captions)
        print(f"wrote {len(captions)} captions -> {path} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("CAPTION_MANY_DONE")


if __name__ == "__main__":
    main()
