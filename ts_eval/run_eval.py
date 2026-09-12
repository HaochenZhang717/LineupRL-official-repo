from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ts_eval.benchmarks import REGISTRY
from ts_eval.caption_answer_harness import ANSWER_STYLES, CAPTION_PROMPT_FAMILIES

DEFAULT_ANSWERER = "./models/Qwen2.5-3B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--captioner-ckpt", default=None,
                    help="RL policy checkpoint dir (e.g. .../global_step650_hf) or base "
                         "Qwen2.5-VL-3B-Instruct dir. Only required without "
                         "--skip-caption-step (reference legs written by "
                         "ts_eval.make_reference_captions have no captioner at all)")
    ap.add_argument("--answerer-ckpt", default=DEFAULT_ANSWERER)
    ap.add_argument("--run-name", default=None, help="defaults to the captioner checkpoint's dir name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default="results/eval_protocol/pipeline1")
    ap.add_argument("--image-root", default=None, help="defaults to <out-dir>/<benchmark>/<run-name>/render")
    ap.add_argument("--i-accept-unclear-license", action="store_true",
                    help="required for --benchmark tsrbench (no confirmed license, see ts_eval/benchmarks/tsrbench.py)")
    ap.add_argument("--hf-token", default=None, help="for --benchmark time_mqa (gated dataset)")
    ap.add_argument("--prompt-family", default="qa", choices=list(CAPTION_PROMPT_FAMILIES),
                    help="prompt the captioner was trained with (see "
                         "ts_eval/caption_answer_harness.py:caption_prompt_pair)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens-caption", type=int, default=1024)
    ap.add_argument("--max-tokens-answer", type=int, default=64)
    ap.add_argument("--max-tokens-judge", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                    help="applied to each step's engine independently, not summed")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size for the CAPTIONER engine only; the answerer "
                         "step stays at 1 (a 14B answerer fits one card, and holding 4 for "
                         "it would idle 3)")
    ap.add_argument("--answer-style", default="caption_qa", choices=list(ANSWER_STYLES),
                    help="how the evidence is framed for the answerer; see "
                         "ts_eval/caption_answer_harness.py:build_answer_messages")
    ap.add_argument("--skip-caption-step", action="store_true",
                    help="reuse an existing captions.jsonl (e.g. re-scoring after an answerer-prompt change)")
    args = ap.parse_args()

    if not args.skip_caption_step and not args.captioner_ckpt:
        raise SystemExit("--captioner-ckpt is required unless --skip-caption-step is given")
    if not args.run_name and not args.captioner_ckpt:
        raise SystemExit("--run-name is required when --captioner-ckpt is omitted")
    run_name = args.run_name or Path(args.captioner_ckpt.rstrip("/")).name
    out_dir = Path(args.out_dir) / args.benchmark / run_name
    image_root = Path(args.image_root) if args.image_root else out_dir / "render"
    captions_path = out_dir / "captions.jsonl"

    shared = ["--benchmark", args.benchmark]
    if args.limit is not None:
        shared += ["--limit", str(args.limit)]
    if args.i_accept_unclear_license:
        shared += ["--i-accept-unclear-license"]
    if args.hf_token:
        shared += ["--hf-token", args.hf_token]

    if not args.skip_caption_step:
        cap_cmd = [
            sys.executable, "-m", "ts_eval.caption_step", *shared,
            "--captioner-ckpt", args.captioner_ckpt,
            "--image-root", str(image_root),
            "--captions-out", str(captions_path),
            "--temperature", str(args.temperature),
            "--max-tokens-caption", str(args.max_tokens_caption),
            "--batch-size", str(args.batch_size),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--prompt-family", args.prompt_family,
            "--tp", str(args.tp),
        ]
        print("=== caption step ===", " ".join(cap_cmd), flush=True)
        subprocess.run(cap_cmd, check=True)
    elif not captions_path.exists():
        raise SystemExit(f"--skip-caption-step given but {captions_path} does not exist")

    ans_cmd = [
        sys.executable, "-m", "ts_eval.answer_step", *shared,
        "--answerer-ckpt", args.answerer_ckpt,
        "--captions-in", str(captions_path),
        "--out-dir", args.out_dir,
        "--run-name", run_name,
        "--temperature", str(args.temperature),
        "--max-tokens-answer", str(args.max_tokens_answer),
        "--max-tokens-judge", str(args.max_tokens_judge),
        "--batch-size", str(args.batch_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--answer-style", args.answer_style,
    ]
    print("=== answer step ===", " ".join(ans_cmd), flush=True)
    subprocess.run(ans_cmd, check=True)


if __name__ == "__main__":
    main()
