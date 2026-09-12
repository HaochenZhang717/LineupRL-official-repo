from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_eval.caption_answer_harness import caption_prompt_pair
from ts_render.render import render_fragment


def render_all(rows: list[dict], image_root: Path) -> list[str]:
    paths = []
    for r in rows:
        stem = r["series_uid"].replace("/", "__")
        p = image_root / f"{stem}.png"
        if not p.exists():
            render_fragment(r["series"], p)
        paths.append(str(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True)
    ap.add_argument("--captioner-ckpt", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--captions-out", required=True)
    ap.add_argument("--prompt-family", default="decodability")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens-caption", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.series, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    image_root = Path(args.image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    paths = render_all(rows, image_root)
    print(f"{len(rows)} series rendered under {image_root}", flush=True)

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(args.captioner_ckpt, trust_remote_code=True)
    llm = LLM(model=args.captioner_ckpt, trust_remote_code=True,
              tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 1})
    params = SamplingParams(n=1, temperature=args.temperature,
                            max_tokens=args.max_tokens_caption)
    sys_prompt, user_prompt = caption_prompt_pair(args.prompt_family)
    print(f"prompt family={args.prompt_family} system={sys_prompt!r}", flush=True)

    out_path = Path(args.captions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            chunk_paths = paths[start:start + args.batch_size]
            inputs = []
            for p in chunk_paths:
                msgs = []
                if sys_prompt:
                    msgs.append({"role": "system", "content": sys_prompt})
                msgs.append({"role": "user", "content": [
                    {"type": "image", "image": p}, {"type": "text", "text": user_prompt}]})
                text = processor.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True)
                imgs, _ = process_vision_info(msgs)
                inputs.append({"prompt": text, "multi_modal_data": {"image": imgs}})
            outs = llm.generate(inputs, params)
            for r, o in zip(chunk, outs):
                fh.write(json.dumps({"id": r["series_uid"],
                                     "caption_key": r["series_uid"],
                                     "caption": o.outputs[0].text.strip()}) + "\n")
            print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
