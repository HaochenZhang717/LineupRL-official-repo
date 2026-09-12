from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT

FLUSH_EVERY = 200


def fix_image_path(path: str) -> str:
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"image not found: {path}")


def load_items(dataset: str, limit: int | None = None) -> list[dict]:
    items = []
    with open(dataset, encoding="utf-8") as fh:
        for line in fh:
            msgs = json.loads(json.loads(line)["message"])
            payload = json.loads(eval(msgs[2]["content"])[0][0])
            image = next(p["image"] for p in msgs[1]["content"] if p.get("type") == "image")
            items.append({"id": payload["id"], "image": fix_image_path(image)})
            if limit is not None and len(items) >= limit:
                break
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="an RL *_messages.jsonl; comma-separated for several splits "
                         "(the 145 GB teacher is then loaded once, not once per split)")
    ap.add_argument("--model", required=True, help="teacher VLM dir or HF id")
    ap.add_argument("--out", required=True, help="output jsonl; one per --dataset entry")
    ap.add_argument("--tp", type=int, default=4, help="tensor parallel size")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    datasets = [d for d in args.dataset.split(",") if d]
    outs = [o for o in args.out.split(",") if o]
    if len(datasets) != len(outs):
        raise SystemExit(f"--dataset has {len(datasets)} entries but --out has {len(outs)}")

    jobs = []
    for ds, out in zip(datasets, outs):
        items = load_items(ds, args.limit)
        if args.resume and os.path.exists(out):
            done = set()
            with open(out, encoding="utf-8") as fh:
                for line in fh:
                    done.add(json.loads(line)["id"])
            items = [it for it in items if it["id"] not in done]
            print(f"{out}: resume, {len(done)} already done, {len(items)} left", flush=True)
        print(f"{out}: {len(items)} captions to generate with {args.model}", flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        if items:
            jobs.append((out, items))
    if not jobs:
        print("nothing to do", flush=True)
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, trust_remote_code=True,
              tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 1})
    params = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_tokens)

    for out_path, items in jobs:
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as out_f:
            for start in range(0, len(items), FLUSH_EVERY):
                chunk = items[start:start + FLUSH_EVERY]
                prompts = []
                for it in chunk:
                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": it["image"]},
                        {"type": "text", "text": TS_CAP_PROMPT}]}]
                    text = processor.apply_chat_template(msgs, tokenize=False,
                                                         add_generation_prompt=True)
                    image_inputs, _ = process_vision_info(msgs)
                    prompts.append({"prompt": text,
                                    "multi_modal_data": {"image": image_inputs}})
                gen = llm.generate(prompts, sampling_params=params)
                for it, o in zip(chunk, gen):
                    out_f.write(json.dumps({
                        "id": it["id"],
                        "image": it["image"],
                        "caption": o.outputs[0].text.strip(),
                    }, ensure_ascii=False) + "\n")
                out_f.flush()
                n = min(start + FLUSH_EVERY, len(items))
                rate = n / max(time.time() - t0, 1e-6)
                eta = (len(items) - n) / max(rate, 1e-6) / 60
                print(f"{os.path.basename(out_path)} progress: {n}/{len(items)}  "
                      f"{rate:.2f} cap/s  eta {eta:.1f} min", flush=True)
        print(f"DONE {len(items)} captions in {(time.time() - t0) / 60:.1f} min "
              f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
