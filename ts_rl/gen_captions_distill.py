import argparse
import json
import os

from ts_rl.eval_val_reward import ensure_processor_config

SYSTEM_PROMPT = "You are an analyst who describes and interprets time series."
USER_PROMPT = "Please describe this image in detail."
BATCH = 2000


def load_rows(manifest, image_root, shard, num_shards, done_ids):
    rows = []
    with open(manifest) as f:
        for i, line in enumerate(f):
            if i % num_shards != shard:
                continue
            r = json.loads(line)
            if r["id"] in done_ids:
                continue
            r["abs_image"] = os.path.join(image_root, r["image_path"])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--processor-src",
                    default="models/Qwen2.5-VL-3B-Instruct")
    args = ap.parse_args()

    ensure_processor_config(args.ckpt, args.processor_src)

    done_ids = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
        print(f"resume: {len(done_ids)} ids already done", flush=True)

    rows = load_rows(args.manifest, args.image_root, args.shard, args.num_shards, done_ids)
    print(f"shard {args.shard}/{args.num_shards}: {len(rows)} captions to generate", flush=True)
    if not rows:
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    llm = LLM(model=args.ckpt, trust_remote_code=True, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1})
    params = SamplingParams(n=1, temperature=args.temperature, max_tokens=args.max_tokens)

    with open(args.out, "a") as out_f:
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            inputs = []
            for r in chunk:
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image", "image": r["abs_image"]},
                        {"type": "text", "text": USER_PROMPT},
                    ]},
                ]
                text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(msgs)
                inputs.append({"prompt": text, "multi_modal_data": {"image": image_inputs}})
            outputs = llm.generate(inputs, sampling_params=params)
            for r, out in zip(chunk, outputs):
                out_f.write(json.dumps({
                    "id": r["id"],
                    "series": r["series"],
                    "len": r["len"],
                    "ds_caption": r["ds_caption"],
                    "caption": out.outputs[0].text,
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"progress: {min(start + BATCH, len(rows))}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
