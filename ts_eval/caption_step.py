from __future__ import annotations

import argparse
from pathlib import Path

from ts_eval.caption_answer_harness import (CAPTION_PROMPT_FAMILIES,
                                            DEFAULT_EVIDENCE_LABEL, caption_items,
                                            caption_key, caption_prompt_pair)
from ts_eval.cli_common import add_benchmark_args, load_items


def _generate_captions_fn(captioner_ckpt: str, temperature: float, max_tokens: int,
                           gpu_memory_utilization: float, prompt_family: str = "qa",
                           tp: int = 1):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(captioner_ckpt, trust_remote_code=True)
    llm = LLM(model=captioner_ckpt, trust_remote_code=True, tensor_parallel_size=tp,
              gpu_memory_utilization=gpu_memory_utilization, limit_mm_per_prompt={"image": 1})
    params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)

    sys_prompt, user_prompt = caption_prompt_pair(prompt_family)
    print(f"caption prompt family={prompt_family} | system={sys_prompt!r}\n  user={user_prompt!r}",
          flush=True)

    def generate(image_paths: list[str]) -> list[str]:
        inputs = []
        for p in image_paths:
            user_msg = {"role": "user", "content": [
                {"type": "image", "image": p},
                {"type": "text", "text": user_prompt},
            ]}
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [user_msg]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(msgs)
            inputs.append({"prompt": text, "multi_modal_data": {"image": image_inputs}})
        outputs = llm.generate(inputs, sampling_params=params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    return generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_benchmark_args(ap)
    ap.add_argument("--captioner-ckpt", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--captions-out", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens-caption", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1,
                    help="vLLM tensor_parallel_size. 1 for a 3B captioner; 4 for the 72B "
                         "teacher, whose weights do not fit one card")
    ap.add_argument("--prompt-family", default="qa", choices=list(CAPTION_PROMPT_FAMILIES),
                    help="which training prompt the captioner was trained with; a "
                         "mismatch measures the prompt, not the model")
    args = ap.parse_args()

    items = load_items(args.benchmark, args.limit, args.i_accept_unclear_license, args.hf_token)
    print(f"loaded {len(items)} scorable items from {args.benchmark}", flush=True)
    if not items:
        raise SystemExit("no items to caption")

    print(f"loading captioner: {args.captioner_ckpt}", flush=True)
    generate_captions = _generate_captions_fn(
        args.captioner_ckpt, args.temperature, args.max_tokens_caption,
        args.gpu_memory_utilization, args.prompt_family, args.tp)
    image_paths, captions = caption_items(items, Path(args.image_root), generate_captions, args.batch_size)
    print(f"captioned {len(captions)} items (images under {args.image_root})", flush=True)

    import json
    out_path = Path(args.captions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item, caption in zip(items, captions):
            f.write(json.dumps({"id": item.id, "caption_key": caption_key(item),
                                "evidence_label": DEFAULT_EVIDENCE_LABEL,
                                "caption": caption}, ensure_ascii=False) + "\n")
    print(f"wrote {len(captions)} captions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
