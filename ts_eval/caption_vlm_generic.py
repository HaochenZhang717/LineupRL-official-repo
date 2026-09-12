from __future__ import annotations

import argparse
import json
from pathlib import Path

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT
from ts_render.render import render_fragment

SPECIAL = {
    "phi4mm": lambda p: f"<|user|><|image_1|>{p}<|end|><|assistant|>",
    "internvl": lambda p: ("<|im_start|>user\n<image>\n" + p + "<|im_end|>\n"
                           "<|im_start|>assistant\n"),
    "mistral3": lambda p: f"<s>[INST][IMG]{p}[/INST]",
}

NO_PROCESSOR: set[str] = set()

SPECIAL_WINS = {"phi4mm", "internvl"}


def family_of(model_id: str) -> str | None:
    m = model_id.lower()
    if "phi-4" in m or "phi4" in m:
        return "phi4mm"
    if "internvl" in m:
        return "internvl"
    if "mistral-small" in m or "pixtral" in m:
        return "mistral3"
    return None


def render_all(rows: list[dict], image_root: Path) -> list[Path]:
    image_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in rows:
        p = image_root / f"{r['series_uid'].replace('/', '__')}.png"
        if not p.exists():
            render_fragment(r["series"], p)
        paths.append(p)
    return paths


def build_prompt(processor, model_id: str, prompt_text: str) -> str:
    fam = family_of(model_id)
    if fam and (fam in SPECIAL_WINS or processor is None):
        return SPECIAL[fam](prompt_text)
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": prompt_text}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tokenizer-mode", default="auto",
                    help="'mistral' for Mistral-native repos that ship tekken.json "
                         "instead of an HF tokenizer")
    ap.add_argument("--config-format", default="auto")
    ap.add_argument("--load-format", default="auto")
    ap.add_argument("--smoke", type=int, default=0,
                    help="caption only N images and print prompt + output, then exit "
                         "WITHOUT writing. Use before every new model.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.series, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if args.smoke:
        rows = rows[: args.smoke]
    paths = render_all(rows, Path(args.image_root))
    print(f"{len(rows)} series, images under {args.image_root}", flush=True)

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    fam = family_of(args.model)
    processor = None
    if fam not in NO_PROCESSOR:
        try:
            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        except Exception as e:
            if fam is None:
                raise
            print(f"AutoProcessor unavailable ({type(e).__name__}); using the {fam} "
                  f"hand-written prompt", flush=True)
    prompt = build_prompt(processor, args.model, TS_CAP_PROMPT)
    print(f"--- rendered prompt for {args.model} ---\n{prompt}\n---", flush=True)
    MARKERS = ("<image", "<|image", "<start_of_image>", "[IMG]", "<|vision_start|>",
               "<img>", "<IMG_CONTEXT>")
    if not any(m in prompt for m in MARKERS):
        print("WARNING: no image marker in the rendered prompt -- if the captions come "
              "back generic, this model needs an entry in SPECIAL.", flush=True)

    llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len,
              tokenizer_mode=args.tokenizer_mode,
              config_format=args.config_format, load_format=args.load_format,
              limit_mm_per_prompt={"image": 1, "video": 0})
    params = SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (out_path.open("w") if not args.smoke else open("/dev/null", "w")) as fh:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            inputs = [{"prompt": prompt,
                       "multi_modal_data": {"image": Image.open(p).convert("RGB")}}
                      for p in paths[start:start + args.batch_size]]
            outs = llm.generate(inputs, params)
            for r, o in zip(chunk, outs):
                cap = o.outputs[0].text.strip()
                if args.smoke:
                    print(f"\n=== {r['series_uid']}\n{cap[:600]}", flush=True)
                else:
                    fh.write(json.dumps({"id": r["series_uid"],
                                         "caption_key": r["series_uid"],
                                         "caption": cap}) + "\n")
                    written += 1
            print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    if args.smoke:
        print("\nSMOKE ONLY -- nothing written. Check the captions actually describe the "
              "charts before running the full pass.", flush=True)
    else:
        print(f"wrote {written} captions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
