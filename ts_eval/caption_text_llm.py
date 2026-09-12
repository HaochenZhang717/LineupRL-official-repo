from __future__ import annotations

import argparse
import json
from pathlib import Path

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT_TEXT


def format_series(values, max_decimals: int = 4) -> str:
    out = []
    for v in values:
        f = float(v)
        out.append(str(int(f)) if f == int(f) and abs(f) < 1e15
                   else f"{f:.{max_decimals}g}")
    return ", ".join(out)


def build_prompt(values) -> str:
    return f"{TS_CAP_PROMPT_TEXT}\n\nValues ({len(values)} points):\n{format_series(values)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run N series, print the rendered prompt and the captions, and "
                         "write nothing. The only reliable way to see that a chat "
                         "template did what you think it did.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.series, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    work = rows[: args.smoke] if args.smoke else rows

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for r in work:
        msg = [{"role": "user", "content": build_prompt(r["series"])}]
        prompts.append(tok.apply_chat_template(msg, tokenize=False,
                                               add_generation_prompt=True))

    if args.smoke:
        print("=== rendered prompt [0] ===")
        print(prompts[0][:2000])
        print("=== end ===")

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    caps = [o.outputs[0].text.strip() for o in outs]
    if args.smoke:
        for r, c in zip(work, caps):
            print(f"\n--- {r['series_uid']} (n={len(r['series'])})\n{c}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with open(out, "w", encoding="utf-8") as f:
        for r, c in zip(work, caps):
            if not c:
                n_empty += 1
            f.write(json.dumps({"id": r["series_uid"], "caption_key": r["series_uid"],
                                "caption": c}, ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(caps)} captions, {n_empty} empty)")
    if n_empty:
        print(f"WARNING: {n_empty} empty captions -- check max_tokens and the template")


if __name__ == "__main__":
    main()
