from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT_CHATTS

MODEL = "bytedance-research/ChatTS-14B"


def _patch_cache_for_old_remote_code() -> None:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

    if not hasattr(DynamicCache, "get_usable_length"):
        def _usable(self, new_seq_length: int, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)
        DynamicCache.get_usable_length = _usable


def build_prompt(n: int) -> str:
    user = TS_CAP_PROMPT_CHATTS.format(n=n)
    return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
            f"<|im_start|>user\n{user}<|im_end|><|im_start|>assistant\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-points", type=int, default=1024,
                    help="series longer than this are skipped and reported, not silently "
                         "truncated: a truncated series is a different series and would "
                         "quietly break the alignment the whole comparison rests on")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --out (this runner is slow enough "
                         "that losing a 3-hour job to an interruption matters)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.series, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = set()
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f}
        print(f"resume: {len(done)} already captioned")

    skipped = [r["series_uid"] for r in rows if len(r["series"]) > args.max_points]
    work = [r for r in rows
            if r["series_uid"] not in done and len(r["series"]) <= args.max_points]
    if skipped:
        print(f"NOTE: skipping {len(skipped)} series longer than {args.max_points} points "
              f"(e.g. {skipped[:3]})")
    if args.smoke:
        work = work[: args.smoke]

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    _patch_cache_for_old_remote_code()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, tokenizer=tok)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, device_map={"": 0},
        torch_dtype=torch.float16)
    model.eval()

    mode = "a" if (args.resume and out_path.exists()) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f_out = None if args.smoke else open(out_path, mode, encoding="utf-8")
    n_empty = 0

    for i in range(0, len(work), args.batch_size):
        chunk = work[i: i + args.batch_size]
        prompts = [build_prompt(len(r["series"])) for r in chunk]
        ts = [np.asarray(r["series"], dtype=np.float32) for r in chunk]
        inputs = proc(text=prompts, timeseries=ts, padding=True, return_tensors="pt")
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        for r, o in zip(chunk, outs):
            cap = tok.decode(o[in_len:], skip_special_tokens=True).strip()
            if not cap:
                n_empty += 1
            if args.smoke:
                print(f"\n--- {r['series_uid']} (n={len(r['series'])})\n{cap}")
            else:
                f_out.write(json.dumps({"id": r["series_uid"],
                                        "caption_key": r["series_uid"],
                                        "caption": cap}, ensure_ascii=False) + "\n")
        if not args.smoke:
            f_out.flush()
            print(f"  {min(i + args.batch_size, len(work))}/{len(work)}", flush=True)

    if f_out:
        f_out.close()
        print(f"wrote {out_path} ({len(work)} new captions, {n_empty} empty, "
              f"{len(skipped)} skipped as too long)")


if __name__ == "__main__":
    main()
