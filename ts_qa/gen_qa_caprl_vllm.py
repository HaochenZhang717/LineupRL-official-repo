from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .gen_qa_caprl import (
    SYS_PROMPT,
    build_user_prompt,
    _load_jsonl,
    _load_progress,
    _save_progress,
    _append_jsonl,
)
from .qwen_client import encode_image

DEFAULT_MODEL = "Qwen/Qwen3-VL-32B-Instruct-FP8"
DEFAULT_BATCH = 256


def _build_conversation(rec: Dict[str, Any], image_root: Optional[Path],
                        with_series: bool) -> Optional[List[Dict[str, Any]]]:
    img = rec["image_path"]
    if image_root is not None:
        img = image_root / img
    if not Path(img).exists():
        logging.warning("missing image, skipping: %s", img)
        return None
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": encode_image(img)}},
            {"type": "text", "text": build_user_prompt(rec.get("series"),
                                                       with_series=with_series)},
        ]},
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="CapRL-style VLM QA generation on local GPU (vLLM).")
    p.add_argument("--in", dest="in_path", required=True, help="fragments.jsonl from rendering")
    p.add_argument("--image-root", default=None, help="root that image_path is relative to")
    p.add_argument("--out-dir", required=True, help="output folder for part_*.jsonl")
    p.add_argument("--limit", type=int, default=None, help="only process the first N fragments")
    p.add_argument("--with-series", action="store_true",
                   help="also feed raw numeric values as text (default: image-only)")

    mdl = p.add_argument_group("vLLM model")
    mdl.add_argument("--model", default=DEFAULT_MODEL,
                     help="HF id, default Qwen/Qwen3-VL-32B-Instruct-FP8")
    mdl.add_argument("--temperature", type=float, default=0.8)
    mdl.add_argument("--max-tokens", type=int, default=4096,
                     help="max NEW tokens per chart; 10 MCQs need ~1600+, keep headroom")
    mdl.add_argument("--top-p", type=float, default=0.9)
    mdl.add_argument("--max-model-len", type=int, default=8192,
                     help="vLLM context window (sys+img+prompt+output must fit)")
    mdl.add_argument("--gpu-mem-util", type=float, default=0.90)
    mdl.add_argument("--tensor-parallel-size", type=int, default=1,
                     help="GPUs per shard (1 = one GPU per shard)")
    mdl.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                     help="charts per checkpoint flush (vLLM still batches within)")

    par = p.add_argument_group("sharding / resume (one shard = one GPU/job)")
    par.add_argument("--part-id", type=int, default=0)
    par.add_argument("--all-parts", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    data = _load_jsonl(Path(args.in_path))
    if args.limit:
        data = data[: args.limit]

    part_size = math.ceil(len(data) / args.all_parts)
    start, end = args.part_id * part_size, min((args.part_id + 1) * part_size, len(data))
    shard = data[start:end]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part_{args.part_id}.jsonl"
    progress_file = out_dir / f"progress_part_{args.part_id}.json"

    resume = _load_progress(progress_file)
    todo = shard[resume:]
    image_root = Path(args.image_root) if args.image_root else None
    logging.info("part %d/%d: shard=%d (global %d:%d), resumed from %d, todo=%d",
                 args.part_id, args.all_parts, len(shard), start, end, resume, len(todo))
    if not todo:
        logging.info("nothing to do for part %d -> %s", args.part_id, out_file)
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
    )

    processed = 0
    for b0 in tqdm(range(0, len(todo), args.batch_size), desc=f"gen-qa-vllm[p{args.part_id}]"):
        chunk = todo[b0: b0 + args.batch_size]
        convs, recs = [], []
        for rec in chunk:
            conv = _build_conversation(rec, image_root, args.with_series)
            if conv is not None:
                convs.append(conv)
                recs.append(rec)

        if convs:
            outputs = llm.chat(convs, sampling, use_tqdm=False)
            records = [{
                "id": rec.get("id"),
                "image_path": rec["image_path"],
                "series": rec.get("series"),
                "qa_response": out.outputs[0].text if out.outputs else "",
            } for rec, out in zip(recs, outputs)]
            _append_jsonl(records, out_file)

        processed += len(chunk)
        _save_progress(progress_file, resume + processed)

    logging.info("Done part %d. Raw QA responses -> %s", args.part_id, out_file)


if __name__ == "__main__":
    main()
