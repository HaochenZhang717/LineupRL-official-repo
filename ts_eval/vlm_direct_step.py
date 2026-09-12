from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_eval.caption_answer_harness import (_ANSWER_INSTRUCTIONS, PredictionRecord,
                                            _run_in_chunks, render_items)
from ts_eval.cli_common import add_benchmark_args, load_items
from ts_eval.report import write_report
from ts_eval import scoring as sc


def build_direct_prompt(item) -> str:
    instr = _ANSWER_INSTRUCTIONS.get(item.scoring_type)
    if instr is None:
        raise ValueError(f"unsupported scoring_type for direct answering: {item.scoring_type!r}")
    return f"{item.question}\n\n{instr}"


def _generate_fn(model_ckpt: str, temperature: float, max_tokens: int,
                 gpu_memory_utilization: float, tp: int, with_image: bool):
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(model_ckpt, trust_remote_code=True)
    llm = LLM(model=model_ckpt, trust_remote_code=True, tensor_parallel_size=tp,
              gpu_memory_utilization=gpu_memory_utilization,
              limit_mm_per_prompt={"image": 1 if with_image else 0})
    params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)

    def generate(payloads: list[str]) -> list[str]:
        inputs = []
        for payload in payloads:
            image_path, _, prompt = payload.partition("\x00")
            content = ([{"type": "image", "image": image_path}] if with_image else []) + \
                      [{"type": "text", "text": prompt}]
            msgs = [{"role": "user", "content": content}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            entry = {"prompt": text}
            if with_image:
                image_inputs, _ = process_vision_info(msgs)
                entry["multi_modal_data"] = {"image": image_inputs}
            inputs.append(entry)
        outputs = llm.generate(inputs, sampling_params=params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    return generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_benchmark_args(ap)
    ap.add_argument("--model-ckpt", required=True, help="the VLM that both sees and answers")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-root", default=None, help="defaults to <out-dir>/<benchmark>/<run-name>/render")
    ap.add_argument("--no-image", action="store_true",
                    help="withhold the chart: the floor for this protocol (option-text prior)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    items = load_items(args.benchmark, args.limit, args.i_accept_unclear_license, args.hf_token)
    print(f"loaded {len(items)} scorable items from {args.benchmark}", flush=True)
    if not items:
        raise SystemExit("no items to answer")

    out_dir = Path(args.out_dir) / args.benchmark / args.run_name
    image_root = Path(args.image_root) if args.image_root else out_dir / "render"
    if args.no_image:
        image_paths = [Path("") for _ in items]
    else:
        image_paths = render_items(items, image_root)

    print(f"loading VLM: {args.model_ckpt} (image={'no' if args.no_image else 'yes'})", flush=True)
    generate = _generate_fn(args.model_ckpt, args.temperature, args.max_tokens,
                            args.gpu_memory_utilization, args.tp, not args.no_image)
    payloads = [f"{p}\x00{build_direct_prompt(item)}" for p, item in zip(image_paths, items)]
    answers = _run_in_chunks(generate, payloads, args.batch_size)

    records = [
        PredictionRecord(
            item_id=item.id, source_benchmark=item.source_benchmark,
            caption="",
            answer_raw=answer, judge_raw=None,
            score=sc.score(answer, item), scoring_type=item.scoring_type,
            task_type=item.task_type, domain=item.domain)
        for item, answer in zip(items, answers)
    ]
    import hashlib
    fingerprint = hashlib.sha256("\x1e".join(
        build_direct_prompt(i) for i in sorted(items, key=lambda i: i.id)[:20]
    ).encode("utf-8")).hexdigest()[:12]
    provenance = {"protocol": "direct", "model": args.model_ckpt,
                  "image": not args.no_image, "prompt_fingerprint": fingerprint}
    report = write_report(records, out_dir, run_name=f"{args.benchmark}/{args.run_name}",
                          provenance=provenance)
    print(json.dumps({k: report[k] for k in ("n_items", "overall_accuracy")}, indent=2))
    print(f"wrote report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
