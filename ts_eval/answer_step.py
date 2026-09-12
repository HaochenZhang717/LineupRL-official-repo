from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_eval.caption_answer_harness import (ANSWER_STYLES, answer_and_score,
                                            prompt_fingerprint)
from ts_eval.cli_common import add_benchmark_args, load_items
from ts_eval.report import write_report

MISSING_CAPTION_TOLERANCE = 0.02


def filter_items_with_captions(items, caption_by_id: dict[str, str], tolerance: float = MISSING_CAPTION_TOLERANCE):
    missing = [item.id for item in items if item.id not in caption_by_id]
    if not missing:
        return items
    frac_missing = len(missing) / len(items)
    print(f"warning: {len(missing)}/{len(items)} ({frac_missing:.1%}) items have no caption "
          f"(first few missing ids: {missing[:5]})", flush=True)
    if frac_missing > tolerance:
        raise ValueError(f"{frac_missing:.1%} missing exceeds the {tolerance:.0%} noise-tolerance "
                          f"threshold -- this looks like a systemic id mismatch (e.g. a "
                          f"non-deterministic benchmark adapter), not incidental per-row flakiness. "
                          f"Investigate before re-running.")
    return [item for item in items if item.id in caption_by_id]


def _generate_text_fns(answerer_ckpt: str, temperature: float, answer_max_tokens: int,
                        judge_max_tokens: int, gpu_memory_utilization: float):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(answerer_ckpt, trust_remote_code=True)
    llm = LLM(model=answerer_ckpt, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization)
    answer_params = SamplingParams(n=1, temperature=temperature, max_tokens=answer_max_tokens)
    judge_params = SamplingParams(n=1, temperature=0.0, max_tokens=judge_max_tokens)

    def apply_chat_template(messages: list[dict]) -> str:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_answers(prompts: list[str]) -> list[str]:
        outputs = llm.generate(prompts, sampling_params=answer_params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def judge_fn(prompts: list[str]) -> list[str]:
        chat_prompts = [apply_chat_template([{"role": "user", "content": p}]) for p in prompts]
        outputs = llm.generate(chat_prompts, sampling_params=judge_params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    return apply_chat_template, generate_answers, judge_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_benchmark_args(ap)
    ap.add_argument("--answerer-ckpt", required=True)
    ap.add_argument("--captions-in", required=True)
    ap.add_argument("--out-dir", required=True, help="base pipeline1 dir; report goes to <out-dir>/<benchmark>/<run-name>")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens-answer", type=int, default=64)
    ap.add_argument("--max-tokens-judge", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--answer-style", default="caption_qa", choices=list(ANSWER_STYLES),
                    help="how the evidence is framed for the answerer; see "
                         "ts_eval/caption_answer_harness.py:build_answer_messages. The "
                         "default reproduces every pipeline-1 number already on disk")
    args = ap.parse_args()

    items = load_items(args.benchmark, args.limit, args.i_accept_unclear_license, args.hf_token)
    print(f"loaded {len(items)} scorable items from {args.benchmark}", flush=True)
    if not items:
        raise SystemExit("no items to answer")

    caption_by_id: dict[str, str] = {}
    label_by_id: dict[str, str | None] = {}
    with open(args.captions_in, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            caption_by_id[row["id"]] = row["caption"]
            label_by_id[row["id"]] = row.get("evidence_label")
    try:
        items = filter_items_with_captions(items, caption_by_id)
    except ValueError as e:
        raise SystemExit(str(e))
    captions = [caption_by_id[item.id] for item in items]
    evidence_labels = [label_by_id.get(item.id) for item in items]

    print(f"loading answerer: {args.answerer_ckpt}", flush=True)
    apply_chat_template, generate_answers, judge_fn = _generate_text_fns(
        args.answerer_ckpt, args.temperature, args.max_tokens_answer, args.max_tokens_judge,
        args.gpu_memory_utilization)
    records = answer_and_score(items, captions, generate_answers, apply_chat_template, judge_fn,
                               args.batch_size, style=args.answer_style,
                               evidence_labels=evidence_labels)
    print(f"answered + scored {len(records)} items", flush=True)

    out_dir = Path(args.out_dir) / args.benchmark / args.run_name
    provenance = {"protocol": "caption_mediated", "answer_style": args.answer_style,
                  "answerer": args.answerer_ckpt,
                  "evidence_label": evidence_labels[0] if evidence_labels else None,
                  "prompt_fingerprint": prompt_fingerprint(items, args.answer_style)}
    report = write_report(records, out_dir, run_name=f"{args.benchmark}/{args.run_name}",
                          provenance=provenance)
    print(json.dumps({k: v for k, v in report.items() if k in ("n_items", "overall_accuracy")}, indent=2))
    print(f"wrote report to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
