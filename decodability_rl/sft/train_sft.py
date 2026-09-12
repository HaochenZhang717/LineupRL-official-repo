from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from datasets import Image as HFImage
from datasets import Sequence as HFSequence
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT


def load_dataset_from_jsonl(path: str) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            records.append({
                "images": [row["image"]],
                "messages": [
                    {"role": "user", "content": TS_CAP_PROMPT},
                    {"role": "assistant", "content": row["caption"]},
                ],
            })
    ds = Dataset.from_list(records)
    return ds.cast_column("images", HFSequence(HFImage()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--base-model", default=str(REPO / "models" / "Qwen2.5-VL-3B-Instruct"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--save-total-limit", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = full epochs")
    ap.add_argument("--ddp-timeout", type=int, default=600,
                    help="seconds before a stuck NCCL collective aborts (the 1800 s "
                         "default lets a dead node waste half an hour)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    train_ds = load_dataset_from_jsonl(args.train)
    val_ds = load_dataset_from_jsonl(args.val)
    print(f"train examples: {len(train_ds)}, val examples: {len(val_ds)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        seed=args.seed,
        ddp_timeout=args.ddp_timeout,
        report_to=args.report_to,
        run_name=args.run_name,
        save_total_limit=args.save_total_limit,
        assistant_only_loss=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
    )

    resume = bool(list(Path(args.output_dir).glob("checkpoint-*")))
    if resume:
        print(f"resuming from the latest checkpoint under {args.output_dir}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved final model to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
