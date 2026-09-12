from __future__ import annotations

import argparse
import json
from pathlib import Path

from decodability_rl.test_reward_design.prompts import build_discriminator_prompt

LETTERS = ("A", "B", "C", "D")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here / "results" / "results.json"))
    ap.add_argument("--out-dir", required=True, help="question files (no answer key here)")
    ap.add_argument("--key", required=True, help="where to write the answer key")
    ap.add_argument(
        "--variant",
        default="hinted",
        choices=["hinted", "neutral"],
        help="hinted = the original template that explains how the distractors were made; "
        "neutral = states the task only",
    )
    ap.add_argument(
        "--caption-source",
        default="ds",
        choices=["ds", "vlm", "empty"],
        help="ds = the dataset's own caption; vlm = the untuned Qwen2.5-VL-3B caption written "
        "from the rendered chart; empty = no description at all. The `empty` "
        "run is the control that says how much of the accuracy comes from the caption "
        "versus from the option construction leaking which candidate is the source.",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = []
    for qi, rec in enumerate(data["records"], 1):
        caption = rec["ds_caption"]
        if not caption.strip():
            raise RuntimeError(f"record {rec['id']} has no ds_caption")
        if args.caption_source == "vlm":
            caption = rec["caption"]
            if not caption.strip():
                raise RuntimeError(f"record {rec['id']} has no VLM caption")
        elif args.caption_source == "empty":
            caption = ""
        for ri, rot in enumerate(rec["rotations"]):
            gold = rot["gold_letter"]
            name = f"q{qi:02d}_v{ri}.txt"
            (out_dir / name).write_text(
                build_discriminator_prompt(
                    caption, rot["option_texts"], args.variant, answer_format="letter"
                ),
                encoding="utf-8",
            )
            key.append(
                {
                    "file": name,
                    "question": qi,
                    "series_id": rec["id"],
                    "gold_letter": gold,
                    "option_kinds": rot["option_kinds"],
                }
            )

    Path(args.key).write_text(json.dumps(key, indent=2))
    print(f"wrote {len(key)} question files to {out_dir}")
    print(f"wrote answer key to {args.key}")


if __name__ == "__main__":
    main()
