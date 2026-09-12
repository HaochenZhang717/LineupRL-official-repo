from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CAP_INSTRUCTION = "Please describe this image in detail."
DEFAULT_SYSTEM = "You are an analyst who describes and interprets time series."


def build_message(system_prompt: str, image_abs: str, qa_tuples: list[tuple[str, str]]) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs},
                {"type": "text", "text": CAP_INSTRUCTION},
            ],
        },
        {"role": "answer", "content": repr(qa_tuples)},
    ]


def self_check(line: str) -> None:
    outer = json.loads(line)
    assert "message" in outer, "line has no 'message' key"
    msg = json.loads(outer["message"])
    assert len(msg) == 3, f"expected 3 messages, got {len(msg)}"
    assert msg[2]["role"] == "answer", f"msg[2].role != answer: {msg[2].get('role')}"
    qas = eval(msg[2]["content"])
    assert isinstance(qas, list) and qas, "qa list empty"
    for q, a in qas:
        assert isinstance(q, str) and isinstance(a, str)
    uc = msg[1]["content"]
    assert uc[0]["type"] == "image" and "image" in uc[0]
    assert uc[1]["type"] == "text" and "text" in uc[1]
    assert os.path.isfile(uc[0]["image"]), f"image not found: {uc[0]['image']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(REPO / "out_10k_xdomain" / "qa_verified_keep.jsonl"),
                    help="KEEP QA bank jsonl")
    ap.add_argument("--out-dir", default=str(REPO / "out_10k_xdomain" / "rl"),
                    help="output dir for train/val message jsonl")
    ap.add_argument("--image-root", default=str(REPO / "out_10k_xdomain"),
                    help="prefix joined with each record's image_path to get the PNG")
    ap.add_argument("--min-qa", type=int, default=2,
                    help="drop images with fewer than this many QA (lower reward variance)")
    ap.add_argument("--val-size", type=int, default=500,
                    help="number of images held out for evaluation (not trained on)")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_in = 0
    dropped_missing = 0
    dropped_minqa = 0
    dropped_badqa = 0
    records = []

    with open(args.input, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            r = json.loads(line)

            image_abs = os.path.abspath(os.path.join(args.image_root, r["image_path"]))
            if not os.path.isfile(image_abs):
                dropped_missing += 1
                continue

            qa_tuples = []
            for qa in r.get("qa_list", []):
                q = qa.get("question")
                a = qa.get("answer")
                if not (isinstance(q, str) and isinstance(a, str) and len(a) == 1 and a.isalpha()):
                    continue
                qa_tuples.append((q, a.upper()))

            if not qa_tuples:
                dropped_badqa += 1
                continue
            if len(qa_tuples) < args.min_qa:
                dropped_minqa += 1
                continue

            records.append(build_message(args.system_prompt, image_abs, qa_tuples))

    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_n = min(args.val_size, len(records))
    val_records = records[:val_n]
    train_records = records[val_n:]

    train_path = out_dir / "train_messages.jsonl"
    val_path = out_dir / "val_messages.jsonl"

    def dump(path: Path, recs: list) -> int:
        n_q = 0
        with open(path, "w", encoding="utf-8") as fh:
            for msg in recs:
                n_q += len(eval(msg[2]["content"]))
                fh.write(json.dumps({"message": json.dumps(msg, ensure_ascii=False)}, ensure_ascii=False) + "\n")
        return n_q

    train_q = dump(train_path, train_records)
    val_q = dump(val_path, val_records)

    if train_records:
        with open(train_path, encoding="utf-8") as fh:
            self_check(fh.readline())

    report = [
        "=== build_rl_dataset report ===",
        f"input:            {args.input}",
        f"image_root:       {args.image_root}",
        f"min_qa:           {args.min_qa}",
        f"seed:             {args.seed}",
        f"images in:        {n_in}",
        f"dropped(missing image): {dropped_missing}",
        f"dropped(bad/empty qa):  {dropped_badqa}",
        f"dropped(<min_qa):       {dropped_minqa}",
        f"images kept:      {len(records)}",
        f"train images/qa:  {len(train_records)} / {train_q}  -> {train_path}",
        f"val   images/qa:  {len(val_records)} / {val_q}  -> {val_path}",
        "self-check on first train line: PASSED" if train_records else "self-check: SKIPPED (no train records)",
    ]
    text = "\n".join(report)
    print(text)
    (out_dir / "build_report.txt").write_text(text + "\n", encoding="utf-8")

    if not train_records:
        print("ERROR: no training records produced", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
