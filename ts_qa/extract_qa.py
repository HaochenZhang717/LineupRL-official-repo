from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

_LEADING_MARKER = re.compile(r"^\s*#+\s*\d+\.\s*")
_ANSWER_LETTER = re.compile(r"Answer:\**\s*([A-F])", re.IGNORECASE)
_FALLBACK_LETTER = re.compile(r"\b([A-F])\b")
_HAS_OPTION = re.compile(r"-\s*[A-F]\)")


def extract_questions_and_answers(text: str) -> List[Dict[str, str]]:
    result: List[Dict[str, str]] = []
    for block in text.split("------"):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        ans_idx = next(
            (i for i, ln in enumerate(lines) if "answer" in ln.lower() and ")" in ln),
            len(lines) - 1,
        )
        answer_line = lines[ans_idx].strip()
        question_block = "\n".join(lines[:ans_idx]).strip()
        question_block = _LEADING_MARKER.sub("", question_block).replace("**", "").strip()

        m = _ANSWER_LETTER.search(answer_line) or _FALLBACK_LETTER.search(answer_line)
        if not m:
            continue
        letter = m.group(1).upper()

        if not _HAS_OPTION.search(question_block):
            continue
        result.append({"question": question_block, "answer": letter})
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Extract structured QAs from raw VLM responses.")
    p.add_argument("--in-dir", default=None, help="folder of part_*.jsonl from gen_qa_caprl")
    p.add_argument("--in", dest="in_files", nargs="*", default=None,
                   help="explicit list of raw jsonl files (alternative to --in-dir)")
    p.add_argument("--out", required=True, help="output qa_raw.jsonl")
    args = p.parse_args()

    files: List[Path] = []
    if args.in_dir:
        files = sorted(
            Path(args.in_dir).glob("part_*.jsonl"),
            key=lambda f: int(re.search(r"part_(\d+)", f.name).group(1)),
        )
    if args.in_files:
        files += [Path(f) for f in args.in_files]
    if not files:
        raise SystemExit("no input files (use --in-dir or --in)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_frag = n_q = n_drop = 0
    with out_path.open("w", encoding="utf-8") as out:
        for fp in files:
            for ln in fp.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                rec: Dict[str, Any] = json.loads(ln)
                qa_list = extract_questions_and_answers(rec.get("qa_response", ""))
                if not qa_list:
                    n_drop += 1
                    continue
                out.write(json.dumps({
                    "image_path": rec["image_path"],
                    "series": rec.get("series"),
                    "qa_list": qa_list,
                }, ensure_ascii=False) + "\n")
                n_frag += 1
                n_q += len(qa_list)

    print(f"Extracted {n_q} questions from {n_frag} fragments "
          f"({n_drop} fragments dropped) -> {out_path}")


if __name__ == "__main__":
    main()
