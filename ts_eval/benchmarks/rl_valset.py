from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ts_eval.benchmarks.base import QAItem

NAME = "rl_valset"
LICENSE = "internal (our own RL-training held-out set, not externally sourced)"

REPO = Path(__file__).resolve().parent.parent.parent
_DEFAULT_FRAGMENTS = REPO / "out_10k_xdomain" / "fragments.jsonl"
_DEFAULT_VAL_MESSAGES = REPO / "out_10k_xdomain" / "rl" / "val_messages.jsonl"


def _load_series_by_image_path(fragments_path: Path) -> dict[str, list[float]]:
    series_by_path = {}
    with open(fragments_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            series_by_path[row["image_path"]] = row["series"]
    return series_by_path


def load(split: str = "test", limit: int | None = None,
         fragments_path: str | Path = _DEFAULT_FRAGMENTS,
         val_messages_path: str | Path = _DEFAULT_VAL_MESSAGES) -> Iterator[QAItem]:
    series_by_path = _load_series_by_image_path(Path(fragments_path))

    n_yielded = 0
    n_skipped_no_series = 0
    with open(val_messages_path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit is not None and n_yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            outer = json.loads(line)
            msg = json.loads(outer["message"])
            image_abs = msg[1]["content"][0]["image"]
            rel_path = "/".join(Path(image_abs).parts[-2:])
            series = series_by_path.get(rel_path)
            if series is None:
                n_skipped_no_series += 1
                continue

            qa_list = eval(msg[2]["content"])
            image_id = Path(image_abs).stem
            for qa_idx, (question, answer) in enumerate(qa_list):
                if limit is not None and n_yielded >= limit:
                    break
                yield QAItem(
                    id=f"{image_id}::{qa_idx}",
                    series=series,
                    question=question,
                    gold=answer,
                    scoring_type="letter_exact",
                    task_type=None,
                    domain=None,
                    source_benchmark=NAME,
                    extra={"line_idx": line_idx},
                )
                n_yielded += 1
    if n_skipped_no_series:
        print(f"rl_valset.load: skipped {n_skipped_no_series} image(s) with no matching "
              f"fragments.jsonl series", flush=True)
