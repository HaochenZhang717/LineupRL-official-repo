from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from ts_bedtime.prompts import (differentiation_question, native_prompt_builder,
                                recognition_question)
from ts_eval.benchmarks.base import QAItem

NAME = "bedtime"
LICENSE = "MIT (HF dataset card: `license: mit`)"

_DATA_DIR = Path(os.environ.get("BEDTIME_DIR", "bench_data/bedtime"))
_STRATEGY = os.environ.get("BEDTIME_STRATEGY", "sbert")

_TASK_SCORING = {"recognition": "tf_exact", "differentiation": "letter_exact"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m ts_bedtime.prepare --strategy {_STRATEGY}` first")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load(split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
    del split
    series_by_uid = {r["series_uid"]: r for r in _read_jsonl(_DATA_DIR / "series.jsonl")}
    items = _read_jsonl(_DATA_DIR / f"items_{_STRATEGY}.jsonl")
    if limit is not None:
        items = items[:limit]

    skipped = 0
    for row in items:
        series_row = series_by_uid.get(row["series_uid"])
        scoring_type = _TASK_SCORING.get(row["task_type"])
        if series_row is None or scoring_type is None:
            skipped += 1
            continue

        if row["task_type"] == "recognition":
            question = recognition_question(row["description"])
            options = None
        else:
            options = list(row["options"])
            question = differentiation_question(options)

        yield QAItem(
            id=row["item_id"],
            series=series_row["series"],
            question=question,
            gold=row["gold"],
            scoring_type=scoring_type,
            options=options,
            task_type=row["task_type"],
            domain=row["dataset"],
            source_benchmark=NAME,
            extra={
                "caption_key": row["series_uid"],
                "series_uid": row["series_uid"],
                "dataset": row["dataset"],
                "cls": row.get("cls"),
                "subclass": row.get("subclass"),
                "description": row.get("description"),
                "strategy": _STRATEGY,
                "native_prompt": native_prompt_builder(
                    row["task_type"], row.get("description", ""), options),
            },
        )

    if skipped:
        print(f"[bedtime.load] skipped {skipped} row(s) with an unknown series_uid or task_type")
