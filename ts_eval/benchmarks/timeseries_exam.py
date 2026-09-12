from __future__ import annotations

from typing import Iterator

from ts_eval.benchmarks.base import QAItem

NAME = "timeseries_exam"
LICENSE = "MIT (Copyright 2024 Auton Lab, Carnegie Mellon University; license text in HF README, no separate LICENSE file)"

_CACHE_DIR = "bench_data/timeseries_exam"


def _build_question(row: dict) -> str:
    options_string = "\n".join(f"{chr(65 + i)}) {opt}" for i, opt in enumerate(row["options"]))
    parts = [row["question"], "", options_string]
    if row.get("format_hint"):
        parts += ["", row["format_hint"]]
    return "\n".join(parts)


def load(split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
    from datasets import load_dataset

    ds = load_dataset("AutonLab/TimeSeriesExam1", split=split, cache_dir=_CACHE_DIR)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    skipped = 0
    for i, row in enumerate(ds):
        options = row["options"]
        answer = row["answer"]
        if not options or answer not in options:
            skipped += 1
            continue

        ts = row.get("ts")
        ts1, ts2 = row.get("ts1"), row.get("ts2")
        if ts:
            series = ts
        elif ts1 and ts2:
            series = [ts1, ts2]
        else:
            skipped += 1
            continue

        letter = chr(65 + options.index(answer))
        gold = f"{letter}) {answer}"

        yield QAItem(
            id=f"timeseries_exam_{split}_{row['id']}",
            series=series,
            question=_build_question(row),
            gold=gold,
            scoring_type="letter_flexible",
            options=options,
            task_type=row["question_type"],
            domain=row["category"],
            source_benchmark="timeseries_exam",
            extra={
                "subcategory": row["subcategory"],
                "difficulty": row["difficulty"],
                "tid": row["tid"],
            },
        )

    if skipped:
        print(f"[timeseries_exam.load] skipped {skipped} malformed row(s) out of {len(ds)}")
