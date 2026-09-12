from __future__ import annotations

import json
from typing import Iterator

from ts_eval.benchmarks.base import QAItem

NAME = "tsaqa"
LICENSE = "MIT (confirmed via LICENSE file at https://huggingface.co/datasets/TSAQA/TSAQA-Benchmark/raw/main/LICENSE)"

_CACHE_DIR = "bench_data/tsaqa"

_QUESTION_TYPE_TO_SCORING = {
    "multiple_choices": "letter_exact",
    "true_or_false": "tf_exact",
    "ordering": "ordering_exact",
}


def _coerce_domain(raw: str) -> str:
    return raw


def load(split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
    from datasets import load_dataset

    ds = load_dataset("TSAQA/TSAQA-Benchmark", split=split, cache_dir=_CACHE_DIR)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    skipped = 0
    for i, row in enumerate(ds):
        scoring_type = _QUESTION_TYPE_TO_SCORING.get(row["question_type"])
        if scoring_type is None:
            skipped += 1
            continue
        try:
            series = json.loads(row["raw_ts"])
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue
        if not series or not row["answer"]:
            skipped += 1
            continue

        yield QAItem(
            id=f"tsaqa_{split}_{i}",
            series=series,
            question=row["question"],
            gold=row["answer"],
            scoring_type=scoring_type,
            options=None,
            task_type=row["task"],
            domain=_coerce_domain(row["domain"]),
            source_benchmark="tsaqa",
            extra={"dataset": row["dataset"], "meta_info": row["meta_info"]},
        )

    if skipped:
        print(f"[tsaqa.load] skipped {skipped} malformed row(s) out of {len(ds)}")
