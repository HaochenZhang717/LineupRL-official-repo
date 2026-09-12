from __future__ import annotations

import ast
import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

from ts_eval.benchmarks.base import QAItem

NAME = "cats_bench"
LICENSE = "MIT (per HF dataset card; note the GitHub code repo itself ships no LICENSE file)"

_REPO_ID = "mhfisher/CaTSBench"
_REPO_TYPE = "dataset"

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "bench_data" / "cats_bench"
_HF_CACHE_DIR = _CACHE_ROOT / "hf_cache"
_EXTRACT_DIR = _CACHE_ROOT / "extracted"

_KNOWN_POOL1_TASK_TYPES = {
    "caption_retrieval_perturbed",
    "plot_retrieval_same_domain",
    "ts_retrieval_perturbed",
    "ts_comparison_amplitude",
    "ts_comparison_mean",
    "ts_comparison_volatility",
    "ts_comparison_peak_earlier",
}

_SERIES_LINE_RE = re.compile(r"Here is a time series:\s*\n([0-9.,\-\s]+)\n\s*\n")
_AB_SERIES_RE = re.compile(r"A:\s*\n\s*(\[[^\]]*\])\s*\n.*?B:\s*\n\s*(\[[^\]]*\])", re.DOTALL)


def _ensure_extracted() -> None:
    _EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    tasks_json = _EXTRACT_DIR / "QA_hard_small" / "tasks.json"
    tm_final = _EXTRACT_DIR / "QA" / "temporal_matching" / "mcqs_final.jsonl"
    tm_mid2 = _EXTRACT_DIR / "QA" / "temporal_matching" / "mcqs_mid2_short_len6.jsonl"
    if tasks_json.exists() and tm_final.exists() and tm_mid2.exists():
        return
    p1 = hf_hub_download(
        repo_id=_REPO_ID,
        filename="QA_hard_small.zip",
        repo_type=_REPO_TYPE,
        cache_dir=str(_HF_CACHE_DIR),
    )
    p2 = hf_hub_download(
        repo_id=_REPO_ID,
        filename="QA.zip",
        repo_type=_REPO_TYPE,
        cache_dir=str(_HF_CACHE_DIR),
    )
    with zipfile.ZipFile(p1) as zf:
        zf.extractall(_EXTRACT_DIR)
    with zipfile.ZipFile(p2) as zf:
        zf.extractall(_EXTRACT_DIR)


def _parse_series_from_prompt(row: dict) -> list:
    task_type = row["task_type"]
    text = row.get("prompt_no_image") or row.get("prompt") or ""

    if task_type in ("caption_retrieval_perturbed", "plot_retrieval_same_domain"):
        m = _SERIES_LINE_RE.search(text)
        if m:
            return [float(x) for x in m.group(1).split(",") if x.strip()]
        return []

    if task_type == "ts_retrieval_perturbed":
        gold = row["ground_truth"].strip().upper()
        m = re.search(r"\(" + re.escape(gold) + r"\)\s*(\[[^\]]*\])", text)
        if m:
            return list(ast.literal_eval(m.group(1)))
        return []

    if task_type.startswith("ts_comparison"):
        m = _AB_SERIES_RE.search(text)
        if m:
            series_a = list(ast.literal_eval(m.group(1)))
            series_b = list(ast.literal_eval(m.group(2)))
            return [series_a, series_b]
        return []

    return []


def _load_pool1(limit: int | None) -> Iterator[QAItem]:
    tasks_path = _EXTRACT_DIR / "QA_hard_small" / "tasks.json"
    with open(tasks_path) as f:
        rows = json.load(f)
    n_yielded = 0
    for row in rows:
        if limit is not None and n_yielded >= limit:
            return
        try:
            question = row.get("prompt_no_image") or row.get("prompt")
            if question is None:
                continue
            series = _parse_series_from_prompt(row)
            item = QAItem(
                id=row["task_id"],
                series=series,
                question=question,
                gold=row["ground_truth"],
                scoring_type="letter_exact",
                task_type=row["task_type"],
                source_benchmark=NAME,
                extra={"pool": "diagnostic_mcq", "ts_name": row.get("ts_name")},
            )
        except Exception:
            continue
        n_yielded += 1
        yield item


def _build_temporal_matching_question(row: dict) -> str:
    options_str = "\n".join(f"({letter}) {value}" for letter, value in row["options"].items())
    return (
        f"Given this {row['sampling_frequency']} time series starting at "
        f"{row['starting_time']}, what is the value at {row['question_time']}?\n"
        f"{options_str}"
    )


def _load_pool2(limit: int | None) -> Iterator[QAItem]:
    files = [
        _EXTRACT_DIR / "QA" / "temporal_matching" / "mcqs_final.jsonl",
        _EXTRACT_DIR / "QA" / "temporal_matching" / "mcqs_mid2_short_len6.jsonl",
    ]
    n_yielded = 0
    for path in files:
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and n_yielded >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    item = QAItem(
                        id=f"{path.stem}::{i}",
                        series=row["time_series"],
                        question=_build_temporal_matching_question(row),
                        gold=row["answer"],
                        scoring_type="letter_exact",
                        task_type="temporal_matching",
                        source_benchmark=NAME,
                        extra={
                            "pool": "temporal_matching",
                            "bucket": row["bucket"],
                            "sampling_frequency": row["sampling_frequency"],
                        },
                    )
                except Exception:
                    continue
                n_yielded += 1
                yield item


def load(split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
    _ensure_extracted()
    n_yielded = 0
    for item in _load_pool1(limit):
        n_yielded += 1
        yield item
    if limit is not None:
        remaining = limit - n_yielded
        if remaining <= 0:
            return
        yield from _load_pool2(remaining)
    else:
        yield from _load_pool2(None)
