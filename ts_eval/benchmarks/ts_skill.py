from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import h5py
from huggingface_hub import hf_hub_download

from ts_eval.benchmarks.base import QAItem

NAME = "ts_skill"
LICENSE = (
    "UNSPECIFIED — no license found on the HF card, GitHub, or README. "
    "Treat as research-use-only; do not redistribute."
)

_REPO_ID = "Anonymous-Dataset-H/TS-Skill"
_REPO_TYPE = "dataset"
_QA_FILENAME = "ts_skill_qa.jsonl"

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "bench_data" / "ts_skill"


def _download_qa_jsonl() -> Path:
    return Path(
        hf_hub_download(
            repo_id=_REPO_ID,
            filename=_QA_FILENAME,
            repo_type=_REPO_TYPE,
            cache_dir=str(_CACHE_ROOT / "hf_cache"),
        )
    )


def _download_h5(ts_file: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=_REPO_ID,
            filename=f"ts/{ts_file}",
            repo_type=_REPO_TYPE,
            cache_dir=str(_CACHE_ROOT / "hf_cache"),
        )
    )


def _read_channel(ts_file: str, channel_idx: int) -> list[float]:
    h5_path = _download_h5(ts_file)
    with h5py.File(h5_path, "r") as f:
        return f["values"][channel_idx].tolist()


def load(split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
    qa_path = _download_qa_jsonl()
    n_yielded = 0
    n_skipped = 0
    with open(qa_path) as f:
        for line_idx, line in enumerate(f):
            if limit is not None and n_yielded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                series = _read_channel(row["ts_file"], row["channel_idx"])
                item = QAItem(
                    id=f"{row['ts_file']}::{row['channel_idx']}::{line_idx}",
                    series=series,
                    question=row["question"],
                    gold=row["answer"],
                    scoring_type="regex_numeric_extract_tolerance",
                    task_type=row["sk_type"],
                    domain=row["category"],
                    source_benchmark=NAME,
                    extra={
                        "sk_level": row["sk_level"],
                        "cluster": row["cluster"],
                        "metric": row["metric"],
                        "cross_channel_indices": row.get("cross_channel_indices"),
                    },
                )
            except Exception as e:
                n_skipped += 1
                print(f"ts_skill.load: skipping line {line_idx} ({e.__class__.__name__}: {e})", flush=True)
                continue
            n_yielded += 1
            yield item
    if n_skipped:
        print(f"ts_skill.load: skipped {n_skipped} malformed/undownloadable row(s)", flush=True)
