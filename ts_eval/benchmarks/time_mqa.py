from __future__ import annotations

import os
import re
from typing import Any, Iterator

from ts_eval.benchmarks.base import QAItem

NAME = "time_mqa"
LICENSE = (
    "Apache-2.0 (confirmed via HF card). Dataset itself is GATED -- requires "
    "HF login + accepting access conditions; set HF_TOKEN env var."
)

_REPO_ID = "Time-MQA/TSQA"
_FILENAME = "Open_Ended_QA/open_ended_QA.csv"
_CACHE_DIR = "bench_data/time_mqa"

_QUESTION_COL_CANDIDATES = ["question", "instruction", "text", "input"]
_ANSWER_COL_CANDIDATES = ["answer", "output", "label"]

_SERIES_RE = re.compile(r"(?:-?\d+\.?\d*[,\s]+){5,}-?\d+\.?\d*")
_QUE_ANS_RE = re.compile(r"<QUE>(.*?)<ANS>(.*?)(?:</END>|$)", re.DOTALL)
_MCQ_OPTION_RE = re.compile(r"\b[A-I]\)")
_MCQ_ANSWER_RE = re.compile(r"Answer:\s*([A-I])\)?", re.IGNORECASE)
_TF_ANSWER_RE = re.compile(r"Answer:\s*(True|False)", re.IGNORECASE)


def _download_csv(hf_token: str | None) -> str:
    from huggingface_hub import hf_hub_download

    token = hf_token or os.environ.get("HF_TOKEN")
    try:
        return hf_hub_download(
            repo_id=_REPO_ID,
            filename=_FILENAME,
            repo_type="dataset",
            token=token,
            cache_dir=_CACHE_DIR,
        )
    except Exception as e:
        raise RuntimeError(
            "Time-MQA/TSQA is gated. To use this adapter: "
            "(1) log into huggingface.co, "
            "(2) visit https://huggingface.co/datasets/Time-MQA/TSQA and accept the "
            "dataset's access conditions, "
            "(3) set the HF_TOKEN env var to a token from that account, "
            "(4) re-run. "
            "NOTE: the exact CSV column parsing in this adapter is UNVERIFIED against "
            "real data (built from paper examples only) -- after first successful "
            "download, re-run this module's test (tests/test_benchmark_time_mqa.py) "
            "and fix field names in ts_eval/benchmarks/time_mqa.py if they don't "
            f"match. Underlying error: {type(e).__name__}: {e}"
        ) from e


def _read_rows(csv_path: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        return df.to_dict(orient="records")
    except ImportError:
        import csv as csv_mod

        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv_mod.DictReader(f))


def _row_text(row: dict[str, Any], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return None


def _split_combined(text: str) -> tuple[str, str | None]:
    m = _QUE_ANS_RE.search(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    idx = text.find("Answer:")
    if idx != -1:
        return text[:idx].strip(), text[idx:].strip()
    return text.strip(), None


def _strip_answer_tail(text: str) -> str:
    idx = text.find("Answer:")
    return text[:idx].strip() if idx != -1 else text.strip()


def _extract_tf_gold(question_raw: str, answer_raw: str | None) -> str | None:
    search_text = question_raw + ("\n" + answer_raw if answer_raw else "")
    m = _TF_ANSWER_RE.search(search_text)
    if m:
        return "T" if m.group(1).upper().startswith("TRUE") else "F"
    if answer_raw:
        m2 = re.search(r"\b(True|False)\b", answer_raw, re.IGNORECASE)
        if m2:
            return "T" if m2.group(1).upper().startswith("TRUE") else "F"
    return None


def _extract_mcq_gold(question_raw: str, answer_raw: str | None) -> str | None:
    search_text = question_raw + ("\n" + answer_raw if answer_raw else "")
    m = _MCQ_ANSWER_RE.search(search_text)
    if m:
        return m.group(1).upper()
    if answer_raw:
        m2 = re.match(r"\s*([A-I])\)", answer_raw)
        if m2:
            return m2.group(1).upper()
    return None


def _parse_row(row: dict[str, Any]) -> dict[str, str] | None:
    q_col = _row_text(row, _QUESTION_COL_CANDIDATES)
    a_col = _row_text(row, _ANSWER_COL_CANDIDATES)

    if q_col is not None:
        question_raw, answer_raw = q_col, a_col
    else:
        first_key = next(iter(row), None)
        if first_key is None or row[first_key] in (None, ""):
            return None
        question_raw, answer_raw = _split_combined(str(row[first_key]))

    is_tf = "true or false" in question_raw.lower() or "true or false" in (answer_raw or "").lower()
    is_mcq = bool(_MCQ_OPTION_RE.search(question_raw))

    if is_tf:
        gold = _extract_tf_gold(question_raw, answer_raw)
        if gold is None:
            return None
        return {
            "question": _strip_answer_tail(question_raw),
            "gold": gold,
            "scoring_type": "tf_exact",
            "task_type": "tf",
        }
    if is_mcq:
        gold = _extract_mcq_gold(question_raw, answer_raw)
        if gold is None:
            return None
        return {
            "question": _strip_answer_tail(question_raw),
            "gold": gold,
            "scoring_type": "letter_exact",
            "task_type": "mcq",
        }
    return None


def _extract_series(row: dict[str, Any]) -> list[float]:
    combined = " ".join(str(v) for v in row.values() if v is not None)
    m = _SERIES_RE.search(combined)
    if not m:
        return []
    out: list[float] = []
    for tok in re.split(r"[,\s]+", m.group(0).strip()):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def load(split: str = "test", limit: int | None = None, hf_token: str | None = None) -> Iterator[QAItem]:
    csv_path = _download_csv(hf_token)
    rows = _read_rows(csv_path)
    if limit is not None:
        rows = rows[:limit]

    n_total = len(rows)
    skipped_unrecognized = 0
    skipped_no_series = 0

    for i, row in enumerate(rows):
        parsed = _parse_row(row)
        if parsed is None:
            skipped_unrecognized += 1
            continue

        series = _extract_series(row)
        if not series:
            skipped_no_series += 1
            continue

        yield QAItem(
            id=f"time_mqa_{split}_{i}",
            series=series,
            question=parsed["question"],
            gold=parsed["gold"],
            scoring_type=parsed["scoring_type"],
            options=None,
            task_type=parsed["task_type"],
            domain=None,
            source_benchmark="time_mqa",
            extra={"raw_row": row},
        )

    if skipped_unrecognized or skipped_no_series:
        print(
            f"[time_mqa.load] {n_total} row(s) read; skipped {skipped_unrecognized} "
            "unrecognized (not TF/MCQ -- presumably one of the open-ended free-text "
            f"rows, or unrecognized header layout) and {skipped_no_series} TF/MCQ "
            "row(s) with no extractable numeric series"
        )
