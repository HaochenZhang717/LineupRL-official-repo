from __future__ import annotations

import json
import string
import sys
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download

from ts_eval.benchmarks.base import QAItem

NAME = "tsrbench"
LICENSE = (
    "UNCONFIRMED — no license found on HF, GitHub, the project page, or arXiv; "
    "this adapter is opt-in only."
)

_REPO_ID = "umd-zhou-lab/TSRBench"
_REPO_TYPE = "dataset"

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "bench_data" / "tsrbench"
_HF_CACHE_DIR = _CACHE_ROOT / "hf_cache"

_TASK_FILES = [
    "decision/qualitative_decision.jsonl",
    "decision/quantitative_decision.jsonl",
    "perception/perception.jsonl",
    "prediction/event_prediction.jsonl",
    "prediction/time_series_forecasting.jsonl",
    "reasoning/abductive_reasoning.jsonl",
    "reasoning/causal_reasoning.jsonl",
    "reasoning/deductive_reasoning.jsonl",
    "reasoning/etiological_reasoning.jsonl",
    "reasoning/inductive_reasoning.jsonl",
    "reasoning/numerical_reasoning.jsonl",
    "reasoning/temporal_relation_reasoning.jsonl",
]


def _require_license_accept(i_accept_unclear_license: bool) -> None:
    if not i_accept_unclear_license:
        raise RuntimeError(
            "TSRBench has no confirmed license (see ts_eval.benchmarks.tsrbench.LICENSE). "
            "Pass i_accept_unclear_license=True to load() to acknowledge this and proceed. "
            "Results from this adapter are opt-in only."
        )


def _ensure_downloaded() -> list[Path]:
    _HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for relpath in _TASK_FILES:
        local_path = hf_hub_download(
            repo_id=_REPO_ID,
            filename=relpath,
            repo_type=_REPO_TYPE,
            cache_dir=str(_HF_CACHE_DIR),
        )
        paths.append(Path(local_path))
    return paths


def _is_scalar(v: object) -> bool:
    return isinstance(v, (str, int, float))


def _resolve_mcq(question: str, answer: object, choices) -> tuple[str, str, str, dict | None, object]:
    ans_str = str(answer).strip()

    if isinstance(choices, list) and len(choices) > 0:
        letters = list(string.ascii_uppercase[: len(choices)])
        all_scalar = all(_is_scalar(c) for c in choices)
        if ans_str.upper() in letters:
            gold = ans_str.upper()
            scoring_type = "letter_exact"
        else:
            match_letter = None
            if all_scalar:
                for letter, text in zip(letters, choices):
                    if str(text).strip() == ans_str:
                        match_letter = letter
                        break
            if match_letter is not None:
                gold = match_letter
                scoring_type = "letter_exact"
            else:
                gold = ans_str
                scoring_type = "regex_numeric_extract_tolerance"
                return question, gold, scoring_type, None, choices

        if all_scalar:
            options = {letter: str(text) for letter, text in zip(letters, choices)}
            options_block = "\n".join(f"{letter}) {text}" for letter, text in options.items())
            question_final = f"{question}\n\nOptions:\n{options_block}"
            return question_final, gold, scoring_type, options, None
        return question, gold, scoring_type, None, choices

    if isinstance(choices, dict) and len(choices) > 0:
        keys_upper = {str(k).strip().upper(): v for k, v in choices.items()}
        if ans_str.upper() in keys_upper:
            gold = ans_str.upper()
            scoring_type = "letter_exact"
            all_scalar = all(_is_scalar(v) for v in choices.values())
            if all_scalar:
                options = {str(k).strip().upper(): str(v) for k, v in choices.items()}
                options_block = "\n".join(f"{letter}) {text}" for letter, text in options.items())
                question_final = f"{question}\n\nOptions:\n{options_block}"
                return question_final, gold, scoring_type, options, None
            return question, gold, scoring_type, None, choices
        return question, ans_str, "regex_numeric_extract_tolerance", None, choices

    if len(ans_str) == 1 and ans_str.isalpha():
        return question, ans_str.upper(), "letter_exact", None, None
    return question, ans_str, "regex_numeric_extract_tolerance", None, None


def _normalize_row(row: dict, default_task_name: str) -> QAItem | None:
    question = row.get("question")
    timeseries = row.get("timeseries")
    answer = row.get("answer")
    if not question or not timeseries or answer is None:
        return None

    choices = row.get("choices")
    question_final, gold, scoring_type, options, choices_raw = _resolve_mcq(question, answer, choices)

    task_type = row.get("task") or row.get("task_type") or row.get("category") or default_task_name
    domain = row.get("domain")
    extra = {"type": row.get("type"), "name_of_series": row.get("name_of_series")}
    if choices_raw is not None:
        extra["choices_raw"] = choices_raw

    return QAItem(
        id=default_task_name,
        series=timeseries,
        question=question_final,
        gold=gold,
        scoring_type=scoring_type,
        options=options,
        task_type=task_type,
        domain=domain,
        source_benchmark=NAME,
        extra=extra,
    )


def load(
    split: str = "test",
    limit: int | None = None,
    i_accept_unclear_license: bool = False,
) -> Iterator[QAItem]:
    _require_license_accept(i_accept_unclear_license)

    paths = _ensure_downloaded()
    n_yielded = 0
    n_skipped = 0
    for path in paths:
        task_name = path.stem
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and n_yielded >= limit:
                    if n_skipped:
                        print(f"[tsrbench] skipped {n_skipped} malformed rows", file=sys.stderr)
                    return
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    item = _normalize_row(row, task_name)
                except Exception:
                    item = None
                if item is None:
                    n_skipped += 1
                    continue
                item.id = f"{task_name}::{i}"
                n_yielded += 1
                yield item
    if n_skipped:
        print(f"[tsrbench] skipped {n_skipped} malformed rows", file=sys.stderr)
