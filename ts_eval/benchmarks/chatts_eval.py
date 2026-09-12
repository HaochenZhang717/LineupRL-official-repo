from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import requests

from ts_eval.benchmarks.base import QAItem

NAME = "chatts_eval"
LICENSE = (
    "Code: MIT (github.com/NetManAIOps/ChatTS). Eval DATA: CC BY 4.0 (Zenodo "
    "record 14349206) — different from the code license; attribute to source "
    "datasets NAB/Weather/Oracle/AIOps per the Zenodo record when publishing "
    "results."
)

_ZENODO_RECORD = "14349206"
_FILES = ["dataset_a.json", "dataset_b.json"]
_DIRECT_URL = f"https://zenodo.org/records/{_ZENODO_RECORD}/files/{{fname}}"
_RECORD_API = f"https://zenodo.org/api/records/{_ZENODO_RECORD}"

_CACHE_ROOT = Path(__file__).resolve().parent.parent.parent / "bench_data" / "chatts_eval"

_ELIGIBLE_ABILITIES = {"trend", "season", "noise", "local", "causal"}

_NO_LOCAL_FLUCTUATIONS = "No local characteristic fluctuations found."


def _resolve_download_url(fname: str) -> str:
    direct = _DIRECT_URL.format(fname=fname)
    resp = requests.head(direct, timeout=30, allow_redirects=True)
    if resp.status_code == 200:
        return direct
    resp = requests.get(_RECORD_API, timeout=30)
    resp.raise_for_status()
    record = resp.json()
    for f in record.get("files", []):
        key = f.get("key") or f.get("filename")
        if key == fname:
            links = f.get("links", {})
            return links.get("self") or links.get("download")
    raise RuntimeError(f"could not resolve a download URL for {fname} via the Zenodo record API")


def _ensure_downloaded() -> list[Path]:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname in _FILES:
        local_path = _CACHE_ROOT / fname
        if not local_path.exists():
            url = _resolve_download_url(fname)
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
        paths.append(local_path)
    return paths


def _type_to_str(t: object) -> str | None:
    if isinstance(t, str) and t.strip():
        return t
    if isinstance(t, list) and t and all(isinstance(x, str) for x in t):
        return "/".join(t)
    return None


def _gold_for(ability: str, attr: object) -> str | None:
    if ability == "causal":
        return attr if isinstance(attr, str) and attr.strip() else None

    if ability in ("trend", "season", "noise"):
        if isinstance(attr, dict):
            return _type_to_str(attr.get("type"))
        return None

    if ability == "local":
        if isinstance(attr, list):
            if not attr:
                return _NO_LOCAL_FLUCTUATIONS
            types = [_type_to_str(d.get("type")) for d in attr if isinstance(d, dict)]
            types = [t for t in types if t]
            return ", ".join(types) if types else None
        if isinstance(attr, dict):
            return _type_to_str(attr.get("type"))
        return None

    return None


def _iter_all_rows(paths: list[Path]):
    for path in paths:
        with open(path) as f:
            items = json.load(f)
        for item_idx, row in enumerate(items):
            yield path.stem, item_idx, row


def load(split: str = "test", limit: int | None = None) -> Iterator[QAItem]:
    paths = _ensure_downloaded()
    n_yielded = 0
    n_skipped_ineligible = 0
    n_skipped_malformed = 0

    for fname, item_idx, row in _iter_all_rows(paths):
        if limit is not None and n_yielded >= limit:
            break
        try:
            timeseries = row["timeseries"]
            cols = row.get("cols")
            question = row["question"]
            ability_types = row["ability_types"]
            attributes = row["attributes"]
        except (KeyError, TypeError):
            continue
        if not timeseries or not question or len(ability_types) != len(attributes):
            continue

        for sub_index, (ability, attr) in enumerate(zip(ability_types, attributes)):
            if limit is not None and n_yielded >= limit:
                break
            if ability not in _ELIGIBLE_ABILITIES:
                n_skipped_ineligible += 1
                continue
            gold = _gold_for(ability, attr)
            if gold is None:
                n_skipped_malformed += 1
                continue
            item = QAItem(
                id=f"{fname}::{item_idx}::{sub_index}",
                series=timeseries,
                question=question,
                gold=gold,
                scoring_type="llm_judge",
                task_type=ability,
                domain=None,
                source_benchmark=NAME,
                extra={
                    "cols": cols,
                    "ability_type": ability,
                    "sub_index": sub_index,
                    "attributes": attr,
                },
            )
            n_yielded += 1
            yield item

    if n_skipped_ineligible or n_skipped_malformed:
        print(
            f"[chatts_eval] skipped {n_skipped_ineligible} ineligible-ability sub-questions "
            f"(*-inductive/deductive/MCQ2) and {n_skipped_malformed} malformed rows",
            file=sys.stderr,
        )
