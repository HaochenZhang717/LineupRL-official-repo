from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np


@dataclass
class Fragment:

    sample_id: int
    length: int
    series: List[float]
    captions: List[str] = field(default_factory=list)

    @property
    def caption(self) -> Optional[str]:
        return self.captions[0] if self.captions else None


def parse_ot(ot: object) -> List[float]:
    if isinstance(ot, (list, tuple, np.ndarray)):
        return [float(v) for v in ot]
    if isinstance(ot, str):
        s = ot.strip()
        if not s:
            raise ValueError("empty OT string")
        return [float(v) for v in ast.literal_eval(s)]
    raise TypeError(f"unsupported OT type: {type(ot)!r}")


def _dedup_rows(rows: Iterable[dict]) -> Iterator[Fragment]:
    by_id: dict[int, Fragment] = {}
    order: list[int] = []
    for r in rows:
        sid = int(r["SampleID"])
        if sid not in by_id:
            by_id[sid] = Fragment(
                sample_id=sid,
                length=int(r.get("TimeInterval", 0)) or len(parse_ot(r["OT"])),
                series=parse_ot(r["OT"]),
                captions=[],
            )
            order.append(sid)
        cap = r.get("Text")
        if isinstance(cap, str) and cap.strip():
            by_id[sid].captions.append(cap.strip())
    for sid in order:
        yield by_id[sid]


def load_fragments(
    source: str = "WinfredGe/TSFragment-600K",
    *,
    split: str = "train",
    limit: Optional[int] = None,
    local_path: Optional[str | Path] = None,
    shuffle: bool = False,
    seed: int = 0,
    shuffle_buffer: int = 10000,
    streaming: bool = True,
    cache_dir: Optional[str | Path] = None,
) -> Iterator[Fragment]:
    rows = _read_rows(source, split=split, local_path=local_path, limit=limit,
                      shuffle=shuffle, seed=seed, shuffle_buffer=shuffle_buffer,
                      streaming=streaming, cache_dir=cache_dir)
    n = 0
    for frag in _dedup_rows(rows):
        yield frag
        n += 1
        if limit is not None and n >= limit:
            return


def _read_rows(
    source: str,
    *,
    split: str,
    local_path: Optional[str | Path],
    limit: Optional[int],
    shuffle: bool = False,
    seed: int = 0,
    shuffle_buffer: int = 10000,
    streaming: bool = True,
    cache_dir: Optional[str | Path] = None,
) -> Iterable[dict]:
    if local_path is not None:
        return _read_local(Path(local_path))

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "`datasets` not installed and no local_path given. "
            "Run `pip install datasets`, or pass local_path=<csv/parquet>."
        ) from e

    ds = load_dataset(source, split=split, streaming=streaming,
                      cache_dir=str(cache_dir) if cache_dir else None)
    if "TextEmbedding" in (ds.column_names or []):
        ds = ds.remove_columns("TextEmbedding")

    if not streaming:
        return ds.shuffle(seed=seed) if shuffle else ds

    if shuffle:
        return ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    if limit is not None:
        ds = ds.take(limit * 5)
    return ds


def _read_local(path: Path) -> Iterable[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    if path.suffix == ".csv":
        import pandas as pd

        return pd.read_csv(path).to_dict("records")
    if path.suffix in (".json", ".jsonl"):
        with path.open() as f:
            if path.suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)
    raise ValueError(f"unsupported local file type: {path.suffix}")


def synthetic_fragments(n: int = 8, seed: int = 0) -> Iterator[Fragment]:
    rng = np.random.default_rng(seed)
    lengths = [24, 48, 96]
    for i in range(n):
        L = lengths[i % len(lengths)]
        t = np.linspace(0, 1, L)
        kind = i % 4
        if kind == 0:
            y = 3.0 * t + rng.normal(0, 0.15, L)
        elif kind == 1:
            y = np.exp(-((t - 0.5) ** 2) / 0.02) * 2.0 + rng.normal(0, 0.1, L)
        elif kind == 2:
            y = np.sin(2 * np.pi * 3 * t) + rng.normal(0, 0.1, L)
        else:
            y = np.where(t < 0.66, 0.0, 1.5) + rng.normal(0, 0.1, L)
        y = y + 10.0
        yield Fragment(sample_id=i, length=L, series=[float(v) for v in y])
