from __future__ import annotations

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

DEFAULT_ROOT = Path("bench_data/bedtime")
DATASETS = ("truce_stock", "truce_synthetic", "sushi", "taxosynth")

CLASSED = ("sushi", "taxosynth")


@dataclass
class SeriesRow:

    series_uid: str
    dataset: str
    idx: str
    series: list[float]
    annotations: list[str]
    cls: str | None = None
    subclass: str | None = None
    extra: dict = field(default_factory=dict)


def parse_series(raw: str) -> list[float]:
    if isinstance(raw, (list, tuple)):
        return [float(v) for v in raw]
    text = str(raw).strip()
    try:
        return [float(v) for v in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        cleaned = text.strip("[]")
        return [float(tok.strip()) for tok in cleaned.split(",") if tok.strip()]


def _is_finite(values: Sequence[float]) -> bool:
    return all(v == v and abs(v) != float("inf") for v in values)


def load_dataset_rows(name: str, root: Path | str = DEFAULT_ROOT) -> list[SeriesRow]:
    import pandas as pd

    path = Path(root) / f"{name}.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    by_uid: dict[str, SeriesRow] = {}
    n_nonfinite = 0
    n_blank_ann = 0
    for record in df.to_dict("records"):
        annotation = record.get("annotations")
        if not isinstance(annotation, str) or not annotation.strip():
            n_blank_ann += 1
            continue
        uid = f"{name}_{record['idx']}"
        row = by_uid.get(uid)
        if row is None:
            values = parse_series(record["series"])
            if not values or not _is_finite(values):
                n_nonfinite += 1
                by_uid[uid] = None
                continue
            row = SeriesRow(
                series_uid=uid,
                dataset=name,
                idx=str(record["idx"]),
                series=values,
                annotations=[],
                cls=record.get("class") if isinstance(record.get("class"), str) else None,
                subclass=record.get("subclass") if isinstance(record.get("subclass"), str) else None,
            )
            by_uid[uid] = row
        if row is None:
            continue
        if annotation.strip() not in row.annotations:
            row.annotations.append(annotation.strip())

    rows = [r for r in by_uid.values() if r is not None and r.annotations]
    if n_nonfinite or n_blank_ann:
        print(f"[bedtime.{name}] dropped {n_nonfinite} series with NaN/Inf values and "
              f"{n_blank_ann} rows with a blank annotation; kept {len(rows)} series",
              flush=True)
    return rows


def sample_rows(rows: Sequence[SeriesRow], per_dataset: int | None, seed: int) -> list[SeriesRow]:
    ordered = sorted(rows, key=lambda r: r.series_uid)
    if per_dataset is None or per_dataset >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, per_dataset), key=lambda r: r.series_uid)


def load_all(root: Path | str = DEFAULT_ROOT, per_dataset: int | None = 500,
             seed: int = 2020, datasets: Sequence[str] = DATASETS) -> dict[str, list[SeriesRow]]:
    out: dict[str, list[SeriesRow]] = {}
    for name in datasets:
        rows = sample_rows(load_dataset_rows(name, root), per_dataset, seed)
        out[name] = rows
        print(f"[bedtime.{name}] sampled {len(rows)} series / "
              f"{sum(len(r.annotations) for r in rows)} annotations", flush=True)
    return out


def synthetic_rows(dataset: str = "synthetic", n: int = 20, seed: int = 0,
                   classed: bool = True) -> list[SeriesRow]:
    rng = random.Random(seed)
    classes = ["rising", "falling", "flat", "spiky"]
    rows = []
    for i in range(n):
        cls = classes[i % len(classes)]
        base = [rng.uniform(-1, 1) for _ in range(12)]
        if cls == "rising":
            values = [b + 3 * t for t, b in enumerate(base)]
        elif cls == "falling":
            values = [b - 3 * t for t, b in enumerate(base)]
        elif cls == "flat":
            values = base
        else:
            values = [b + (30 if t == 6 else 0) for t, b in enumerate(base)]
        rows.append(SeriesRow(
            series_uid=f"{dataset}_{i}",
            dataset=dataset,
            idx=str(i),
            series=values,
            annotations=[f"the series is {cls}", f"a {cls} pattern over time"],
            cls=cls if classed else None,
            subclass=f"{cls}_sub" if classed else None,
        ))
    return rows
