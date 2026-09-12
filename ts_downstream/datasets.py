from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class DownstreamItem:
    id: str
    series: np.ndarray
    task: str
    split: str
    dataset: str
    label: Optional[int] = None
    future: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def make_synthetic(
    task: str = "classification",
    n: int = 240,
    length: int = 96,
    horizon: int = 24,
    n_classes: int = 3,
    seed: int = 0,
) -> list[DownstreamItem]:
    rng = np.random.default_rng(seed)
    items: list[DownstreamItem] = []
    t = np.arange(length, dtype=np.float64)

    def split_of(i: int) -> str:
        r = i / n
        return "train" if r < 0.6 else ("val" if r < 0.8 else "test")

    if task == "classification":
        for i in range(n):
            cls = i % n_classes
            noise = rng.normal(0, 0.3, size=length)
            if cls == 0:
                s = 2.0 - 0.03 * t + noise
            elif cls == 1:
                s = -2.0 + 0.03 * t + noise
            else:
                s = 1.5 * np.sin(2 * np.pi * t / 12.0) + noise
            items.append(DownstreamItem(
                id=f"syn_cls_{i:05d}", series=s.astype(np.float32),
                task="classification", split=split_of(i), dataset="synthetic", label=int(cls)))
    elif task == "forecasting":
        for i in range(n):
            period = rng.choice([8, 12, 16])
            amp = rng.uniform(0.5, 2.0)
            slope = rng.uniform(-0.02, 0.02)
            phase = rng.uniform(0, 2 * np.pi)
            full = np.arange(length + horizon, dtype=np.float64)
            base = amp * np.sin(2 * np.pi * full / period + phase) + slope * full
            base = base + rng.normal(0, 0.1, size=full.shape)
            s = base[:length]
            fut = base[length:length + horizon]
            items.append(DownstreamItem(
                id=f"syn_fc_{i:05d}", series=s.astype(np.float32),
                task="forecasting", split=split_of(i), dataset="synthetic",
                future=fut.astype(np.float32),
                extra={"period": int(period)}))
    else:
        raise ValueError(f"unknown task {task!r}")
    return items


def load_ucr(name: str, val_frac: float = 0.2, seed: int = 2020,
             limit: Optional[int] = None) -> list[DownstreamItem]:
    from sktime.datasets import load_UCR_UEA_dataset

    def _to_arrays(X, y):
        series = []
        for _, row in X.iterrows():
            cell = row.iloc[0]
            series.append(np.asarray(cell, dtype=np.float32))
        labels = np.asarray(y)
        classes = sorted(set(labels.tolist()))
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        return series, [cls_to_idx[c] for c in labels], cls_to_idx

    Xtr, ytr = load_UCR_UEA_dataset(name, split="train", return_X_y=True)
    Xte, yte = load_UCR_UEA_dataset(name, split="test", return_X_y=True)
    tr_series, tr_lab, cls_map = _to_arrays(Xtr, ytr)
    te_series, te_lab, _ = _to_arrays(Xte, yte)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(tr_series))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_idx = set(idx[:n_val].tolist())

    items: list[DownstreamItem] = []
    for j, (s, lab) in enumerate(zip(tr_series, tr_lab)):
        split = "val" if j in val_idx else "train"
        items.append(DownstreamItem(id=f"ucr_{name}_tr_{j:05d}", series=s,
                                    task="classification", split=split,
                                    dataset=f"ucr:{name}", label=int(lab)))
    for j, (s, lab) in enumerate(zip(te_series, te_lab)):
        items.append(DownstreamItem(id=f"ucr_{name}_te_{j:05d}", series=s,
                                    task="classification", split="test",
                                    dataset=f"ucr:{name}", label=int(lab)))
    if limit:
        keep = {"train": [], "val": [], "test": []}
        for it in items:
            if len(keep[it.split]) < limit:
                keep[it.split].append(it)
        items = keep["train"] + keep["val"] + keep["test"]
    return items


def load_ett(csv_path: str, target: str = "OT", lookback: int = 96, horizon: int = 96,
             stride: int = 1, max_windows: Optional[int] = None,
             seed: int = 2020) -> list[DownstreamItem]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in {list(df.columns)}")
    vals = df[target].to_numpy(dtype=np.float64)
    n = len(vals)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_end, val_end = n_train, n_train + n_val

    mu = float(vals[:train_end].mean())
    sigma = float(vals[:train_end].std() + 1e-8)
    norm = (vals - mu) / sigma

    def split_of(start_of_target: int) -> Optional[str]:
        if start_of_target < train_end:
            return "train"
        if start_of_target < val_end:
            return "val"
        if start_of_target + horizon <= n:
            return "test"
        return None

    items: list[DownstreamItem] = []
    tag = Path(csv_path).stem
    i = 0
    for start in range(0, n - lookback - horizon + 1, stride):
        tgt_start = start + lookback
        split = split_of(tgt_start)
        if split is None:
            continue
        s = norm[start:start + lookback].astype(np.float32)
        fut = norm[tgt_start:tgt_start + horizon].astype(np.float32)
        items.append(DownstreamItem(id=f"ett_{tag}_{start:07d}", series=s,
                                    task="forecasting", split=split,
                                    dataset=f"ett:{tag}", future=fut,
                                    extra={"mu": mu, "sigma": sigma}))
        i += 1
    if max_windows and len(items) > max_windows:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(len(items), size=max_windows, replace=False))
        items = [items[k] for k in keep_idx]
    return items


def load_dataset(spec: str, task: str, *, lookback: int = 96, horizon: int = 96,
                 stride: int = 1, max_windows: Optional[int] = None,
                 val_frac: float = 0.2, seed: int = 2020,
                 limit: Optional[int] = None) -> list[DownstreamItem]:
    if spec == "synthetic":
        return make_synthetic(task="forecasting" if task == "reconstruction" else task,
                              horizon=horizon, seed=seed)
    if spec.startswith("ucr:"):
        return load_ucr(spec[len("ucr:"):], val_frac=val_frac, seed=seed, limit=limit)
    if spec.startswith("ett:"):
        rest = spec[len("ett:"):]
        parts = rest.split(":")
        csv_path = parts[0]
        target = parts[1] if len(parts) > 1 else "OT"
        return load_ett(csv_path, target=target, lookback=lookback, horizon=horizon,
                        stride=stride, max_windows=max_windows, seed=seed)
    raise ValueError(f"unknown dataset spec {spec!r}")
