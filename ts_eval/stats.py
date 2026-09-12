from __future__ import annotations

import numpy as np


def boot_mean(x, n_boot: int = 2000, seed: int = 0, ci: float = 0.95):
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def boot_paired_delta(a, b, n_boot: int = 2000, seed: int = 0, ci: float = 0.95):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in length: {a.shape} vs {b.shape}")
    return boot_mean(a - b, n_boot=n_boot, seed=seed, ci=ci)


def cohen_kappa(y1, y2) -> float:
    y1 = list(y1)
    y2 = list(y2)
    if len(y1) != len(y2):
        raise ValueError("label sequences differ in length")
    n = len(y1)
    if n == 0:
        return float("nan")
    labels = sorted(set(y1) | set(y2), key=str)
    k = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=float)
    for u, v in zip(y1, y2):
        m[k[u], k[v]] += 1
    po = np.trace(m) / n
    pe = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / (n * n)
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))
