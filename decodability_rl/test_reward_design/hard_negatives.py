from __future__ import annotations

import numpy as np


def summary_features(series) -> np.ndarray:
    y = np.asarray(series, dtype=float)
    return np.array([y.mean(), y.std(), y.min(), y.max()])


def build_pool(rows: list[dict], length: int) -> tuple[list[dict], np.ndarray]:
    pool = [r for r in rows if r["len"] == length]
    feats = np.stack([summary_features(r["series"]) for r in pool])
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd == 0] = 1.0
    return pool, (feats - mu) / sd


def nearest_negatives(
    target: dict,
    pool: list[dict],
    zfeats: np.ndarray,
    n: int = 3,
    exclude_ids: set | None = None,
) -> list[dict]:
    exclude = set(exclude_ids or ())
    exclude.add(target["id"])

    feats = np.stack([summary_features(r["series"]) for r in pool])
    m, s = feats.mean(0), feats.std(0)
    s[s == 0] = 1.0
    zt = (summary_features(target["series"]) - m) / s

    d = np.linalg.norm(zfeats - zt, axis=1)
    order = np.argsort(d)
    out = []
    for i in order:
        cand = pool[int(i)]
        if cand["id"] in exclude:
            continue
        out.append(cand)
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} negatives available for id={target['id']}")
    return out
