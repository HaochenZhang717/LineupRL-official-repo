from __future__ import annotations

from typing import Sequence

from ts_bedtime.data import CLASSED, SeriesRow

STRATEGIES = ("sbert", "euclid")
SBERT_MODEL = "all-MiniLM-L6-v2"
EUCLID_RESAMPLE_LEN = 128


def _annotation_pool(rows: Sequence[SeriesRow]) -> list[tuple[str, str, str | None]]:
    seen = set()
    pool = []
    for row in rows:
        for annotation in row.annotations:
            key = (annotation, row.series_uid)
            if key in seen:
                continue
            seen.add(key)
            pool.append((annotation, row.series_uid, row.cls))
    return sorted(pool)


def _eligible(cand_uid: str, cand_cls: str | None, row: SeriesRow, constrained: bool) -> bool:
    if cand_uid == row.series_uid:
        return False
    if constrained and cand_cls is not None and row.cls is not None and cand_cls == row.cls:
        return False
    return True


def sbert_negatives(rows: Sequence[SeriesRow], top_n: int = 3,
                    model_name: str = SBERT_MODEL, device: str | None = None,
                    batch_size: int = 256) -> dict[tuple[str, str], list[str]]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    pool = _annotation_pool(rows)
    if not pool:
        return {}
    texts = [a for a, _, _ in pool]
    uids = [u for _, u, _ in pool]
    classes = [c for _, _, c in pool]

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    sim = emb @ emb.T

    first_slot: dict[str, int] = {}
    for i, text in enumerate(texts):
        first_slot.setdefault(text, i)

    constrained = rows[0].dataset in CLASSED
    uid_arr = np.asarray(uids)
    cls_arr = np.asarray([c if c is not None else "" for c in classes])

    out: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        mask = uid_arr != row.series_uid
        if constrained and row.cls is not None:
            mask &= cls_arr != row.cls
        cand_idx = np.flatnonzero(mask)
        if cand_idx.size == 0:
            raise ValueError(f"no eligible distractor for {row.series_uid}; pool too small")
        order = cand_idx[np.argsort(sim[:, cand_idx][[first_slot[a] for a in row.annotations]],
                                    axis=1, kind="stable")]
        for annotation, row_order in zip(row.annotations, order):
            out[(row.series_uid, annotation)] = _pick_distinct(
                [texts[j] for j in row_order], top_n, exclude=row.annotations)
    return out


def _pick_distinct(ordered: Sequence[str], top_n: int, exclude: Sequence[str] = ()) -> list[str]:
    blocked = set(exclude)
    out: list[str] = []
    for text in ordered:
        if text in blocked or text in out:
            continue
        out.append(text)
        if len(out) == top_n:
            return out
    raise ValueError(f"only {len(out)} distinct distractors available, need {top_n}")


def _resampled_znorm(values: Sequence[float], length: int):
    import numpy as np

    y = np.asarray(values, dtype=float)
    std = y.std()
    y = (y - y.mean()) / (std if std > 0 else 1.0)
    if y.size == length:
        return y
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, y.size), y)


def euclid_negatives(rows: Sequence[SeriesRow], top_n: int = 3,
                     length: int = EUCLID_RESAMPLE_LEN) -> dict[tuple[str, str], list[str]]:
    import numpy as np

    if not rows:
        return {}
    mat = np.stack([_resampled_znorm(r.series, length) for r in rows])
    dist = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1)

    constrained = rows[0].dataset in CLASSED
    out: dict[tuple[str, str], list[str]] = {}
    for i, row in enumerate(rows):
        order = np.argsort(-dist[i], kind="stable")
        far_texts: list[str] = []
        for j in order:
            cand = rows[int(j)]
            if not _eligible(cand.series_uid, cand.cls, row, constrained):
                continue
            far_texts.extend(sorted(cand.annotations))
            if len(set(far_texts) - set(row.annotations)) >= top_n + 2:
                break
        for annotation in row.annotations:
            out[(row.series_uid, annotation)] = _pick_distinct(
                far_texts, top_n, exclude=row.annotations)
    return out


def build_negatives(rows: Sequence[SeriesRow], strategy: str, top_n: int = 3,
                    device: str | None = None) -> dict[tuple[str, str], list[str]]:
    if strategy == "sbert":
        return sbert_negatives(rows, top_n=top_n, device=device)
    if strategy == "euclid":
        return euclid_negatives(rows, top_n=top_n)
    raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
