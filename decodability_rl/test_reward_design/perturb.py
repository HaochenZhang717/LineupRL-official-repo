from __future__ import annotations

import numpy as np

DISTRACTOR_TYPES = ("shuffle", "reverse", "segment_swap")


def _shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return y[rng.permutation(y.size)]


def _reverse(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return y[::-1].copy()


def _segment_swap(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = y.size
    if n < 8:
        raise ValueError("segment_swap needs at least 8 points")
    edges = np.linspace(0, n, 5, dtype=int)
    blocks = [y[a:b].copy() for a, b in zip(edges[:-1], edges[1:])]
    i, j = rng.choice(4, size=2, replace=False)
    blocks[i], blocks[j] = blocks[j], blocks[i]
    return np.concatenate(blocks)


_MAKERS = {
    "shuffle": _shuffle,
    "reverse": _reverse,
    "segment_swap": _segment_swap,
}


def make_distractor(
    y: np.ndarray,
    kind: str,
    rng: np.random.Generator,
    *,
    max_tries: int = 20,
    min_frac_moved: float = 0.25,
) -> np.ndarray:
    maker = _MAKERS[kind]
    for _ in range(max_tries):
        z = maker(np.asarray(y, dtype=float), rng)
        moved = float(np.mean(~np.isclose(z, y, rtol=0, atol=1e-9)))
        if moved >= min_frac_moved:
            return z
        if kind == "reverse":
            break
    raise ValueError(f"{kind} is degenerate on this series (moved<{min_frac_moved})")


def make_all_distractors(
    y: np.ndarray, rng: np.random.Generator, kinds=DISTRACTOR_TYPES
) -> dict[str, np.ndarray]:
    out = {}
    for kind in kinds:
        z = make_all_distractors_guard(y, kind, rng, out)
        out[kind] = z
    return out


def make_all_distractors_guard(y, kind, rng, sofar) -> np.ndarray:
    for _ in range(20):
        z = make_distractor(y, kind, rng)
        if all(not np.allclose(z, prev) for prev in sofar.values()):
            return z
        if kind == "reverse":
            break
    raise ValueError(f"{kind} collides with another option on this series")
