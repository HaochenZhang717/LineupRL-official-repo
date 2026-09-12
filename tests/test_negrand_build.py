from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from decodability_rl.rl.build_decodability_dataset import longest_common_run
from decodability_rl.test_reward_design.hard_negatives import summary_features

L = 24


def _make_fragments(root: Path) -> dict[int, list[float]]:
    rng = random.Random(7)
    src = [round(rng.uniform(-5, 5), 2) for _ in range(L + 2)]
    series = {1: src[:L], 2: src[2 : L + 2]}
    for i in range(3, 13):
        series[i] = [round(rng.uniform(-5, 5), 2) for _ in range(L)]
    assert longest_common_run(series[1], series[2]) == L - 2

    (root / "render").mkdir(parents=True)
    with open(root / "fragments.jsonl", "w", encoding="utf-8") as fh:
        for i, s in series.items():
            img = f"render/{i:06d}.png"
            (root / img).touch()
            fh.write(json.dumps({"id": i, "image_path": img, "series": s, "len": L}) + "\n")
    return series


def _build(root: Path, out: str, *extra: str) -> Path:
    out_dir = root / out
    cmd = [
        sys.executable, "-m", "decodability_rl.rl.build_decodability_dataset",
        "--input", str(root / "fragments.jsonl"),
        "--image-root", str(root),
        "--out-dir", str(out_dir),
        "--val-size", "6",
        *extra,
    ]
    subprocess.run(cmd, cwd=REPO, check=True, capture_output=True)
    return out_dir


def _payloads(out_dir: Path) -> dict[str, list[dict]]:
    parts = {}
    for name in ("train_messages.jsonl", "val_messages.jsonl"):
        rows = []
        for line in (out_dir / name).read_text(encoding="utf-8").splitlines():
            msg = json.loads(json.loads(line)["message"])
            rows.append(json.loads(eval(msg[2]["content"])[0][0]))
        parts[name.split("_")[0]] = rows
    return parts


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("negrand")
    series = _make_fragments(root)
    outs = {
        "nearest": _payloads(_build(root, "nearest")),
        "random": _payloads(_build(root, "rand_a", "--negatives-order", "random")),
        "random_same": _payloads(_build(root, "rand_b", "--negatives-order", "random")),
        "random_other_seed": _payloads(
            _build(root, "rand_c", "--negatives-order", "random", "--neg-seed", "999")
        ),
    }
    return series, outs


def _all_items(parts: dict) -> list[dict]:
    return parts["val"] + parts["train"]


def test_random_order_still_filters_same_source_windows(built):
    series, outs = built
    for name in ("random", "random_other_seed", "nearest"):
        for p in _all_items(outs[name]):
            if p["true"] == series[1]:
                partner = series[2]
            elif p["true"] == series[2]:
                partner = series[1]
            else:
                continue
            pool = p["distractors"] + p.get("distractor_pool", [])
            assert partner not in pool, f"{name}: overlapping window entered the pool"


def test_nearest_path_is_the_original_argsort(built):
    series, outs = built
    ids = sorted(series)
    feats = np.stack([summary_features(series[i]) for i in ids])
    mu, sd = feats.mean(0), feats.std(0)
    sd[sd == 0] = 1.0
    z = (feats - mu) / sd
    by_true = {tuple(p["true"]): p for p in _all_items(outs["nearest"])}
    for pos, i in enumerate(ids):
        p = by_true[tuple(series[i])]
        order = np.argsort(np.linalg.norm(z - z[pos], axis=1))
        expected = []
        for j in order:
            cand = ids[int(j)]
            if cand == i:
                continue
            if longest_common_run(series[i], series[cand]) > int(0.2 * L):
                continue
            expected.append(series[cand])
            if len(expected) == 3:
                break
        assert p["distractors"] == expected, f"id {i}: nearest order changed"


def test_split_is_identical_across_orders(built):
    _, outs = built
    for part in ("val", "train"):
        near, rand = outs["nearest"][part], outs["random"][part]
        assert [p["id"] for p in near] == [p["id"] for p in rand]
        assert [p["true"] for p in near] == [p["true"] for p in rand]
        assert [p["decimals"] for p in near] == [p["decimals"] for p in rand]


def test_random_actually_differs_from_nearest(built):
    _, outs = built
    near = [p["distractors"] for p in _all_items(outs["nearest"])]
    rand = [p["distractors"] for p in _all_items(outs["random"])]
    assert near != rand


def test_random_is_deterministic_per_seed(built):
    _, outs = built
    a = [p["distractors"] for p in _all_items(outs["random"])]
    b = [p["distractors"] for p in _all_items(outs["random_same"])]
    c = [p["distractors"] for p in _all_items(outs["random_other_seed"])]
    assert a == b, "same --neg-seed must rebuild byte-identically"
    assert a != c, "a different --neg-seed must pick different distractors"


def test_distractors_stay_same_length_as_true(built):
    _, outs = built
    for p in _all_items(outs["random"]):
        assert all(len(d) == len(p["true"]) for d in p["distractors"])
