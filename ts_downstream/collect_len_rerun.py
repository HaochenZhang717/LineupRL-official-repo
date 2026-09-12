from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "results" / "eval_protocol" / "pipeline3_fusion" / "gemma"

PUBLISHED_256 = {
    ("ETTh2", "base"): 0.3106, ("ETTh2", "decod550"): 0.1766,
    ("ETTh2", "sft"): 0.3322, ("ETTh2", "teacher72b"): 0.3344,
    ("ETTm2", "base"): 0.1021, ("ETTm2", "decod550"): 0.0494,
    ("ETTm2", "sft"): 0.1052, ("ETTm2", "teacher72b"): 0.1376,
    ("saugeen", "base"): 0.9721, ("saugeen", "decod550"): 0.8224,
    ("saugeen", "sft"): 0.9765, ("saugeen", "teacher72b"): 0.9698,
    ("aus_elec", "base"): 0.5790, ("aus_elec", "decod550"): 0.2062,
    ("aus_elec", "sft"): 0.4082, ("aus_elec", "teacher72b"): 0.4499,
}
ARMS = ["base", "decod550", "sft", "teacher72b"]
ALL_DATASETS = ["ETTh2", "ETTm2", "saugeen", "aus_elec"]

PUBLISHED_EPOCH = {
    ("ETTh2", "base"): 2, ("ETTh2", "decod550"): 14,
    ("ETTh2", "sft"): 2, ("ETTh2", "teacher72b"): 1,
    ("ETTm2", "base"): 9, ("ETTm2", "decod550"): 15,
    ("ETTm2", "sft"): 22, ("ETTm2", "teacher72b"): 3,
    ("saugeen", "base"): 7, ("saugeen", "decod550"): 1,
    ("saugeen", "sft"): 1, ("saugeen", "teacher72b"): 5,
    ("aus_elec", "base"): 6, ("aus_elec", "decod550"): 21,
    ("aus_elec", "sft"): 8, ("aus_elec", "teacher72b"): 23,
}


def load() -> dict:
    out = defaultdict(lambda: defaultdict(list))
    for p in sorted(ROOT.glob("*/*/caption_len*/seed*/metrics.json")):
        m = json.loads(p.read_text())
        ds, arm = p.parents[3].name, p.parents[2].name
        length = int(p.parents[1].name.replace("caption_len", ""))
        out[(ds, arm, length)]["mse"].append(m["test"]["mse"])
        out[(ds, arm, length)]["mae"].append(m["test"]["mae"])
        out[(ds, arm, length)]["seed"].append(int(p.parent.name.replace("seed", "")))
    return out


def _agg(vals: list[float]) -> tuple[float, float]:
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lens", nargs=2, type=int, default=[256, 1400])
    args = ap.parse_args()
    old_len, new_len = args.lens

    data = load()
    if not data:
        print(f"No runs under {ROOT}")
        return 1
    datasets = sorted({ds for ds, _, _ in data})

    print("=== reproduction anchor (len=%d, seed 2020 vs published) ===" % old_len)
    hard, lottery = 0, 0
    for p in sorted(ROOT.glob(f"*/*/caption_len{old_len}/seed2020/metrics.json")):
        ds, arm = p.parents[3].name, p.parents[2].name
        m = json.loads(p.read_text())
        got, got_ep = m["test"]["mse"], m.get("best_epoch")
        want = PUBLISHED_256.get((ds, arm))
        if want is None:
            continue
        pub = PUBLISHED_EPOCH.get((ds, arm))
        if abs(got - want) < 1e-3:
            verdict = "MATCH"
        elif pub is not None and got_ep != pub:
            verdict = f"checkpoint-lottery (ep {pub}->{got_ep})"
            lottery += 1
        else:
            verdict = "DIVERGED at same epoch"
            hard += 1
        print(f"  {ds:9} {arm:11} got {got:.4f}  published {want:.4f}  {verdict}")
    print(f"  -> {'all reproduce' if not (hard or lottery) else ''}"
          f"{f'{hard} hard mismatch(es) ' if hard else ''}"
          f"{f'{lottery} checkpoint-lottery difference(s)' if lottery else ''}\n")
    mismatches = hard

    ranking_changed = []
    for ds in datasets:
        print(f"--- {ds}")
        print(f"    {'arm':11} {'len':>5} {'n':>2} {'MSE':>9} {'+/-':>7} {'MAE':>8}")
        per_len = {}
        for length in (old_len, new_len):
            row = {}
            for arm in ARMS:
                d = data.get((ds, arm, length))
                if not d:
                    continue
                mu, sd = _agg(d["mse"])
                mae, _ = _agg(d["mae"])
                row[arm] = (mu, sd, mae, len(d["mse"]))
                print(f"    {arm:11} {length:5d} {len(d['mse']):2d} {mu:9.4f} {sd:7.4f} {mae:8.4f}")
            per_len[length] = row
            print()

        old_row, new_row = per_len.get(old_len, {}), per_len.get(new_len, {})
        shared = [a for a in ARMS if a in old_row and a in new_row]
        if len(shared) < 2:
            print(f"    (ranking pending -- only {len(shared)} arm(s) have both lengths)\n")
            continue
        if len(shared) < len(ARMS):
            print(f"    NB: ranking over {len(shared)}/{len(ARMS)} arms so far "
                  f"({', '.join(shared)}); the rest are still running.")
        old_rank = sorted(shared, key=lambda a: old_row[a][0])
        new_rank = sorted(shared, key=lambda a: new_row[a][0])
        same = old_rank == new_rank
        print(f"    ranking @{old_len}: {' < '.join(old_rank)}")
        print(f"    ranking @{new_len}: {' < '.join(new_rank)}   "
              f"{'(unchanged)' if same else '<-- CHANGED'}")
        if not same:
            ranking_changed.append(ds)
        if new_rank and new_rank[0] != "decod550" and "decod550" in shared:
            print(f"    NOTE: at {new_len}, best arm is {new_rank[0]}, not decod550")
        print()

    n_complete = sum(1 for ds in ALL_DATASETS
                     if all((ds, a, L) in data for a in ARMS for L in (old_len, new_len)))
    missing = [ds for ds in ALL_DATASETS if ds not in datasets]
    print("=== Gate 0 verdict ===")
    print(f"  ({n_complete}/{len(ALL_DATASETS)} datasets have all {len(ARMS)} arms at both "
          f"lengths"
          + (f"; not started: {', '.join(missing)}" if missing else "")
          + ")")
    if n_complete < len(ALL_DATASETS):
        print("  VERDICT IS PROVISIONAL.")
    if mismatches:
        print(f"  BLOCKED: {mismatches} cell(s) fail to reproduce the published 256-token "
              f"number. Fix that before reading anything else.")
        return 1
    if ranking_changed:
        print(f"  Arm ranking CHANGED on {len(ranking_changed)} dataset(s): "
              f"{', '.join(ranking_changed)}.")
        print("  -> the original forecasting conclusion must be corrected: it was "
              "measured with sft/teacher72b truncated at 256.")
    else:
        print("  Arm ranking unchanged once captions are no longer truncated -- the "
              "published conclusion survives, and now with an explicit control for the "
              "length confound rather than by luck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
