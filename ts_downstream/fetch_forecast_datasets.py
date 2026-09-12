from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

ETT_BASE = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small")
ETT = {"ETTh2": "ETT-small/ETTh2.csv", "ETTm2": "ETT-small/ETTm2.csv"}
GLUON = {"saugeen":  ("saugeenday", 0, "saugeen/saugeen.csv"),
         "aus_elec": ("australian_electricity_demand", 2, "aus_elec/aus_elec.csv")}


def write_ot_csv(path: Path, values) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "OT"])
        for i, v in enumerate(values):
            w.writerow([i, repr(float(v))])
    return len(values)


def fetch_ett(out_root: Path) -> None:
    for name, rel in ETT.items():
        dst = out_root / rel
        if dst.exists():
            print(f"  {name}: already at {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ETT_BASE}/{name}.csv"
        print(f"  {name}: downloading {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"  {name}: {sum(1 for _ in open(dst)) - 1} rows -> {dst}")


def fetch_gluon(out_root: Path) -> None:
    from gluonts.dataset.repository import get_dataset
    for name, (ds_name, idx, rel) in GLUON.items():
        dst = out_root / rel
        if dst.exists():
            print(f"  {name}: already at {dst}")
            continue
        print(f"  {name}: gluonts get_dataset({ds_name!r}) series {idx}  (downloads on first use)")
        ds = get_dataset(ds_name)
        series = [r["target"] for r in ds.train]
        if idx >= len(series):
            raise SystemExit(f"{ds_name} has {len(series)} series; index {idx} out of range")
        target = series[idx]
        n = write_ot_csv(dst, target)
        print(f"  {name}: {n} rows -> {dst}")


def verify(out_root: Path, ref_root: Path) -> int:
    import numpy as np
    bad = 0
    for rel in list(ETT.values()) + [t[-1] for t in GLUON.values()]:
        a, b = out_root / rel, ref_root / rel
        if not b.exists():
            print(f"  {rel}: no reference, skipped"); continue
        va = [float(r[-1]) for r in list(csv.reader(open(a)))[1:]]
        vb = [float(r[-1]) for r in list(csv.reader(open(b)))[1:]]
        if len(va) != len(vb):
            print(f"  {rel}: LENGTH {len(va)} vs {len(vb)}"); bad += 1; continue
        same = np.array_equal(np.float32(va), np.float32(vb))
        print(f"  {rel}: {len(va)} rows, float32-identical: {same}")
        bad += 0 if same else 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="TSF root; files land at <root>/<dataset>/")
    ap.add_argument("--verify-against", default=None,
                    help="an existing TSF root to compare against instead of downloading")
    args = ap.parse_args()
    out = Path(args.out_dir)

    if args.verify_against:
        print(f"verifying {out} against {args.verify_against}")
        sys.exit(1 if verify(out, Path(args.verify_against)) else 0)

    print(f"writing into {out}")
    fetch_ett(out)
    fetch_gluon(out)
    print("done. Point the readouts at it with --dataset ett:<root>/<name>/<name>.csv.")


if __name__ == "__main__":
    main()
