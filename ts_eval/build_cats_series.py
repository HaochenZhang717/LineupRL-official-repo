from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ts_eval.benchmarks.cats_bench import _SERIES_LINE_RE, _parse_series_from_prompt

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "bench_data" / "cats_bench" / "extracted" / "QA" / "tasks.json"
OUT = REPO / "bench_data" / "cats_bench" / "series.jsonl"
MIN_POINTS = 8


def domain_of(ts_name: str) -> str:
    parts = ts_name.split("_")
    while parts and (parts[-1].isdigit() or parts[-1] in ("test", "train", "val")):
        parts.pop()
    return "_".join(parts) or "unknown"


def extract(pool_path: Path, min_points: int = MIN_POINTS) -> list[dict]:
    rows = json.loads(pool_path.read_text())
    series: dict[str, list[float]] = {}
    for r in rows:
        if r["task_type"].startswith("ts_comparison"):
            continue
        name = r["ts_name"]
        if name in series:
            continue
        try:
            s = _parse_series_from_prompt(r)
        except Exception:
            s = None
        if not s:
            m = _SERIES_LINE_RE.search(r.get("prompt_no_image") or r.get("prompt") or "")
            s = [float(x) for x in m.group(1).split(",") if x.strip()] if m else None
        if not s or isinstance(s[0], list) or len(s) < min_points:
            continue
        series[name] = s
    return [{"series_uid": k, "dataset": domain_of(k), "series": v}
            for k, v in sorted(series.items())]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-points", type=int, default=MIN_POINTS)
    args = ap.parse_args()

    rows = extract(Path(args.pool), args.min_points)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    lens = sorted(len(r["series"]) for r in rows)
    groups = Counter(len(r["series"]) for r in rows)
    orphan = sum(c for c in groups.values() if c < 4)
    print(f"wrote {len(rows)} series to {out}")
    print(f"  length min/median/max: {lens[0]}/{lens[len(lens) // 2]}/{lens[-1]}")
    print(f"  domains: {len(Counter(r['dataset'] for r in rows))}")
    print(f"  series in a length-group smaller than 4: {orphan}")


if __name__ == "__main__":
    main()
