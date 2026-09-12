from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERIES = REPO / "bench_data" / "cats_bench" / "series.jsonl"
REFS = REPO / "data" / "cats" / "cats_agnostic_refs.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default=str(SERIES))
    ap.add_argument("--refs", default=str(REFS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", action="store_true",
                    help="the references may cover only some of the series; write just "
                         "those (for the human-rewritten CaTS subset). Every reference "
                         "must still match a series.")
    ap.add_argument("--ids", default=None,
                    help="optional file of ids (one per line or jsonl with 'id'): restrict "
                         "the output to these series, e.g. to score the synthetic references "
                         "on exactly the human subset")
    args = ap.parse_args()

    series = [json.loads(l) for l in open(args.series, encoding="utf-8")]
    refs = {}
    for line in open(args.refs, encoding="utf-8"):
        r = json.loads(line)
        refs[r["id"]] = r["rewritten"]

    if args.ids:
        keep = set()
        for line in open(args.ids, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            keep.add(json.loads(line)["id"] if line.startswith("{") else line)
        refs = {k: v for k, v in refs.items() if k in keep}
    ids = {r["series_uid"] for r in series}
    missing = sorted(ids - set(refs))
    extra = sorted(set(refs) - ids)
    if args.subset:
        series = [r for r in series if r["series_uid"] in refs]
        missing = []
    if missing or extra:
        raise SystemExit(
            f"reference/series mismatch: {len(missing)} series without a reference "
            f"(e.g. {missing[:3]}), {len(extra)} references without a series "
            f"(e.g. {extra[:3]})")

    empty = [i for i, t in refs.items() if not t.strip()]
    if empty:
        raise SystemExit(f"{len(empty)} references are empty, e.g. {empty[:3]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in series:
            f.write(json.dumps({
                "series_uid": r["series_uid"],
                "dataset": r["dataset"],
                "cls": None,
                "annotations": [refs[r["series_uid"]]],
                "series": r["series"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out}  ({len(series)} series, 1 reference each)")


if __name__ == "__main__":
    main()
