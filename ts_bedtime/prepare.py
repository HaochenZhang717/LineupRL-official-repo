from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

from ts_bedtime.data import DATASETS, DEFAULT_ROOT, SeriesRow, load_all, synthetic_rows
from ts_bedtime.negatives import STRATEGIES, build_negatives
from ts_bedtime.prompts import LETTERS

N_OPTIONS = len(LETTERS)


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", text).strip("-")


def build_items(rows: Sequence[SeriesRow], negatives: dict[tuple[str, str], list[str]],
                seed: int) -> list[dict]:
    items: list[dict] = []
    for row in rows:
        for k, annotation in enumerate(row.annotations):
            distractors = negatives[(row.series_uid, annotation)]
            stem = f"bedtime_{row.dataset}_{_slug(row.idx)}_a{k}"
            common = {
                "dataset": row.dataset,
                "series_uid": row.series_uid,
                "cls": row.cls,
                "subclass": row.subclass,
            }

            items.append({**common, "item_id": f"{stem}_rec_pos", "task_type": "recognition",
                          "description": annotation, "gold": "True"})
            items.append({**common, "item_id": f"{stem}_rec_neg", "task_type": "recognition",
                          "description": distractors[0], "gold": "False"})

            options = [annotation] + list(distractors[:N_OPTIONS - 1])
            random.Random(f"{seed}:{stem}").shuffle(options)
            items.append({**common, "item_id": f"{stem}_diff", "task_type": "differentiation",
                          "options": options, "gold": LETTERS[options.index(annotation)],
                          "description": annotation})
    return items


def _write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="where the four CSVs live")
    ap.add_argument("--out-dir", default=str(DEFAULT_ROOT))
    ap.add_argument("--strategy", default="sbert", choices=STRATEGIES)
    ap.add_argument("--per-dataset", type=int, default=500,
                    help="unique series sampled per dataset; 0 = all")
    ap.add_argument("--seed", type=int, default=2020,
                    help="pinned to 2020, the seed shared with the other evaluation pipelines")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--device", default=None, help="sentence-transformers device, e.g. cuda:0")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="skip the CSVs and use N generated series per fake dataset (tests)")
    args = ap.parse_args()

    names = [n for n in args.datasets.split(",") if n]
    if args.synthetic:
        by_dataset = {n: synthetic_rows(n, args.synthetic, seed=args.seed,
                                        classed=n in ("sushi", "taxosynth"))
                      for n in names}
    else:
        by_dataset = load_all(args.root, args.per_dataset or None, args.seed, names)

    out_dir = Path(args.out_dir)
    all_items: list[dict] = []
    all_rows: list[SeriesRow] = []
    for name, rows in by_dataset.items():
        if not rows:
            print(f"[bedtime.{name}] no rows, skipping", flush=True)
            continue
        negatives = build_negatives(rows, args.strategy, top_n=N_OPTIONS - 1, device=args.device)
        items = build_items(rows, negatives, args.seed)
        all_items.extend(items)
        all_rows.extend(rows)
        n_rec = sum(1 for i in items if i["task_type"] == "recognition")
        print(f"[bedtime.{name}] {len(items)} items ({n_rec} recognition / "
              f"{len(items) - n_rec} differentiation)", flush=True)

    n_series = _write_jsonl(out_dir / "series.jsonl", (
        {"series_uid": r.series_uid, "dataset": r.dataset, "idx": r.idx,
         "cls": r.cls, "subclass": r.subclass, "annotations": r.annotations,
         "series": r.series}
        for r in all_rows))
    items_path = out_dir / f"items_{args.strategy}.jsonl"
    n_items = _write_jsonl(items_path, all_items)
    print(f"wrote {n_series} series to {out_dir / 'series.jsonl'}", flush=True)
    print(f"wrote {n_items} items to {items_path}", flush=True)


if __name__ == "__main__":
    main()
