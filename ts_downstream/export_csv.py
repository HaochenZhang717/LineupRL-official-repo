from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EP = REPO / "results" / "eval_protocol"

SOURCES = [
    ("forecasting", EP / "pipeline3_forecast_notrunc"),
    ("forecasting", EP / "pipeline3_forecast_notrunc_seed2021"),
    ("forecasting", EP / "pipeline3_forecast_notrunc_seed2022"),
    ("forecasting", EP / "pipeline3_forecast_notrunc_seed2023"),
    ("forecasting", EP / "pipeline3_forecast_notrunc_seed2024"),
    ("reconstruction", EP / "ts_recon"),
    ("reconstruction", EP / "ts_recon_seed2021"),
    ("reconstruction", EP / "ts_recon_seed2022"),
    ("reconstruction", EP / "ts_recon_seed2023"),
    ("reconstruction", EP / "ts_recon_seed2024"),
]

FIELDS = ["study", "dataset", "condition", "arm", "control", "seed", "train_seed",
          "mse", "mae", "naive_skill", "best_epoch",
          "n_captions", "median_caption_tokens", "max_caption_tokens", "n_truncated",
          "n_train", "n_val", "n_test", "seconds", "source"]


def rows_from(study: str, root: Path):
    if not root.exists():
        return
    for mp in sorted(root.glob("*/*/metrics.json")):
        m = json.loads(mp.read_text())
        a = m.get("args", {})
        ct = m.get("caption_tokens") or {}
        t = m.get("test", {})
        cond = mp.parent.name
        if cond == "null":
            arm, control = None, "null"
        elif cond.endswith("_shuffled"):
            arm, control = cond[: -len("_shuffled")], "shuffled"
        elif cond.startswith("tmpl_"):
            arm, control = None, cond[len("tmpl_"):]
        else:
            arm, control = cond, None
        if ct.get("n_truncated"):
            raise SystemExit(
                f"{mp}: {ct['n_truncated']} captions were truncated. This export is "
                f"untruncated runs only -- fix the run rather than the export.")
        yield {
            "study": study,
            "dataset": mp.parent.parent.name,
            "condition": cond,
            "arm": arm,
            "control": control,
            "seed": a.get("seed"),
            "train_seed": a.get("train_seed") if a.get("train_seed") is not None
            else a.get("seed"),
            "n_captions": ct.get("n"),
            "median_caption_tokens": ct.get("median_tokens"),
            "max_caption_tokens": ct.get("max_tokens"),
            "n_truncated": ct.get("n_truncated"),
            "mse": t.get("mse"),
            "mae": t.get("mae"),
            "naive_skill": t.get("naive_skill"),
            "best_epoch": m.get("best_epoch"),
            "n_train": m.get("n_train"), "n_val": m.get("n_val"), "n_test": m.get("n_test"),
            "seconds": m.get("seconds"),
            "source": str(mp.relative_to(REPO)),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EP / "readout_results.csv"))
    args = ap.parse_args()

    rows = [r for study, root in SOURCES for r in rows_from(study, root)]
    rows.sort(key=lambda r: (r["study"], r["dataset"], r["condition"], r["train_seed"] or 0))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}  ({len(rows)} rows)")
    by: dict[tuple, int] = {}
    for r in rows:
        k = (r["study"], r["train_seed"])
        by[k] = by.get(k, 0) + 1
    per_study: dict[str, int] = {}
    for (study, _), n in by.items():
        per_study[study] = max(per_study.get(study, 0), n)
    for k in sorted(by, key=str):
        print(f"  study={k[0]:15s} train_seed={k[1]}  n={by[k]}/{per_study[k[0]]}")


if __name__ == "__main__":
    main()
