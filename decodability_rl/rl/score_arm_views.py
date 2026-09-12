from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "decodability_rl" / "mask_ab"


def score_arm(url: str, rows: list[dict], batch: int, scale: float, timeout: float):
    rewards: list[float] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        prompts = [[r["caption"], [[json.dumps(r["payload"]), ""]]] for r in chunk]
        resp = requests.post(
            url, json={"prompts": prompts, "query": [], "labels": []}, timeout=timeout
        )
        resp.raise_for_status()
        rewards += resp.json()["rewards"]
        print(f"    {len(rewards)}/{len(rows)}", flush=True)
    arr = np.asarray(rewards, dtype=float)
    return arr, arr / scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--view", required=True, choices=["full", "masked", "blend"],
                    help="label only -- it must match how the server was started")
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path/to/captions.jsonl, repeatable")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for spec in args.arm:
        name, _, path = spec.partition("=")
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        print(f"=== {name} [{args.view}] {len(rows)} captions from {path}", flush=True)
        arr, acc = score_arm(args.url, rows, args.batch, args.reward_scale, args.timeout)

        with (out_dir / f"items_{name}_{args.view}.jsonl").open("w") as fh:
            for r, a in zip(rows, acc):
                fh.write(json.dumps({"id": r["id"], "acc": float(a)}) + "\n")

        summary.setdefault(name, {})[args.view] = {
            "n": int(len(arr)),
            "accuracy": float(acc.mean()),
            "reward_mean": float(arr.mean()),
            "reward_sd": float(arr.std()),
            "frac_perfect": float((acc >= 0.999).mean()),
            "frac_zero": float((acc <= 0.001).mean()),
            "mean_caption_chars": float(np.mean([len(r["caption"]) for r in rows])),
        }
        s = summary[name][args.view]
        print(f"  {name:12} {args.view:6} acc={s['accuracy']:.4f} "
              f"reward={s['reward_mean']:.3f} perfect={s['frac_perfect']:.3f} "
              f"zero={s['frac_zero']:.3f}", flush=True)
        summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
