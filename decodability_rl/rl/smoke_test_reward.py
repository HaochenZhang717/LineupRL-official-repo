from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_items(path: Path, n: int) -> list[tuple[str, dict]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            msg = json.loads(json.loads(line)["message"])
            payload = json.loads(eval(msg[2]["content"])[0][0])
            out.append((msg[1]["content"][0]["image"], payload))
            if len(out) == n:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument(
        "--dataset", default=str(REPO / "out_10k_xdomain" / "rl_decod" / "val_messages.jsonl")
    )
    ap.add_argument(
        "--fragments", default=str(REPO / "out_10k_xdomain" / "fragments.jsonl"),
        help="source of ds_caption for the same ids",
    )
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    items = load_items(Path(args.dataset), args.n)
    ds_caption = {}
    with open(args.fragments, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ds_caption[r["id"]] = r.get("ds_caption", "")

    generic = (
        "This image is a line plot depicting a time series dataset. The x-axis represents "
        "time and the y-axis represents the value. The series shows some fluctuations over "
        "time, with periods of increase and decrease."
    )

    conditions = {
        "ds_caption": lambda p: ds_caption.get(p["id"], ""),
        "generic": lambda p: generic,
        "empty": lambda p: "",
    }

    print(f"{args.n} items from {args.dataset}\n")
    results = {}
    for name, fn in conditions.items():
        payload = {
            "prompts": [[fn(p), [[json.dumps(p), ""]]] for _, p in items],
            "query": [""] * len(items),
            "labels": [""] * len(items),
        }
        r = requests.post(args.url, json=payload, timeout=1800)
        r.raise_for_status()
        rewards = r.json()["rewards"]
        results[name] = rewards
        print(
            f"{name:12s} mean={statistics.mean(rewards):.4f} "
            f"min={min(rewards):.4f} max={max(rewards):.4f}"
        )

    print()
    gap = statistics.mean(results["ds_caption"]) - statistics.mean(results["empty"])
    print(f"ds_caption - empty = {gap:+.4f}   (this is the signal RL can optimise)")
    print(
        f"generic    - empty = "
        f"{statistics.mean(results['generic']) - statistics.mean(results['empty']):+.4f}"
        "   (a template-collapsed caption should be close to 0 here)"
    )
    ok = gap > 0.15
    print("\nVERDICT:", "looks usable" if ok else "TOO WEAK -- do not start training")


if __name__ == "__main__":
    main()
