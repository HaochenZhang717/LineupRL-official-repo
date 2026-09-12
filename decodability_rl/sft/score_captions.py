from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_payloads(dataset: str) -> dict[int, list]:
    out = {}
    with open(dataset, encoding="utf-8") as fh:
        for line in fh:
            msgs = json.loads(json.loads(line)["message"])
            qa = eval(msgs[2]["content"])
            out[json.loads(qa[0][0])["id"]] = qa
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True, help="jsonl with {id, caption}")
    ap.add_argument("--dataset", required=True, help="the matching RL *_messages.jsonl")
    ap.add_argument("--url", required=True)
    ap.add_argument("--label", default=None, help="name for this caption source")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", default=None, help="jsonl to append the summary line to")
    ap.add_argument("--per-item-out", default=None, help="jsonl for per-caption rewards")
    args = ap.parse_args()

    payloads = load_payloads(args.dataset)
    rows = []
    with open(args.captions, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] in payloads and (r.get("caption") or "").strip():
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"scoring {len(rows)} captions from {args.captions}", flush=True)

    rewards = []
    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        resp = requests.post(args.url, timeout=args.timeout, json={
            "prompts": [[r["caption"], payloads[r["id"]]] for r in chunk],
            "query": [], "labels": [],
        })
        resp.raise_for_status()
        rewards.extend(resp.json()["rewards"])
        print(f"  {len(rewards)}/{len(rows)}  running mean "
              f"{statistics.mean(rewards):.4f}", flush=True)

    rec = {
        "label": args.label or Path(args.captions).stem,
        "captions": args.captions,
        "n": len(rows),
        "mean_reward": round(statistics.mean(rewards), 4),
        "accuracy": round(statistics.mean(rewards) / 2, 4),
        "std_reward": round(statistics.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "mean_caption_chars": round(sum(len(r["caption"]) for r in rows) / len(rows), 1),
        "seconds": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    print("SCORE_RESULT", json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    if args.per_item_out:
        with open(args.per_item_out, "w", encoding="utf-8") as fh:
            for r, rew in zip(rows, rewards):
                fh.write(json.dumps({"id": r["id"], "reward": rew,
                                     "chars": len(r["caption"])}) + "\n")


if __name__ == "__main__":
    main()
