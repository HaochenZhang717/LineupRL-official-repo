from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from decodability_rl.test_reward_design import prompts as P
from decodability_rl.test_reward_design.hard_negatives import summary_features

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT

CAP_INSTRUCTION = TS_CAP_PROMPT
DEFAULT_SYSTEM = "You are an analyst who describes and interprets time series."


def longest_common_run(a: list[float], b: list[float], ndigits: int = 4) -> int:
    ar = [round(x, ndigits) for x in a]
    br = [round(x, ndigits) for x in b]
    prev = [0] * (len(br) + 1)
    best = 0
    for i in range(len(ar)):
        cur = [0] * (len(br) + 1)
        for j in range(len(br)):
            if ar[i] == br[j]:
                cur[j + 1] = prev[j] + 1
                best = max(best, cur[j + 1])
        prev = cur
    return best


def build_message(system_prompt: str, image_abs: str, payload: str) -> list:
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_abs},
                {"type": "text", "text": CAP_INSTRUCTION},
            ],
        },
        {"role": "answer", "content": repr([(payload, "")])},
    ]


def self_check(line: str, n_distractors: int = 3) -> None:
    msg = json.loads(json.loads(line)["message"])
    assert len(msg) == 3 and msg[2]["role"] == "answer"
    qas = eval(msg[2]["content"])
    assert isinstance(qas, list) and len(qas) == 1
    payload, _ = qas[0]
    d = json.loads(payload)
    assert len(d["distractors"]) == n_distractors
    assert all(len(x) == len(d["true"]) for x in d["distractors"])
    uc = msg[1]["content"]
    assert uc[0]["type"] == "image" and os.path.isfile(uc[0]["image"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default=str(REPO / "out_10k_xdomain" / "fragments.jsonl"))
    ap.add_argument("--image-root", default=str(REPO / "out_10k_xdomain"))
    ap.add_argument("--out-dir", default=str(REPO / "out_10k_xdomain" / "rl_decod"))
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--min-len", type=int, default=24)
    ap.add_argument(
        "--max-overlap-frac",
        type=float,
        default=0.2,
        help="drop a negative sharing a contiguous run longer than this fraction of the "
        "series with the true one (same-source sliding windows)",
    )
    ap.add_argument("--n-distractors", type=int, default=3)
    ap.add_argument(
        "--pool-size",
        type=int,
        default=8,
        help="how many nearest real series to store per training item as a distractor "
        "pool; the reward server resamples --n-distractors of them each step, so the "
        "caption cannot overfit one fixed set of negatives. VAL items always keep a "
        "FIXED set (the first --n-distractors) so the val curve is comparable across "
        "checkpoints.",
    )
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    ap.add_argument(
        "--negatives-order",
        choices=("nearest", "random"),
        default="nearest",
        help="'nearest' scans candidates by feature-space distance (the decod550 "
        "design); 'random' scans them in a random order, so the pool is a uniform "
        "sample of same-length, non-overlapping series instead of the hardest ones. "
        "Both apply the same length bucket and --max-overlap-frac filter, so the "
        "kept-item set -- and therefore the train/val split under --seed -- is "
        "identical between the two orders.",
    )
    ap.add_argument(
        "--neg-seed",
        type=int,
        default=1234,
        help="RNG seed for --negatives-order random. Deliberately separate from "
        "--seed: reusing --seed would leave the split alone (it only feeds the item "
        "shuffle) but ties two unrelated choices together.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = use all fragments")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["len"] >= args.min_len]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} fragments after length filter")

    by_len: dict[int, list[dict]] = {}
    for r in rows:
        by_len.setdefault(r["len"], []).append(r)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    dropped_missing = dropped_negs = 0
    pool_size = max(args.pool_size, args.n_distractors)
    for length, group in sorted(by_len.items()):
        feats = np.stack([summary_features(r["series"]) for r in group])
        mu, sd = feats.mean(0), feats.std(0)
        sd[sd == 0] = 1.0
        z = (feats - mu) / sd
        print(f"  len={length}: {len(group)} fragments", flush=True)

        for i, r in enumerate(group):
            image_abs = os.path.abspath(os.path.join(args.image_root, r["image_path"]))
            if not os.path.isfile(image_abs):
                dropped_missing += 1
                continue

            if args.negatives_order == "random":
                rng = random.Random(f"{args.neg_seed}:{r['id']}")
                order = rng.sample(range(len(group)), len(group))
            else:
                d = np.linalg.norm(z - z[i], axis=1)
                order = np.argsort(d)
            max_run = max(1, int(args.max_overlap_frac * length))
            negs = []
            for j in order:
                j = int(j)
                if j == i:
                    continue
                cand = group[j]
                if longest_common_run(r["series"], cand["series"]) > max_run:
                    continue
                negs.append([float(x) for x in cand["series"]])
                if len(negs) == pool_size:
                    break
            if len(negs) < args.n_distractors:
                dropped_negs += 1
                continue

            items.append(
                {
                    "id": r["id"],
                    "true": [float(x) for x in r["series"]],
                    "pool": negs,
                    "decimals": P.decimals_for(r["series"]),
                    "image_abs": image_abs,
                }
            )

    pool_lens = [len(it["pool"]) for it in items]
    print(
        f"kept {len(items)}  (dropped: missing image {dropped_missing}, "
        f"not enough clean negatives {dropped_negs}); "
        f"pool size mean {np.mean(pool_lens):.1f} min {min(pool_lens)} max {max(pool_lens)}"
    )

    random.Random(args.seed).shuffle(items)
    val_items = items[: args.val_size]
    train_items = items[args.val_size :]

    def to_message(it, with_pool: bool) -> list:
        payload = {
            "id": it["id"],
            "true": it["true"],
            "distractors": it["pool"][: args.n_distractors],
            "decimals": it["decimals"],
        }
        if with_pool and len(it["pool"]) > args.n_distractors:
            payload["distractor_pool"] = it["pool"]
        return build_message(
            args.system_prompt, it["image_abs"], json.dumps(payload, separators=(",", ":"))
        )

    val = [to_message(it, with_pool=False) for it in val_items]
    train = [to_message(it, with_pool=True) for it in train_items]
    for name, part in (("train_messages.jsonl", train), ("val_messages.jsonl", val)):
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as fh:
            for msg in part:
                fh.write(json.dumps({"message": json.dumps(msg)}, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(part)} items)")
        with open(path, encoding="utf-8") as fh:
            self_check(fh.readline(), args.n_distractors)
    print("self-check passed (trainer-side parse reproduced)")


if __name__ == "__main__":
    main()
