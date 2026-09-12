from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path

GENERIC = ("The time series shows values that change over time. There are increases and "
           "decreases at various points, with an overall pattern across the range.")
EMPTY = ""


def load_items(path: Path, n: int) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msg = row["message"]
            msg = json.loads(msg) if isinstance(msg, str) else msg
            payload = json.loads(ast.literal_eval(msg[-1]["content"])[0][0])
            items.append(payload)
            if len(items) >= n:
                break
    return items


def describe(payload: dict) -> str:
    v = payload["true"]
    lo, hi = min(v), max(v)
    i_lo, i_hi = v.index(lo), v.index(hi)
    n = len(v)

    def where(i):
        f = i / max(n - 1, 1)
        return ("at the very start" if f < 0.1 else "early on" if f < 0.35 else
                "around the middle" if f < 0.65 else "in the later part" if f < 0.9 else
                "at the very end")

    trend = ("rises overall" if v[-1] > v[0] * 1.02 else
             "falls overall" if v[-1] < v[0] * 0.98 else "ends near where it started")
    return (f"The series {trend}, starting at {v[0]:.2f} and ending at {v[-1]:.2f}. "
            f"Its lowest point, {lo:.2f}, comes {where(i_lo)}, and its highest, {hi:.2f}, "
            f"comes {where(i_hi)}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-pretrain", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    from decodability_rl.rl.reward_server_judge import JudgeRewardModel, run_selftest

    model = JudgeRewardModel(args)
    print("\n=== rubric ordering self-test ===")
    run_selftest(model)

    items = load_items(Path(args.dataset), args.n)
    print(f"\n=== {len(items)} real val items, three caption conditions ===")
    conditions = {
        "described": [describe(p) for p in items],
        "generic": [GENERIC] * len(items),
        "empty": [EMPTY] * len(items),
    }
    print("\nexample described caption:\n  " + conditions["described"][0])

    out = {}
    for name, caps in conditions.items():
        prompts = [[c, [[json.dumps(p), ""]]] for c, p in zip(caps, items)]
        rewards, _ = model.get_reward(prompts)
        out[name] = rewards
        print(f"\n{name:10s} mean={statistics.mean(rewards):.4f} "
              f"min={min(rewards):.4f} max={max(rewards):.4f} "
              f"sd={statistics.pstdev(rewards):.4f}")

    gap = statistics.mean(out["described"]) - statistics.mean(out["empty"])
    sd = statistics.pstdev(out["described"])
    print(f"\ndescribed - empty = {gap:+.4f}   (the signal RL can optimise)")
    print(f"generic   - empty = "
          f"{statistics.mean(out['generic']) - statistics.mean(out['empty']):+.4f}")
    print(f"sd within `described` = {sd:.4f}   (the spread RLOO actually trains on)")

    verdict = []
    if gap <= 0.15:
        verdict.append("gap too small -- the judge is barely reading the caption")
    if sd < 0.02:
        verdict.append("no spread within a condition -- RLOO would see zero advantage")
    print("\nVERDICT:", "looks usable" if not verdict else "NOT USABLE: " + "; ".join(verdict))


if __name__ == "__main__":
    main()
