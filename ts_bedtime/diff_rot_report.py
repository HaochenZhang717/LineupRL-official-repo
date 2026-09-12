from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DIR = REPO / "results" / "eval_protocol" / "bedtime_diff_rot"

ARMS = [
    ("L0", "no evidence (floor)"),
    ("base", "untuned Qwen2.5-VL-3B"),
    ("sft", "SFT, distilled from the 72B teacher"),
    ("teacher72b", "the 72B teacher itself"),
    ("rl650", "RL v1, self-generated MCQ reward"),
    ("decod550", "RL v2, decodability reward"),
    ("L2", "raw series values (ceiling)"),
]


def load(arm: str) -> dict | None:
    p = DIR / f"{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def per_item(arm: str) -> dict[str, float]:
    p = DIR / f"{arm}.jsonl"
    if not p.exists():
        return {}
    acc: dict[str, list[float]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            acc.setdefault(r["item_id"], []).append(r["correct"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def sign_test(a: dict, b: dict) -> tuple[int, int, float]:
    keys = [k for k in a if k in b]
    pos = sum(1 for k in keys if a[k] > b[k])
    neg = sum(1 for k in keys if a[k] < b[k])
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    z = (abs(pos - neg) - 1) / math.sqrt(n)
    return pos, neg, math.erfc(z / math.sqrt(2))


def cell(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {arm: load(arm) for arm, _ in ARMS}
    have = [(arm, what) for arm, what in ARMS if res[arm]]
    if not have:
        raise SystemExit(f"no results in {DIR}")
    datasets = sorted({d for arm, _ in have for d in res[arm]["by_dataset"]})

    L: list[str] = []
    A = L.append
    A("# BEDTime differentiation — every question asked at all 4 gold positions")
    A("")
    A("> Part of the picture. `bedtime_cats_all_models.md` holds every model and every\n"
      "> BEDTime/CaTS metric in one table, generated from the same JSON as this file.")
    A("")
    A("**This is a separate measurement from `bedtime_three_metrics.md`.** Same items, same")
    A("reader, same captions — but BEDTime ships each question with its options in one")
    A("fixed shuffle, so the gold sits wherever `prepare.py` put it. Here every item is")
    A("asked four times, the gold moved to A, B, C and D with the three distractors held in")
    A("their order. An item scores 0, .25, .5, .75 or 1. A reader that just likes a letter")
    A("scores exactly 0.25 by construction.")
    A("")
    A("Recognition is not run: its negatives paste another series' annotation on and label")
    A("it False without checking it is false there, so a generic borrowed sentence makes the")
    A("gold wrong. See `bedtime_answer_examples.md`.")
    A("")
    n_items = res[have[0][0]]["n_items"]
    A(f"{n_items} items x 4 rotations per arm. Chance is 0.25. All four datasets are")
    A("included — the answerer reads only the caption, so sushi's 2048-point series costs")
    A("nothing in context here.")
    A("")

    A("## Overall")
    A("")
    A("`all 4` is the share of items answered correctly at **every** position — the number a")
    A("fixed-shuffle score cannot give you. It separates a caption that identifies the")
    A("series from one that got a lucky placement.")
    A("")
    A("| arm | what it is | accuracy | all 4 | none |")
    A("|---|---|---|---|---|")
    for arm, what in have:
        r = res[arm]
        A(f"| `{arm}` | {what} | {cell(r['accuracy'])} | {cell(r['frac_all_four'])} "
          f"| {cell(r['frac_none'])} |")
    A("")

    A("## By dataset")
    A("")
    A("The four sources ask structurally different questions: `truce_stock` and")
    A("`truce_synthetic` are short crowd phrases about shape and extremum location,")
    A("`sushi` is a wordier trend paraphrase, and `taxosynth` is a templated claim about a")
    A("statistical property (fat tails, stationarity, seasonality) that a shape description")
    A("is not really built to carry.")
    A("")
    A("| arm | " + " | ".join(datasets) + " |")
    A("|---" * (len(datasets) + 1) + "|")
    for arm, _ in have:
        cells = [cell(res[arm]["by_dataset"].get(d, {}).get("accuracy")) for d in datasets]
        A(f"| `{arm}` | " + " | ".join(cells) + " |")
    A("")

    A("## Position controls")
    A("")
    A("Accuracy split by where the gold was placed, and separately what the reader actually")
    A("answered. Under full rotation the answer distribution should be near-uniform; a")
    A("spike is the position prior, now measured instead of absorbed into the score.")
    A("")
    A("| arm | gold@A | gold@B | gold@C | gold@D | answered A/B/C/D |")
    A("|---|---|---|---|---|---|")
    for arm, _ in have:
        r = res[arm]
        pos = " | ".join(cell(r["by_gold_position"][k]) for k in "ABCD")
        dist = " / ".join(f"{r['answer_distribution'][k]:.2f}" for k in "ABCD")
        A(f"| `{arm}` | {pos} | {dist} |")
    A("")

    scores = {arm: per_item(arm) for arm, _ in have}
    if len([a for a in scores if scores[a]]) > 1:
        A("## Paired comparisons")
        A("")
        A("Every arm answers the same items, so each is scored 0-1 over its 4 rotations and")
        A("the sign test counts the items where one arm beats the other. Ties are dropped.")
        A("")
        A("| comparison | better | worse | p |")
        A("|---|---|---|---|")
        ref = "base" if "base" in scores else have[0][0]
        pairs = [(a, ref) for a, _ in have if a != ref and scores[a]]
        pairs += [(a, b) for a, b in (("decod550", "sft"), ("decod550", "teacher72b"),
                                      ("teacher72b", "sft"))
                  if a in scores and b in scores and scores[a] and scores[b]]
        for a, b in pairs:
            pos, neg, p = sign_test(scores[a], scores[b])
            A(f"| {a} vs {b} | {pos} | {neg} | {p:.1e} |")
        A("")

    worst = max(res[arm]["unparsed_rate"] for arm, _ in have)
    A(f"Unparsed replies: at most {worst:.4f} across arms.")
    A("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
