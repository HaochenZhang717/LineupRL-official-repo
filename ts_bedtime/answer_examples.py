from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_eval.benchmarks import bedtime
from ts_eval.caption_answer_harness import (
    DEFAULT_EVIDENCE_LABEL,
    build_answer_messages,
)

REPO = Path(__file__).resolve().parents[1]
BEDTIME = REPO / "results" / "eval_protocol" / "bedtime"
OUT = REPO / "results" / "eval_protocol" / "bedtime_answer_examples.md"

LEGS = [
    ("L0", "no evidence (floor)", None),
    ("base", "untuned Qwen2.5-VL-3B", DEFAULT_EVIDENCE_LABEL),
    ("decod550", "RL v2, decodability reward", DEFAULT_EVIDENCE_LABEL),
    ("L2", "raw series values (ceiling)", "Time series values"),
]


def load_predictions(arm: str) -> dict[str, dict]:
    path = BEDTIME / arm / "predictions.jsonl"
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--per-task", type=int, default=2,
                    help="examples per task type, split between one every leg got right "
                         "and one where the legs disagree")
    args = ap.parse_args()

    items = {i.id: i for i in bedtime.load(split="test", limit=None)}
    preds = {arm: load_predictions(arm) for arm, _, _ in LEGS}

    def legible(it) -> bool:
        s = it.series
        return 8 <= len(s) <= 60 and len(set(s)) > 3

    chosen: list[str] = []
    for task in ("recognition", "differentiation"):
        by_ds: dict[str, list[str]] = {}
        for i, it in sorted(items.items()):
            if it.task_type != task or not legible(it):
                continue
            if not all(i in preds[a] for a, _, _ in LEGS):
                continue
            by_ds.setdefault(it.domain, []).append(i)
        for ds in sorted(by_ds):
            ids = by_ds[ds]
            split = [i for i in ids
                     if preds["decod550"][i]["score"] != preds["base"][i]["score"]]
            chosen += (split or ids)[: args.per_task]

    L: list[str] = []
    A = L.append
    A("# BEDTime Protocol A — answering from the caption alone")
    A("")
    A("Worked examples. Every block below is the verbatim text sent to the answerer")
    A("(Qwen2.5-14B-Instruct, greedy, 16 max new tokens) and the raw string it returned.")
    A("")
    A("The captioner never sees the question, and the answerer never sees the chart or the")
    A("numbers. That separation is the whole design: the caption is the only channel")
    A("between them, so the answerer's accuracy is a measurement of the caption.")
    A("")
    A("| leg | what it is | evidence block |")
    A("|---|---|---|")
    for arm, what, label in LEGS:
        A(f"| `{arm}` | {what} | {'(none)' if label is None else label} |")
    A("")
    A("`L0` and `L2` are not captioners: L0 answers with no evidence at all and shows what")
    A("the options alone give away, L2 gets the raw values and shows the same answerer's")
    A("ceiling. A caption leg is only interesting relative to those two.")
    A("")
    A("## A caveat these examples make visible")
    A("")
    A("Recognition items come in balanced pairs: `_rec_pos` shows the series' own")
    A("annotation and is labelled True, `_rec_neg` shows **another series' annotation** and")
    A("is labelled False by construction (`ts_bedtime/prepare.py`, `distractors[0]`).")
    A("")
    A("Nothing checks that the borrowed annotation is actually false of this series. When")
    A("the borrowed sentence is generic -- \"exhibits no clear seasonal pattern\", \"trends")
    A("upward\" -- it can easily be true of the series it was pasted onto, and then the gold")
    A("label is wrong and an answerer that reasons correctly is scored as having failed.")
    A("Example 1 below is exactly this case. This is a property of the benchmark, not of")
    A("our arms, and it hits every leg equally, so it does not distort the comparison")
    A("between arms -- but it does put a ceiling on the absolute numbers, and it is part of")
    A("why even the L2 oracle leg only reaches 0.626.")
    A("")

    for n, item_id in enumerate(chosen, 1):
        item = items[item_id]
        A(f"## Example {n} — {item.task_type}")
        A("")
        A(f"- item: `{item_id}`")
        A(f"- dataset: {item.domain}")
        A(f"- gold answer: **{item.gold}**")
        A("")
        A("Underlying series (never shown to the answerer except in the L2 leg):")
        A("")
        A("```")
        A(", ".join(f"{v:g}" for v in item.series))
        A("```")
        A("")
        for arm, what, label in LEGS:
            rec = preds[arm][item_id]
            msg = build_answer_messages(item, rec["caption"], "evidence", label)
            verdict = "correct" if rec["score"] == 1.0 else "wrong"
            A(f"### `{arm}` — {what} → {verdict}")
            A("")
            A("**Input to the answerer:**")
            A("")
            A("```text")
            A(msg[0]["content"])
            A("```")
            A("")
            A(f"**Raw output:** `{rec['answer_raw']}`  |  gold `{item.gold}`  |  "
              f"score {rec['score']}")
            A("")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(chosen)} examples, {len(LEGS)} legs each)")


if __name__ == "__main__":
    main()
