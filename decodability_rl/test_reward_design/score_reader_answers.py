from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

def kind_label(kind: str) -> str:
    if kind in KIND_LABEL:
        return KIND_LABEL[kind]
    if kind.startswith("real_"):
        return f"distractor: another real series {kind.split('(', 1)[-1].rstrip(')')}"
    return kind


KIND_LABEL = {
    "true": "TRUE series",
    "shuffle": "distractor: shuffled",
    "reverse": "distractor: time-reversed",
    "segment_swap": "distractor: segments swapped",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results" / "results.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    answers = {a["file"]: a for a in json.loads(Path(args.answers).read_text())}
    key = json.loads(Path(args.key).read_text())
    local = json.loads(Path(args.results).read_text())
    local_by_id = {r["id"]: r for r in local["records"]}

    rows = []
    for item in key:
        a = answers.get(item["file"], {})
        letter = a.get("letter")
        correct = letter == item["gold_letter"]
        picked_kind = (
            item["option_kinds"]["ABCD".index(letter)] if letter in "ABCD" and letter else None
        )
        rows.append({**item, "pred": letter, "correct": correct,
                     "picked_kind": picked_kind, "confidence": a.get("confidence"),
                     "why": a.get("why", "")})

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    by_q: dict[int, list] = {}
    for r in rows:
        by_q.setdefault(r["question"], []).append(r)

    L: list[str] = []
    A = L.append
    A("# Strong-reader re-test: an external LLM as the reader\n")
    A(
        "**Why this was run**: the local 3B discriminator reaches only 42.5% even on the "
        "oracle description (programmatic, guaranteed to identify the true series), so a low "
        "score cannot be attributed to the caption rather than to the reader. Here a much "
        "stronger reader (one independent LLM call per question file, no shared state) "
        "re-measures that ceiling.\n"
    )
    A("**Setup**\n")
    A("- The description is always the dataset's own caption (`ds_caption`, shipped with "
      "TSFragment), not a VLM-generated one")
    A("- Questions, options and prompt are identical to the local experiment: the four "
      "candidates are the true series plus three rearrangements of it (shuffled / "
      "time-reversed / segments swapped), all with the same multiset of values")
    A(f"- {len(by_q)} series x 4 position rotations (true series at A/B/C/D in turn) = "
      f"**{n} questions**, chance = 25%")
    A("- Each reader call sees only its own question file; the answer key lives in a "
      "separate directory and the file names do not encode the gold letter\n")

    A("## Overall result\n")
    A("| reader | description | accuracy |")
    A("|---|---|---:|")
    A(f"| **strong external reader** | ds_caption | **{acc:.1%}** ({sum(r['correct'] for r in rows)}/{n}) |")
    lc = local["summary"]["conditions"]
    A(f"| local Qwen2.5-3B-Instruct | ds_caption | {lc['ds_caption']['accuracy']:.1%} |")
    A(f"| local Qwen2.5-3B-Instruct | oracle (programmatic, guaranteed decisive) | {lc['oracle']['accuracy']:.1%} |")
    A(f"| local Qwen2.5-3B-Instruct | VLM caption | {lc['caption']['accuracy']:.1%} |")
    A("| chance | -- | 25.0% |")
    A("")

    picks = Counter(r["pred"] for r in rows)
    A(f"**Option-position distribution** (after 4 rotations an unbiased reader is close to "
      f"uniform): {dict(picks)}\n")

    wrong = Counter(r["picked_kind"] for r in rows if not r["correct"] and r["picked_kind"])
    if wrong:
        A("**Which distractor was picked when the answer was wrong**\n")
        A("| distractor picked | count |")
        A("|---|---:|")
        for kind, cnt in wrong.most_common():
            A(f"| {kind_label(kind)} | {cnt} |")
        A("")

    conf = Counter(r["confidence"] for r in rows)
    conf_acc = {
        c: sum(r["correct"] for r in rows if r["confidence"] == c) / max(1, sum(1 for r in rows if r["confidence"] == c))
        for c in conf
    }
    A("**Self-reported confidence vs actual accuracy**\n")
    A("| confidence | questions | accuracy |")
    A("|---|---:|---:|")
    for c, cnt in conf.most_common():
        A(f"| {c} | {cnt} | {conf_acc[c]:.0%} |")
    A("")

    import re

    pat = re.compile(
        r"derive|derived|original|source|must be the|reversal of [ABCD]|"
        r"reverse of [ABCD]|is [ABCD] with",
        re.I,
    )
    leaks = [r for r in rows if pat.search(r["why"] or "")]
    real_mode = any(k.startswith("real_") for k in rows[0]["option_kinds"])
    if real_mode:
        A("## Where the distractors come from\n")
        A(
            "The three distractors are **other real series** (nearest neighbours after "
            "standardising mean/std/min/max), so all four candidates are equally natural and "
            "there is no \"original vs corrupted\" structure to reverse-engineer. "
            f"{len(leaks)}/{n} rationales still contain structural wording, listed below; "
            "usually they point out that two candidates are overlapping windows of the same "
            "source series.\n"
        )
        for r in leaks[:8]:
            A(f"- `{r['file']}` ({'correct' if r['correct'] else 'wrong'}): {r['why'][:200]}")
        A("")
        A("## Per-question results\n")
        A("| series id | all 4 correct? | per rotation (gold->pred) | accuracy |")
        A("|---|---|---|---:|")
        for q in sorted(by_q):
            rs = sorted(by_q[q], key=lambda r: r["gold_letter"])
            a = sum(r["correct"] for r in rs) / len(rs)
            detail = " ".join(
                f"{r['gold_letter']}->{r['pred']}{'ok' if r['correct'] else 'X'}" for r in rs
            )
            flag = "yes" if a == 1 else ("no" if a == 0 else "partly")
            A(f"| {rs[0]['series_id']} | {flag} | {detail} | {a:.0%} |")
        A("")
        Path(args.out).write_text("\n".join(L), encoding="utf-8")
        print(f"accuracy = {acc:.3f} ({sum(r['correct'] for r in rows)}/{n})")
        print(f"wrote {args.out}")
        return
    A("## A hole that must be recorded: the derivation structure among the options leaks the answer\n")
    A(
        "All three distractors are **derived from the true series** (reversed, shuffled, "
        "segments swapped), so a strong enough reader can infer the source purely from the "
        "**relations among the options** -- without using the caption at all. "
        f"**{len(leaks)}/{n}** rationales explicitly contain this reasoning:\n"
    )
    for r in leaks:
        A(f"- `{r['file']}` ({'correct' if r['correct'] else 'wrong'}): {r['why'][:220]}")
    A("")
    A(
        "So the accuracy above is an **upper bound on the upper bound**: it mixes \"what the "
        "caption contributes\" with \"what the option construction leaks\", and this run does "
        "not separate the two. **The control that has to follow**: the same reader answers "
        "the same 40 questions **without a caption** (or with a mismatched one). The "
        "difference is the caption's real contribution, and the only part usable as a reward.\n"
    )
    A(
        "If the leak is substantial, the fix is to stop the four candidates sharing one "
        "source: e.g. draw hard negatives from other real series matched on value "
        "distribution, or perturb all four candidates (the true one included) from a hidden "
        "base series.\n"
    )

    A("## Per-question results\n")
    A("| series id | all 4 correct? | per rotation (gold->pred) | accuracy |")
    A("|---|---|---|---:|")
    for q in sorted(by_q):
        rs = sorted(by_q[q], key=lambda r: r["gold_letter"])
        a = sum(r["correct"] for r in rs) / len(rs)
        detail = " ".join(
            f"{r['gold_letter']}->{r['pred']}{'ok' if r['correct'] else 'X'}" for r in rs
        )
        flag = "yes" if a == 1 else ("no" if a == 0 else "partly")
        A(f"| {rs[0]['series_id']} | {flag} | {detail} | {a:.0%} |")
    A("")

    A("## Caption used per question, and the reader's rationale\n")
    for q in sorted(by_q):
        rs = sorted(by_q[q], key=lambda r: r["gold_letter"])
        sid = rs[0]["series_id"]
        A(f"### series id={sid} (question {q})\n")
        A(f"![series {sid}](figs/series_{sid}.png)\n")
        A("**ds_caption (the dataset's own caption; the question was built on it)**\n")
        A("```")
        A(local_by_id[sid]["ds_caption"].strip())
        A("```\n")
        A("| gold | reader | correct | confidence | rationale |")
        A("|---|---|---|---|---|")
        for r in rs:
            A(
                f"| {r['gold_letter']} | {r['pred']} | {'ok' if r['correct'] else 'X'} | "
                f"{r['confidence']} | {r['why'].replace('|', '/')} |"
            )
        A("")

    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"accuracy = {acc:.3f} ({sum(r['correct'] for r in rows)}/{n})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
