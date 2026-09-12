from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BEDTIME = REPO / "results" / "eval_protocol" / "bedtime"
MCQ = REPO / "results" / "eval_protocol" / "bedtime_mcq"

ARMS = [
    ("base", "untuned Qwen2.5-VL-3B"),
    ("sft", "SFT, distilled from the 72B teacher"),
    ("teacher72b", "the 72B teacher itself"),
    ("rl650", "RL v1, self-generated MCQ reward"),
    ("decod550", "RL v2, decodability reward"),
]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def per_series_scores(arm: str) -> dict[tuple[str, str], float]:
    path = MCQ / f"{arm}.jsonl"
    if not path.exists():
        return {}
    acc: dict[tuple[str, str], list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            acc.setdefault((r["metric"], r["series_uid"]), []).append(r["correct"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def sign_test(a: dict, b: dict, metric: str) -> tuple[int, int, float]:
    keys = [k for k in a if k[0] == metric and k in b]
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
    global MCQ
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--mcq-dir", default=str(MCQ),
                    help="where <arm>.json from mcq_metrics lives")
    ap.add_argument("--title", default="BEDTime — three caption metrics")
    ap.add_argument("--no-nli", action="store_true",
                    help="omit the gen->gt column. Set for CaTS, whose reference captions "
                         "carry dates, domain and population statistics that a caption "
                         "written from a leak-proofed chart cannot entail")
    args = ap.parse_args()
    MCQ = Path(args.mcq_dir)

    L: list[str] = []
    A = L.append
    A(f"# {args.title}")
    A("")
    A("")
    A("> Part of the picture. `bedtime_cats_all_models.md` holds every model and every\n"
      "> BEDTime/CaTS metric in one table, generated from the same JSON as this file.")
    A("")
    if args.no_nli:
        A("| arm | what it is | caption→series | series→caption |")
        A("|---|---|---|---|")
    else:
        A("| arm | what it is | 1. gen→gt | 2. caption→series | 3. series→caption |")
        A("|---|---|---|---|---|")
    rows = []
    for arm, what in ARMS:
        nli = load(BEDTIME / arm / "generation_deployed.json")
        mcq = load(MCQ / f"{arm}.json")
        g = nli["overall"]["gen_entails_gt"] if nli else None
        a = mcq["A"]["accuracy"] if mcq and "A" in mcq else None
        b = mcq["B"]["accuracy"] if mcq and "B" in mcq else None
        rows.append((arm, g, a, b, mcq))
        if args.no_nli:
            if mcq is None:
                continue
            A(f"| `{arm}` | {what} | {cell(a)} | {cell(b)} |")
        else:
            A(f"| `{arm}` | {what} | {cell(g)} | {cell(a)} | {cell(b)} |")
    A("")
    n_series = next((m["A"]["n_series"] for *_, m in rows if m and "A" in m), None)
    A("Chance is 0.25 on both MCQ metrics.")
    if not args.no_nli:
        A("gen→gt has no chance level; its floor is an empty caption (0.003) and its")
        A("ceiling is the reference scored against itself (1.000).")
    A("")
    A("**caption→series is decod550's training objective**, asked of the same frozen 14B")
    A("reader it was trained against, in the same prompt template. Its score there is a")
    A("home-field number and is not evidence of generalisation. series→caption inverts the")
    A("question and no arm was trained on it.")
    A("")
    if args.no_nli:
        A(f"Coverage: {n_series} unique CaTS series (lengths 8-150) pulled from the QA pool")
        A("by `ts_eval.build_cats_series` and captioned from the same leak-proofed chart.")
        A("Distractors are the 3 nearest series in z-scored (mean, std, min, max) space at")
        A("the SAME length, pooled across domains -- CaTS's domain label lives only in")
        A("metadata the captioner never sees, and some domains hold fewer than 4 series of a")
        A("given length.")
        A("")
        A("**gen→gt is deliberately absent.** CaTS's reference captions are not a usable")
        A("target for these arms: all 100 sampled gold captions state explicit dates, 56%")
        A("name the phenomenon (\"daily COVID-19 deaths in China\") and 56% quote")
        A("population-level statistics. Our charts carry no title, dates or domain, so a")
        A("caption cannot entail any of it, and the score would measure the metadata we")
        A("removed on purpose rather than caption quality.")
    else:
        A(f"Coverage: the MCQ metrics run on {n_series} series (truce_stock,")
        A("truce_synthetic, taxosynth); sushi is excluded because its 2048-point series need")
        A("~57k tokens for a 4-way option set against the reader's 32k context. gen→gt")
        A("covers all 2000 series. Distractors are the 3 nearest series in z-scored")
        A("(mean, std, min, max) space within the same dataset and the same length, chosen")
        A("from the series ids alone and therefore byte-identical across arms.")
    A("")

    have = [r for r in rows if r[4]]
    if have:
        A("## Per-dataset breakdown")
        A("")
        for metric, title in (("A", "2. caption→series"), ("B", "3. series→caption")):
            datasets = sorted({d for _, _, _, _, m in have if m and metric in m
                               for d in m[metric]["by_dataset"]})
            if not datasets:
                continue
            A(f"### {title}")
            A("")
            A("| arm | " + " | ".join(datasets) + " |")
            A("|---" * (len(datasets) + 1) + "|")
            for arm, _, _, _, m in have:
                if not m or metric not in m:
                    continue
                cells = [cell(m[metric]["by_dataset"].get(d, {}).get("accuracy"))
                         for d in datasets]
                A(f"| `{arm}` | " + " | ".join(cells) + " |")
            A("")

        A("## Position control")
        A("")
        A("Accuracy by which slot held the true option. All four rotations are asked for")
        A("every question, so a flat row means the score is not a position prior; a peaked")
        A("row means the reader is guessing a letter.")
        A("")
        A("| arm | metric | A | B | C | D |")
        A("|---|---|---|---|---|---|")
        for arm, _, _, _, m in have:
            for metric in ("A", "B"):
                if not m or metric not in m:
                    continue
                p = m[metric]["by_gold_position"]
                A(f"| `{arm}` | {metric} | " + " | ".join(cell(p[k]) for k in "ABCD") + " |")
        A("")
        worst = max((m["unparsed_rate"] for _, _, _, _, m in have if m), default=0.0)
        A(f"Unparsed replies: at most {worst:.4f} of prompts across arms.")
        A("")

        A("## Paired comparisons")
        A("")
        A(f"Every arm answers the same {n_series} questions, so these are paired: each series")
        A("is scored 0-1 by how many of its 4 rotations were right, and the sign test counts")
        A("the series where one arm beats the other. Ties are uninformative and dropped.")
        A("")
        scores = {arm: per_series_scores(arm) for arm, *_ in have}
        ref = "base" if "base" in scores else next(iter(scores))
        pairs = [(a, b) for a in scores for b in scores
                 if a != b and (b == ref or (a, b) == ("decod550", "sft"))]
        A("| metric | comparison | better | worse | p |")
        A("|---|---|---|---|---|")
        for metric in ("A", "B"):
            for a, b in pairs:
                if not scores[a] or not scores[b]:
                    continue
                pos, neg, p = sign_test(scores[a], scores[b], metric)
                if pos + neg == 0:
                    continue
                A(f"| {metric} | {a} vs {b} | {pos} | {neg} | {p:.1e} |")
        A("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
