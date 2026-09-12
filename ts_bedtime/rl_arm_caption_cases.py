from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

ARMS = ["rl650", "decod550", "judge550"]
ARM_LABEL = {
    "rl650": "`rl650` — RL v1 / QA-accuracy reward",
    "decod550": "`decod550` — RL v2 / decodability (4-way identification) reward",
    "judge550": "`judge550` — RL v3' / LLM-as-judge score reward",
}
DATASETS = ["truce_stock", "truce_synthetic", "sushi", "taxosynth"]

_ANCHOR_ALTS = r"""
    (?:time\s*(?:steps?|points?|index|indices|stamps?)?|timesteps?|steps?
       |index|indices|positions?|points?|x)
        \s*(?:=|:|\#)?\s*(\d+)\b
  | \bt\s*=\s*(\d+)\b
  | \b(\d+)\s*(?:st|nd|rd|th)\b
  | \b(\d+)\s*(?:mark|tick)\b
"""
TIME_PAT = re.compile(_ANCHOR_ALTS, re.IGNORECASE | re.VERBOSE)
TAIL_PAT = re.compile(r"<T>\s*(?:-|–|—|to|and|through|until)\s*(\d+)\b", re.IGNORECASE)
NUM_PAT = re.compile(r"-?\d+(?:\.\d+)?")

PROSE_POS_PAT = re.compile(
    r"\b(?:at|near|toward[s]?|in|by|around|from)\s+the\s+"
    r"(?:very\s+)?(?:beginning|start|end|middle|midpoint|outset|close|first|last|"
    r"early|late)\b|\b(?:first|second|last|final|middle)\s+(?:half|third|quarter|"
    r"portion|segment|part)\b|\bmid[- ]?(?:way|point|series)\b",
    re.IGNORECASE,
)


LIST_MARKER_PAT = re.compile(r"(?m)^\s{0,6}\d+\.\s")


def parse_anchors(text: str) -> tuple[list[int], str]:
    text = LIST_MARKER_PAT.sub(" ", text)
    anchors: list[int] = []

    def _sub(m):
        anchors.append(int(next(g for g in m.groups() if g is not None)))
        return " <T> "

    stripped = TIME_PAT.sub(_sub, text)
    while True:
        m = TAIL_PAT.search(stripped)
        if not m:
            break
        anchors.append(int(m.group(1)))
        stripped = stripped[:m.start()] + " <T> " + stripped[m.end():]
    return anchors, stripped


AXIS_SENT_PAT = re.compile(
    r"(?:the\s+)?[xy]-axis\b[^,.;]{0,40}?\branges?\b[^,.;]*"
    r"|\bincrements?\s+of\s+-?\d+(?:\.\d+)?",
    re.IGNORECASE,
)


def strip_axis_talk(text: str) -> str:
    return AXIS_SENT_PAT.sub(" ", text)


def value_claims(text: str) -> list[float]:
    _, stripped = parse_anchors(text)
    stripped = re.sub(r"-?\d+(?:\.\d+)?\s*%", " ", stripped)
    return [float(x) for x in NUM_PAT.findall(stripped)]


SELF_PRAISE_PAT = re.compile(
    r"captures every detail|would be able to recreate|all details are precise|"
    r"no ambiguity|every aspect of the chart is|there are no nuances|"
    r"exactly as (?:described|depicted|shown)|this level of detail ensures",
    re.IGNORECASE,
)


FURNITURE_PAT = re.compile(
    r"blue line|blue markers?|circular markers?|marker(?:s)? at each|"
    r"axis labeled|labeled \"?(?:value|time)\"?|the plot shows|the chart shows a",
    re.IGNORECASE,
)


SENT_PAT = re.compile(r"[^.!?]+[.!?]|\S+$")


def boilerplate_frac(text: str) -> float:
    sents = [x for x in SENT_PAT.findall(text) if x.strip()]
    boiler = sum(len(x) for x in sents
                 if SELF_PRAISE_PAT.search(x) or FURNITURE_PAT.search(x))
    return boiler / max(len(text), 1)


def max_shingle(text: str, n: int = 10) -> int:
    words = text.split()
    if len(words) < n:
        return 1
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(grams.values())


def entropy(counts) -> float:
    tot = sum(counts)
    if tot == 0:
        return 0.0
    ps = [c / tot for c in counts if c]
    return -sum(p * math.log2(p) for p in ps)


def caption_measures(caption: str, series: np.ndarray) -> dict:
    s = np.asarray(series, dtype=float)
    n = len(s)
    lo, hi = float(s.min()), float(s.max())
    rng = hi - lo if hi > lo else 1.0
    sd = float(s.std()) or 1.0

    anchors, _ = parse_anchors(caption)
    n_prose = len(PROSE_POS_PAT.findall(caption))
    in_axis = [a for a in anchors if a < n]
    vals = value_claims(caption)
    vals_na = value_claims(strip_axis_talk(caption))

    def _fit(vs):
        if not vs:
            return float("nan"), float("nan")
        d = np.abs(np.asarray(vs)[:, None] - s[None, :]).min(axis=1) / sd
        return (float(np.median(d)),
                float(np.mean([(v < lo - 0.1 * rng) or (v > hi + 0.1 * rng) for v in vs])))

    near_med, oob = _fit(vals)
    near_med_na, oob_na = _fit(vals_na)

    if in_axis:
        bins = {min(9, int(a / n * 10)) for a in in_axis}
        cov = len(bins) / 10
        ent = entropy(Counter(min(9, int(a / n * 10)) for a in in_axis).values())
    else:
        cov, ent = 0.0, 0.0

    return {
        "chars": len(caption),
        "rep10": max_shingle(caption),
        "truncated": float(not caption.rstrip().endswith((".", "!", "?", '"', ")"))),
        "self_praise": float(bool(SELF_PRAISE_PAT.search(caption))),
        "furniture": float(bool(FURNITURE_PAT.search(caption))),
        "boiler_frac": boilerplate_frac(caption),
        "n_vals": len(vals_na),
        "n_anchors": len(anchors),
        "n_prose": n_prose,
        "anchor_oob_rate": (1 - len(in_axis) / len(anchors)) if anchors else float("nan"),
        "coverage10": cov,
        "anchor_entropy": ent,
        "val_dist_med": near_med_na,
        "val_oob_rate": oob_na,
        "val_dist_med_raw": near_med,
        "val_oob_rate_raw": oob,
        "n_vals_raw": len(vals),
    }


def truth_block(s: np.ndarray) -> str:
    d = np.abs(np.diff(s)) if len(s) > 1 else np.array([0.0])
    return (
        f"- length **{len(s)}** · first **{s[0]:.2f}** · last **{s[-1]:.2f}**\n"
        f"- max **{s.max():.2f} @ t={int(s.argmax())}** · min **{s.min():.2f} @ t={int(s.argmin())}**\n"
        f"- mean {s.mean():.2f} · std {s.std():.2f} · largest single-step jump {d.max():.2f} @ t={int(d.argmax())}\n"
    )


def fmt(x, nd=2):
    return "—" if (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def load_captions(root: Path, arm: str) -> dict[str, str]:
    p = root / "bedtime" / arm / "captions.jsonl"
    out: dict[str, str] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.setdefault(d["caption_key"], d["caption"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="bench_data/bedtime/series.jsonl")
    ap.add_argument("--results-root", default="results/eval_protocol")
    ap.add_argument("--out", default="results/eval_protocol/rl_reward_caption_cases.md")
    ap.add_argument("--qa-bank", default="out_10k_xdomain/qa_verified_keep.jsonl",
                    help="the KEEP question bank rl650 was trained on; used to state what its reward tests")
    ap.add_argument("--per-dataset", type=int, default=2, help="random cases per dataset")
    ap.add_argument("--seed", type=int, default=2020)
    args = ap.parse_args()

    root = Path(args.results_root)
    caps = {a: load_captions(root, a) for a in ARMS}
    series = {}
    for line in open(args.series, encoding="utf-8"):
        d = json.loads(line)
        series[d["series_uid"]] = d

    keys = sorted(set.intersection(*(set(c) for c in caps.values())) & set(series))
    print(f"{len(keys)} series shared by all arms")

    per: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    for k in keys:
        s = np.asarray(series[k]["series"], dtype=float)
        for a in ARMS:
            per[a][k] = caption_measures(caps[a][k], s)

    def agg(arm: str, subset: list[str], field: str) -> float:
        vals = [per[arm][k][field] for k in subset]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        return float(np.median(vals)) if vals else float("nan")

    def mean(arm: str, subset: list[str], field: str) -> float:
        vals = [per[arm][k][field] for k in subset]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    qa_note = "The question bank is not available on this machine; this item was not computed."
    qa_path = Path(args.qa_bank)
    if qa_path.exists():
        kinds: Counter = Counter()
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                for q in json.loads(line)["qa_list"]:
                    kinds[q.get("verify", {}).get("kind", "?")] += 1
        tot = sum(kinds.values())
        top = ", ".join(f"`{k}` {v/tot:.1%}" for k, v in kinds.most_common(3))
        qa_note = (
            f"{tot:,} questions in total; by verifier kind the top types are {top}. "
            f"A `region` question reads \"Between t=0 and t=40, in which interval is the "
            f"average value lowest? A) t=0–10 …\" -- **all four options are time intervals, "
            f"none is a value**. So {kinds['region']/tot:.1%} of this reward tests position only; "
            f"reporting a number on the y axis earns nothing."
        )

    out: list[str] = []
    W = out.append
    W("# How captions differ under three RL rewards: BEDTime case study\n")
    W("Same policy (`Qwen2.5-VL-3B-Instruct`), same series, **byte-identical images** "
      "(md5-checked), **same caption prompt** (`decodability` family = "
      "`decodability_rl/rl/cap_prompt.py:TS_CAP_PROMPT`, no system message), "
      "**greedy decoding** (`--temperature 0.0`), caption cap 1024 tokens.\n")
    W("**The inference side is fully aligned**, so every difference on this page comes from "
      "what the model chooses to say. The training side is not fully aligned -- see the three "
      "caveats in §0, which must be kept in mind when reading the conclusions.\n")
    W("The \"ground truth\" block is **computed from the raw values**, not stated by any model, "
      "so it can be used directly to check who is making things up.\n")
    W(f"Generated by `python -m ts_bedtime.rl_arm_caption_cases`, "
      f"random-case seed `{args.seed}`.\n")
    W("---\n")

    W("## 0. The three arms\n")
    W("| arm | reward form | checkpoint |")
    W("|---|---|---|")
    W("| `rl650` | **RL v1**: the model writes 5 MCQs about its own image, a frozen text LLM answers "
      "from the caption alone, **answer accuracy** is the reward | `outputs/answerability/ckpt/global_step650_hf` |")
    W("| `decod550` | **RL v2**: a frozen reader uses the caption to **pick the described series out of 4 "
      "candidates**; a correct pick scores | `outputs/lineuprl/ckpt/global_step550_hf` |")
    W("| `judge550` | **RL v3'**: the same frozen 14B **scores the caption 1–10** against the true values, "
      "`reward=(g-1)/9*2` | `outputs/judge/ckpt/global_step550_hf` |")
    W("")
    W("The policy (`Qwen2.5-VL-3B-Instruct`), the RLOO recipe and the frozen model used as reward "
      "(`Qwen2.5-14B-Instruct`) are shared. `decod550` and `judge550` even share the training set "
      "(`out_10k_xdomain/rl_decod`, 9,500 items), so **between these two arms the reward form is the only variable**.\n")
    W("**Three caveats that must come first:**\n")
    W("1. **`rl650` was trained with a different caption prompt** (`Please describe this image in "
      "detail.`, upstream CapRL's original line), while the BEDTime evaluation uses `TS_CAP_PROMPT` "
      "for all arms. So `rl650` is **read under a prompt it was never trained on** -- part of its gap "
      "to the other two arms comes from this mismatch and cannot all be charged to the reward. "
      "`decod550` / `judge550` use the same prompt in training and evaluation.")
    W("2. **`rl650`'s training set is not the same either**: it used the QA-bank export "
      "`out_10k_xdomain/rl/train_messages.jsonl` (**7,243 items**), "
      "`decod550`/`judge550` used `rl_decod` (**9,500 items**); the two share **6,888 images**. "
      "So the `rl650` vs. others comparison is **\"the full RL v1 route\" vs. \"RL v2/v3'\"**, "
      "not a single-variable control.")
    W("3. **`judge550`'s reward has collapsed to full marks**: from step 410 on the judge gives 10 "
      "to everything (`results/decodability_rl/judge_reward_collapse.md`); step 550 is the "
      "\"val 1.0000\" best. It represents captions trained under **an already saturated judge signal**, "
      "which is itself one of the phenomena this case study is meant to show.\n")
    W("---\n")

    W("## 1. Corpus-level measurements (all 2000 unique series, not a sample)\n")
    W("Every column is computed from the caption text + raw values; definitions in §4 at the end.\n")
    W("| dataset | arm | n | chars (median) | value claims (median) | numeric anchors (median) | verbal position words (median) | "
      "time-axis coverage (10 bins) | anchor entropy (bit) | value error (median σ) | value out-of-range rate | out-of-range rate incl. axis sentences | anchor out-of-range rate |")
    W("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for ds in DATASETS:
        sub = [k for k in keys if series[k]["dataset"] == ds]
        for a in ARMS:
            W(f"| {ds} | `{a}` | {len(sub)} | {agg(a,sub,'chars'):.0f} | "
              f"{agg(a,sub,'n_vals'):.0f} | {agg(a,sub,'n_anchors'):.0f} | "
              f"{agg(a,sub,'n_prose'):.0f} | "
              f"{mean(a,sub,'coverage10'):.2f} | {mean(a,sub,'anchor_entropy'):.2f} | "
              f"{fmt(agg(a,sub,'val_dist_med'))} | {mean(a,sub,'val_oob_rate'):.3f} | "
              f"{mean(a,sub,'val_oob_rate_raw'):.3f} | "
              f"{mean(a,sub,'anchor_oob_rate'):.3f} |")
    W("")
    W("**All datasets pooled:**\n")
    W("| arm | chars (mean) | numeric density (numbers / 100 chars) | value claims (median) | numeric anchors (median) | "
      "verbal position words (median) | value error (median σ) | value out-of-range rate | out-of-range rate incl. axis sentences | "
      "time-axis coverage | anchor entropy | share of most common opening | vocabulary (first 300) |")
    W("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for a in ARMS:
        chars = [per[a][k]["chars"] for k in keys]
        nums = [per[a][k]["n_vals"] + per[a][k]["n_anchors"] for k in keys]
        density = 100 * sum(nums) / sum(chars)
        openings = Counter(" ".join(caps[a][k].split()[:8]) for k in keys)
        vocab = set()
        for k in keys[:300]:
            vocab.update(w.lower().strip(".,;:()") for w in caps[a][k].split())
        W(f"| `{a}` | {np.mean(chars):.0f} | {density:.3f} | "
          f"{agg(a,keys,'n_vals'):.0f} | {agg(a,keys,'n_anchors'):.0f} | "
          f"{agg(a,keys,'n_prose'):.0f} | "
          f"{fmt(agg(a,keys,'val_dist_med'))} | {mean(a,keys,'val_oob_rate'):.3f} | "
          f"{mean(a,keys,'val_oob_rate_raw'):.3f} | "
          f"{mean(a,keys,'coverage10'):.2f} | {mean(a,keys,'anchor_entropy'):.2f} | "
          f"{max(openings.values())/len(keys):.1%} | {len(vocab)} |")
    W("")
    W("**The gap between the two out-of-range columns is entirely judge550.** Almost every one of "
      "its captions opens with "
      "`The y-axis ranges approximately from 0 to 70 and marks increments of 10.` "
      "-- those are **axis ticks**, not claims about series values, and ticks may legitimately lie "
      "outside the data range. Counting them, its out-of-range rate is "
      f"{mean('judge550',keys,'val_oob_rate_raw'):.3f} (highest of the three); "
      f"with such sentences removed it is {mean('judge550',keys,'val_oob_rate'):.3f} "
      "(**lowest of the three**). The left column (axis sentences removed) is the one that "
      "measures whether it invents numbers that do not exist.\n")
    W("**Degenerate behaviour (whole corpus, not a sample):**\n")
    W("| arm | repetition rate (some 10-gram occurs ≥3 times) | max repeats (median / max) | "
      "cut off at 1024 tokens | contains self-praise | describes the figure itself (blue line / markers / axis labels) | "
      "**share of text in these two kinds of sentence** (median) | chars ≥3000 |")
    W("|---|--:|--:|--:|--:|--:|--:|--:|")
    for a in ARMS:
        rep = [per[a][k]["rep10"] for k in keys]
        W(f"| `{a}` | {np.mean([r >= 3 for r in rep]):.1%} | "
          f"{int(np.median(rep))} / {max(rep)} | "
          f"{mean(a,keys,'truncated'):.1%} | {mean(a,keys,'self_praise'):.1%} | "
          f"{mean(a,keys,'furniture'):.1%} | {agg(a,keys,'boiler_frac'):.1%} | "
          f"{np.mean([per[a][k]['chars'] >= 3000 for k in keys]):.1%} |")
    W("")
    W("\"Self-praise\" = the caption contains sentences like `captures every detail` / `would be able to recreate` / "
      "`no ambiguity` / `exactly as described`. These **do not describe the series**; they tell the "
      "scoring judge that the caption is complete -- exactly the trace a reward that collapsed to 10/10 "
      "should leave.\n")
    W("The three arms' **downstream answer accuracy** on BEDTime (same frozen answerer, same 11,874 questions, "
      "from each arm's `report.json`): `rl650` **0.610** / `decod550` **0.595** / "
      "`judge550` **0.583**. Corpus styles this different, yet only 2.7pp apart on this measure -- "
      "**BEDTime's MCQs cannot separate these three rewards**, which is why the captions themselves are examined below.\n")
    W("---\n")

    rng = random.Random(args.seed)
    picked: list[tuple[str, str]] = []
    for ds in DATASETS:
        sub = sorted(k for k in keys if series[k]["dataset"] == ds)
        for k in rng.sample(sub, min(args.per_dataset, len(sub))):
            picked.append((k, "random"))

    def spread(k: str, field: str) -> float:
        vs = [per[a][k][field] for a in ARMS]
        if any(isinstance(v, float) and math.isnan(v) for v in vs):
            return -1.0
        return max(vs) - min(vs)

    chosen = {k for k, _ in picked}
    for field, why in [("n_vals", "largest spread in number of value claims"),
                       ("coverage10", "largest spread in time-axis coverage"),
                       ("val_dist_med", "largest spread in value error")]:
        ranked = sorted((k for k in keys if k not in chosen),
                        key=lambda k: spread(k, field), reverse=True)
        for k in ranked[:1]:
            picked.append((k, why))
            chosen.add(k)

    def g(a, f, kind="med"):
        return agg(a, keys, f) if kind == "med" else mean(a, keys, f)

    W("## 2. Three rewards push the caption to three different places\n")
    W("Every number below comes from the tables in §1, not from an impression of the cases.\n")
    W(f"**`rl650` (answer-accuracy reward) → positions only, no values.** "
      f"Median value claims **{g('rl650','n_vals'):.0f}**, median numeric anchors "
      f"**{g('rl650','n_anchors'):.0f}** -- the only arm with far more anchors than values "
      f"(the other two are {g('decod550','n_vals'):.0f}/{g('decod550','n_anchors'):.0f} and "
      f"{g('judge550','n_vals'):.0f}/{g('judge550','n_anchors'):.0f}). "
      f"It writes sentences like \"the highest values are around time 1 to time 2\": "
      f"which segment rises, which falls, where the peak is, but almost never how much on the y axis.\n")
    W("**This is not a guess; count its question bank.** `rl650`'s reward comes from the verified questions in "
      f"`{args.qa_bank}`: " + qa_note + "\n")
    W(f"**`decod550` (4-way identification reward) → shortest, highest information per character.** "
      f"Mean chars **{np.mean([per['decod550'][k]['chars'] for k in keys]):.0f}**, "
      f"{1 - np.mean([per['decod550'][k]['chars'] for k in keys]) / np.mean([per['rl650'][k]['chars'] for k in keys]):.0%} shorter than `rl650`, "
      f"{1 - np.mean([per['decod550'][k]['chars'] for k in keys]) / np.mean([per['judge550'][k]['chars'] for k in keys]):.0%} shorter than `judge550`; "
      f"median value error **{g('decod550','val_dist_med'):.2f}σ**, value out-of-range rate "
      f"**{g('decod550','val_oob_rate','mean'):.1%}**; "
      f"numeric density **{100*sum(per['decod550'][k]['n_vals']+per['decod550'][k]['n_anchors'] for k in keys)/sum(per['decod550'][k]['chars'] for k in keys):.2f}** numbers / 100 chars, "
      f"highest of the three. Values and time points are almost equal in count "
      f"({g('decod550','n_vals'):.0f} vs {g('decod550','n_anchors'):.0f}) -- "
      f"it writes **paired \"when it is how much\"**.\n")
    W("To pick the described series out of 4 candidates, **only true numbers bound to a time point are discriminative**: "
      "inventing a value that does not exist pushes the reader toward the wrong candidate, and a sentence "
      "that does not help discriminate earns nothing. This reward penalises both fabrication and filler, "
      "so the caption is short and dense.\n")
    W(f"**`judge550` (LLM-score reward, collapsed to full marks) → longest, widest coverage, accurate values, "
      f"but {g('judge550','boiler_frac'):.0%} of the text is not about this series.** "
      f"Mean chars **{np.mean([per['judge550'][k]['chars'] for k in keys]):.0f}** (longest of the three, "
      f"{np.mean([per['judge550'][k]['chars'] for k in keys])/np.mean([per['decod550'][k]['chars'] for k in keys]):.1f}x `decod550`), "
      f"time-axis coverage **{g('judge550','coverage10','mean'):.2f}** and anchor entropy "
      f"**{g('judge550','anchor_entropy','mean'):.2f}** both highest of the three -- "
      f"it is the arm that **skips the fewest segments**. Value error **{g('judge550','val_dist_med'):.2f}σ** and "
      f"value out-of-range rate **{g('judge550','val_oob_rate','mean'):.1%}** are also fine.\n")
    W(f"The cost is in the other two columns: **{mean('judge550',keys,'self_praise'):.1%} of captions carry self-praise** "
      f"(`captures every detail`, `no ambiguity`, `would be able to recreate`), "
      f"**{mean('judge550',keys,'furniture'):.1%} describe the figure itself** "
      f"(`a blue line with circular blue markers`, `the y-axis labeled \"value\"`); "
      f"for the other two arms these are {mean('rl650',keys,'self_praise'):.1%} / "
      f"{mean('rl650',keys,'furniture'):.1%} and {mean('decod550',keys,'self_praise'):.1%} / "
      f"{mean('decod550',keys,'furniture'):.1%}. The renderer is the same for every image, "
      f"so \"blue line\" and \"circular markers\" contribute **zero** to telling series apart; "
      f"self-praise describes nothing at all and only tells the judge the caption is complete.\n")
    W("This is the shape a collapsed reward leaves: from step 410 the judge gives 10 to everything, "
      "**the gradient no longer separates good from bad, only \"what text makes the judge say full marks\"**. "
      "It did not break the model -- values that should be accurate still are -- but it spent a large share "
      "of the budget on **satisfying the scorer** rather than **letting a reader recover the series**.\n")
    W("**In one sentence:** same model, same image, same prompt, three rewards push the caption to "
      "three places -- the identification reward forces **compact value-time binding** "
      f"(shortest, highest numeric density); the QA-accuracy reward forces **pure position** "
      f"(median value claims {g('rl650','n_vals'):.0f}, almost no numbers); "
      f"the collapsed judge reward forces **performative thoroughness** (longest, widest coverage, but "
      f"{mean('judge550',keys,'self_praise'):.1%} with self-praise, "
      f"{g('judge550','boiler_frac'):.0%} of characters unrelated to this series). "
      "**This is the evidence the claim \"the reward decides what information a caption carries\" needs.**\n")
    W("---\n")

    W("## 3. Case study\n")
    W(f"The first {len(DATASETS)*args.per_dataset} cases are **random** "
      f"(`random.Random({args.seed})`, {args.per_dataset} per dataset, not hand-picked); "
      "the last 3 are **selected by a computable disagreement metric**, with the reason stated. Captions are unabridged.\n")

    for k, why in picked:
        row = series[k]
        s = np.asarray(row["series"], dtype=float)
        W(f"### `{k}`  ·  {row['dataset']}, len={len(s)}  ·  *[{why}]*\n")
        if row.get("cls"):
            W(f"class `{row['cls']}` / `{row.get('subclass')}`\n")
        W("**Ground truth (computed)**\n")
        W(truth_block(s))
        W("**BEDTime's own crowd/template annotations**: " +
          " / ".join(f"\"{a}\"" for a in row["annotations"]) + "\n")
        W(f"Image: `results/eval_protocol/bedtime/decod550/render/bedtime_{k}.png`"
          " (identical md5 across the three arms)\n")
        W("| arm | chars | value claims | numeric anchors | verbal position words | coverage | value error (σ) | value out-of-range | "
          "max repeats | self-praise |")
        W("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for a in ARMS:
            m = per[a][k]
            W(f"| `{a}` | {m['chars']} | {m['n_vals']} | {m['n_anchors']} | "
              f"{m['n_prose']} | {m['coverage10']:.1f} | {fmt(m['val_dist_med'])} | "
              f"{fmt(m['val_oob_rate'])} | {m['rep10']}× | "
              f"{'yes' if m['self_praise'] else '—'} |")
        W("")
        for a in ARMS:
            W(f"- **{ARM_LABEL[a]}** ({per[a][k]['chars']} chars): {caps[a][k]}\n")
        W("---\n")

    W("## 4. Metric definitions (all recomputable)\n")
    W("- **Numeric anchors**: integers in the text that point at a time step. The three arms spell them "
      "differently, so all three forms are caught: "
      "`time step 5` / `time steps 7 and 8` (decod550), `time 5` / `from time 0 to "
      "time 3` (rl650), `around the 2nd point` / `around the 400 mark` (judge550), "
      "plus `t=5` / `index N` / `position N`. Markdown ordered-list numbers are stripped first. "
      "**The numbers left after removing these count as value claims** (percentages excluded). "
      "This step matters: a regex that only recognises decod550's spelling counts all of rl650's "
      "time points as values and inflates its value error to 5.96σ -- an artefact of the measure, not of the arm.")
    W("- **Value error**: distance from each value claim to the **nearest actual value** in the series, "
      "divided by the series std σ; median within a caption, then median over the corpus. It measures "
      "whether the stated number exists on this series, not whether it was placed at the right time step.")
    W("- **Value out-of-range rate**: fraction of value claims outside `[min − 0.1·range, max + 0.1·range]` -- "
      "numbers that **do not exist** on this series at all.")
    W("- **Anchor out-of-range rate**: fraction of time anchors ≥ series length (pointing at a nonexistent step).")
    W("- **Time-axis coverage**: split the time axis into 10 equal bins; the number of bins the caption's anchors fall into (÷10). "
      "Measures **completeness**: skipping a whole segment lowers it.")
    W("- **Verbal position words**: occurrences of **number-free position phrases** such as "
      "`near the end` / `in the middle` / `the first half`. They give positions too but cannot enter the coverage column, "
      "so they are counted separately.")
    W("- **Self-praise / describes the figure**: the former is `captures every detail`, `no ambiguity` and the like -- "
      "statements to the scorer that the caption is complete; the latter is `a blue line with circular blue markers`, "
      "`the y-axis labeled \"value\"` and the like -- descriptions of **rendering style**. The renderer is the same for every image, "
      "so both kinds contribute zero to recognising this series from the caption. "
      "\"Share of text\" = characters in sentences matching either kind ÷ total caption characters, median.")
    W("- **Max repeats**: how many times the most frequent 10-word n-gram in the caption occurs. "
      "Normal descriptions give 1–2; ≥3 is essentially decoding stuck in a loop.")
    W("- **Anchor entropy**: Shannon entropy (bit) of the anchors over the 10 bins, maximum log2(10)=3.32. "
      "Coverage measures how many bins are touched; entropy measures whether they are touched evenly.")
    W("")
    W("**Known limitations of the measures**, stated here so they are not taken as conclusions:\n")
    W("1. Value error is **nearest-neighbour matching only**, so \"27 placed at t=0\" and \"27 placed at t=5\" "
      "score the same -- it does not penalise **misplacement**, only whether the number exists on this series. "
      "Measuring misplacement needs an aligned reconstruction error, which is a different exercise.")
    W("2. Purely verbal position phrases (`near the end`) cannot enter the coverage column. They are counted "
      "separately as \"verbal position words\"; the three arms' medians are 0/0/1, small enough that the ordering "
      "of the coverage column is not overturned -- but coverage should still be read as covering "
      "**numbered** positions.")
    W("3. Value claims include axis descriptions (judge550 likes to write "
      "`the y-axis ranges approximately from 0 to 70`). These numbers do describe the figure and may indeed lie "
      "off the series, so they are not excluded from the raw column -- part of judge550's raw out-of-range rate "
      "comes from this phrasing rather than hallucination.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  ({len(picked)} cases)")


if __name__ == "__main__":
    main()
