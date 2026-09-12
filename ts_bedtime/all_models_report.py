from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EP = REPO / "results" / "eval_protocol"
ZOO = EP / "vlm_zoo"
TEXTZOO = EP / "text_zoo"
CATS_GENGT = EP / "cats_gengt" / "nli_cats"

ARMS = [
    ("base", "untuned Qwen2.5-VL-3B, our starting point", "3B"),
    ("sft", "SFT, distilled from the 72B teacher", "3B"),
    ("rl650", "RL v1, reward = self-generated MCQs", "3B"),
    ("decod550", "RL v2, reward = decodability", "3B"),
    ("valmask400", "RL v2 with numerals masked in the reward", "3B"),
    ("teacher72b", "the 72B teacher itself", "72B"),
    ("judge550", "RL v3, reward = a frozen 14B grading the caption 1-10", "3B"),
]
ZOO_MODELS = [
    ("phi4mm", "Phi-4-multimodal", "Microsoft", "5.6B"),
    ("idefics3-8b", "Idefics3", "HuggingFace", "8B"),
    ("qwen3vl-8b", "Qwen3-VL", "Alibaba", "8B"),
    ("gemma3-12b", "Gemma-3", "Google", "12B"),
    ("internvl3-14b", "InternVL3", "Shanghai AI Lab", "14B"),
]
TEXT_MODELS = [
    ("phi35-mini", "Phi-3.5-mini", "Microsoft", "3.8B"),
    ("qwen25-7b", "Qwen2.5-7B", "Alibaba", "7B"),
    ("qwen25-14b", "Qwen2.5-14B", "Alibaba", "14B"),
]
TSLM_MODELS = [
    ("chattime-7b", "ChatTime-1-Chat", "BUPT", "7B"),
    ("chatts-14b", "ChatTS", "ByteDance", "14B"),
]
MODALITY = {"arm": "chart", "zoo": "chart", "text": "digits", "tslm": "series encoder"}

DIFF_BASELINES = [("L0", "no evidence (floor)"), ("L2", "the true series values (oracle)")]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def paths_for(tag: str, group: str) -> dict[str, Path]:
    if group == "arm":
        return {"nli": EP / "bedtime" / tag / "generation_deployed.json",
                "bt": EP / "bedtime_mcq" / f"{tag}.json",
                "ct": EP / "cats_mcq" / f"{tag}.json",
                "bt_rows": EP / "bedtime_mcq" / f"{tag}.jsonl",
                "ct_rows": EP / "cats_mcq" / f"{tag}.jsonl",
                "diff": EP / "bedtime_diff_rot" / f"{tag}.json"}
    root = ZOO if group == "zoo" else TEXTZOO
    return {"nli": root / "nli_bedtime" / f"{tag}_deployed.json",
            "bt": root / "mcq_bedtime" / f"{tag}.json",
            "ct": root / "mcq_cats" / f"{tag}.json",
            "bt_rows": root / "mcq_bedtime" / f"{tag}.jsonl",
            "ct_rows": root / "mcq_cats" / f"{tag}.jsonl",
            "diff": EP / "bedtime_diff_rot" / f"{tag}.json"}


def row_for(tag: str, group: str) -> dict:
    p = paths_for(tag, group)
    nli, bt, ct, diff = load(p["nli"]), load(p["bt"]), load(p["ct"]), load(p["diff"])
    cq = load(CATS_GENGT / f"{tag}_qualitative.json")
    cg = load(CATS_GENGT / f"{tag}_agnostic.json")
    return {
        "ct_gen_gt": cq["overall"]["gen_entails_gt"] if cq else None,
        "ct_gen_gt_num": cg["overall"]["gen_entails_gt"] if cg else None,
        "ct_gen_gt_floor": (cq.get("controls", {}).get("empty", {}).get("gen_entails_gt")
                            if cq else None),
        "tag": tag, "group": group,
        "gen_gt": nli["overall"]["gen_entails_gt"] if nli else None,
        "gt_gen": nli["overall"]["gt_entails_gen"] if nli else None,
        "chars": nli["overall"]["mean_caption_chars"] if nli else None,
        "bt_a": bt["A"]["accuracy"] if bt and "A" in bt else None,
        "bt_b": bt["B"]["accuracy"] if bt and "B" in bt else None,
        "ct_a": ct["A"]["accuracy"] if ct and "A" in ct else None,
        "ct_b": ct["B"]["accuracy"] if ct and "B" in ct else None,
        "bt_unparsed": bt.get("unparsed_rate") if bt else None,
        "ct_unparsed": ct.get("unparsed_rate") if ct else None,
        "diff_acc": diff.get("accuracy") if diff else None,
        "diff_all4": diff.get("frac_all_four") if diff else None,
        "diff_none": diff.get("frac_none") if diff else None,
        "_bt": bt, "_ct": ct, "_diff": diff, "_nli": nli,
    }


DEVIANT: set[str] = set()


def check_alignment(rows: list[dict]) -> list[str]:
    notes: list[str] = []
    for field, label in (("_bt", "BEDTime MCQ"), ("_ct", "CaTS MCQ")):
        seen: dict[tuple, list[str]] = {}
        for r in rows:
            d = r[field]
            if not d:
                continue
            key = (d["reader"], d["n_distractors"], d["rotations"],
                   d["A"]["n_series"], d["B"]["n_series"])
            seen.setdefault(key, []).append(r["tag"])
        if seen:
            main = max(seen, key=lambda k: len(seen[k]))
            reader, nd, rot, na, nb = main
            notes.append(f"{label}: {len(seen[main])} models scored identically -- reader "
                         f"{reader}, {nd} distractors, {rot} rotations, n_series {na}/{nb}")
            for key, tags in seen.items():
                if key == main:
                    continue
                DEVIANT.update(tags)
                notes.append(f"  ⚠️ {label}: {', '.join(tags)} differ -> reader {key[0]}, "
                             f"{key[1]} distractors, {key[2]} rotations, "
                             f"n_series {key[3]}/{key[4]}; excluded from the paired tests")
    seen_nli: dict[tuple, list[str]] = {}
    for r in rows:
        d = r["_nli"]
        if d:
            seen_nli.setdefault((d["model"], d["threshold"], d["overall"]["n"]),
                                []).append(r["tag"])
    if seen_nli:
        main = max(seen_nli, key=lambda k: len(seen_nli[k]))
        m, th, n = main
        notes.append(f"gen→gt: {len(seen_nli[main])} models, {m} at threshold {th}, n={n}")
        for key, tags in seen_nli.items():
            if key != main:
                notes.append(f"  ⚠️ gen→gt: {', '.join(tags)} scored on n={key[2]} pairs, "
                             f"not {n} -- not comparable to the other rows")
    return notes


def per_series(path: Path) -> dict[tuple[str, str], float]:
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


def c(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = ([row_for(t, "arm") for t, *_ in ARMS]
            + [row_for(t, "zoo") for t, *_ in ZOO_MODELS]
            + [row_for(t, "text") for t, *_ in TEXT_MODELS]
            + [row_for(t, "tslm") for t, *_ in TSLM_MODELS])
    by_tag = {r["tag"]: r for r in rows}
    align = check_alignment(rows)
    meta = {t: (d, s, "arm") for t, d, s in ARMS}
    meta.update({t: (f"{n} ({org})", s, "zoo") for t, n, org, s in ZOO_MODELS})
    meta.update({t: (f"{n} ({org})", s, "text") for t, n, org, s in TEXT_MODELS})
    meta.update({t: (f"{n} ({org})", s, "tslm") for t, n, org, s in TSLM_MODELS})

    L: list[str] = []
    A = L.append
    A("# BEDTime and CaTS — all models, all metrics")
    A("")
    A("Generated by `python -m ts_bedtime.all_models_report`. It is the union of the")
    A("per-slice reports (`bedtime_three_metrics.md`, `cats_two_metrics.md`,")
    A("`bedtime_differentiation.md`, `caption_metrics_full.md`), read from the same JSON,")
    A("so no number here is a second version of a number there.")
    A("")

    A("## The five metrics")
    A("")
    A("| # | metric | question | chance |")
    A("|---|---|---|---|")
    A("| 1 | gen→gt | does the generated caption **entail** BEDTime's reference caption? "
      "(NLI, generated as premise) | none |")
    A("| 2 | BEDTime cap→ser | given the caption, pick the true series out of 4 | 0.25 |")
    A("| 3 | BEDTime ser→cap | given the series, pick the true caption out of 4 | 0.25 |")
    A("| 4 | CaTS gen→gt | same as 1, against CaTS references rewritten to carry no domain "
      "and **no numbers** (`cats_qualitative_refs.jsonl`) | none |")
    A("| 5 | CaTS cap→ser | same as 2, on CaTS-Bench | 0.25 |")
    A("| 6 | CaTS ser→cap | same as 3, on CaTS-Bench | 0.25 |")
    A("")
    A("gen→gt has no chance level: its floor is an empty caption (0.003) and its ceiling")
    A("the reference scored against itself (1.000).")
    A("")
    A("**Read metrics 3 and 5 first.** `cap→ser` on BEDTime is decod550's *training")
    A("objective*, asked of the same frozen 14B reader in the same prompt template, so its")
    A("score there is a home-field number rather than evidence of generalisation. `ser→cap`")
    A("inverts the question and no model was trained on it.")
    A("")

    A("## All models")
    A("")
    A("| model | what it is | size | sees | 1. BT gen→gt | 2. BT cap→ser | 3. BT ser→cap "
      "| 4. CaTS gen→gt | 5. CaTS cap→ser | 6. CaTS ser→cap |")
    A("|---|---|---|---|---|---|---|---|---|---|")

    def emit(tag: str, what: str, size: str, group: str) -> None:
        r = by_tag[tag]
        mark = " ⚠️" if tag in DEVIANT else ""
        A(f"| `{tag}`{mark} | {what} | {size} | {MODALITY[group]} | {c(r['gen_gt'])} | "
          f"{c(r['bt_a'])} | {c(r['bt_b'])} | {c(r['ct_gen_gt'])} | {c(r['ct_a'])} | "
          f"{c(r['ct_b'])} |")

    A("| | **trained in this project — read the chart** | | | | | | | | |")
    for tag, what, size in ARMS:
        emit(tag, what, size, "arm")
    A("| | **untuned off-the-shelf VLMs — read the chart** | | | | | | | | |")
    for tag, name, org, size in ZOO_MODELS:
        emit(tag, f"{name} ({org})", size, "zoo")
    A("| | **text LLMs — read the values as digits** | | | | | | | | |")
    for tag, name, org, size in TEXT_MODELS:
        emit(tag, f"{name} ({org})", size, "text")
    A("| | **time-series LMs — read the values through a series encoder** | | | | | | | | |")
    for tag, name, org, size in TSLM_MODELS:
        emit(tag, f"{name} ({org})", size, "tslm")
    A("")
    A("⚠️ marks a model whose question set differs from the rest; see the notes below.")
    A("")
    for n in align:
        A(f"- {n}")
    A("")

    A("## `judge550`: the metrics disagree about it, and that is the point")
    A("")
    A("The third RL arm replaced the decodability reward with an LLM-as-judge one -- a")
    A("frozen Qwen2.5-14B grading the caption 1-10 against the values, same policy, same")
    A("data, same RLOO recipe. By step 550 the judge scored **every** val caption a perfect")
    A("1.0000. The external metrics split cleanly on whether they ask a machine or a person:")
    A("")
    A("| metric | what it asks | judge550 | base (untuned) | decod550 |")
    A("|---|---|---|---|---|")
    A("| BEDTime cap→ser | can a reader identify the series | 0.717 | 0.669 | 0.795 |")
    A("| CaTS cap→ser | same, on CaTS | 0.921 | 0.876 | 0.968 |")
    import statistics as _st
    from ts_downstream.seed_report import DATASETS as _DS, FORECAST as _FC, load as _load
    _g = _load()
    _seeds = sorted({s_ for v in _g.values() for s_ in v})

    def _rank(arm: str) -> float:
        rs = []
        for ds in _DS:
            order = sorted(_FC, key=lambda a: _st.mean(list(_g[("forecasting", ds, a)].values())))
            rs.append(order.index(arm) + 1)
        return sum(rs) / len(rs)

    def _recon(arm: str) -> float:
        return _st.mean(list(_g[("reconstruction", "ETTm2", arm)].values()))

    A(f"| forecasting (mean rank of 7) | is the caption usable by a regressor | "
      f"{_rank('judge550'):.2f} | {_rank('base'):.2f} | {_rank('decod550'):.2f} |")
    A(f"| reconstruction ETTm2 MSE | same, predicting the window | {_recon('judge550'):.4f} "
      f"| — | {_recon('decod550'):.4f} |")
    A("| **BEDTime gen→gt** | **does it entail a human description** | **0.118** | 0.296 | 0.400 |")
    A("| **CaTS gen→gt** | **same, on CaTS** | **0.033** | 0.352 | 0.611 |")
    A("")
    A("Everything a model reads goes up relative to the untuned baseline. Everything a")
    A("person would read goes **down by a factor of three to ten**, below every other row in")
    A("this file including the untuned model.")
    A("")
    A("The captions say why. judge550 describes the CHART rather than the series:")
    A("")
    A("> The line chart depicts a single time series with a y-axis labeled \"value\" and an")
    A("> x-axis labeled \"time\". The y-axis ranges approximately from 85 to 110 and marks")
    A("> increments of 2 [...] The plot shows a blue line with circular blue markers at each")
    A("> data point. The line begins slightly below 92 [...]")
    A("")
    A("against decod550 on the same series:")
    A("")
    A("> The time series starts with an initial value of approximately 91.2. It then shows a")
    A("> gradual increase, reaching about 92.5 by time step 2 [...]")
    A("")
    A("Axis ranges, tick spacing, line colour and marker shape are true of the image and")
    A("unfalsifiable, so the judge rewards them as specific and detailed. They are also true")
    A("of *every* chart in the dataset, so they entail none of BEDTime's \"line decreases")
    A("near the end\" descriptions. The numeric detail that survives is enough for an")
    A("embedding readout to work with, which is why the machine-facing metrics hold up.")
    A("")
    A("Read this as the cost of an absolute-grading reward, not as the arm being broken: it")
    A("optimised exactly what it was told to, and what it was told to optimise turned out to")
    A("be one frozen model's notion of a thorough description. See")
    A("`results/decodability_rl/judge_reward_collapse.md` for the training-side story.")
    A("")

    A("## CaTS gen→gt — two reference sets, and why only one of them works")
    A("")
    A("CaTS-Bench ships one reference caption per series and none of it is usable as")
    A("written: it states calendar years, names the phenomenon, and compares the series to")
    A("statistics of the source dataset. Our captioner sees an unlabelled chart, so scoring")
    A("against that measures the metadata we removed on purpose. The references were")
    A("therefore rewritten twice, and the difference between the two is the whole story.")
    A("")
    A("| reference set | domain removed | numbers removed | share containing an exact number |")
    A("|---|---|---|---|")
    A("| `cats_agnostic_refs.jsonl` | yes | **no** — values kept verbatim | 99.9% "
      "(median 4 per reference) |")
    A("| `cats_qualitative_refs.jsonl` | yes | **yes** | 0.0% |")
    A("| BEDTime's own references, for scale | n/a | n/a | 1.3% |")
    A("")
    A("Keeping the values was right for faithfulness and fatal for the metric. A model")
    A("reading a chart with no value labels cannot entail *\"starting at 91.29 and reaching")
    A("105.27\"* however well it describes the shape, so that column ranked access to")
    A("digits. The second rewrite replaces each quantity with the comparative fact it")
    A("supports — *\"starting low and ending higher\"* — and keeps every position word, so")
    A("the target is the kind of statement BEDTime uses.")
    A("")
    A("**Column 4 of the headline table is the qualitative one.** Both are shown here")
    A("because the contrast is the evidence that the first was measuring the wrong thing.")
    A("")
    A("| model | sees | qualitative refs | numeric refs |")
    A("|---|---|---|---|")
    ranked = sorted((r for r in rows if r["ct_gen_gt"] is not None),
                    key=lambda r: -r["ct_gen_gt"])
    for r in ranked:
        A(f"| `{r['tag']}` | {MODALITY[r['group']]} | {r['ct_gen_gt']:.3f} | "
          f"{c(r['ct_gen_gt_num'])} |")
    floors = [r["ct_gen_gt_floor"] for r in rows if r.get("ct_gen_gt_floor") is not None]
    if floors:
        A(f"| *empty caption (control)* | — | {floors[0]:.3f} | 0.007 |")
    A("")
    A("Under the numeric references every chart-reading model sat at or below the")
    A("empty-caption floor — `gemma3-12b` 0.002, `rl650` 0.005, `sft` 0.008, `teacher72b`")
    A("0.011 — and the three text LLMs led. Under the qualitative references the same")
    A("models score 0.222 to 0.611 and the ordering no longer tracks modality: text LLMs")
    A("are spread through the middle of the table rather than at the top.")
    A("")
    A("That is the useful result. **On the MCQ metrics a text LLM reading digits beats")
    A("every chart reader by a wide margin; on qualitative gen→gt it does not.** The two")
    A("kinds of metric reward different things — identifying a series, and describing it")
    A("the way a person would — and exact values only help with the first.")
    A("")

    A("## Ranked by the metrics nobody trained on")
    A("")
    A("`ser→cap` on both benchmarks: the reader sees the series and picks the caption. No")
    A("model in this table was trained against it, on either benchmark.")
    A("")
    for key, title in (("bt_b", "BEDTime ser→cap"), ("ct_b", "CaTS ser→cap")):
        ranked = sorted((r for r in rows if r[key] is not None),
                        key=lambda r: -r[key])
        A(f"**{title}**  ")
        A("| rank | model | score | group |")
        A("|---|---|---|---|")
        for i, r in enumerate(ranked, 1):
            A(f"| {i} | `{r['tag']}` | {c(r[key])} | {meta[r['tag']][2]} |")
        A("")

    A("## Caption length")
    A("")
    A("Mean characters per caption, over the same 3958 BEDTime items. Included because it")
    A("is the first thing to suspect when a score moves: a longer caption has more chances")
    A("to entail the reference and more chances to name the distinguishing feature.")
    A("decod550 is the SHORTEST caption in the table and still leads metrics 3 and 5, which")
    A("is the main reason its lead is not a length artifact.")
    A("")
    A("| model | mean chars | gen→gt | BT ser→cap |")
    A("|---|---|---|---|")
    for r in sorted((r for r in rows if r["chars"] is not None), key=lambda r: r["chars"]):
        A(f"| `{r['tag']}` | {r['chars']:.0f} | {c(r['gen_gt'])} | {c(r['bt_b'])} |")
    A("")

    for field, bench in (("_bt", "BEDTime"), ("_ct", "CaTS")):
        A(f"## {bench} — per-dataset")
        A("")
        for metric, title in (("A", "cap→ser"), ("B", "ser→cap")):
            datasets = sorted({d for r in rows if r[field]
                               for d in r[field][metric]["by_dataset"]})
            if not datasets:
                continue
            A(f"### {bench} {title}")
            A("")
            A("| model | " + " | ".join(datasets) + " |")
            A("|---" * (len(datasets) + 1) + "|")
            for r in rows:
                if not r[field]:
                    continue
                cells = [c(r[field][metric]["by_dataset"].get(d, {}).get("accuracy"))
                         for d in datasets]
                A(f"| `{r['tag']}` | " + " | ".join(cells) + " |")
            A("")

    A("## BEDTime differentiation (every question at all 4 gold positions)")
    A("")
    A("A separate measurement from metrics 2-5: BEDTime ships each question in one fixed")
    A("shuffle, so the gold sits wherever `prepare.py` put it. Here each item is asked four")
    A("times with the gold moved to A, B, C, D and the distractors held in order, so an")
    A("item scores 0, .25, .5, .75 or 1 and a reader that merely likes a letter scores")
    A("exactly 0.25. `all 4` — correct at *every* position — is the number a fixed-shuffle")
    A("score cannot give you.")
    A("")
    A("| model | what it is | accuracy | all 4 | none |")
    A("|---|---|---|---|---|")
    for tag, what in DIFF_BASELINES:
        d = load(EP / "bedtime_diff_rot" / f"{tag}.json")
        if d:
            A(f"| `{tag}` | {what} | {c(d['accuracy'])} | {c(d['frac_all_four'])} | "
              f"{c(d['frac_none'])} |")
    for r in rows:
        if r["diff_acc"] is None:
            continue
        A(f"| `{r['tag']}` | {meta[r['tag']][0]} | {c(r['diff_acc'])} | "
          f"{c(r['diff_all4'])} | {c(r['diff_none'])} |")
    A("")
    A("**decod550 does not lead here.** That is the single most useful line in this file:")
    A("the arm that wins metrics 2-5 by a wide margin is mid-pack on differentiation, so")
    A("its advantage is specific to describing a series in a way that identifies it, and")
    A("is not a general improvement in chart reading.")
    A("")

    A("## BEDTime's own protocol (fixed shuffle, both task types)")
    A("")
    A("BEDTime as it ships: 11874 items, each with its options in the one shuffle")
    A("`prepare.py` produced. Kept because it is the published protocol and because it is")
    A("the only table `valmask400` appears in besides gen→gt. It is also the weakest of the")
    A("measurements here — a single gold position per item means part of each score is")
    A("placement luck, which is exactly what the rotated table above removes.")
    A("")
    A("`recognition` is reported but should not be read as a capability score: its negatives")
    A("paste another series' annotation on and label it False without checking that it is")
    A("false there, so a generic borrowed sentence makes the gold wrong. See")
    A("`bedtime_answer_examples.md`.")
    A("")
    A("| model | what it is | overall | differentiation | recognition |")
    A("|---|---|---|---|---|")
    for tag, what in DIFF_BASELINES:
        d = load(EP / "bedtime" / tag / "report.json")
        if d:
            t = d["by_task_type"]
            A(f"| `{tag}` | {what} | {c(d['overall_accuracy'])} | "
              f"{c(t.get('differentiation', {}).get('accuracy'))} | "
              f"{c(t.get('recognition', {}).get('accuracy'))} |")
    for tag, what, _size in ARMS:
        d = load(EP / "bedtime" / tag / "report.json")
        if not d:
            continue
        t = d["by_task_type"]
        A(f"| `{tag}` | {what} | {c(d['overall_accuracy'])} | "
          f"{c(t.get('differentiation', {}).get('accuracy'))} | "
          f"{c(t.get('recognition', {}).get('accuracy'))} |")
    A("")
    A("**The two differentiation protocols disagree, and the disagreement is unexplained.**")
    A("`teacher72b` is last among the six arms here (0.551) while leading the rotated table")
    A("(0.637); the 3B arms sit within 0.011 of each other here against a 0.030 spread")
    A("there. Same captions, same reader, same items — the only difference is that this")
    A("protocol asks each question at one gold position and the other asks all four. That")
    A("makes the rotated numbers the ones to quote, but it does not by itself explain the")
    A("reversal, and nothing in this repo currently does.")
    A("")

    A("## Position control")
    A("")
    A("Accuracy by which slot held the true option, over all four rotations of every")
    A("question. A flat row means the score is not a position prior; a peaked row means the")
    A("reader is partly guessing a letter.")
    A("")
    A("| model | bench | metric | A | B | C | D |")
    A("|---|---|---|---|---|---|---|")
    for r in rows:
        for field, bench in (("_bt", "BEDTime"), ("_ct", "CaTS")):
            if not r[field]:
                continue
            for metric in ("A", "B"):
                p = r[field][metric]["by_gold_position"]
                A(f"| `{r['tag']}` | {bench} | {metric} | "
                  + " | ".join(c(p[k]) for k in "ABCD") + " |")
    A("")
    worst = max((x for r in rows for x in (r["bt_unparsed"], r["ct_unparsed"])
                 if x is not None), default=0.0)
    A(f"Unparsed reader replies: at most {worst:.4f} of prompts across all models and both")
    A("benchmarks, so nothing here is decided by parse failures.")
    A("")

    A("## Paired comparisons")
    A("")
    A("Every model answers the same questions, so these are paired: each series is scored")
    A("0-1 by how many of its 4 rotations were right, and the sign test counts the series")
    A("where one model beats the other. Ties are uninformative and dropped. Compared")
    A("against `base` (what tuning bought) and against the strongest untuned model on each")
    A("benchmark's ser→cap (whether tuning beats simply using a bigger off-the-shelf VLM).")
    A("")
    A("| bench | metric | comparison | better | worse | p |")
    A("|---|---|---|---|---|---|")
    for field, rowsfile, bench in (("_bt", "bt_rows", "BEDTime"), ("_ct", "ct_rows", "CaTS")):
        scores = {r["tag"]: per_series(paths_for(r["tag"], r["group"])[rowsfile])
                  for r in rows if r[field]}
        key = "bt_b" if field == "_bt" else "ct_b"
        zoo_best = max((r for r in rows if r["group"] == "zoo" and r[key] is not None),
                       key=lambda r: r[key], default=None)
        refs = [t for t in ("base", zoo_best["tag"] if zoo_best else None) if t in scores]
        for metric in ("A", "B"):
            for tag in scores:
                for ref in refs:
                    if tag == ref:
                        continue
                    if tag in DEVIANT or ref in DEVIANT:
                        continue
                    pos, neg, p = sign_test(scores[tag], scores[ref], metric)
                    if pos + neg == 0:
                        continue
                    A(f"| {bench} | {metric} | {tag} vs {ref} | {pos} | {neg} | {p:.1e} |")
    A("")

    A("## Gaps")
    A("")
    A("Empty cells above, and why:")
    A("")
    missing = []
    for r in rows:
        for key, what in (("gen_gt", "gen→gt"), ("bt_a", "BEDTime MCQ"),
                          ("ct_a", "CaTS MCQ"), ("diff_acc", "differentiation")):
            if r[key] is None:
                missing.append((r["tag"], what))
    if missing:
        A("| model | missing | reason |")
        A("|---|---|---|")
        for tag, what in missing:
            if tag == "valmask400" and what == "differentiation":
                why = ("only the ROTATED sweep is missing; valmask400 does have BEDTime's "
                       "own fixed-shuffle differentiation score (0.620), in the table above")
            elif tag == "valmask400":
                why = ("the masked-reward ablation was run as a reward-design check against "
                       "decod550, on gen→gt and BEDTime's own protocol; the MCQ metrics "
                       "were never run for it")
            elif what == "differentiation":
                why = ("the rotated differentiation sweep was only run for the trained "
                       "arms; the zoo captions exist, so this is a scoring job away, not a "
                       "re-captioning one")
            else:
                why = "not run"
            A(f"| `{tag}` | {what} | {why} |")
    else:
        A("None — every model has every metric.")
    A("")
    A("Two organisations are absent from the zoo for environment reasons, not oversight:")
    A("the Mistral vision models fail under the pinned vLLM because `mistral_common` 1.11.0")
    A("moved `ImageChunk`, and Molmo's remote code requires TensorFlow.")
    A("")

    A("## Coverage and caveats")
    A("")
    A("- **BEDTime MCQ excludes sushi.** Its 2048-point series need ~57k tokens for a 4-way")
    A("  option set against the reader's 32k context. gen→gt and differentiation read only")
    A("  the caption, so they keep all four datasets.")
    A("- **CaTS gen→gt exists now but is quarantined**, see the section above. Its shipped")
    A("  references state dates, name the phenomenon and quote dataset statistics, so they")
    A("  were rewritten to strip the domain; the rewrite keeps values verbatim, and 99.9% of")
    A("  the results contain an exact number against BEDTime's 1.3%. Every chart-reading")
    A("  model lands at or below the empty-caption floor as a result.")
    A("- **CaTS distractors are pooled across domains**, not within: the domain label lives")
    A("  only in metadata the captioner never sees, and some domains hold fewer than 4")
    A("  series of a given length.")
    A("- The zoo captions come from a different runner (`caption_vlm_generic.py`) because")
    A("  those models need their own chat templates. Everything downstream of the caption —")
    A("  distractors, reader, prompt, rotations, scoring code — is shared.")
    A("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(L)} lines)")
    else:
        print(text)


if __name__ == "__main__":
    main()
