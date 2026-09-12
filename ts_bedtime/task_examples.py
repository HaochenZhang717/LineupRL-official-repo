from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_eval.benchmarks import bedtime
from ts_eval.vlm_direct_step import build_direct_prompt

REPO = Path(__file__).resolve().parents[1]
BEDTIME = REPO / "results" / "eval_protocol" / "bedtime"
DIRECT = REPO / "results" / "eval_protocol" / "bedtime_direct" / "bedtime"
OUT = REPO / "results" / "eval_protocol" / "bedtime_task_examples.md"

DIRECT_LEGS = [
    ("base", "untuned 3B, sees the chart"),
    ("base_noimg", "untuned 3B, chart REMOVED (control)"),
    ("decod550", "decodability arm, sees the chart"),
    ("decod550_noimg", "decodability arm, chart REMOVED (control)"),
]
NLI_ARMS = [("base", "untuned"), ("sft", "SFT"), ("decod550", "decodability arm")]


def load_jsonl_by(path: Path, key: str) -> dict:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r[key]] = r
    return out


def load_captions(arm: str) -> dict[str, str]:
    caps = {}
    with open(BEDTIME / arm / "captions.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            caps.setdefault(r["caption_key"], r["caption"])
    return caps


def legible(it) -> bool:
    return 8 <= len(it.series) <= 60 and len(set(it.series)) > 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-direct", type=int, default=2, help="examples per task type")
    ap.add_argument("--n-nli", type=int, default=3)
    args = ap.parse_args()

    items = {i.id: i for i in bedtime.load(split="test", limit=None)}
    direct = {leg: load_jsonl_by(DIRECT / leg / "predictions.jsonl", "item_id")
              for leg, _ in DIRECT_LEGS}

    L: list[str] = []
    A = L.append
    A("# BEDTime — the other two evaluation families")
    A("")
    A("Protocol A (answering from the caption) is in `bedtime_answer_examples.md`. This")
    A("file covers the two it does not: **Protocol B**, where the captioner answers the")
    A("question itself from the chart with no caption in between, and **Task 3**, where")
    A("there is no question at all and the caption is scored against a reference")
    A("description by an NLI model.")
    A("")

    A("## Protocol B — the model answers from the chart directly")
    A("")
    A("No caption, no text-only answerer: the captioner checkpoint is shown the PNG and")
    A("the question, and answers itself. The `_noimg` legs send the identical prompt with")
    A("the image removed, which measures how much of the score the question wording alone")
    A("gives away.")
    A("")
    A("This is the control on the whole caption-mediated idea. If a model answers better")
    A("directly than through its own caption, the caption is a lossy channel.")
    A("")
    A("**Read the `_noimg` floor carefully.** Measured over all recognition items, with the")
    A("chart removed the model answers `True` to **99%** of them, which on a balanced")
    A("True/False set scores exactly 0.50. So the no-image number is not \"half the answer")
    A("is in the wording\" -- it is the arithmetic of a degenerate always-True responder.")
    A("The examples below show this directly: the same chart's positive and negative item")
    A("both get `True` without the image.")
    A("")
    A("A weaker version survives WITH the chart: `True` is still 74% of answers against a")
    A("50% base rate, for both the untuned model and the decodability arm. Recognition")
    A("accuracy is therefore carried partly by a yes-bias, in every leg equally.")
    A("")
    A("The prompt is `build_direct_prompt(item)` — the same question text and the same")
    A("response-format line Protocol A appends, so a gap between protocols is the evidence")
    A("and not the wording. There is no system turn, matching the arms' training")
    A("distribution.")
    A("")

    chosen: list[str] = []
    for task in ("recognition", "differentiation"):
        ids = sorted(i for i, it in items.items()
                     if it.task_type == task and legible(it)
                     and all(i in direct[leg] for leg, _ in DIRECT_LEGS))
        contrast = [i for i in ids
                    if direct["decod550"][i]["score"] != direct["decod550_noimg"][i]["score"]]
        chosen += (contrast or ids)[: args.n_direct]

    for n, item_id in enumerate(chosen, 1):
        item = items[item_id]
        A(f"### B{n} — {item.task_type}")
        A("")
        A(f"- item: `{item_id}` | dataset: {item.domain} | gold: **{item.gold}**")
        A(f"- chart shown to the model: `bedtime_{item.extra['series_uid']}.png`")
        A("")
        A("Underlying series (the model sees only the rendered chart of this):")
        A("")
        A("```")
        A(", ".join(f"{v:g}" for v in item.series))
        A("```")
        A("")
        A("**Input sent with the chart image:**")
        A("")
        A("```text")
        A(build_direct_prompt(item))
        A("```")
        A("")
        A("| leg | image? | raw output | gold | score |")
        A("|---|---|---|---|---|")
        for leg, what in DIRECT_LEGS:
            r = direct[leg][item_id]
            has_img = "no" if leg.endswith("_noimg") else "yes"
            A(f"| `{leg}` | {has_img} | `{r['answer_raw']}` | {item.gold} | {r['score']} |")
        A("")

    A("## Task 3 — open generation, scored by NLI")
    A("")
    A("There is no question and no multiple choice. The captioner writes a description of")
    A("the chart, and an NLI model (`tasksource/deberta-base-long-nli`) is asked whether")
    A("that description entails the dataset's reference sentence, and separately whether")
    A("the reference entails the description. A pair counts as entailed at P >= 0.5.")
    A("")
    A("| direction | premise | hypothesis | what it asks |")
    A("|---|---|---|---|")
    A("| `gen→gt` | the generated caption | the reference sentence | did it say the true thing |")
    A("| `gt→gen` | the reference sentence | the generated caption | did it say ONLY true things |")
    A("")
    A("`gt→gen` is near-impossible by construction here: the references average 54")
    A("characters and the captions run to hundreds, and a short sentence cannot imply a")
    A("long one. That is why that column sits at 0.04-0.09 for every arm.")
    A("")

    nli = {arm: load_jsonl_by(BEDTIME / arm / "generation_deployed.jsonl", "series_uid")
           for arm, _ in NLI_ARMS}
    caps = {arm: load_captions(arm) for arm, _ in NLI_ARMS}
    uids = sorted(u for u in nli["decod550"]
                  if all(u in nli[a] and u in caps[a] for a, _ in NLI_ARMS)
                  and not u.startswith("sushi"))
    split = [u for u in uids
             if (nli["decod550"][u]["p_gen_entails_gt"] >= 0.5)
             != (nli["base"][u]["p_gen_entails_gt"] >= 0.5)]
    for n, uid in enumerate((split or uids)[: args.n_nli], 1):
        A(f"### T{n} — series `{uid}`")
        A("")
        A(f"**Reference sentence (the hypothesis for `gen→gt`):**")
        A("")
        A(f"> {nli['decod550'][uid]['ground_truth']}")
        A("")
        for arm, what in NLI_ARMS:
            r = nli[arm][uid]
            A(f"**`{arm}` ({what}) — the generated caption, used as the premise:**")
            A("")
            A("```text")
            A(caps[arm][uid].strip())
            A("```")
            A("")
            A(f"P(caption entails reference) = **{r['p_gen_entails_gt']:.3f}** "
              f"→ {'entailed' if r['p_gen_entails_gt'] >= 0.5 else 'not entailed'}  |  "
              f"P(reference entails caption) = {r['p_gt_entails_gen']:.3f}")
            A("")

    A("### A caveat this scorer has")
    A("")
    A("Entailment is monotone in the premise: adding text cannot destroy a conclusion that")
    A("already followed. This scorer does not obey that. Duplicating a caption verbatim")
    A("adds no information and collapses the entailment rate from 0.227 to 0.043, while")
    A("appending irrelevant boilerplate changes nothing at all")
    A("(`ts_bedtime/nli_length_probe.py`). Treat the ordering as meaningful and the")
    A("magnitudes as a similarity score wearing an entailment label.")
    A("")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
