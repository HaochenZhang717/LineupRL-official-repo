from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EP = REPO / "results" / "eval_protocol"
ZOO = EP / "vlm_zoo"

ARMS = [
    ("base", "untuned Qwen2.5-VL-3B", "3B", "arm"),
    ("sft", "SFT from the 72B teacher", "3B", "arm"),
    ("rl650", "RL v1, self-generated MCQs", "3B", "arm"),
    ("decod550", "RL v2, decodability", "3B", "arm"),
    ("teacher72b", "the 72B teacher", "72B", "arm"),
]
ZOO_MODELS = [
    ("phi4mm", "Phi-4-multimodal (Microsoft)", "5.6B"),
    ("idefics3-8b", "Idefics3 (HuggingFace)", "8B"),
    ("qwen3vl-8b", "Qwen3-VL (Alibaba)", "8B"),
    ("gemma3-12b", "Gemma-3 (Google)", "12B"),
    ("internvl3-14b", "InternVL3 (Shanghai AI Lab)", "14B"),
]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def row_for(tag: str, kind: str) -> dict:
    if kind == "arm":
        nli = load(EP / "bedtime" / tag / "generation_deployed.json")
        bt = load(EP / "bedtime_mcq" / f"{tag}.json")
        ct = load(EP / "cats_mcq" / f"{tag}.json")
    else:
        nli = load(ZOO / "nli_bedtime" / f"{tag}_deployed.json")
        bt = load(ZOO / "mcq_bedtime" / f"{tag}.json")
        ct = load(ZOO / "mcq_cats" / f"{tag}.json")
    return {
        "gen_gt": nli["overall"]["gen_entails_gt"] if nli else None,
        "chars": nli["overall"]["mean_caption_chars"] if nli else None,
        "bt_a": bt["A"]["accuracy"] if bt and "A" in bt else None,
        "bt_b": bt["B"]["accuracy"] if bt and "B" in bt else None,
        "ct_a": ct["A"]["accuracy"] if ct and "A" in ct else None,
        "ct_b": ct["B"]["accuracy"] if ct and "B" in ct else None,
    }


def c(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L: list[str] = []
    A = L.append
    A("# Caption metrics — trained arms and untuned VLMs")
    A("")
    A("")
    A("> Part of the picture. `bedtime_cats_all_models.md` holds every model and every\n"
      "> BEDTime/CaTS metric in one table, generated from the same JSON as this file.")
    A("")
    A("| model | what | size | gen→gt | BEDTime cap→ser | BEDTime ser→cap "
      "| CaTS cap→ser | CaTS ser→cap |")
    A("|---|---|---|---|---|---|---|---|")
    for tag, what, size, kind in ARMS:
        r = row_for(tag, kind)
        A(f"| `{tag}` | {what} | {size} | {c(r['gen_gt'])} | {c(r['bt_a'])} | "
          f"{c(r['bt_b'])} | {c(r['ct_a'])} | {c(r['ct_b'])} |")
    A("| | | | | | | | |")
    for tag, what, size in ZOO_MODELS:
        r = row_for(tag, "zoo")
        A(f"| `{tag}` | {what} | {size} | {c(r['gen_gt'])} | {c(r['bt_a'])} | "
          f"{c(r['bt_b'])} | {c(r['ct_a'])} | {c(r['ct_b'])} |")
    A("")
    A("Chance is 0.25 on the four MCQ columns. gen→gt has no chance level: its floor is an")
    A("empty caption (0.003) and its ceiling the reference against itself (1.000).")
    A("")
    A("**BEDTime cap→ser is decod550's training objective**, asked of the same frozen 14B")
    A("reader it trained against. Every other column is a transfer measurement, and")
    A("**ser→cap is the one no model was trained on** — the column to quote.")
    A("")
    A("Coverage: BEDTime MCQ excludes sushi (2048-point series overflow the reader's 32k")
    A("context in a 4-way option set); gen→gt covers all 2000 series; CaTS covers 2969.")
    A("Distractors come from `build_negatives`, a pure function of the series ids and")
    A("values, so every row above answers byte-identical questions.")
    A("")
    A("Two organisations are absent for environment reasons, not oversight: the Mistral")
    A("vision models fail under the pinned vLLM because mistral_common 1.11.0 moved")
    A("`ImageChunk`, and Molmo's remote code requires TensorFlow.")
    A("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
