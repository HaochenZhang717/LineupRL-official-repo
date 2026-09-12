from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LETTERS = ("A", "B", "C", "D")
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
COND_LABEL = {
    "caption": ("caption", "the VLM's caption for **this** series <- what the experiment measures"),
    "ds_caption": ("ds_caption", "the dataset's own caption (stronger caption reference)"),
    "oracle": ("oracle", "**programmatic, guaranteed-decisive** description (reader ceiling)"),
    "mismatched": ("mismatched", "the caption of a **different** series (pairing control)"),
    "empty": ("empty", "no description at all (leakage control)"),
}
COND_ORDER = ("oracle", "ds_caption", "caption", "mismatched", "empty")
PLOT_LABEL = {
    "true": "TRUE series",
    "shuffle": "distractor: shuffled",
    "reverse": "distractor: time-reversed",
    "segment_swap": "distractor: segments swapped",
}


def plot_options(rec: dict, out_path: Path) -> None:
    series = rec["display_series"]
    lo = min(min(s) for s in series)
    hi = max(max(s) for s in series)
    pad = 0.08 * ((hi - lo) or 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 4.6), dpi=110)
    for ax, vals, kind in zip(axes.ravel(), series, rec["display_kinds"]):
        is_true = kind == "true"
        ax.plot(range(len(vals)), vals, color="#1f77b4", lw=1.4, marker="o", ms=2.5)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_title(
            PLOT_LABEL.get(kind, "distractor: another real series"),
            fontsize=9,
            color="#b8860b" if is_true else "#333333",
            fontweight="bold" if is_true else "normal",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#b8860b" if is_true else "#cccccc")
            spine.set_linewidth(1.8 if is_true else 0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--results", default=str(here / "results" / "results.json"))
    ap.add_argument("--image-root", default=str(here.parents[1] / "out_10k_xdomain"))
    ap.add_argument("--title", default="test_reward_design: decodability reward feasibility pilot")
    args = ap.parse_args()

    res_path = Path(args.results)
    data = json.loads(res_path.read_text())
    summary, records = data["summary"], data["records"]
    out_dir = res_path.parent
    figs = out_dir / "figs"
    figs.mkdir(exist_ok=True)

    import shutil

    for i, rec in enumerate(records, 1):
        src = Path(args.image_root) / rec["image_path"]
        if src.exists():
            shutil.copyfile(src, figs / f"series_{rec['id']}.png")
        plot_options(rec, figs / f"q{i:02d}_options.png")

    L: list[str] = []
    A = L.append
    cfg = summary["config"]

    A(f"# {args.title}\n")
    A(
        "**What is being tested**: if a caption really describes a series, then reading the "
        "caption alone should be enough to pick that series out of several candidates whose "
        "values are nearly identical and only differ in temporal order. That identification "
        "probability is the quantity we intend to use as an RL reward; this pilot checks "
        "whether it discriminates at all.\n"
    )

    A("## Setup\n")
    A(f"- **captioner**: `{Path(cfg['captioner']).name}`, sees only the rendered line chart, "
      "greedy decoding")
    A(
        f"- **discriminator (answerer)**: `{Path(cfg['discriminator']).name}`, **text only**, "
        "sees a description plus the raw values of four candidate series, never a chart. "
        "Reading values rather than charts keeps its own chart perception out of the reward."
    )
    A(
        f"- **{summary['n_questions']} series / {summary['n_questions']} 4-way questions** "
        f"(series length {records[0]['len']}, seed {cfg['seed']}), chance = 25%"
    )
    real_mode = any(k.startswith("real_") for k in records[0]["display_kinds"])
    if real_mode:
        A(
            "- **The three distractors are three other real series**, the nearest neighbours "
            "after standardising (mean, std, min, max), i.e. the hardest negatives available. "
            "All four candidates are real series, so there is no \"one original + three "
            "corrupted copies\" structure to reverse-engineer."
        )
    else:
        A(
            "- **The three distractors are rearrangements of the true series** (shuffled / "
            "time-reversed / segments swapped): the four options have **exactly the same "
            "multiset of values**, so \"values around 38\" or \"range 12 to 18\" earns nothing; "
            "only saying what happens when can win."
        )
    A(
        f"- **Each question is asked 4 times**, with the true series at A, B, C and D in turn, "
        f"and the score averaged ({summary['n_calls_per_condition']} calls per condition). "
        "**This rotation is forced by the first version of the experiment**: asked once with "
        "the gold at a random position, the discriminator answered A on 8 of 10 questions and "
        "the empty-caption control scored 50% -- pure option-position preference. With "
        "rotation a pure position prior scores exactly 25%.\n"
    )

    A("## 0. Overall results\n")
    A("| condition | meaning | accuracy | mean p(gold) | letters picked |")
    A("|---|---|---:|---:|---|")
    for cond in COND_ORDER:
        if cond not in summary["conditions"]:
            continue
        s = summary["conditions"][cond]
        name, desc = COND_LABEL[cond]
        picks = " ".join(f"{k}x{v}" for k, v in s["pick_distribution"].items() if k)
        A(f"| **{name}** | {desc} | **{s['accuracy']:.1%}** | {s['mean_p_gold']:.3f} | {picks} |")
    A("")
    A("> `p(gold)`: the discriminator's first-token probability mass on the correct option, "
      "normalised over A/B/C/D.")
    A("> A binary right/wrong reward produces many zero-variance RLOO groups; this continuous "
      "quantity is what a reward would use.\n")

    cap_acc = summary["conditions"]["caption"]["accuracy"]
    ora_acc = summary["conditions"]["oracle"]["accuracy"]
    A("**How to read the table**\n")
    A(
        f"1. **oracle reaches only {ora_acc:.1%}** -- the most important number. The oracle "
        "description is **generated from the raw values** and states the first/last values, "
        "the max/min with their time indices and the four quarter means, which is enough "
        "information to **uniquely determine** the true series (first/last values rule out "
        "the reversal, quarter means rule out the shuffle and the segment swap). A reader "
        f"that understands it should be near 100%. At {ora_acc:.1%}, **the bottleneck is the "
        "discriminator, not the caption**."
    )
    A(
        f"2. **caption condition {cap_acc:.1%}, about chance** -- with the oracle at only "
        f"{ora_acc:.1%} this number has no explanatory power: it could mean the caption "
        "carries no information or that the discriminator cannot read it, and this experiment "
        "cannot tell the two apart."
    )
    A(
        "3. **The discriminator's letter distribution is heavily skewed** (last column): even "
        "with 4 rotations it puts most of its mass on one letter, i.e. it is largely **not "
        "reading the candidates at all**.\n"
    )

    wrong = Counter()
    for rec in records:
        for rot in rec["rotations"]:
            ans = rot["conditions"]["caption"]
            if not ans["correct"] and ans["letter"] in LETTERS:
                wrong[rot["option_kinds"][LETTERS.index(ans["letter"])]] += 1
    if wrong:
        A("**Which distractor was picked when the caption condition was wrong**\n")
        A("| distractor picked | count |")
        A("|---|---:|")
        for kind, cnt in wrong.most_common():
            A(f"| {kind_label(kind)} | {cnt} |")
        A("")

    A("---\n")
    A(f"## 1. Captions of the {summary['n_questions']} series\n")
    A("Captioner prompt (identical for every series):\n")
    A("```text")
    A("[system] You are an analyst who describes and interprets time series.")
    A("[user]   <image> This image is a line chart of a time series. Please describe this time series.")
    A("```\n")
    for i, rec in enumerate(records, 1):
        A(f"### Series {i} (id={rec['id']}, length {rec['len']})\n")
        A(f"![series {rec['id']}](figs/series_{rec['id']}.png)\n")
        A(f"**captioner ({Path(cfg['captioner']).name}) output:**\n")
        A("```")
        A(rec["caption"].strip())
        A("```\n")
        A("<details><summary>Oracle description of the same series (programmatic; reader ceiling)</summary>\n")
        A("```")
        A(rec["oracle_caption"])
        A("```")
        A("</details>\n")

    A("---\n")
    A("## 2. The four candidates of each question, and the discriminator's answers\n")
    A(
        "Each figure shows the four candidates of a question (gold = true series). **The "
        "discriminator sees neither the figure nor the labels**; it only gets four value "
        "lists. Each row of the table below is one rotation: where the true series was "
        "placed, what the discriminator answered, and whether it was right.\n"
    )
    for i, rec in enumerate(records, 1):
        acc = rec["scores"]["caption"]["accuracy"]
        mark = ("all correct" if acc == 1 else
                ("all wrong" if acc == 0 else f"partly correct ({acc:.0%})"))
        A(f"### Question {i} (series id={rec['id']}) -- caption condition: {mark}\n")
        A(f"![q{i} options](figs/q{i:02d}_options.png)\n")
        A("| true series at | discriminator answer | correct | what it actually picked | p(gold) |")
        A("|---|---|---|---|---:|")
        for rot in rec["rotations"]:
            ans = rot["conditions"]["caption"]
            picked = (
                kind_label(rot["option_kinds"][LETTERS.index(ans["letter"])])
                if ans["letter"] in LETTERS
                else "(no letter parsed)"
            )
            A(
                f"| {rot['gold_letter']} | {ans['letter']} | "
                f"{'ok' if ans['correct'] else 'X'} | {picked} | {ans['p_gold']:.3f} |"
            )
        A("")
        others = "  ".join(
            f"{COND_LABEL[c][0]} {rec['scores'][c]['accuracy']:.0%}"
            for c in COND_ORDER
            if c in rec["scores"] and c != "caption"
        )
        A(f"- This question under the other conditions (mean over 4 rotations): {others}")
        A("")
        A("<details><summary>The four value lists the discriminator saw (rotation with the true series at A)</summary>\n")
        rot0 = rec["rotations"][0]
        for letter, text, kind in zip(LETTERS, rot0["option_texts"], rot0["option_kinds"]):
            A(f"`{letter})` ({kind_label(kind)}) `{text}`\n")
        A("</details>\n")

    A("---\n")
    A("## 3. Example discriminator prompt (question 1, true series at A)\n")
    A("```text")
    A(records[0]["prompt"])
    A("```\n")

    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out_dir/'report.md'} and {len(records)*2} figures")


if __name__ == "__main__":
    main()
