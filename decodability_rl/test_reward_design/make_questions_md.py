from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LETTERS = ("A", "B", "C", "D")


def label_of(kind: str, ascii_only: bool = False) -> str:
    if kind == "true":
        return "TRUE series" if ascii_only else "**TRUE series**"
    if kind.startswith("real_"):
        sid = kind.split("(", 1)[-1].rstrip(")")
        return f"other real series {sid}" if ascii_only else f"another real series ({sid})"
    table = {
        "shuffle": ("shuffled", "true series, shuffled"),
        "reverse": ("time-reversed", "true series, time-reversed"),
        "segment_swap": ("segments swapped", "true series, two segments swapped"),
    }
    short, long = table.get(kind, (kind, kind))
    return short if ascii_only else long


def plot_panel(rec: dict, out_path: Path) -> None:
    series = rec["display_series"]
    kinds = rec["display_kinds"]
    lo = min(min(s) for s in series)
    hi = max(max(s) for s in series)
    pad = 0.08 * ((hi - lo) or 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 5), dpi=115)
    for ax, letter, vals, kind in zip(axes.ravel(), LETTERS, series, kinds):
        is_true = kind == "true"
        ax.plot(range(len(vals)), vals, color="#1f77b4", lw=1.4, marker="o", ms=2.5)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{letter})  {label_of(kind, ascii_only=True)}",
            fontsize=10,
            color="#b8860b" if is_true else "#333333",
            fontweight="bold" if is_true else "normal",
        )
        for spine in ax.spines.values():
            spine.set_edgecolor("#b8860b" if is_true else "#cccccc")
            spine.set_linewidth(2.0 if is_true else 0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--answers", required=True, help="strong reader, with the caption")
    ap.add_argument("--answers-empty", default=None, help="strong reader, no caption")
    ap.add_argument("--key", required=True)
    ap.add_argument("--image-root", default=str(here.parents[1] / "out_10k_xdomain"))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--caption-field",
        default="ds_caption",
        choices=["ds_caption", "caption"],
        help="which description the graded run was given: the dataset's own caption, "
        "or the untuned VLM's caption",
    )
    ap.add_argument("--caption-name", default="the dataset's own caption")
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    records = data["records"]
    local = data["summary"]["conditions"]
    key = json.loads(Path(args.key).read_text())
    ans = {a["file"]: a for a in json.loads(Path(args.answers).read_text())}
    ans_e = (
        {a["file"]: a for a in json.loads(Path(args.answers_empty).read_text())}
        if args.answers_empty
        else {}
    )

    out_path = Path(args.out)
    figs = out_path.parent / "figs_q"
    figs.mkdir(parents=True, exist_ok=True)

    import shutil

    by_q: dict[int, list] = {}
    for item in key:
        by_q.setdefault(item["question"], []).append(item)

    total = sum(1 for it in key if ans[it["file"]]["letter"] == it["gold_letter"])
    total_e = sum(
        1 for it in key if ans_e and ans_e[it["file"]]["letter"] == it["gold_letter"]
    )
    n = len(key)

    real_mode = any(k.startswith("real_") for k in records[0]["display_kinds"])

    L: list[str] = []
    A = L.append
    A("# Questions / option charts / accuracy\n")
    A(
        "One 4-way question per series: **given a description, pick the described series "
        "out of four candidates**. The candidates are given to the discriminator as raw value "
        "lists; it never sees a chart.\n"
    )
    A("**Distractors in this version**: " + (
        "three other **real series**, the nearest neighbours after standardising "
        "(mean, std, min, max).\n"
        if real_mode else
        "three rearrangements of the true series itself (shuffled / time-reversed / "
        "segments swapped).\n"
    ))
    A(
        "**Each question is asked 4 times**, with the true series at A, B, C and D in turn, "
        "and the score averaged -- so a pure position preference scores 25%. The letters in "
        "each figure below are the **first rotation's** (true series at A); the other three "
        "rotations show the same four candidates with the letters permuted.\n"
    )

    A("## Overall accuracy\n")
    A("| reader | description | accuracy |")
    A("|---|---|---:|")
    A(f"| **strong external reader** | {args.caption_name} | **{total}/{n} = {total/n:.1%}** |")
    if ans_e:
        A(f"| strong external reader | **no description** (leakage control) | "
          f"{total_e}/{n} = {total_e/n:.1%} |")
    A(f"| local Qwen2.5-3B-Instruct | the dataset's own caption | {local['ds_caption']['accuracy']:.1%} |")
    A(f"| local Qwen2.5-3B-Instruct | VLM-generated caption | {local['caption']['accuracy']:.1%} |")
    A(f"| local Qwen2.5-3B-Instruct | oracle (programmatic, guaranteed decisive) | {local['oracle']['accuracy']:.1%} |")
    A("| chance | -- | 25.0% |")
    A("")

    A("## Per-question accuracy\n")
    A("| q | series id | length | strong reader (with description) | strong reader (no description) |")
    A("|---:|---:|---:|---:|---:|")
    for q in sorted(by_q):
        items = by_q[q]
        c = sum(ans[it["file"]]["letter"] == it["gold_letter"] for it in items)
        ce = (
            sum(ans_e[it["file"]]["letter"] == it["gold_letter"] for it in items)
            if ans_e else None
        )
        A(
            f"| {q} | {items[0]['series_id']} | {records[q-1]['len']} | "
            f"{c}/4 = {c/4:.0%} | " + (f"{ce}/4 = {ce/4:.0%} |" if ans_e else "-- |")
        )
    A("")

    A("---\n")
    for q in sorted(by_q):
        rec = records[q - 1]
        items = sorted(by_q[q], key=lambda it: it["gold_letter"])
        c = sum(ans[it["file"]]["letter"] == it["gold_letter"] for it in items)

        src = Path(args.image_root) / rec["image_path"]
        if src.exists():
            shutil.copyfile(src, figs / f"series_{rec['id']}.png")
        plot_panel(rec, figs / f"q{q:02d}.png")

        A(f"## Question {q}: series id={rec['id']} (length {rec['len']}) -- accuracy {c}/4 = {c/4:.0%}\n")
        A(f"**Description given to the discriminator** ({args.caption_name})\n")
        A("```")
        A(rec[args.caption_field].strip())
        A("```\n")
        A("**The four candidates** (letters as in the first rotation)\n")
        A(f"![q{q} options](figs_q/q{q:02d}.png)\n")
        A("| option | what it is |")
        A("|---|---|")
        for letter, kind in zip(LETTERS, rec["display_kinds"]):
            A(f"| {letter} | {label_of(kind)} |")
        A("")
        A("**Discriminator answers** (4 position rotations)\n")
        A("| rotation | true series at | strong reader | correct | no-caption answer | correct |")
        A("|---:|---|---|---|---|---|")
        for r_i, it in enumerate(items, 1):
            a = ans[it["file"]]
            ok = a["letter"] == it["gold_letter"]
            if ans_e:
                ae = ans_e[it["file"]]
                oke = ae["letter"] == it["gold_letter"]
                tail = f"| {ae['letter']} | {'ok' if oke else 'X'} |"
            else:
                tail = "| -- | -- |"
            A(
                f"| {r_i} | {it['gold_letter']} | {a['letter']} | "
                f"{'ok' if ok else 'X'} " + tail
            )
        A("")
        A("<details><summary>The four value lists the discriminator actually saw (first rotation)</summary>\n")
        rot0 = rec["rotations"][0]
        for letter, text, kind in zip(LETTERS, rot0["option_texts"], rot0["option_kinds"]):
            A(f"`{letter})` {label_of(kind)}\n")
            A(f"`{text}`\n")
        A("</details>\n")
        A("**Original line chart** (the one the captioner saw)\n")
        A(f"![series {rec['id']}](figs_q/series_{rec['id']}.png)\n")
        A("---\n")

    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out_path} ({total}/{n} = {total/n:.1%})")


if __name__ == "__main__":
    main()
