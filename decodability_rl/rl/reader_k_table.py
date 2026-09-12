from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def paired_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = b - a
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d))
    return m, m - half, m + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--arm", default="decod550")
    ap.add_argument("--out", default=None, help="markdown path")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    d = Path(args.in_dir)
    pat = re.compile(rf"^summary_{re.escape(args.arm)}_k(\d+)\.json$")
    hits = [(int(m.group(1)), p) for p in d.glob(f"summary_{args.arm}_k*.json")
            if (m := pat.match(p.name))]
    if not hits:
        raise SystemExit(f"no summary_{args.arm}_k<N>.json in {d}")
    rows = [json.loads(p.read_text()) for _, p in sorted(hits)]

    per_item: dict[int, np.ndarray] = {}
    for r in rows:
        k = r["n_options"]
        f = d / f"items_{args.arm}_k{k}.jsonl"
        if f.exists():
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            per_item[k] = np.asarray([x["acc_vs_chance"] for x in recs], dtype=float)

    lines = [
        f"# Reader difficulty vs candidate count ({args.arm} captions)",
        "",
        f"Reader: {rows[0]['reader']}. One fixed caption set of {rows[0]['n']}, scored at "
        "every k; the distractors are nested (arm k's are a prefix of arm k+2's), so a "
        "larger k only ADDS candidates.",
        "",
        "| k | distractors | chance | accuracy | acc_vs_chance | frac_perfect | frac_zero | unparsed |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        p = r.get("parsing") or {}
        unp = f"{p['unparsed_frac']:.4f}" if p else "n/a"
        lines.append(
            f"| {r['n_options']} | {r['n_distractors']} | {r['chance']:.3f} | "
            f"{r['accuracy']:.4f} ± {r['accuracy_sem']:.4f} | "
            f"{r['acc_vs_chance']:.4f} ± {r['acc_vs_chance_sem']:.4f} | "
            f"{r['frac_perfect']:.3f} | {r['frac_zero']:.3f} | {unp} |"
        )

    if len(per_item) > 1:
        base = min(per_item)
        lines += ["", f"Paired change in acc_vs_chance, against k={base} on the same captions:",
                  "", "| comparison | Δ acc_vs_chance | 95% CI |", "|---|---:|---|"]
        for k in sorted(per_item):
            if k == base:
                continue
            m, lo, hi = paired_delta(per_item[base], per_item[k])
            lines.append(f"| k={base} → k={k} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        lines += [
            "",
            "A CI that excludes 0 means the extra candidates really did make the question "
            "harder for a caption that did not change. One that contains 0 means they were "
            "free, and the negatives ladder was not trading caption quality for difficulty.",
        ]

    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md)
    if args.csv:
        cols = ["arm", "reader", "n_options", "n_distractors", "chance", "n", "accuracy",
                "accuracy_sem", "acc_vs_chance", "acc_vs_chance_sem", "frac_perfect",
                "frac_zero", "mean_caption_chars", "seconds"]
        out = [",".join(cols + ["unparsed_frac"])]
        for r in rows:
            p = r.get("parsing") or {}
            out.append(",".join([str(r.get(c, "")) for c in cols]
                                + [str(p.get("unparsed_frac", ""))]))
        Path(args.csv).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
