from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_domain(d: Path, arm: str) -> tuple[list[dict], dict[int, np.ndarray]]:
    pat = re.compile(rf"^summary_{re.escape(arm)}_k(\d+)\.json$")
    hits = [(int(m.group(1)), p) for p in d.glob(f"summary_{arm}_k*.json")
            if (m := pat.match(p.name))]
    rows = [json.loads(p.read_text()) for _, p in sorted(hits)]
    per_item: dict[int, np.ndarray] = {}
    for r in rows:
        k = r["n_options"]
        f = d / f"items_{arm}_k{k}.jsonl"
        if f.exists():
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            per_item[k] = np.asarray([x["acc_vs_chance"] for x in recs], dtype=float)
    return rows, per_item


def paired_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d = b - a
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d))
    return m, m - half, m + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", action="append", required=True,
                    help="label=dir[:arm], repeatable; arm defaults to <label>_decod550 "
                         "for benchmarks and decod550 for the RL val split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    domains = []
    for spec in args.domain:
        label, _, rest = spec.partition("=")
        path, _, arm = rest.partition(":")
        domains.append((label, Path(path), arm or "decod550"))

    loaded = []
    for label, path, arm in domains:
        rows, per_item = load_domain(path, arm)
        if not rows:
            raise SystemExit(f"no summary_{arm}_k*.json under {path}")
        loaded.append((label, rows, per_item))

    lines = ["# Reader difficulty vs candidate count, across domains", ""]
    lines.append("Same reader (Qwen2.5-14B-Instruct), same recipe, same nested-negatives "
                 "rule everywhere; the captions are decod550's in every domain. Only k moves.")
    lines.append("")

    ks = sorted({r["n_options"] for _, rows, _ in loaded for r in rows})
    head = "| k | chance |" + "".join(f" {lab} acc | {lab} acc_vs_chance |"
                                      for lab, _, _ in loaded)
    sep = "|---:|---:|" + "---:|" * (2 * len(loaded))
    lines += [head, sep]
    for k in ks:
        chance = 1.0 / k
        cells = ""
        for _, rows, _ in loaded:
            r = next((x for x in rows if x["n_options"] == k), None)
            cells += (f" {r['accuracy']:.4f} | {r['acc_vs_chance']:.4f} |" if r
                      else " n/a | n/a |")
        lines.append(f"| {k} | {chance:.3f} |{cells}")
    lines.append("")

    lines.append("Items scored per domain (the negatives rule keeps only series whose "
                 "same-length group can supply every candidate):")
    lines.append("")
    lines.append("| domain | n | mean caption chars |")
    lines.append("|---|---:|---:|")
    for lab, rows, _ in loaded:
        lines.append(f"| {lab} | {rows[0]['n']} | {rows[0]['mean_caption_chars']:.0f} |")
    lines.append("")

    lines += ["Paired change in acc_vs_chance against the smallest k, within each domain:",
              "", "| domain | " + " | ".join(f"k={k}" for k in ks[1:]) + " |",
              "|---|" + "---:|" * (len(ks) - 1)]
    for lab, _, per_item in loaded:
        if len(per_item) < 2:
            lines.append(f"| {lab} | " + " | ".join("n/a" for _ in ks[1:]) + " |")
            continue
        base = min(per_item)
        cells = []
        for k in ks[1:]:
            if k in per_item and k != base:
                m, lo, hi = paired_delta(per_item[base], per_item[k])
                cells.append(f"{m:+.4f} [{lo:+.3f}, {hi:+.3f}]")
            else:
                cells.append("n/a")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per two extra candidates, averaged over the ladder:")
    lines.append("")
    lines.append("| domain | mean Δ acc_vs_chance per +2 candidates |")
    lines.append("|---|---:|")
    for lab, rows, _ in loaded:
        by_k = {r["n_options"]: r["acc_vs_chance"] for r in rows}
        got = sorted(by_k)
        if len(got) < 2:
            continue
        slope = (by_k[got[-1]] - by_k[got[0]]) / ((got[-1] - got[0]) / 2)
        lines.append(f"| {lab} | {slope:+.4f} |")

    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
