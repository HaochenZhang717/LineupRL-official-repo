from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MIN_CHARS = 80


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def classify(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keep, drop = [], []
    for r in rows:
        cap = (r.get("caption") or "").strip()
        if len(cap) < MIN_CHARS:
            r["_reason"] = f"too short ({len(cap)} chars)"
            drop.append(r)
        elif not cap.endswith((".", "!", "?", '"', ")")):
            r["_reason"] = "no sentence-final punctuation (likely truncated at max-tokens)"
            drop.append(r)
        else:
            r["caption"] = cap
            keep.append(r)
    return keep, drop


def stats(rows: list[dict]) -> dict:
    caps = [r["caption"] for r in rows]
    chars = sorted(len(c) for c in caps)
    words = sorted(len(c.split()) for c in caps)
    openings = Counter(" ".join(c.split()[:8]) for c in caps)
    top_open, top_n = openings.most_common(1)[0]

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return {
        "n": len(caps),
        "chars_p10": pct(chars, 0.10), "chars_median": pct(chars, 0.50),
        "chars_p90": pct(chars, 0.90), "chars_max": chars[-1],
        "words_median": pct(words, 0.50),
        "distinct_openings": len(openings),
        "top_opening_rate": round(top_n / len(caps), 4),
        "top_opening": top_open[:90],
        "exact_duplicate_rate": round(1 - len(set(caps)) / len(caps), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="dir holding captions_{train,val}.jsonl")
    ap.add_argument("--n-samples", type=int, default=8, help="captions to quote in the QC md")
    args = ap.parse_args()
    d = Path(args.dir)

    report = ["# Teacher caption QC\n",
              f"Source: `{d}`  · rejects go to `rejected_<split>.jsonl`\n"]
    for split in ("train", "val"):
        src = d / f"captions_{split}.jsonl"
        if not src.exists():
            print(f"skip {split}: {src} missing")
            continue
        rows = load(src)
        keep, drop = classify(rows)
        with open(d / f"sft_{split}.jsonl", "w", encoding="utf-8") as fh:
            for r in keep:
                fh.write(json.dumps({"id": r["id"], "image": r["image"],
                                     "caption": r["caption"]}, ensure_ascii=False) + "\n")
        if drop:
            with open(d / f"rejected_{split}.jsonl", "w", encoding="utf-8") as fh:
                for r in drop:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = stats(keep)
        print(f"{split}: {len(rows)} in -> {len(keep)} kept, {len(drop)} dropped", flush=True)

        report.append(f"\n## {split}: {len(rows)} generated, "
                      f"**{len(keep)} kept**, {len(drop)} dropped\n")
        if drop:
            reasons = Counter(r["_reason"].split(" (")[0] for r in drop)
            report.append("Drop reasons: " +
                          ", ".join(f"`{k}` ×{v}" for k, v in reasons.most_common()) + "\n")
        report.append("\n| metric | value |\n|---|---|\n")
        for k, v in s.items():
            report.append(f"| {k} | {v} |\n")
        report.append(f"\n<details><summary>{args.n_samples} sample captions</summary>\n\n")
        for r in keep[:args.n_samples]:
            report.append(f"**id {r['id']}** (`{Path(r['image']).name}`)\n\n> "
                          + r["caption"].replace("\n", "\n> ") + "\n\n")
        report.append("</details>\n")

    (d / "teacher_qc.md").write_text("".join(report), encoding="utf-8")
    print(f"wrote {d / 'teacher_qc.md'}")


if __name__ == "__main__":
    main()
