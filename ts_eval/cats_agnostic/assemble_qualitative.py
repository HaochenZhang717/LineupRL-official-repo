from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WORK = Path("work/cats_qual")
K = 100
DIGIT = re.compile(r"\d")
WORDNUM = re.compile(r"\b(percent|per cent)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write the file even if some references still contain digits. "
                         "For inspecting a partial run, never for scoring.")
    args = ap.parse_args()
    work = Path(args.work)

    ids = [json.loads(l)["id"] for l in open(work / "items.jsonl", encoding="utf-8")]
    out, problems, dirty = [], [], []
    for b in range((len(ids) + K - 1) // K):
        exp = ids[b * K:(b + 1) * K]
        p = work / "shards" / f"batch_{b:03d}.jsonl"
        try:
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        except Exception as e:
            problems.append(f"batch {b:03d}: {e}")
            continue
        got = [r.get("id") for r in rows]
        if got != exp:
            i = next((j for j, (a, c) in enumerate(zip(got, exp)) if a != c), None)
            problems.append(f"batch {b:03d}: ids differ (n={len(got)} vs {len(exp)}, "
                            f"first mismatch at {i})")
            continue
        for r in rows:
            t = r.get("rewritten", "")
            if not t.strip():
                problems.append(f"{r['id']}: empty rewritten")
            elif DIGIT.search(t) or WORDNUM.search(t):
                dirty.append((r["id"], t))
        out.extend(rows)

    print(f"joined {len(out)}/{len(ids)} records; {len(problems)} structural problems, "
          f"{len(dirty)} still containing a number")
    for p in problems[:20]:
        print("  !", p)
    for i, t in dirty[:10]:
        print(f"  # {i}: {t[:110]}")

    if problems:
        raise SystemExit("structural problems -- fix the shards and re-run")
    if dirty and not args.allow_dirty:
        raise SystemExit(f"{len(dirty)} references still contain a number. Re-run those "
                         f"shards; do not strip the digits mechanically, which would leave "
                         f"the sentence broken.")
    assert [r["id"] for r in out] == ids, "ORDER BROKEN"
    print("order check: ids match items.jsonl exactly, in order  OK")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({"id": r["id"], "rewritten": r["rewritten"].strip()},
                               ensure_ascii=False) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
