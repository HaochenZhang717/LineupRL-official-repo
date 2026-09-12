from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "results" / "decodability_rl" / "reader_k_sweep"


def score(url: str, rows: list[dict], n_options: int, batch: int, scale: float,
          timeout: float) -> np.ndarray:
    acc: list[float] = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        prompts = [[r["caption"], [[json.dumps(r["payload"], separators=(",", ":")), ""]]]
                   for r in chunk]
        resp = requests.post(
            url,
            json={"prompts": prompts, "query": [], "labels": [], "full_rotations": True},
            timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        served = int(body.get("n_options", n_options))
        if served != n_options:
            raise SystemExit(
                f"server is running --n-options {served} but this sweep was asked for "
                f"{n_options}; start one server per k or fix --n-options"
            )
        got = body.get("accuracies")
        if got and len(got) == len(chunk):
            acc += list(got)
        else:
            acc += [r / scale for r in body["rewards"]]
        print(f"    {len(acc)}/{len(rows)}", flush=True)
    return np.asarray(acc, dtype=float)


UNPARSED_RE = re.compile(r"unparsed=(\d+)/(\d+)")


def unparsed_from_log(path: Path) -> dict | None:
    if not path.exists():
        return None
    bad = tot = seen = 0
    for m in UNPARSED_RE.finditer(path.read_text(errors="replace")):
        bad += int(m.group(1))
        tot += int(m.group(2))
        seen += 1
    if not tot:
        return None
    return {"unparsed": bad, "questions_logged": tot,
            "unparsed_frac": bad / tot, "batches_logged": seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--n-options", type=int, required=True,
                    help="must match the --n-options the server was started with")
    ap.add_argument("--items", required=True,
                    help="jsonl of {id, caption, payload} where payload carries ALL nine "
                         "distractors; the server keeps the first n_options-1")
    ap.add_argument("--arm", default="decod550", help="which caption set this is")
    ap.add_argument("--reader", default="Qwen2.5-14B-Instruct", help="label only")
    ap.add_argument("--server-log", default=None,
                    help="the server's stdout, for the unparsed rate")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()]
    short = [r["id"] for r in rows if len(r["payload"]["distractors"]) < args.n_options - 1]
    if short:
        raise SystemExit(
            f"{len(short)} items carry fewer than {args.n_options - 1} distractors "
            f"(first: id {short[0]}); rebuild the items file with nine"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.arm} | k={args.n_options} | {len(rows)} captions | {args.items}",
          flush=True)

    t0 = time.time()
    acc = score(args.url, rows, args.n_options, args.batch, args.reward_scale, args.timeout)
    elapsed = time.time() - t0

    chance = 1.0 / args.n_options
    vs_chance = (acc - chance) / (1.0 - chance)

    with (out_dir / f"items_{args.arm}_k{args.n_options}.jsonl").open("w") as fh:
        for r, a, c in zip(rows, acc, vs_chance):
            fh.write(json.dumps({"id": r["id"], "acc": float(a),
                                 "acc_vs_chance": float(c)}) + "\n")

    row = {
        "arm": args.arm,
        "reader": args.reader,
        "n_options": args.n_options,
        "n_distractors": args.n_options - 1,
        "chance": chance,
        "n": int(len(acc)),
        "accuracy": float(acc.mean()),
        "accuracy_sem": float(acc.std(ddof=1) / np.sqrt(len(acc))),
        "acc_vs_chance": float(vs_chance.mean()),
        "acc_vs_chance_sem": float(vs_chance.std(ddof=1) / np.sqrt(len(acc))),
        "reward_mean": float(acc.mean() * args.reward_scale),
        "frac_perfect": float((acc >= 0.999).mean()),
        "frac_zero": float((acc <= 0.001).mean()),
        "mean_caption_chars": float(np.mean([len(r["caption"]) for r in rows])),
        "seconds": round(elapsed, 1),
    }
    if args.server_log:
        row["parsing"] = unparsed_from_log(Path(args.server_log))

    (out_dir / f"summary_{args.arm}_k{args.n_options}.json").write_text(
        json.dumps(row, indent=2)
    )
    print(f"  acc={row['accuracy']:.4f} (chance {chance:.3f})  "
          f"acc_vs_chance={row['acc_vs_chance']:.4f}  "
          f"perfect={row['frac_perfect']:.3f} zero={row['frac_zero']:.3f}  "
          f"[{elapsed:.0f}s]", flush=True)
    if row.get("parsing"):
        p = row["parsing"]
        print(f"  unparsed {p['unparsed']}/{p['questions_logged']} "
              f"({p['unparsed_frac']:.4f}) over {p['batches_logged']} logged batches",
              flush=True)


if __name__ == "__main__":
    main()
