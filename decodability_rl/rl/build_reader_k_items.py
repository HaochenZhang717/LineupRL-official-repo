from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from decodability_rl.test_reward_design import prompts as P

DEFAULT_CAPTIONS = None
DEFAULT_OUT = REPO / "out_10k_xdomain" / "reader_k_items"


def read_val_payloads(val_jsonl: Path) -> list[dict]:
    out = []
    for line in val_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        msg = json.loads(json.loads(line)["message"])
        payload, _ = eval(msg[2]["content"])[0]
        out.append(json.loads(payload))
    return out


def negatives_by_uid(rows: list[dict], n_distractors: int, max_overlap_frac: float,
                     uid_key: str = "series_uid") -> tuple[dict[str, list], dict]:
    from decodability_rl.rl.build_decodability_dataset import longest_common_run
    from decodability_rl.test_reward_design.hard_negatives import summary_features

    by_len: dict[int, list[dict]] = {}
    for r in rows:
        by_len.setdefault(len(r["series"]), []).append(r)

    out: dict[str, list] = {}
    dropped_small_group = dropped_negs = 0
    for length, group in sorted(by_len.items()):
        if len(group) <= n_distractors:
            dropped_small_group += len(group)
            continue
        feats = np.stack([summary_features(r["series"]) for r in group])
        sd = feats.std(0)
        sd[sd == 0] = 1.0
        z = (feats - feats.mean(0)) / sd
        max_run = max(1, int(max_overlap_frac * length))
        for i, r in enumerate(group):
            order = np.argsort(np.linalg.norm(z - z[i], axis=1))
            negs, seen = [], {tuple(round(float(x), 6) for x in r["series"])}
            for j in order:
                j = int(j)
                if j == i:
                    continue
                if longest_common_run(r["series"], group[j]["series"]) > max_run:
                    continue
                key = tuple(round(float(x), 6) for x in group[j]["series"])
                if key in seen:
                    continue
                seen.add(key)
                negs.append([float(x) for x in group[j]["series"]])
                if len(negs) == n_distractors:
                    break
            if len(negs) < n_distractors:
                dropped_negs += 1
                continue
            out[r[uid_key]] = negs
    stats = {"kept": len(out), "dropped_small_group": dropped_small_group,
             "dropped_not_enough_clean": dropped_negs, "total": len(rows)}
    return out, stats


def build_benchmark(args) -> None:
    series = [json.loads(l) for l in Path(args.series).read_text().splitlines() if l.strip()]
    caps = [json.loads(l) for l in Path(args.captions).read_text().splitlines() if l.strip()]

    if args.max_series_len:
        too_long = [r for r in series if len(r["series"]) > args.max_series_len]
        if too_long:
            import collections
            by_ds = collections.Counter(r.get("dataset", "?") for r in too_long)
            print(f"excluding {len(too_long)} series longer than {args.max_series_len} "
                  f"points (a k={args.n_distractors + 1} question would not fit the "
                  f"reader's context): {dict(by_ds)}")
            series = [r for r in series if len(r["series"]) <= args.max_series_len]

    cap_by_uid: dict[str, str] = {}
    for c in caps:
        uid = c.get(args.caption_id_field) or c.get("id")
        text = c["caption"]
        if uid in cap_by_uid and cap_by_uid[uid] != text:
            raise SystemExit(
                f"series {uid} carries two different captions; the caption is supposed to "
                "be a property of the series, so folding is not safe here"
            )
        cap_by_uid[uid] = text
    print(f"{len(caps)} caption rows -> {len(cap_by_uid)} unique series")

    series = [r for r in series if r[args.uid_field] in cap_by_uid]
    print(f"{len(series)} series have a caption")

    negs, stats = negatives_by_uid(series, args.n_distractors, args.max_overlap_frac,
                                   args.uid_field)
    print(f"negatives: kept {stats['kept']}/{stats['total']} "
          f"(dropped {stats['dropped_small_group']} in length groups too small to supply "
          f"{args.n_distractors}, {stats['dropped_not_enough_clean']} without enough clean "
          f"negatives)")
    if not negs:
        raise SystemExit("no item could be given a full set of negatives")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"items_k{args.n_distractors}_{args.arm}.jsonl"
    n = 0
    with out.open("w") as fh:
        for r in series:
            uid = r[args.uid_field]
            if uid not in negs:
                continue
            fh.write(json.dumps(
                {"id": uid, "caption": cap_by_uid[uid],
                 "payload": {"id": uid, "true": [float(x) for x in r["series"]],
                             "distractors": negs[uid],
                             "decimals": P.decimals_for(r["series"])}},
                separators=(",", ":")) + "\n")
            n += 1
    print(f"wrote {n} items to {out}")
    print(f"  {args.n_distractors} distractors each, serving k up to {args.n_distractors + 1}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="tsfragment", choices=["tsfragment", "jsonl"],
                    help="tsfragment = rebuild the RL val split; jsonl = a benchmark with "
                         "its own series file (BEDTime, CaTS)")
    ap.add_argument("--series", help="--source jsonl: series jsonl with uid + series")
    ap.add_argument("--uid-field", default="series_uid")
    ap.add_argument("--caption-id-field", default="caption_key")
    ap.add_argument("--max-overlap-frac", type=float, default=0.2,
                    help="same default as build_decodability_dataset")
    ap.add_argument("--dedupe-headroom", type=int, default=4,
                    help="--source tsfragment: extra candidates to request so that, after "
                         "dropping numerically identical ones, n_distractors remain")
    ap.add_argument("--max-series-len", type=int, default=0,
                    help="--source jsonl: drop series longer than this, because a k-way "
                         "question holds k of them in one prompt (0 = no limit)")
    ap.add_argument("--n-distractors", type=int, default=11,
                    help="negatives stored per item; serves every k up to this + 1")
    ap.add_argument("--captions", required=True,
                    help="jsonl of {id, caption[, payload]}: the arm's greedy val captions")
    ap.add_argument("--fragments",
                    default=str(REPO / "out_10k_xdomain" / "fragments.jsonl"))
    ap.add_argument("--image-root", default=str(REPO / "out_10k_xdomain"))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--arm", default="decod550")
    args = ap.parse_args()

    if args.source == "jsonl":
        if not args.series:
            raise SystemExit("--source jsonl needs --series")
        build_benchmark(args)
        return

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "decodability_rl.rl.build_decodability_dataset",
            "--input", args.fragments, "--image-root", args.image_root,
            "--out-dir", tmp,
            "--n-distractors", str(args.n_distractors + args.dedupe_headroom),
            "--pool-size", str(args.n_distractors + args.dedupe_headroom),
        ]
        print("$", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        print(r.stdout, end="")
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            raise SystemExit("dataset rebuild failed")
        if "not enough clean negatives 0" not in r.stdout:
            raise SystemExit(
                "the rebuild dropped items for want of negatives, so this val split is "
                "NOT the one the arms trained against; lower --n-distractors"
            )
        payloads = read_val_payloads(Path(tmp) / "val_messages.jsonl")

    n_trimmed = 0
    for pl in payloads:
        pl["_raw_distractors"] = list(pl["distractors"])
        seen = {tuple(round(float(x), 6) for x in pl["true"])}
        keep = []
        for d in pl["distractors"]:
            key = tuple(round(float(x), 6) for x in d)
            if key in seen:
                continue
            seen.add(key)
            keep.append(d)
        if len(keep) < args.n_distractors:
            raise SystemExit(
                f"item {pl['id']} has only {len(keep)} distinct negatives after dedupe; "
                f"raise --dedupe-headroom above {args.dedupe_headroom}"
            )
        if len(keep) != len(pl["distractors"]):
            n_trimmed += 1
        pl["distractors"] = keep[: args.n_distractors]
    print(f"dedupe: {n_trimmed} val items had a numerically repeated candidate")

    by_id = {p["id"]: p for p in payloads}
    caps = [json.loads(l) for l in Path(args.captions).read_text().splitlines() if l.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"items_k{args.n_distractors}_{args.arm}.jsonl"

    checked = 0
    with out.open("w") as fh:
        for c in caps:
            p = by_id[c["id"]]
            assert len(p["distractors"]) == args.n_distractors, c["id"]
            if "payload" in c:
                assert p["true"] == c["payload"]["true"], f"true series differs at {c['id']}"
                assert p["decimals"] == c["payload"]["decimals"], c["id"]
                n_old = len(c["payload"]["distractors"])
                a = {tuple(x) for x in p["_raw_distractors"][:n_old]}
                b = {tuple(x) for x in c["payload"]["distractors"]}
                assert a == b, f"the first {n_old} distractors differ at {c['id']}"
                checked += 1
            fh.write(json.dumps(
                {"id": c["id"], "caption": c["caption"],
                 "payload": {"id": p["id"], "true": p["true"],
                             "distractors": p["distractors"], "decimals": p["decimals"]}},
                separators=(",", ":")) + "\n")

    print(f"wrote {len(caps)} items to {out}")
    print(f"  {args.n_distractors} distractors each, serving k up to {args.n_distractors + 1}")
    print(f"  cross-checked against the shipped payload for {checked}/{len(caps)} items")


if __name__ == "__main__":
    main()
