from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VAL = REPO / "out_10k_xdomain" / "rl_decod" / "val_messages.jsonl"
FRAGMENTS = REPO / "out_10k_xdomain" / "fragments.jsonl"
OUT_DIR = REPO / "results" / "decodability_rl" / "negatives_v2"

SETTINGS = [
    "current",
    "num_masked",
    "num_coarse",
    "random",
    "same_source",
    "rich_stat",
    "shape_hard",
    "n8",
]


def summary4(y: np.ndarray) -> np.ndarray:
    return np.array([y.mean(), y.std(), y.min(), y.max()])


def summary8(y: np.ndarray) -> np.ndarray:
    n = len(y)
    t = np.arange(n)
    slope = float(np.polyfit(t, y, 1)[0]) if n > 1 else 0.0
    d = np.diff(y)
    ac1 = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if n > 2 and y.std() > 0 else 0.0
    return np.array([
        y.mean(), y.std(), y.min(), y.max(),
        slope,
        ac1,
        float(np.mean(np.abs(d))) if n > 1 else 0.0,
        float(np.sum(np.diff(np.sign(d)) != 0)) if n > 2 else 0.0,
    ])


def zscore_rows(mat: np.ndarray) -> np.ndarray:
    mu, sd = mat.mean(axis=0), mat.std(axis=0)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_NUMWORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
    "sixty seventy eighty ninety hundred thousand million billion "
    "first second third fourth fifth sixth seventh eighth ninth tenth "
    "half quarter third-quarter point negative minus plus"
).split()
_NUMWORD_RE = re.compile(r"\b(" + "|".join(sorted(_NUMWORDS, key=len, reverse=True)) + r")\b",
                         re.IGNORECASE)


def mask_numbers(caption: str, token: str = "<num>", words: bool = True) -> str:
    out = _NUM.sub(token, caption)
    return _NUMWORD_RE.sub(token, out) if words else out


def coarsen_numbers(caption: str, sig: int = 1) -> str:
    from math import floor, log10

    def r(m):
        v = float(m.group())
        if v == 0:
            return "0"
        return f"{round(v, -int(floor(log10(abs(v)))) + (sig - 1)):g}"

    return _NUM.sub(r, caption)


def numeric_density(caption: str) -> float:
    toks = caption.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if _NUM.search(t)) / len(toks)


def load_val() -> list[dict]:
    import ast

    rows = []
    for line in VAL.read_text().splitlines():
        if not line.strip():
            continue
        msg = json.loads(line)["message"]
        msg = json.loads(msg) if isinstance(msg, str) else msg
        payload = json.loads(ast.literal_eval(msg[2]["content"])[0][0])
        image = next(c["image"] for c in msg[1]["content"] if c.get("type") == "image")
        rows.append({"id": payload["id"], "image": image, "true": payload["true"],
                     "distractors": payload["distractors"],
                     "decimals": payload.get("decimals", 0)})
    return rows


def load_pool() -> tuple[list[int], list[np.ndarray]]:
    ids, series = [], []
    for line in FRAGMENTS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ids.append(r["id"])
        series.append(np.asarray(r["series"], dtype=float))
    return ids, series


def longest_common_run(a: np.ndarray, b: np.ndarray) -> int:
    sa = {round(float(v), 6) for v in a}
    return sum(1 for v in b if round(float(v), 6) in sa)


def pick(setting: str, item: dict, pool_ids, pool_series, feats4, feats8,
         rng: random.Random, n_distract: int = 3, max_overlap_frac: float = 0.3):
    true = np.asarray(item["true"], dtype=float)
    L = len(true)
    same_len = [i for i, s in enumerate(pool_series)
                if len(s) == L and pool_ids[i] != item["id"]]

    def ok(i):
        return longest_common_run(true, pool_series[i]) <= max_overlap_frac * L

    if setting in ("current", "num_masked", "num_coarse"):
        return item["distractors"]

    if setting == "random":
        cand = [i for i in same_len if ok(i)]
        return [pool_series[i].tolist() for i in rng.sample(cand, min(n_distract, len(cand)))]

    if setting == "same_source":
        scored = []
        for i in same_len:
            run = longest_common_run(true, pool_series[i])
            if 0 < run <= max_overlap_frac * L:
                scored.append((run, i))
        scored.sort(reverse=True)
        chosen = [i for _, i in scored[:n_distract]]
        if len(chosen) < n_distract:
            extra = [i for i in same_len if ok(i) and i not in chosen]
            chosen += rng.sample(extra, min(n_distract - len(chosen), len(extra)))
        return [pool_series[i].tolist() for i in chosen]

    if setting in ("rich_stat", "n8"):
        feats = feats8 if setting == "rich_stat" else feats4
        f = (summary8(true) if setting == "rich_stat" else summary4(true))
        k = n_distract if setting == "rich_stat" else 7
        cand = [i for i in same_len if ok(i)]
        fz = feats[cand]
        d = np.linalg.norm(fz - ((f - feats.mean(0)) / np.where(feats.std(0) == 0, 1, feats.std(0))), axis=1)
        order = np.argsort(d)[:k]
        return [pool_series[cand[j]].tolist() for j in order]

    if setting == "shape_hard":
        lo, hi = true.min(), true.max()
        span = max(hi - lo, 1e-9)
        cand = [i for i in same_len if ok(i)
                and abs(pool_series[i].mean() - true.mean()) < 0.5 * span]
        if len(cand) < n_distract:
            cand = [i for i in same_len if ok(i)]
        cors = []
        for i in cand:
            s = pool_series[i]
            c = float(np.corrcoef(true, s)[0, 1]) if s.std() > 0 and true.std() > 0 else 0.0
            cors.append((c if np.isfinite(c) else 0.0, i))
        cors.sort(reverse=True)
        return [pool_series[i].tolist() for _, i in cors[:n_distract]]

    raise ValueError(f"unknown setting {setting!r}")


def cmd_caption(args):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT

    rows = load_val()
    if args.limit:
        rows = rows[: args.limit]
    proc = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)
    llm = LLM(model=args.ckpt, trust_remote_code=True,
              tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_util,
              limit_mm_per_prompt={"image": 1})

    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": r["image"]},
            {"type": "text", "text": TS_CAP_PROMPT}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        prompts.append({"prompt": text, "multi_modal_data": {"image": imgs}})

    outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.0,
                                                max_tokens=args.max_tokens))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r, o in zip(rows, outs):
            fh.write(json.dumps({"id": r["id"],
                                 "caption": o.outputs[0].text.strip()}) + "\n")
    lens = sorted(len(o.outputs[0].text.strip()) for o in outs)
    print(f"wrote {len(rows)} captions to {out} | median {lens[len(lens)//2]} chars")


def cmd_build(args):
    rows = load_val()
    if args.limit:
        rows = rows[: args.limit]
    caps = {}
    for line in Path(args.captions).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            caps[int(r["id"])] = r["caption"]
    missing = [r["id"] for r in rows if r["id"] not in caps]
    if missing:
        raise SystemExit(f"{len(missing)} val items have no caption, e.g. {missing[:3]}")

    pool_ids, pool_series = load_pool()
    f4 = zscore_rows(np.stack([summary4(s) for s in pool_series]))
    f8 = zscore_rows(np.stack([summary8(s) for s in pool_series]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for setting in (args.settings or SETTINGS):
        rng = random.Random(2020)
        out = OUT_DIR / f"items_{setting}.jsonl"
        with out.open("w") as fh:
            for r in rows:
                cap = caps[r["id"]]
                if setting == "num_masked":
                    cap = mask_numbers(cap)
                elif setting == "num_coarse":
                    cap = coarsen_numbers(cap)
                d = pick(setting, r, pool_ids, pool_series, f4, f8, rng)
                fh.write(json.dumps({"id": r["id"], "caption": cap,
                                     "payload": {"id": r["id"], "true": r["true"],
                                                 "distractors": d,
                                                 "decimals": r["decimals"]}}) + "\n")
        print(f"  wrote {out.name}  ({len(rows)} items, {len(d)} distractors each)")


def cmd_score(args):
    import requests

    results = {}
    for setting in (args.settings or SETTINGS):
        path = OUT_DIR / f"items_{setting}.jsonl"
        if not path.exists():
            print(f"  skip {setting}: not built"); continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        rewards = []
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            prompts = [[r["caption"], [[json.dumps(r["payload"]), ""]]] for r in chunk]
            resp = requests.post(args.url, json={"prompts": prompts, "query": [], "labels": []},
                                 timeout=1800)
            resp.raise_for_status()
            rewards += resp.json()["rewards"]
        arr = np.asarray(rewards, dtype=float)
        acc = arr / args.reward_scale
        results[setting] = {
            "n": len(arr), "accuracy": float(acc.mean()), "reward_mean": float(arr.mean()),
            "reward_sd": float(arr.std()),
            "frac_perfect": float((acc >= 0.999).mean()),
            "frac_zero": float((acc <= 0.001).mean()),
        }
        print(f"  {setting:14} acc={acc.mean():.4f}  reward={arr.mean():.3f}  "
              f"perfect={results[setting]['frac_perfect']:.3f}")
    out = OUT_DIR / "summary.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(results)
    out.write_text(json.dumps(prev, indent=2))
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("caption")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--tp", type=int, default=1)
    c.add_argument("--gpu-util", type=float, default=0.9)
    c.add_argument("--max-tokens", type=int, default=1024)
    c.add_argument("--limit", type=int, default=None)
    c.set_defaults(fn=cmd_caption)

    b = sub.add_parser("build"); b.add_argument("--captions", required=True)
    b.add_argument("--settings", nargs="*"); b.add_argument("--limit", type=int, default=None)
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("score"); s.add_argument("--url", required=True)
    s.add_argument("--settings", nargs="*"); s.add_argument("--batch", type=int, default=32)
    s.add_argument("--reward-scale", type=float, default=2.0)
    s.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
