from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

from decodability_rl.test_reward_design import prompts as P

REPO = Path(__file__).resolve().parents[1]
SERIES = REPO / "bench_data" / "bedtime" / "series.jsonl"
BEDTIME = REPO / "results" / "eval_protocol" / "bedtime"
OUT_DIR = REPO / "results" / "eval_protocol" / "bedtime_mcq"
CATS_OUT_DIR = REPO / "results" / "eval_protocol" / "cats_mcq"
XVERIFIER_DIR = REPO / "results" / "eval_protocol" / "xverifier"
PUBLISHED_DIRS = (OUT_DIR, CATS_OUT_DIR)

DEFAULT_READER = "Qwen/Qwen2.5-14B-Instruct"

LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"\b([ABCD])\b")
TAG_RE = re.compile(r"<answer>\s*\(?([ABCD])\)?\s*</answer>")
ANSWER_RE = re.compile(r"(?i:answer)\s*(?i:is|:)?\s*\**\s*\(?([ABCD])\)?(?![A-Za-z])")

CONTROLS = ("none", "empty", "mismatch")
ORDERS = ("rotation", "random")

EXCLUDE_DATASETS = ("sushi",)

MCQ_B_SYSTEM = "You are a careful time-series analyst. You answer with a single letter."

MCQ_B_TEMPLATE = """Below is one time series, given as its raw values in time order.

Time series:
{series}

Here are four written descriptions. Each was written about some time series.

A) {opt_a}

B) {opt_b}

C) {opt_c}

D) {opt_d}

Exactly one of these four descriptions was written about the time series above.
Which one is it? Answer with a single letter: A, B, C, or D."""


def summary4(y: np.ndarray) -> np.ndarray:
    return np.array([y.mean(), y.std(), y.min(), y.max()])


def group_key(r: dict, same_dataset: bool) -> tuple:
    return (r["dataset"], len(r["series"])) if same_dataset else (len(r["series"]),)


def build_negatives(rows: list[dict], n_distract: int = 3,
                    same_dataset: bool = True) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(group_key(r, same_dataset), []).append(r)

    for key, group in groups.items():
        if len(group) <= n_distract:
            raise ValueError(
                f"group {key} has only {len(group)} series, cannot draw {n_distract} "
                f"distractors of the same length (pass same_dataset=False to pool across "
                f"datasets, or drop the short group)")
        feats = np.stack([summary4(np.asarray(r["series"], dtype=float)) for r in group])
        sd = feats.std(axis=0)
        sd[sd == 0] = 1.0
        z = (feats - feats.mean(axis=0)) / sd
        for i, r in enumerate(group):
            d = np.linalg.norm(z - z[i], axis=1)
            d[i] = np.inf
            out[r["series_uid"]] = [group[j]["series_uid"]
                                    for j in np.argsort(d)[:n_distract]]
    return out


def rotations(true_txt: str, distractor_txt: list[str]) -> list[tuple[int, list[str]]]:
    return [(g, distractor_txt[:g] + [true_txt] + distractor_txt[g:]) for g in range(4)]


def placements(true_txt: str, distractor_txt: list[str], order: str = "rotation",
               key: str = "", seed: int = 0) -> list[tuple[int, list[str]]]:
    if order == "rotation":
        return rotations(true_txt, distractor_txt)
    if order != "random":
        raise ValueError(f"order must be one of {ORDERS}, got {order!r}")
    if len(distractor_txt) != 3:
        raise ValueError(f"expected 3 distractors, got {len(distractor_txt)}")
    perm = list(range(4))
    random.Random(f"{seed}|{key}").shuffle(perm)
    all4 = [true_txt] + list(distractor_txt)
    return [(perm.index(0), [all4[j] for j in perm])]


def mismatch_donors(pool: list[dict], negs: dict[str, list[str]],
                    same_dataset: bool = True) -> tuple[dict[str, str], list[str]]:
    groups: dict[tuple, list[str]] = {}
    for r in pool:
        groups.setdefault(group_key(r, same_dataset), []).append(r["series_uid"])
    out: dict[str, str] = {}
    fallback: list[str] = []
    for uids in groups.values():
        n = len(uids)
        for i, uid in enumerate(uids):
            chosen = None
            for step in range(1, n):
                d = uids[(i + step) % n]
                if d not in negs.get(uid, ()) and uid not in negs.get(d, ()):
                    chosen = d
                    break
            if chosen is None:
                chosen = uids[(i + 1) % n]
                fallback.append(uid)
            out[uid] = chosen
    return out, fallback


def control_captions(caps: dict[str, str], pool: list[dict], control: str,
                     negs: dict[str, list[str]] | None = None,
                     same_dataset: bool = True) -> tuple[dict[str, str], dict]:
    if control == "none":
        return caps, {}
    if control == "empty":
        return {r["series_uid"]: P.EMPTY_CAPTION for r in pool}, {}
    if control == "mismatch":
        if negs is None:
            raise ValueError("mismatch needs the candidate sets")
        donors, fallback = mismatch_donors(pool, negs, same_dataset)
        out = {uid: caps[d] for uid, d in donors.items() if d in caps}
        return out, {"mismatch_donor_in_candidates": len(fallback),
                     "mismatch_fallback_uids": fallback,
                     "mismatch_donor": donors}
    raise ValueError(f"control must be one of {CONTROLS}, got {control!r}")


def load_captions(arm: str, path: Path | None = None) -> dict[str, str]:
    path = path or BEDTIME / arm / "captions.jsonl"
    by_series: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = r["caption_key"]
            previous = by_series.setdefault(key, r["caption"])
            if previous != r["caption"]:
                raise ValueError(f"{key} has two different captions in {path}")
    return by_series


def build_prompts(rows: list[dict], caps: dict[str, str], negs: dict[str, list[str]],
                  metrics: tuple[str, ...],
                  pool: list[dict] | None = None,
                  order: str = "rotation", seed: int = 0,
                  control: str = "none") -> tuple[list[str], list[dict]]:
    by_uid = {r["series_uid"]: r for r in (pool if pool is not None else rows)}
    prompts: list[str] = []
    meta: list[dict] = []
    for r in rows:
        uid = r["series_uid"]
        caption = caps.get(uid)
        if caption is None:
            continue
        others = negs[uid]
        decimals = P.decimals_for(r["series"])
        true_series_txt = P.format_series(r["series"], decimals)

        if "A" in metrics:
            distr = [P.format_series(by_uid[o]["series"], decimals) for o in others]
            for gold, opts in placements(true_series_txt, distr, order, f"{uid}|A", seed):
                prompts.append(P.build_discriminator_prompt(caption, opts, "neutral"))
                meta.append({"series_uid": uid, "dataset": r["dataset"],
                             "metric": "A", "gold": gold,
                             "control": control, "order": order})

        if "B" in metrics:
            distr_caps = [caps.get(o, "") for o in others]
            if any(not c for c in distr_caps):
                continue
            for gold, opts in placements(caption, distr_caps, order, f"{uid}|B", seed):
                prompts.append(MCQ_B_TEMPLATE.format(
                    series=true_series_txt, opt_a=opts[0], opt_b=opts[1],
                    opt_c=opts[2], opt_d=opts[3]))
                meta.append({"series_uid": uid, "dataset": r["dataset"],
                             "metric": "B", "gold": gold,
                             "control": control, "order": order})
    return prompts, meta


def parse_reply(text: str) -> int | None:
    text = text.strip()
    short = len(text) <= 40
    for rx in (TAG_RE, ANSWER_RE):
        hits = list(rx.finditer(text))
        if hits:
            return LETTERS.index((hits[0] if short else hits[-1]).group(1).upper())
    hits = list(LETTER_RE.finditer(text))
    if hits:
        return LETTERS.index((hits[0] if short else hits[-1]).group(1).upper())
    return None


def summarise(meta: list[dict], correct: list[float], unparsed: int) -> dict:
    def rate(rows: list[tuple[dict, float]]) -> dict:
        if not rows:
            return {"n": 0}
        return {"n": len(rows),
                "n_series": len({m["series_uid"] for m, _ in rows}),
                "accuracy": float(np.mean([c for _, c in rows]))}

    pairs = list(zip(meta, correct))
    out = {"unparsed": unparsed, "unparsed_rate": unparsed / max(len(meta), 1)}
    for metric in ("A", "B"):
        sel = [(m, c) for m, c in pairs if m["metric"] == metric]
        if not sel:
            continue
        entry = rate(sel)
        entry["by_dataset"] = {
            ds: rate([(m, c) for m, c in sel if m["dataset"] == ds])
            for ds in sorted({m["dataset"] for m, _ in sel})}
        entry["by_gold_position"] = {
            LETTERS[g]: rate([(m, c) for m, c in sel if m["gold"] == g])["accuracy"]
            for g in range(4)}
        out[metric] = entry
    return out


def output_name(arm: str, control: str, order: str) -> str:
    stem = "empty" if control == "empty" else arm
    if control == "mismatch":
        stem += "_mismatch"
    if order == "random":
        stem += "_random"
    return stem


def bench_from_series(path: str) -> str:
    p = str(path)
    if "cats" in p:
        return "cats"
    if "bedtime" in p:
        return "bedtime"
    raise SystemExit(f"cannot tell the benchmark from {path}; pass --bench")


def published_dir_guard(out_dir: Path, reader: str, controls: tuple[str, ...],
                        orders: tuple[str, ...], allow: bool) -> None:
    if allow:
        return
    published = any(out_dir.resolve() == d.resolve() for d in PUBLISHED_DIRS)
    unpublished_config = (reader != DEFAULT_READER or tuple(controls) != ("none",)
                          or tuple(orders) != ("rotation",))
    if published and unpublished_config:
        raise SystemExit(
            f"refusing to write a non-published configuration (reader={reader}, "
            f"controls={controls}, orders={orders}) into {out_dir}. Pass --reader-tag to "
            f"route the outputs under {XVERIFIER_DIR}, or --allow-overwrite-published if "
            f"you really mean to replace the paper's numbers.")


def thinking_off_for(reader: str) -> bool:
    return "qwen3" in reader.lower()


def make_chats(tok, prompts: list[str], thinking_off: bool = False) -> tuple[list[str], bool]:
    kw = {"enable_thinking": False} if thinking_off else {}
    msgs = [{"role": "system", "content": MCQ_B_SYSTEM}, {"role": "user", "content": "x"}]
    try:
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        system_ok = True
    except Exception:
        system_ok = False

    def chat(p: str) -> str:
        if system_ok:
            m = [{"role": "system", "content": MCQ_B_SYSTEM}, {"role": "user", "content": p}]
        else:
            m = [{"role": "user", "content": MCQ_B_SYSTEM + "\n\n" + p}]
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)

    return [chat(p) for p in prompts], system_ok


def run_reader(chats: list[str], reader: str, tp: int = 1, max_model_len: int = 32768,
               gpu_memory_utilization: float = 0.90, max_tokens: int = 8,
               mm_zero: bool = False) -> list[str]:
    from vllm import LLM, SamplingParams

    kwargs = dict(model=reader, tensor_parallel_size=tp,
                  gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len)
    if mm_zero:
        kwargs["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(**kwargs)
    outs = llm.generate(chats, SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens))
    return [o.outputs[0].text for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--series", default=str(SERIES),
                    help="series.jsonl; defaults to BEDTime's")
    ap.add_argument("--captions", default=None,
                    help="captions jsonl; defaults to the arm's BEDTime captions.jsonl")
    ap.add_argument("--pool-negatives-across-datasets", action="store_true",
                    help="draw distractors from any dataset of the same length. Needed "
                         "for CaTS, whose domain groups can hold fewer than 4 series")
    ap.add_argument("--exclude-datasets", default=",".join(EXCLUDE_DATASETS),
                    help="comma-separated; empty string keeps everything. BEDTime needs "
                         "sushi excluded (2048-point series); CaTS needs nothing excluded")
    ap.add_argument("--reader", default=DEFAULT_READER)
    ap.add_argument("--reader-tag", default=None,
                    help="label for the reader; routes outputs to xverifier/<bench>/<tag>/")
    ap.add_argument("--bench", default=None, choices=("bedtime", "cats"),
                    help="only needed when it cannot be read off --series")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--controls", default="none",
                    help="comma-separated subset of " + ",".join(CONTROLS))
    ap.add_argument("--order", default="rotation",
                    help="comma-separated subset of " + ",".join(ORDERS))
    ap.add_argument("--seed", type=int, default=0, help="seed for --order random")
    ap.add_argument("--metrics", default="AB", help="A, B, or AB")
    ap.add_argument("--limit", type=int, default=None, help="first N series, for smoke tests")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score even if the caption file does not cover every series. Off "
                         "by default: partial coverage makes arms incomparable silently")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="reply budget; 8 for the Qwen readers, 16 for verbose ones")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--mm-zero", action="store_true",
                    help="pass limit_mm_per_prompt={image: 0} (multimodal checkpoints)")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to bedtime_mcq/ (published) or, with --reader-tag, "
                         "xverifier/<bench>/<tag>/")
    ap.add_argument("--allow-overwrite-published", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompts and print counts; load no model, write nothing")
    args = ap.parse_args()
    metrics = tuple(args.metrics)
    controls = tuple(c for c in args.controls.split(",") if c)
    orders = tuple(o for o in args.order.split(",") if o)
    for c in controls:
        if c not in CONTROLS:
            raise SystemExit(f"unknown control {c!r}; choose from {CONTROLS}")
    for o in orders:
        if o not in ORDERS:
            raise SystemExit(f"unknown order {o!r}; choose from {ORDERS}")

    bench = args.bench or bench_from_series(args.series)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.reader_tag:
        out_dir = XVERIFIER_DIR / bench / args.reader_tag
    else:
        out_dir = OUT_DIR
    published_dir_guard(out_dir, args.reader, controls, orders, args.allow_overwrite_published)

    excluded = tuple(d for d in args.exclude_datasets.split(",") if d)
    rows = [json.loads(l) for l in open(args.series, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["dataset"] not in excluded]
    same_dataset = not args.pool_negatives_across_datasets
    negs = build_negatives(rows, same_dataset=same_dataset)
    pool = rows
    if args.limit:
        rows = rows[: args.limit]
    caps = load_captions(args.arm, Path(args.captions) if args.captions else None)

    missing = [r["series_uid"] for r in rows if r["series_uid"] not in caps]
    if missing and not args.allow_partial:
        raise SystemExit(
            f"{len(missing)}/{len(rows)} series have no caption in this arm's file "
            f"(e.g. {missing[:3]}). Every arm must be scored on the same items, so this "
            f"refuses rather than quietly shrinking the item set. Re-run the captioner, "
            f"or pass --allow-partial if a smaller set really is intended.")

    all_prompts: list[str] = []
    all_meta: list[dict] = []
    spans: list[tuple[str, str, int, int, dict]] = []
    for control in controls:
        ccaps, extra = control_captions(caps, pool, control, negs, same_dataset)
        for order in orders:
            prompts, meta = build_prompts(rows, ccaps, negs, metrics, pool=pool,
                                          order=order, seed=args.seed, control=control)
            spans.append((control, order, len(all_prompts),
                          len(all_prompts) + len(prompts), extra))
            all_prompts += prompts
            all_meta += meta
            print(f"{args.arm} [{control}/{order}]: {len(rows)} series -> {len(prompts)} "
                  f"prompts ({', '.join(sorted({m['metric'] for m in meta}))})", flush=True)
    if args.dry_run:
        print(f"dry run: {len(all_prompts)} prompts total, would write to {out_dir}")
        return

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.reader)
    chats, system_ok = make_chats(tok, all_prompts, thinking_off=thinking_off_for(args.reader))
    single_token = {L: len(tok.encode(L, add_special_tokens=False)) == 1 for L in LETTERS}
    if not all(single_token.values()):
        print(f"note: option letters are not all single tokens for {args.reader}: "
              f"{single_token} (harmless in generate mode, recorded in the result)",
              flush=True)
    too_long = [i for i, c in enumerate(chats)
                if len(tok(c)["input_ids"]) > args.max_model_len - 16]
    if too_long:
        raise SystemExit(
            f"{len(too_long)} prompts exceed the {args.max_model_len}-token context "
            f"(first: {all_meta[too_long[0]]}). Refusing to run: vLLM would silently drop "
            f"them and the accuracy would be computed over a different item set per arm.")

    replies = run_reader(chats, args.reader, tp=args.tp, max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         max_tokens=args.max_tokens, mm_zero=args.mm_zero)

    out_dir.mkdir(parents=True, exist_ok=True)
    for control, order, a, b, extra in spans:
        meta = all_meta[a:b]
        reps = replies[a:b]
        correct, unparsed = [], 0
        for text, m in zip(reps, meta):
            idx = parse_reply(text)
            if idx is None:
                unparsed += 1
                correct.append(0.0)
            else:
                correct.append(float(idx == m["gold"]))

        result = {"arm": args.arm, "reader": args.reader,
                  "reader_tag": args.reader_tag, "tp": args.tp, "bench": bench,
                  "control": control, "order": order, "seed": args.seed,
                  "max_tokens": args.max_tokens, "system_role_supported": system_ok,
                  "thinking_off": thinking_off_for(args.reader),
                  "letters_single_token": single_token,
                  "series_file": args.series,
                  "series_requested": len(rows), "series_captioned": len(rows) - len(missing),
                  "excluded_datasets": list(excluded),
                  "n_distractors": 3, "rotations": 4 if order == "rotation" else 1,
                  **{k: v for k, v in extra.items() if k != "mismatch_donor"},
                  **summarise(meta, correct, unparsed)}
        name = output_name(args.arm, control, order)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))
        donors = extra.get("mismatch_donor", {})
        fell_back = set(extra.get("mismatch_fallback_uids", ()))
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for m, c, text in zip(meta, correct, reps):
                row = {**m, "correct": c, "reply": text.strip()}
                if donors:
                    row["donor"] = donors.get(m["series_uid"])
                    row["donor_is_candidate"] = m["series_uid"] in fell_back
                f.write(json.dumps(row) + "\n")
        print(f"[{control}/{order}] -> {out_dir / name}.json",
              json.dumps({k: v for k, v in result.items()
                          if k in ("A", "B", "unparsed_rate")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
