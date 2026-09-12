from __future__ import annotations

import re

SYSTEM = (
    "You read descriptions of time series and report the numbers in them. "
    "You answer in exactly the format asked for, with no commentary."
)

PROMPT = """You are given a fragment of a description of a time series, and you list the
numbers in it.

Write one line for every number in the fragment, in the order it appears. Each line is the
number exactly as written, a space, then:

  1  if it is a VALUE OF THE SERIES -- a reading on the y-axis: a level, a height, a peak
     or a trough, an end of a range of values, an amount the series rose or fell by
  0  if it is anything else -- a position or duration on the time axis (a time step, an
     index, "the 8th point", "for 6 steps"), list numbering, a count, a percentage

List every occurrence separately. If the same number appears three times, write three
lines. A range is two numbers, so it is two lines: "values around 9.5 to 10.2" is

9.5 1
10.2 1

and "by time steps 18-21" is

18 0
21 0

Example of the format:
3300 1
12 0
21.2 1

Write nothing except these lines.

----- fragment -----
\"\"\"{caption}\"\"\"

It contains exactly {n} numbers, so write exactly {n} lines."""


_PAIR_RE = re.compile(r"^\s*(?:\d+[.)]\s+)?(-?\d+(?:\.\d+)?)[\s:,]+([01])\s*$")
_RANGE_PAIR_RE = re.compile(
    r"^\s*(?:\d+[.)]\s+)?(-?\d+(?:\.\d+)?)\s*(?:-|–|to|and)\s*(-?\d+(?:\.\d+)?)"
    r"[\s:,]+([01])\s*$",
    re.IGNORECASE,
)


def build_prompt(chunk: str) -> str:
    return PROMPT.format(caption=chunk, n=count_numbers(chunk))


def parse_reply(reply: str) -> tuple[list[str], list[int]] | None:
    numbers, flags = [], []
    for ln in reply.splitlines():
        ln = ln.strip().strip("-*").strip()
        m = _PAIR_RE.match(ln)
        if m:
            numbers.append(m.group(1))
            flags.append(int(m.group(2)))
            continue
        m = _RANGE_PAIR_RE.match(ln)
        if m:
            numbers += [m.group(1), m.group(2)]
            flags += [int(m.group(3))] * 2
    if not numbers:
        return None
    return numbers, flags


_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")


def split_for_masking(caption: str, max_numbers: int = 5) -> list[str]:
    parts = _BOUNDARY_RE.split(caption)
    seps = _BOUNDARY_RE.findall(caption)
    pieces = [p + (seps[i] if i < len(seps) else "") for i, p in enumerate(parts)]

    chunks, cur, n = [], "", 0
    for piece in pieces:
        k = count_numbers(piece)
        if cur and n + k > max_numbers:
            chunks.append(cur)
            cur, n = piece, k
        else:
            cur += piece
            n += k
    if cur:
        chunks.append(cur)
    return chunks or [caption]


_DIGIT_RUN = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?")


def count_numbers(caption: str) -> int:
    return len(_DIGIT_RUN.findall(caption))


def token_budget(caption: str) -> int:
    return min(1536, 96 + 8 * count_numbers(caption))


def mask_all_numbers(caption: str, token: str = "<val>") -> str:
    return _DIGIT_RUN.sub(token, caption)


def mask_one(chunk: str, reply: str) -> tuple[str, int]:
    parsed = parse_reply(reply)
    numbers, flags = parsed if parsed is not None else ([], [])
    from collections import defaultdict, deque

    def key(lit: str) -> float:
        try:
            return float(lit)
        except ValueError:
            return float("nan")

    pending: dict[float, deque] = defaultdict(deque)
    for lit, fl in zip(numbers, flags):
        pending[key(lit)].append(fl)

    out, last, unnamed = [], 0, 0
    for m in _DIGIT_RUN.finditer(chunk):
        k = key(m.group(0))
        queue = pending.get(k)
        if not queue:
            queue = pending.get(-k)
        if not queue:
            unnamed += 1
            continue
        hide = bool(queue.popleft())
        if hide:
            out.append(chunk[last : m.start()])
            out.append("<val>")
            last = m.end()
    out.append(chunk[last:])
    return "".join(out), unnamed


def numeric_density(caption: str) -> float:
    toks = caption.split()
    if not toks:
        return 0.0
    return sum(1 for t in toks if _DIGIT_RUN.search(t)) / len(toks)


def _load_captions(limit: int):
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    out = []
    p = repo / "results/decodability_rl/negatives_v2/val_captions_decod550.jsonl"
    if p.exists():
        for line in p.read_text().splitlines()[: limit // 2]:
            out.append(("decod550", json.loads(line)["caption"]))
    return out[:limit]


_TIMEREF = re.compile(r"(?:time[\s-]*)?steps?\s+\d+|\d+(?:st|nd|rd|th)", re.IGNORECASE)


def main() -> None:
    import argparse
    import hashlib
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    print("PROMPT SHA1:", hashlib.sha1(PROMPT.encode()).hexdigest()[:12], flush=True)
    print("PROMPT HEAD:", PROMPT.split("\n")[0][:70], flush=True)

    rows = _load_captions(args.limit)
    print(f"{len(rows)} captions", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")
    model.eval()

    chunks, owner = [], []
    for i, (_, cap) in enumerate(rows):
        for ch in split_for_masking(cap):
            chunks.append(ch); owner.append(i)
    print(f"{len(chunks)} chunks ({len(chunks)/len(rows):.1f} per caption)", flush=True)

    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": build_prompt(ch)}],
            tokenize=False, add_generation_prompt=True)
        for ch in chunks
    ]

    replies, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            sl = slice(i, i + args.batch_size)
            enc = tok(prompts[sl], return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc,
                                 max_new_tokens=max(token_budget(c) for c in chunks[sl]),
                                 do_sample=False, pad_token_id=tok.pad_token_id)
            replies += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
            print(f"  {min(i+args.batch_size, len(prompts))}/{len(prompts)}", flush=True)
    elapsed = time.time() - t0

    masked_chunks, unnamed = [], 0
    for ch, reply in zip(chunks, replies):
        m, n_un = mask_one(ch, reply)
        masked_chunks.append(m); unnamed += n_un

    masked = [""] * len(rows)
    for m, i in zip(masked_chunks, owner):
        masked[i] += m

    leaks = [
        (re.findall(r"(?<![\d.])\d+\.\d+(?!\d)", m)[:4], m[:110])
        for m in masked if re.search(r"(?<![\d.])\d+\.\d+(?!\d)", m)
    ]
    print(f"\n=== {args.model} ===")
    print(f"captions            : {len(rows)}")
    print(f"chunks              : {len(chunks)}")
    total_numbers = sum(count_numbers(c) for _, c in rows)
    print(f"numbers never named : {unnamed}/{total_numbers} "
          f"({unnamed/max(total_numbers,1):.1%})  <- left unmasked")
    print(f"TIME REFS KEPT      : "
          f"{sum(len(_TIMEREF.findall(m)) for m in masked)}/"
          f"{sum(len(_TIMEREF.findall(c)) for _, c in rows)}")
    print(f"numeric density     : "
          f"{sum(numeric_density(c) for _, c in rows)/len(rows):.3f} -> "
          f"{sum(map(numeric_density, masked))/len(masked):.3f}")
    print(f"captions with a decimal still exposed: {len(leaks)} / {len(rows)}")
    print(f"time                : {elapsed:.1f}s, {elapsed/len(rows)*1000:.0f} ms/caption "
          f"(batch {args.batch_size}, 1 GPU)")
    for left, ctx in leaks[:5]:
        print(f"   LEAK {left}  ...{ctx}...")
    for i in range(min(2, len(rows))):
        print("\n" + "-" * 96)
        print(masked[i][:400])

    print("\n" + "=" * 96)
    print("RAW REPLIES FOR CHUNKS WITH UNNAMED NUMBERS")
    shown = 0
    for ch, reply in zip(chunks, replies):
        _, n_un = mask_one(ch, reply)
        if not n_un or shown >= 6:
            continue
        shown += 1
        parsed = parse_reply(reply)
        print("\n--- chunk (%d numbers): %s" % (count_numbers(ch), ch.strip()[:180]))
        print("    reply   : %s" % reply.strip()[:220].replace("\n", " | "))
        print("    parsed  : %s" % (list(zip(*parsed)) if parsed else None))


if __name__ == "__main__":
    main()
