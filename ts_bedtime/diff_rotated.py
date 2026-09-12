from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import numpy as np

from ts_bedtime.prompts import differentiation_question
from ts_eval.benchmarks import bedtime
from ts_eval.caption_answer_harness import DEFAULT_EVIDENCE_LABEL, build_answer_messages

REPO = Path(__file__).resolve().parents[1]
BEDTIME = REPO / "results" / "eval_protocol" / "bedtime"
OUT_DIR = REPO / "results" / "eval_protocol" / "bedtime_diff_rot"

LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"\b([ABCD])\b")

EVIDENCE_LABEL = {"L0": None, "L2": "Time series values"}


def load_captions(arm: str) -> dict[str, str]:
    path = BEDTIME / arm / "captions.jsonl"
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = r["caption_key"]
            prev = out.setdefault(key, r["caption"])
            if prev != r["caption"]:
                raise ValueError(f"{key} has two different captions in {path}")
    return out


def rotated_items(item):
    opts = list(item.options)
    gold_idx = LETTERS.index(item.gold)
    gold_text = opts[gold_idx]
    others = [o for i, o in enumerate(opts) if i != gold_idx]
    out = []
    for g in range(4):
        placed = others[:g] + [gold_text] + others[g:]
        out.append((g, dataclasses.replace(
            item, question=differentiation_question(placed), gold=LETTERS[g],
            options=placed)))
    return out


def summarise(meta: list[dict], correct: list[float], unparsed: int) -> dict:
    def rates(rows):
        if not rows:
            return {"n": 0}
        by_item: dict[str, list[float]] = {}
        for m, c in rows:
            by_item.setdefault(m["item_id"], []).append(c)
        per_item = [sum(v) / len(v) for v in by_item.values()]
        return {"n_rotations": len(rows), "n_items": len(by_item),
                "accuracy": float(np.mean([c for _, c in rows])),
                "frac_all_four": float(np.mean([p >= 0.999 for p in per_item])),
                "frac_none": float(np.mean([p <= 0.001 for p in per_item]))}

    pairs = list(zip(meta, correct))
    out = rates(pairs)
    out["unparsed"] = unparsed
    out["unparsed_rate"] = unparsed / max(len(meta), 1)
    out["by_dataset"] = {ds: rates([(m, c) for m, c in pairs if m["dataset"] == ds])
                         for ds in sorted({m["dataset"] for m, _ in pairs})}
    out["by_gold_position"] = {
        LETTERS[g]: rates([(m, c) for m, c in pairs if m["gold_index"] == g])["accuracy"]
        for g in range(4)}
    picks = {L: 0 for L in LETTERS}
    for m, _ in pairs:
        if m.get("picked") in picks:
            picks[m["picked"]] += 1
    tot = sum(picks.values()) or 1
    out["answer_distribution"] = {k: round(v / tot, 4) for k, v in picks.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--reader", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    items = [i for i in bedtime.load(split="test", limit=None)
             if i.task_type == "differentiation"]
    items.sort(key=lambda i: i.id)
    if args.limit:
        items = items[: args.limit]
    caps = load_captions(args.arm)
    label = EVIDENCE_LABEL.get(args.arm, DEFAULT_EVIDENCE_LABEL)

    missing = [i.id for i in items if i.extra["caption_key"] not in caps]
    if missing:
        raise SystemExit(f"{len(missing)} items have no caption for arm {args.arm}")

    prompts, meta = [], []
    for item in items:
        caption = caps[item.extra["caption_key"]]
        for gold_index, rot in rotated_items(item):
            prompts.append(build_answer_messages(rot, caption, "evidence", label)[0]["content"])
            meta.append({"item_id": item.id, "dataset": item.domain,
                         "gold_index": gold_index})
    print(f"{args.arm}: {len(items)} items -> {len(prompts)} prompts", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.reader)
    chats = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    llm = LLM(model=args.reader, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    outs = llm.generate(chats, SamplingParams(n=1, temperature=0.0, max_tokens=8))

    correct, unparsed = [], 0
    for o, m in zip(outs, meta):
        hit = LETTER_RE.search(o.outputs[0].text.strip())
        if hit is None:
            unparsed += 1
            correct.append(0.0)
            m["picked"] = None
        else:
            m["picked"] = hit.group(1)
            correct.append(float(LETTERS.index(hit.group(1)) == m["gold_index"]))

    result = {"arm": args.arm, "reader": args.reader, "task": "differentiation",
              "rotations": 4, "answer_style": "evidence", "evidence_label": label,
              **summarise(meta, correct, unparsed)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.arm}.json").write_text(json.dumps(result, indent=1))
    with (out_dir / f"{args.arm}.jsonl").open("w") as f:
        for m, c in zip(meta, correct):
            f.write(json.dumps({**m, "correct": c}) + "\n")
    print(json.dumps({k: result[k] for k in
                      ("accuracy", "frac_all_four", "by_dataset", "answer_distribution")},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
