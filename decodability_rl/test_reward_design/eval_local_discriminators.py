from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decodability_rl.test_reward_design import prompts as P

LETTERS = ("A", "B", "C", "D")
LETTER_RE = re.compile(r"\b([ABCD])\b")
CONDITIONS = ("ds_caption", "caption", "oracle", "empty")
COND_FIELD = {"ds_caption": "ds_caption", "caption": "caption", "oracle": "oracle_caption"}


def build_items(records: list[dict]) -> list[dict]:
    items = []
    for qi, rec in enumerate(records, 1):
        for rot in rec["rotations"]:
            for cond in CONDITIONS:
                caption = "" if cond == "empty" else rec[COND_FIELD[cond]]
                items.append(
                    {
                        "question": qi,
                        "series_id": rec["id"],
                        "gold_letter": rot["gold_letter"],
                        "condition": cond,
                        "prompt": P.build_discriminator_prompt(
                            caption, rot["option_texts"], variant="neutral",
                            answer_format="letter"
                        ),
                    }
                )
    return items


class Scorer:
    def __init__(self, model_id: str, dtype: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map="auto"
        )
        self.model.eval()
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"{model_id}: {letter!r} is not a single token")
            self.letter_ids.append(ids[0])

    def letter_probs(self, prompt: str) -> list[float]:
        text = self.tok.apply_chat_template(
            [
                {"role": "system", "content": P.DISCRIM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return self.torch.softmax(logits[self.letter_ids].float(), dim=-1).tolist()

    def generate_answer(
        self, prompts: list[str], enable_thinking: bool, max_new_tokens: int
    ) -> list[dict]:
        texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": P.DISCRIM_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            kwargs = {"tokenize": False, "add_generation_prompt": True}
            try:
                texts.append(
                    self.tok.apply_chat_template(
                        messages, enable_thinking=enable_thinking, **kwargs
                    )
                )
            except TypeError:
                texts.append(self.tok.apply_chat_template(messages, **kwargs))

        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        inputs = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        results = []
        for row in gen:
            n_tokens = int((row != self.tok.pad_token_id).sum())
            text = self.tok.decode(row, skip_special_tokens=True)
            closed = "</think>" in text
            if enable_thinking and not closed:
                results.append(
                    {
                        "pred": None,
                        "truncated": True,
                        "n_gen_tokens": n_tokens,
                        "thinking_tokens": n_tokens,
                        "raw_tail": text.strip()[-200:],
                    }
                )
                continue
            tail = text.rsplit("</think>", 1)[-1] if closed else text
            m = LETTER_RE.findall(tail)
            results.append(
                {
                    "pred": m[-1] if m else None,
                    "truncated": False,
                    "n_gen_tokens": n_tokens,
                    "thinking_tokens": len(
                        self.tok.encode(text.split("</think>")[0], add_special_tokens=False)
                    )
                    if closed
                    else 0,
                    "raw_tail": tail.strip()[:200],
                }
            )
        return results

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(here / "results_realneg" / "results.json"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument(
        "--mode",
        default="logits",
        choices=["logits", "think", "nothink"],
        help="logits = read the A/B/C/D distribution at the first assistant token "
        "(one forward pass, non-thinking models only); think = generate with the "
        "reasoning block enabled and parse the final letter; nothink = generate with "
        "thinking explicitly disabled (the like-for-like control for hybrid models)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag-suffix", default="", help="appended to the output filename")
    args = ap.parse_args()

    records = json.loads(Path(args.results).read_text())["records"]
    items = build_items(records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(records)} questions x 4 rotations x {len(CONDITIONS)} conditions "
          f"= {len(items)} forward passes per model", flush=True)

    for model_id in args.models:
        tag = model_id.replace("/", "__") + args.tag_suffix
        out_path = out_dir / f"{tag}.json"
        if out_path.exists():
            print(f"[skip] {model_id} (already scored)", flush=True)
            continue
        print(f"\n=== {model_id}", flush=True)
        t0 = time.time()
        try:
            scorer = Scorer(model_id, dtype=args.dtype)
        except Exception as e:
            print(f"[FAIL] could not load {model_id}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"    loaded in {time.time()-t0:.0f}s", flush=True)

        rows = []
        t0 = time.time()
        if args.mode == "logits":
            for i, it in enumerate(items, 1):
                probs = scorer.letter_probs(it["prompt"])
                pred = LETTERS[int(np.argmax(probs))]
                rows.append(
                    {
                        k: it[k]
                        for k in ("question", "series_id", "gold_letter", "condition")
                    } | {
                        "pred": pred,
                        "correct": pred == it["gold_letter"],
                        "p_gold": round(probs[LETTERS.index(it["gold_letter"])], 4),
                        "letter_probs": {l: round(p, 4) for l, p in zip(LETTERS, probs)},
                    }
                )
                if i % 40 == 0:
                    print(f"    {i}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
        else:
            think = args.mode == "think"
            for start in range(0, len(items), args.batch_size):
                batch = items[start : start + args.batch_size]
                outs = scorer.generate_answer(
                    [it["prompt"] for it in batch], think, args.max_new_tokens
                )
                for it, o in zip(batch, outs):
                    rows.append(
                        {
                            k: it[k]
                            for k in ("question", "series_id", "gold_letter", "condition")
                        } | {
                            "pred": o["pred"],
                            "correct": o["pred"] == it["gold_letter"],
                            "p_gold": 1.0 if o["pred"] == it["gold_letter"] else 0.0,
                            "truncated": o["truncated"],
                            "n_gen_tokens": o["n_gen_tokens"],
                            "thinking_tokens": o["thinking_tokens"],
                            "raw_tail": o["raw_tail"],
                        }
                    )
                done = min(start + args.batch_size, len(items))
                print(f"    {done}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
        elapsed = time.time() - t0
        scorer.close()

        summary = {}
        for cond in CONDITIONS:
            sub = [r for r in rows if r["condition"] == cond]
            summary[cond] = {
                "n": len(sub),
                "accuracy": round(float(np.mean([r["correct"] for r in sub])), 4),
                "mean_p_gold": round(float(np.mean([r["p_gold"] for r in sub])), 4),
                "pick_distribution": {
                    l: sum(1 for r in sub if r["pred"] == l) for l in LETTERS
                },
            }
        summary["separation_ds_minus_empty"] = round(
            summary["ds_caption"]["accuracy"] - summary["empty"]["accuracy"], 4
        )
        summary["separation_vlmcap_minus_empty"] = round(
            summary["caption"]["accuracy"] - summary["empty"]["accuracy"], 4
        )
        summary["mode"] = args.mode
        summary["seconds_total"] = round(elapsed, 1)
        summary["unparsed"] = sum(1 for r in rows if r["pred"] is None)
        summary["truncated"] = sum(1 for r in rows if r.get("truncated"))
        parsed = [r for r in rows if r["pred"] is not None]
        summary["accuracy_parsed_only"] = (
            round(float(np.mean([r["correct"] for r in parsed])), 4) if parsed else None
        )
        if args.mode != "logits":
            summary["mean_gen_tokens"] = round(
                float(np.mean([r["n_gen_tokens"] for r in rows])), 1
            )
            summary["mean_thinking_tokens"] = round(
                float(np.mean([r["thinking_tokens"] for r in rows])), 1
            )
        out_path.write_text(
            json.dumps(
                {"model": model_id, "mode": args.mode, "summary": summary, "rows": rows},
                indent=2,
            )
        )
        print(f"    {model_id}: " + "  ".join(
            f"{c}={summary[c]['accuracy']:.1%}" for c in CONDITIONS
        ), flush=True)
        print(f"    separation (ds - empty) = {summary['separation_ds_minus_empty']:+.1%}",
              flush=True)
        print(f"    wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
