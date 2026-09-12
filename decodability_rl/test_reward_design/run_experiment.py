from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decodability_rl.test_reward_design import prompts as P
from decodability_rl.test_reward_design.perturb import (
    DISTRACTOR_TYPES,
    make_all_distractors,
)
from decodability_rl.test_reward_design.oracle import oracle_caption
from decodability_rl.test_reward_design.hard_negatives import (
    build_pool,
    nearest_negatives,
)

LETTERS = ("A", "B", "C", "D")


def select_fragments(
    path: Path,
    n: int,
    seed: int,
    min_len: int,
    max_len: int = 10**9,
    distractors: str = "derived",
) -> list[dict]:
    all_rows = [json.loads(line) for line in path.open()]
    rows = [r for r in all_rows if min_len <= r["len"] <= max_len]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))

    picked: list[dict] = []
    for idx in order:
        row = rows[int(idx)]
        y = np.asarray(row["series"], dtype=float)
        try:
            dset = make_all_distractors(y, np.random.default_rng(seed * 1000 + int(idx)))
        except ValueError:
            continue
        row = dict(row)
        row["_distractors"] = {k: v.tolist() for k, v in dset.items()}
        picked.append(row)
        if len(picked) == n:
            break
    if len(picked) < n:
        raise RuntimeError(f"only found {len(picked)}/{n} usable fragments")

    if distractors == "real":
        chosen_ids = {r["id"] for r in picked}
        for row in picked:
            pool, zfeats = build_pool(all_rows, row["len"])
            negs = nearest_negatives(row, pool, zfeats, n=3, exclude_ids=chosen_ids)
            row["_distractors"] = {
                f"real_{i+1}(id={neg['id']})": list(map(float, neg["series"]))
                for i, neg in enumerate(negs)
            }
    return picked


def build_question(row: dict, rng: np.random.Generator) -> dict:
    true_series = list(map(float, row["series"]))
    decimals = P.decimals_for(true_series)

    kinds = list(row["_distractors"].keys())
    d_order = [kinds[int(i)] for i in rng.permutation(len(kinds))]
    distractors = [(k, row["_distractors"][k]) for k in d_order]

    rotations = []
    for gold_idx in range(4):
        slots: list[tuple[str, list]] = list(distractors)
        slots.insert(gold_idx, ("true", true_series))
        rotations.append(
            {
                "gold_letter": LETTERS[gold_idx],
                "option_kinds": [kind for kind, _ in slots],
                "option_texts": [P.format_series(v, decimals) for _, v in slots],
            }
        )
    return {
        "id": row["id"],
        "image_path": row["image_path"],
        "len": row["len"],
        "decimals": decimals,
        "display_kinds": ["true"] + [k for k, _ in distractors],
        "display_series": [true_series] + [v for _, v in distractors],
        "rotations": rotations,
    }


def caption_all(image_paths: list[Path], model_dir: str, max_new_tokens: int) -> list[str]:
    from ts_cap.qwen_vl_local import QwenVLLocal

    vlm = QwenVLLocal(
        model_dir,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        device_map="cuda:0",
    )
    captions: list[str] = []
    for path in image_paths:
        captions.append(
            vlm.generate(P.CAPTION_PROMPT, image_path=path, system=P.CAPTION_SYSTEM)
        )
    del vlm
    import torch

    torch.cuda.empty_cache()
    return captions


class Discriminator:

    def __init__(self, model_dir: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype="auto", device_map="cuda:0"
        )
        self.model.eval()
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"letter {letter!r} is not a single token")
            self.letter_ids.append(ids[0])

    def ask(self, prompt: str) -> dict:
        messages = [
            {"role": "system", "content": P.DISCRIM_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        raw = self.tok.decode(
            out.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        first = out.scores[0][0]
        probs = self.torch.softmax(first[self.letter_ids].float(), dim=-1).tolist()
        match = re.search(r"\b([ABCD])\b", raw) or re.search(r"([ABCD])", raw)
        return {
            "raw": raw,
            "letter": match.group(1) if match else None,
            "letter_probs": {l: round(p, 4) for l, p in zip(LETTERS, probs)},
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", default=str(REPO_ROOT / "out_10k_xdomain" / "fragments.jsonl"))
    ap.add_argument("--image-root", default=str(REPO_ROOT / "out_10k_xdomain"))
    ap.add_argument("--captioner", default=str(REPO_ROOT / "models" / "Qwen2.5-VL-3B-Instruct"))
    ap.add_argument("--discriminator", default=str(REPO_ROOT / "models" / "Qwen2.5-3B-Instruct"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=10**9)
    ap.add_argument(
        "--distractors",
        default="derived",
        choices=["derived", "real"],
        help="derived = rearrangements of the true series (leaks: a strong reader picks "
        "the 'un-corrupted' one without the caption); real = other real series matched "
        "on mean/std/min/max, so every option is equally plausible",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] selecting {args.n} fragments ...", flush=True)
    rows = select_fragments(
        Path(args.fragments), args.n, args.seed, args.min_len, args.max_len,
        args.distractors,
    )
    image_paths = [Path(args.image_root) / r["image_path"] for r in rows]
    print("      ids:", [r["id"] for r in rows], flush=True)

    print("[2/4] captioning with the VLM ...", flush=True)
    cache_path = out_dir / "captions.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if all(str(r["id"]) in cache for r in rows):
        print("      (reusing cached captions)", flush=True)
        captions = [cache[str(r["id"])] for r in rows]
    else:
        captions = caption_all(image_paths, args.captioner, args.max_new_tokens)
        cache.update({str(r["id"]): c for r, c in zip(rows, captions)})
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    for row, cap in zip(rows, captions):
        row["caption"] = cap

    print("[3/4] building questions ...", flush=True)
    rng = np.random.default_rng(args.seed + 7)
    questions = [build_question(r, rng) for r in rows]

    print("[4/4] querying the discriminator ...", flush=True)
    disc = Discriminator(args.discriminator)
    mismatched = captions[1:] + captions[:1]

    conds = ("caption", "ds_caption", "oracle", "mismatched", "empty")
    records = []
    for i, (row, q) in enumerate(zip(rows, questions)):
        rec = {
            **q,
            "caption": row["caption"],
            "ds_caption": row.get("ds_caption", ""),
            "oracle_caption": oracle_caption(row["series"], q["decimals"]),
            "scores": {},
        }
        caps = {
            "caption": row["caption"],
            "ds_caption": rec["ds_caption"],
            "oracle": rec["oracle_caption"],
            "mismatched": mismatched[i],
            "empty": "",
        }
        for rot in rec["rotations"]:
            rot["conditions"] = {}
            for cond in conds:
                prompt = P.build_discriminator_prompt(caps[cond], rot["option_texts"])
                ans = disc.ask(prompt)
                ans["correct"] = ans["letter"] == rot["gold_letter"]
                ans["p_gold"] = ans["letter_probs"][rot["gold_letter"]]
                if cond == "caption" and rot["gold_letter"] == "A":
                    rec["prompt"] = prompt
                rot["conditions"][cond] = ans
        for cond in conds:
            rec["scores"][cond] = {
                "accuracy": round(
                    float(np.mean([r["conditions"][cond]["correct"] for r in rec["rotations"]])), 4
                ),
                "mean_p_gold": round(
                    float(np.mean([r["conditions"][cond]["p_gold"] for r in rec["rotations"]])), 4
                ),
            }
        records.append(rec)
        picks = "".join(r["conditions"]["caption"]["letter"] or "?" for r in rec["rotations"])
        print(
            f"      q{i+1:02d} id={q['id']} picks(A/B/C/D rot)={picks} "
            f"acc={rec['scores']['caption']['accuracy']:.2f} "
            f"p_gold={rec['scores']['caption']['mean_p_gold']:.3f}",
            flush=True,
        )

    def _flat(cond, key):
        return [r["conditions"][cond][key] for rec in records for r in rec["rotations"]]

    from collections import Counter

    summary = {
        "n_questions": len(records),
        "n_rotations": 4,
        "n_calls_per_condition": len(records) * 4,
        "chance": 0.25,
        "conditions": {
            cond: {
                "accuracy": round(float(np.mean(_flat(cond, "correct"))), 4),
                "mean_p_gold": round(float(np.mean(_flat(cond, "p_gold"))), 4),
                "pick_distribution": dict(
                    Counter(_flat(cond, "letter")).most_common()
                ),
            }
            for cond in conds
        },
        "config": vars(args),
    }
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary["conditions"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {out_dir/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
