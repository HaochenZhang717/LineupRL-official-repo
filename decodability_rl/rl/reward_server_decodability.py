from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from decodability_rl.test_reward_design import prompts as P

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("reward_server_decodability")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_N_OPTIONS = 4


def letter_regexes(n_options: int) -> tuple[re.Pattern, re.Pattern]:
    letters = "".join(P.letters_for(n_options))
    bare = re.compile(rf"^\W*([{letters}])\W*$")
    return bare, re.compile(rf"\b([{letters}])\b")


_PRONOUN_I = re.compile(r"\s+[a-z]")


def answer_tag_re(n_options: int) -> re.Pattern:
    letters = "".join(P.letters_for(n_options))
    return re.compile(rf"<answer>\s*([{letters}])\s*</answer>", re.I)


def parse_letter(
    reply: str,
    bare_re: re.Pattern,
    fallback_re: re.Pattern,
    tag_re: re.Pattern | None = None,
) -> tuple[str | None, bool]:
    if tag_re is not None:
        m = tag_re.search(reply)
        if m:
            return m.group(1).upper(), True
    m = bare_re.match(reply.strip())
    if m:
        return m.group(1), False
    for m in fallback_re.finditer(reply):
        if m.group(1) == "I" and _PRONOUN_I.match(reply[m.end():]):
            continue
        return m.group(1), False
    return None, False


class DecodabilityRewardModel:
    def __init__(self, args) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.n_options = int(getattr(args, "n_options", DEFAULT_N_OPTIONS))
        self.letters = P.letters_for(self.n_options)
        self.bare_re, self.letter_re = letter_regexes(self.n_options)
        self.answer_format = getattr(args, "answer_format", "tag")
        self.tag_re = answer_tag_re(self.n_options) if self.answer_format == "tag" else None
        self.n_tagged = 0
        self.n_parsed = 0
        self.tok = AutoTokenizer.from_pretrained(args.reward_pretrain)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.reward_pretrain, dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.letter_ids = []
        for letter in self.letters:
            ids = self.tok.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"{letter!r} is not a single token for this tokenizer")
            self.letter_ids.append(ids[0])
        self.n_calls = 0
        self.picks = dict.fromkeys(self.letters, 0)
        logger.info(
            "loaded %s (%d-way, %s answers)",
            args.reward_pretrain, self.n_options, self.answer_format,
        )

        self._api = None
        if getattr(args, "mask_api_model", ""):
            from openai import OpenAI

            key = Path(args.mask_api_key_file).read_text().strip()
            self._api = OpenAI(api_key=key, base_url=args.mask_api_base)
            logger.info("masking via %s at %s (concurrency %d)", args.mask_api_model,
                        args.mask_api_base, args.mask_api_concurrency)
        if float(getattr(args, "number_mask_weight", 0.0) or 0.0) > 0:
            self._selftest_llm_mask()

    def _selftest_llm_mask(self) -> None:
        probes = [
            ("The series peaks at 3650 around time step 18.", "3650", "time step 18"),
            ("It reaches a low point of 0.42 at approximately time 12.", "0.42", "time 12"),
        ]
        try:
            masked, fb, _ = self._mask_batch([p[0] for p in probes])
            ok = all(
                val not in m and keep in m for m, (_, val, keep) in zip(masked, probes)
            )
        except Exception as e:
            ok, fb = False, len(probes)
            logger.error("LLM mask self-test raised %s: %s", type(e).__name__, e)
        if ok and fb == 0:
            logger.info("mask self-test PASSED")
            for m in masked:
                logger.info("    %s", m)
        else:
            raise RuntimeError(
                f"mask self-test FAILED (failsafe={fb}). There is no second masker to fall "
                "back to, so a broken masker would silently mask every number of every "
                "caption for the whole run. Refusing to start."
            )

    def _rotation_prompts(self, caption: str, payload: dict) -> list[str]:
        decimals = payload.get("decimals", 2)
        true_txt = P.format_series(payload["true"], decimals)
        distractors = payload["distractors"]
        if len(distractors) < self.n_options - 1:
            raise ValueError(
                f"item {payload.get('id')} has {len(distractors)} distractors but "
                f"--n-options {self.n_options} needs {self.n_options - 1}; rebuild the "
                "dataset with build_decodability_dataset --n-distractors"
            )
        distr_txt = [
            P.format_series(s, decimals) for s in distractors[: self.n_options - 1]
        ]
        golds = payload.get("rotations") or list(range(self.n_options))
        prompts = []
        for gold_idx in golds:
            options = list(distr_txt)
            options.insert(gold_idx, true_txt)
            prompts.append(
                P.build_discriminator_prompt(
                    caption, options, variant="neutral", answer_format=self.answer_format
                )
            )
        return prompts

    def _chat(self, prompt: str) -> str:
        return self.tok.apply_chat_template(
            [
                {"role": "system", "content": P.SYSTEMS[self.answer_format]},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def _score_generate(
        self, texts: list[str], gold_idx: list[int]
    ) -> tuple[list[float], int]:
        out: list[float] = []
        unparsed = 0
        bs = self.args.batch_size
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            inputs = self.tok(chunk, return_tensors="pt", padding=True).to(
                self.model.device
            )
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            new = gen[:, inputs["input_ids"].shape[1] :]
            replies = self.tok.batch_decode(new, skip_special_tokens=True)
            for reply, g in zip(replies, gold_idx[start : start + bs]):
                letter, tagged = parse_letter(
                    reply, self.bare_re, self.letter_re, self.tag_re
                )
                if letter is None:
                    unparsed += 1
                    out.append(0.0)
                else:
                    self.picks[letter] += 1
                    self.n_parsed += 1
                    self.n_tagged += int(tagged)
                    out.append(1.0 if letter == self.letters[g] else 0.0)
        return out, unparsed

    @torch.no_grad()
    def _score_logits(self, texts: list[str], gold_idx: list[int]) -> tuple[list[float], int]:
        out = []
        bs = self.args.batch_size
        for start in range(0, len(texts), bs):
            chunk = texts[start : start + bs]
            inputs = self.tok(chunk, return_tensors="pt", padding=True).to(
                self.model.device
            )
            logits = self.model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits[:, self.letter_ids].float(), dim=-1)
            for row, g in zip(probs, gold_idx[start : start + bs]):
                out.append(float(row[g]))
        return out, 0

    def _mask_replies_api(self, chunks: list[str]) -> list[str] | None:
        import concurrent.futures as cf

        from decodability_rl.rl import llm_mask as LM

        def ask(chunk: str) -> str | None:
            for attempt in range(4):
                try:
                    r = self._api.chat.completions.create(
                        model=self.args.mask_api_model,
                        temperature=0,
                        reasoning_effort="none",
                        messages=[{"role": "system", "content": LM.SYSTEM},
                                  {"role": "user", "content": LM.build_prompt(chunk)}],
                    )
                    return r.choices[0].message.content or ""
                except Exception:
                    if attempt == 3:
                        return None
                    time.sleep(2 ** attempt)
            return None

        with cf.ThreadPoolExecutor(max_workers=self.args.mask_api_concurrency) as ex:
            replies = list(ex.map(ask, chunks))

        n_failed = sum(r is None for r in replies)
        if n_failed > 0.1 * len(replies):
            logger.error(
                "masking API failed on %d/%d chunks -- falling back to the local reader "
                "for this batch", n_failed, len(replies),
            )
            return None
        if n_failed:
            logger.warning("masking API failed on %d/%d chunks", n_failed, len(replies))
        return ["" if r is None else r for r in replies]

    @torch.no_grad()
    def _mask_batch(self, captions: list[str]) -> tuple[list[str], int, int]:
        if float(getattr(self.args, "number_mask_weight", 0.0) or 0.0) <= 0:
            return list(captions), 0, 0

        from decodability_rl.rl import llm_mask as LM

        self._unnamed_example = None
        chunks, owner = [], []
        for i, cap in enumerate(captions):
            for ch in LM.split_for_masking(cap):
                chunks.append(ch)
                owner.append(i)

        masked_chunks: list[str] = [""] * len(chunks)
        unnamed = 0

        if getattr(self, "_api", None) is not None:
            replies = self._mask_replies_api(chunks)
            if replies is not None:
                for i, (ch, reply) in enumerate(zip(chunks, replies)):
                    masked_chunks[i], n_un = LM.mask_one(ch, reply)
                    unnamed += n_un
                out = [""] * len(captions)
                for ch, i in zip(masked_chunks, owner):
                    out[i] += ch
                return out, unnamed, sum(LM.count_numbers(c) for c in captions)

        texts = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": LM.SYSTEM},
                 {"role": "user", "content": LM.build_prompt(ch)}],
                tokenize=False, add_generation_prompt=True)
            for ch in chunks
        ]

        bs = self.args.batch_size
        for start in range(0, len(texts), bs):
            sl = slice(start, start + bs)
            enc = self.tok(texts[sl], return_tensors="pt", padding=True).to(
                self.model.device
            )
            try:
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max(LM.token_budget(c) for c in chunks[sl]),
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
                replies = self.tok.batch_decode(
                    gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
            except Exception as e:
                logger.error("masking failed (%s: %s) -- hiding every number in this "
                             "chunk instead", type(e).__name__, e)
                replies = [""] * len(chunks[sl])

            for j, reply in enumerate(replies):
                ch = chunks[start + j]
                out_chunk, n_unnamed = LM.mask_one(ch, reply)
                masked_chunks[start + j] = out_chunk
                unnamed += n_unnamed
                if n_unnamed and self._unnamed_example is None:
                    self._unnamed_example = (ch[:200], (reply or "").strip()[:200])

        out = [""] * len(captions)
        for ch, i in zip(masked_chunks, owner):
            out[i] += ch
        return out, unnamed, sum(LM.count_numbers(c) for c in captions)

    def _caption_views(self, caption: str, masked: str) -> list[tuple[str, float]]:
        w = float(getattr(self.args, "number_mask_weight", 0.0) or 0.0)
        if w <= 0:
            return [(caption, 1.0)]
        if w >= 1:
            return [(masked, 1.0)]
        return [(caption, 1.0 - w), (masked, w)]

    def get_reward(self, prompts, queries, labels):
        t0 = time.time()
        texts: list[str] = []
        golds: list[int] = []
        owner: list[int] = []
        weight: list[float] = []
        view_of: list[int] = []
        masked_all, n_unnamed, n_numbers = self._mask_batch(
            [str(c) for c, _ in prompts]
        )
        for i, (caption, qa) in enumerate(prompts):
            payload = json.loads(qa[0][0])
            golds_used = payload.get("rotations") or list(range(self.n_options))
            for v, (view, w) in enumerate(self._caption_views(caption, masked_all[i])):
                for gold_idx, prompt in zip(golds_used, self._rotation_prompts(view, payload)):
                    texts.append(self._chat(prompt))
                    golds.append(gold_idx)
                    owner.append(i)
                    weight.append(w)
                    view_of.append(v)

        scorer = (
            self._score_generate if self.args.score_mode == "generate" else self._score_logits
        )
        p, unparsed = scorer(texts, golds)

        per_sample: list[list[tuple[float, float]]] = [[] for _ in prompts]
        for idx, val, w in zip(owner, p, weight):
            per_sample[idx].append((val, w))

        rewards, accuracies = [], []
        chance = 1.0 / self.n_options
        for vals in per_sample:
            tot_w = sum(w for _, w in vals) or 1.0
            mean_p = float(sum(v * w for v, w in vals) / tot_w)
            accuracies.append(mean_p)
            if self.args.reward_transform == "chance":
                mean_p = max(0.0, (mean_p - chance) / (1.0 - chance))
            elif self.args.reward_transform == "centered":
                mean_p = (mean_p - chance) / (1.0 - chance)
            rewards.append(mean_p * self.args.reward_scale)

        self.n_calls += 1
        if self.n_calls % self.args.log_every == 0 or self.n_calls == 1:
            logger.info(
                "batch=%d captions | %d questions | %.1fs | reward mean=%.4f min=%.4f "
                "max=%.4f | unparsed=%d/%d",
                len(prompts), len(texts), time.time() - t0,
                float(np.mean(rewards)), float(np.min(rewards)), float(np.max(rewards)),
                unparsed, len(texts),
            )
            if any(v for v in view_of):
                from decodability_rl.rl.llm_mask import numeric_density

                full = [x for x, v in zip(p, view_of) if v == 0]
                mask = [x for x, v in zip(p, view_of) if v == 1]
                logger.info(
                    "  acc_full=%.4f (n=%d) | acc_masked=%.4f (n=%d)",
                    float(np.mean(full)) if full else float("nan"), len(full),
                    float(np.mean(mask)) if mask else float("nan"), len(mask),
                )

                raws = [str(c) for c, _ in prompts]
                masks = masked_all
                logger.info(
                    "  MASK | density %.3f -> %.3f | unnamed numbers=%d/%d (%.0f%%)",
                    float(np.mean([numeric_density(r) for r in raws])),
                    float(np.mean([numeric_density(m) for m in masks])),
                    n_unnamed, n_numbers, 100.0 * n_unnamed / max(n_numbers, 1),
                )
                if n_unnamed > 0.2 * n_numbers:
                    logger.warning(
                        "  %d/%d numbers were never named by the reader and are therefore "
                        "left unmasked -- values may be getting through; check its replies",
                        n_unnamed, n_numbers,
                    )
                if self._unnamed_example:
                    ch, rep = self._unnamed_example
                    logger.info("  UNNAMED e.g. chunk: %s", ch.replace("\n", " "))
                    logger.info("               reply: %s", rep.replace("\n", " | "))
                logger.info("  masked sample : %s", masks[0][:200].replace("\n", " "))
            if self.answer_format == "tag":
                logger.info(
                    "  tagged replies %d/%d (%.0f%%)",
                    self.n_tagged, self.n_parsed,
                    100.0 * self.n_tagged / max(self.n_parsed, 1),
                )
            tot = sum(self.picks.values()) or 1
            logger.info(
                "  picks %s  (uniform = %.0f%% each; a drift here is the reward turning "
                "into a position prior)",
                " ".join(f"{k}:{100.0 * v / tot:.0f}%" for k, v in self.picks.items()),
                100.0 / self.n_options,
            )
            logger.info("  sample caption: %s", str(prompts[0][0])[:200].replace("\n", " "))
            logger.info("  rotations as (p, view_weight): %s",
                        [(round(v, 3), round(w, 2)) for v, w in per_sample[0]])
        if unparsed > 0.05 * len(texts):
            logger.warning(
                "%d/%d replies had no A-D letter (>5%%); reward is being driven to 0",
                unparsed, len(texts),
            )
        return rewards, accuracies


def run_worker(args) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_rank)
    port = args.worker_base_port + args.worker_rank
    logger.info("worker %d starting on GPU %d, port %d", args.worker_rank, args.worker_rank, port)

    model = DecodabilityRewardModel(args)
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.post("/get_reward")
    async def get_reward(request: dict):
        rewards, accuracies = model.get_reward(
            request.get("prompts"), request.get("query"), request.get("labels")
        )
        return JSONResponse(
            {"rewards": rewards, "accuracies": accuracies,
             "n_options": model.n_options, "worker": args.worker_rank}
        )

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "model": args.reward_pretrain})

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def log_group_stats(prompts, rewards, jsonl_path: str | None = None) -> None:
    by_id: dict = {}
    for pr, rw in zip(prompts, rewards):
        try:
            pid = json.loads(pr[1][0][0]).get("id")
        except Exception:
            return
        by_id.setdefault(pid, []).append(rw)

    groups = {k: v for k, v in by_id.items() if len(v) > 1}
    if not groups:
        return
    stds = np.array([float(np.std(v)) for v in groups.values()])
    means = np.array([float(np.mean(v)) for v in groups.values()])
    zero = int((stds < 1e-9).sum())

    logger.info(
        "  groups=%d size~%.1f | zero-variance(no gradient)=%d/%d=%.0f%% | "
        "within-group std mean=%.3f | across-group mean std=%.3f",
        len(groups), float(np.mean([len(v) for v in groups.values()])),
        zero, len(groups), 100.0 * zero / len(groups),
        float(stds.mean()), float(means.std()),
    )
    if zero > 0.5 * len(groups):
        logger.warning(
            "  %.0f%% of groups have zero advantage (group means %.2f) -- "
            "%s", 100.0 * zero / len(groups), float(means.mean()),
            "policy may be collapsing" if means.mean() < 1.0 else "reward may be saturating",
        )

    if not jsonl_path:
        return
    rec = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_groups": len(groups),
        "groups": [{"id": k, "r": [round(float(x), 4) for x in v]} for k, v in groups.items()],
        "within_std_mean": round(float(stds.mean()), 4),
        "across_mean_std": round(float(means.std()), 4),
        "zero_var": zero,
        "reward_mean": round(float(np.mean(rewards)), 4),
    }
    try:
        with open(jsonl_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:
        logger.warning("could not append to %s: %s", jsonl_path, e)


def run_master(args) -> None:
    import asyncio

    import random

    import httpx

    urls = [
        f"http://127.0.0.1:{args.worker_base_port + r}/get_reward"
        for r in range(args.num_workers)
    ]
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def pick_rotations(prompts, full: bool = False) -> int:
        k = args.n_options if full else args.n_rotations
        if k >= args.n_options:
            return 0
        per_id, n = {}, 0
        for pr in prompts:
            qa = pr[1]
            payload = json.loads(qa[0][0])
            key = payload.get("id")
            if key not in per_id:
                per_id[key] = sorted(random.sample(range(args.n_options), k))
            payload["rotations"] = per_id[key]
            qa[0][0] = json.dumps(payload, separators=(",", ":"))
            n += 1
        return n

    def resample_negatives(prompts) -> int:
        n_draw = args.resample_negatives
        if n_draw <= 0:
            return 0
        per_id: dict = {}
        n_resampled = 0
        for pr in prompts:
            qa = pr[1]
            payload = json.loads(qa[0][0])
            pool = payload.get("distractor_pool")
            if not pool or len(pool) < n_draw:
                continue
            key = payload.get("id")
            if key not in per_id:
                idx = random.sample(range(len(pool)), n_draw)
                per_id[key] = [pool[i] for i in idx]
            payload["distractors"] = per_id[key]
            payload.pop("distractor_pool", None)
            qa[0][0] = json.dumps(payload, separators=(",", ":"))
            n_resampled += 1
        return n_resampled

    @app.post("/get_reward")
    async def get_reward(request: dict):
        prompts = request.get("prompts") or []
        if not prompts:
            return JSONResponse({"rewards": []})
        t0 = time.time()
        n_resampled = resample_negatives(prompts)
        pick_rotations(prompts, full=bool(request.get("full_rotations")))
        n = len(prompts)
        size = (n + args.num_workers - 1) // args.num_workers
        chunks = [prompts[i : i + size] for i in range(0, n, size)]

        async with httpx.AsyncClient(timeout=1800) as client:
            resps = await asyncio.gather(
                *[
                    client.post(urls[i], json={"prompts": c, "query": None, "labels": None})
                    for i, c in enumerate(chunks)
                ],
                return_exceptions=True,
            )

        rewards: list[float] = []
        accuracies: list[float] = []
        for i, r in enumerate(resps):
            if isinstance(r, Exception) or r.status_code != 200:
                logger.error("worker %d failed: %s", i, r)
                return JSONResponse(status_code=500, content={"error": f"worker {i} failed"})
            body = r.json()
            rewards.extend(body["rewards"])
            accuracies.extend(body.get("accuracies") or [])
        logger.info(
            "master: %d captions over %d workers in %.1fs (mean reward %.4f, resampled %d)",
            n, len(chunks), time.time() - t0, float(np.mean(rewards)), n_resampled,
        )
        log_group_stats(prompts, rewards, args.group_stats_jsonl)
        out = {"rewards": rewards, "n_options": args.n_options}
        if len(accuracies) == len(rewards):
            out["accuracies"] = accuracies
        return JSONResponse(out)

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "workers": args.num_workers})

    logger.info("master on 0.0.0.0:%d fanning out to %s", args.port, urls)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reward_pretrain",
        default=str(REPO / "models" / "Qwen2.5-14B-Instruct"),
        help="the discriminator; Qwen2.5-14B-Instruct was selected by the pilot in "
        "the pilot's reader-selection table (decodability_rl/test_reward_design/README.md)",
    )
    ap.add_argument(
        "--n-options",
        type=int,
        default=DEFAULT_N_OPTIONS,
        help="how many candidate series the reader chooses between (1 true + n-1 "
        "distractors). 4 is decod550; the negatives ablation runs 6/8/10. The dataset "
        "must carry at least n-1 distractors per item, and chance moves to 1/n.",
    )
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--score-mode",
        default="generate",
        choices=["generate", "logits"],
        help="generate = the reader answers and we parse the letter (1/0 per rotation); "
        "logits = diagnostic, read p(gold) at the first assistant token without decoding",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="the answer is one letter; the prompt asks for exactly that, so a few "
        "tokens are enough. Thinking models are out of scope here (they need "
        "thousands and are ~800x slower). 16 rather than 8 "
        "because <answer>C</answer> is 6 tokens plus EOS for the Qwen tokenizers, and 8 "
        "cut the closing tag off as soon as the reader prefixed anything; generation "
        "still stops at EOS, so a bare-letter reply costs what it always did.",
    )
    ap.add_argument(
        "--mask-api-model",
        default="",
        help="hosted model to do the masking, e.g. deepseek-v4-flash. Empty = use the "
             "local reader. Measured on the same 40 captions: the local Qwen2.5-14B never "
             "names 4.7% of numbers and leaks a value in 1 caption of 40; deepseek-v4-flash "
             "names all of them and leaks none. Falls back to the local reader on failure.",
    )
    ap.add_argument("--mask-api-base", default="https://api.deepseek.com/v1")
    ap.add_argument(
        "--mask-api-key-file",
        default=os.environ.get("MASK_API_KEY_FILE", "mask_api_key.txt"),
        help="file holding the API key for --mask-api-model (read only when that is set); "
        "defaults to $MASK_API_KEY_FILE",
    )
    ap.add_argument("--mask-api-concurrency", type=int, default=32)
    ap.add_argument(
        "--number-mask-weight",
        type=float,
        default=0.0,
        help="0 = single-view behaviour. >0 asks the SAME reader the SAME question a "
             "second time on a masked view of the caption, and routes that share of the "
             "reward to it, so the caption cannot win by quoting values that identify the "
             "series. 1.0 = masked only. Any value >0 doubles reader calls per sample.",
    )
    ap.add_argument(
        "--n-rotations",
        type=int,
        default=4,
        help="how many of the 4 positions the true series is tested at, per caption view. "
             "4 is the full rotation; fewer trades reward resolution for speed, which is "
             "the dominant term in the step time. The positions are drawn per image per "
             "request and shared across that image's rollouts, so a lucky position cancels "
             "in the RLOO baseline rather than adding noise to the advantage.",
    )
    ap.add_argument("--reward-scale", type=float, default=2.0)
    ap.add_argument(
        "--reward-transform", default="raw", choices=["raw", "chance", "centered"],
        help="raw = accuracy, and the only correct choice for TRAINING: RLOO already "
        "cancels the chance floor, so raw is k-invariant in the gradient. centered = "
        "(accuracy - 1/k)/(1 - 1/k) rescales advantages by 1/(1 - 1/k) with nothing "
        "downstream to undo it, i.e. a k-dependent learning-rate multiplier; it is for "
        "reporting, not training. chance = centered clipped at 0, not even affine.",
    )
    ap.add_argument(
        "--answer-format", default="tag", choices=["tag", "letter"],
        help="tag = ask for the letter inside <answer></answer>, so the reply parses by "
        "construction; letter = decod550's original bare-letter question.",
    )
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument(
        "--role",
        default="single",
        choices=["single", "master", "worker"],
        help="single = one process, one GPU; master/worker = data-parallel replicas, "
        "one per GPU, with the master splitting each batch across them",
    )
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-rank", type=int, default=0)
    ap.add_argument("--worker-base-port", type=int, default=8900)
    ap.add_argument(
        "--group-stats-jsonl",
        default=None,
        help="append one JSON line per request with EVERY group's full reward vector "
        "(for plotting how the within-group spread evolves). Console output stays a "
        "summary either way.",
    )
    ap.add_argument(
        "--resample-negatives",
        type=int,
        default=-1,
        help="master-only: draw this many negatives from each training item's "
        "distractor_pool per request (0 = use the fixed distractors, -1 = n_options-1, "
        "the only value that yields a well-formed question). Items without a pool -- "
        "i.e. val -- are unaffected.",
    )
    args = ap.parse_args()

    P.letters_for(args.n_options)
    if args.answer_format == "tag" and args.score_mode == "logits":
        raise SystemExit("--score-mode logits requires --answer-format letter")
    if args.resample_negatives < 0:
        args.resample_negatives = args.n_options - 1
    elif 0 < args.resample_negatives != args.n_options - 1:
        raise SystemExit(
            f"--resample-negatives {args.resample_negatives} would build a "
            f"{args.resample_negatives + 1}-way question while --n-options says "
            f"{args.n_options}; pass -1 to track n_options."
        )

    if args.role == "worker":
        run_worker(args)
    elif args.role == "master":
        run_master(args)
    else:
        args.worker_rank = 0
        args.worker_base_port = args.port
        run_worker(args)


if __name__ == "__main__":
    main()
