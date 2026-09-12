from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from decodability_rl.rl.judge_prompts import (
    JUDGE_SYSTEM,
    build_judge_prompt,
    parse_score,
)
from decodability_rl.rl.reward_server_decodability import log_group_stats
from decodability_rl.test_reward_design import prompts as P

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("judge-reward")

SCORE_LO, SCORE_HI = 1, 10


class JudgeRewardModel:
    def __init__(self, args) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.reward_pretrain)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.reward_pretrain, dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.n_calls = 0
        self.hist = dict.fromkeys(range(SCORE_LO, SCORE_HI + 1), 0)
        logger.info("loaded %s", args.reward_pretrain)

    def _chat(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    def _prompt_for(self, caption: str, payload: dict) -> str:
        decimals = payload.get("decimals", 2)
        series_text = P.format_series(payload["true"], decimals)
        return self._chat(build_judge_prompt(caption, series_text))

    @torch.no_grad()
    def _grade(self, texts: list[str]) -> tuple[list[float | None], int]:
        out: list[float | None] = []
        unparsed = 0
        bs = self.args.batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            gen = self.model.generate(
                **enc, max_new_tokens=self.args.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
            for row in gen[:, enc["input_ids"].shape[1]:]:
                reply = self.tok.decode(row, skip_special_tokens=True).strip()
                s = parse_score(reply, SCORE_LO, SCORE_HI)
                if s is None:
                    unparsed += 1
                else:
                    self.hist[int(s)] += 1
                out.append(s)
        return out, unparsed

    def get_reward(self, prompts, queries=None, labels=None):
        t0 = time.time()
        captions = [str(c) for c, _ in prompts]
        texts = [
            self._prompt_for(c, json.loads(qa[0][0])) for c, qa in prompts
        ]
        scores, unparsed = self._grade(texts)

        floor = float(SCORE_LO)
        filled = [floor if s is None else s for s in scores]
        norm = [(s - SCORE_LO) / (SCORE_HI - SCORE_LO) for s in filled]
        rewards = [v * self.args.reward_scale for v in norm]

        self.n_calls += 1
        if self.n_calls % self.args.log_every == 0 or self.n_calls == 1:
            lens = np.array([len(c) for c in captions], dtype=float)
            r = np.array(rewards, dtype=float)
            rho = _spearman(lens, r)
            used = {k: v for k, v in self.hist.items() if v}
            logger.info(
                "batch=%d captions | %.1fs | reward mean=%.4f min=%.4f max=%.4f | "
                "unparsed=%d/%d | rho(len,reward)=%s | scores so far=%s",
                len(prompts), time.time() - t0, float(np.mean(r)), float(np.min(r)),
                float(np.max(r)), unparsed, len(texts),
                "n/a" if rho is None else f"{rho:+.3f}", used,
            )
        return rewards, {}


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return None
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    return r


SELFTEST = [
    ("accurate+specific",
     "The series rises steadily from about 2 to about 9 over the first two thirds, then "
     "flattens near 9 for the rest, with a small dip just after the midpoint."),
    ("generic",
     "The series changes over time, with some ups and downs and a general pattern."),
    ("false",
     "The series falls steadily throughout, ending far below where it started, with a "
     "sharp spike at the very beginning."),
]
SELFTEST_SERIES = [2.0, 2.6, 3.3, 4.1, 4.8, 5.6, 6.3, 5.9, 6.8, 7.5, 8.2, 8.8,
                   9.0, 8.9, 9.1, 9.0, 9.1, 8.9, 9.0, 9.1]


def run_selftest(model: JudgeRewardModel) -> None:
    payload = {"true": SELFTEST_SERIES, "decimals": 1}
    texts = [model._prompt_for(c, payload) for _, c in SELFTEST]
    scores, unparsed = model._grade(texts)
    for (label, _), s in zip(SELFTEST, scores):
        logger.info("  selftest %-18s -> %s", label, s)
    if unparsed:
        raise RuntimeError(f"judge produced {unparsed} unparseable replies on the "
                           f"self-test; the rubric or the chat template is wrong")
    if not all(s is not None for s in scores):
        raise RuntimeError("self-test scores incomplete")
    acc, gen, false = scores
    if not (acc > gen and acc > false):
        raise RuntimeError(
            f"judge does not rank an accurate caption above a generic one ({acc} vs "
            f"{gen}) and a false one ({false}). A rubric that cannot separate these "
            "cannot separate a policy's samples either. Refusing to start."
        )
    logger.info("self-test PASSED (accurate %.0f > generic %.0f, false %.0f)",
                acc, gen, false)


def run_worker(args) -> None:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_rank)
    port = args.worker_base_port + args.worker_rank
    logger.info("worker %d on GPU %d, port %d", args.worker_rank, args.worker_rank, port)

    model = JudgeRewardModel(args)
    if not args.no_selftest:
        run_selftest(model)

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.post("/get_reward")
    async def get_reward(request: dict):
        rewards, _ = model.get_reward(request.get("prompts"), request.get("query"),
                                      request.get("labels"))
        return JSONResponse({"rewards": rewards, "worker": args.worker_rank})

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "model": args.reward_pretrain})

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def run_master(args) -> None:
    import asyncio

    import httpx

    urls = [f"http://127.0.0.1:{args.worker_base_port + r}/get_reward"
            for r in range(args.num_workers)]
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    @app.post("/get_reward")
    async def get_reward(request: dict):
        prompts = request.get("prompts") or []
        if not prompts:
            return JSONResponse({"rewards": []})
        t0 = time.time()
        n = len(prompts)
        size = (n + args.num_workers - 1) // args.num_workers
        chunks = [prompts[i : i + size] for i in range(0, n, size)]
        async with httpx.AsyncClient(timeout=1800) as client:
            resps = await asyncio.gather(
                *[client.post(urls[i], json={"prompts": c, "query": None, "labels": None})
                  for i, c in enumerate(chunks)],
                return_exceptions=True,
            )
        rewards: list[float] = []
        for i, r in enumerate(resps):
            if isinstance(r, Exception) or r.status_code != 200:
                logger.error("worker %d failed: %s", i, r)
                return JSONResponse(status_code=500,
                                    content={"error": f"worker {i} failed"})
            rewards.extend(r.json()["rewards"])
        logger.info("master: %d captions over %d workers in %.1fs (mean %.4f)",
                    n, len(chunks), time.time() - t0, float(np.mean(rewards)))
        log_group_stats(prompts, rewards, args.group_stats_jsonl)
        return JSONResponse({"rewards": rewards})

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "workers": args.num_workers})

    logger.info("master on 0.0.0.0:%d fanning out to %s", args.port, urls)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reward-pretrain", default="Qwen/Qwen2.5-14B-Instruct",
                    help="the judge. Defaults to the model the decodability reward uses "
                         "as its reader, so the two rewards differ in form and not in "
                         "who is judging.")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=8,
                    help="the rubric asks for an integer alone; more only invites prose "
                         "the parser then has to survive")
    ap.add_argument("--reward-scale", type=float, default=2.0,
                    help="matches the decodability server, which multiplies its accuracy "
                         "by 2 as upstream CapRL did")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--group-stats-jsonl", default=None)
    ap.add_argument("--role", choices=["master", "worker"], default="master")
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-rank", type=int, default=0)
    ap.add_argument("--worker-base-port", type=int, default=8900)
    ap.add_argument("--no-selftest", action="store_true",
                    help="skip the ordering check. Only for debugging the transport; a "
                         "training run should never set it.")
    ap.add_argument("--selftest", action="store_true",
                    help="load the judge, run the ordering check, print the scores and "
                         "exit without serving")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(JudgeRewardModel(args))
        return
    if args.role == "worker":
        run_worker(args)
    else:
        run_master(args)


if __name__ == "__main__":
    main()
