import argparse
import json
import logging
import re
import time

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import torch.multiprocessing as mp

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import os
import random
import numpy as np
import httpx
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("reward_server_ts")


def get_response_from_query(q: str):
    response_prefix = r"<\|im_start\|>assistant\n"
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end():]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def shuffle_options(question, answer):
    lines = question.split('\n')
    q_text = lines[0]
    options = [o for o in lines[1:] if len(o.strip())]

    pattern = r'-\s*([A-F])\)\s*(.+)'
    original_options = {}
    for opt in options:
        match = re.search(pattern, opt.strip())
        if match:
            original_options[match.group(1)] = match.group(2)
        else:
            raise ValueError(f"ERROR parsing option: {opt!r}")

    if answer not in original_options:
        raise ValueError(f"{answer} not in the candidates")
    correct_answer_text = original_options[answer]

    shuffled_items = list(original_options.items())
    random.shuffle(shuffled_items)

    new_labels = ['A', 'B', 'C', 'D', 'E', 'F']
    new_options = {}
    new_answer = ''
    for i, (_, content) in enumerate(shuffled_items):
        label = new_labels[i]
        new_options[label] = content
        if content == correct_answer_text:
            new_answer = label

    new_question_lines = [q_text]
    for label in new_options:
        new_question_lines.append(f"   - {label}) {new_options[label]}")

    return ['\n'.join(new_question_lines), new_answer]


sampling_params = SamplingParams(
    n=1,
    temperature=0.6,
    top_p=1.0,
    repetition_penalty=1.0,
    max_tokens=10,
    stop_token_ids=[],
)


def parse_easy(answer, gt):
    res = re.compile(r'[A-I]').findall(answer)
    if len(res) > 0:
        return 1 if res[0] == gt else 0
    return 0


class RewardModelProxy:
    def __init__(self, args):
        self.qa_num = args.qa_num
        self.shuffle_qa = args.shuffle_qa
        self.all_qa = args.all_qa
        assert not (self.shuffle_qa and self.all_qa)

        self.llm = LLM(
            model=args.reward_pretrain,
            tensor_parallel_size=args.tp,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(args.reward_pretrain, trust_remote_code=True)

    def get_reward(self, prompts, queries, labels):
        prompt = '''<|im_start|>user
You will be given a caption describing a univariate time series (a line chart of a value versus a time index).
Your task is to answer the multiple-choice question strictly based on the caption, even if the answer may seem obvious from prior knowledge or the question wording.

Ignore any outside knowledge. Do not assume anything the caption does not explicitly or implicitly state.

Example:
Caption: <Caption Start> The series rises steadily from about 10 at the start to a peak near 40 around the middle, then declines to roughly 25 by the end. <Caption End>
Question: At which point in time is the series at its highest value?
- A) the very beginning
- B) one quarter of the way through
- C) around the middle
- D) at the very end

The answer is C.

Now, answer the question based on the following caption:

Caption: <Caption Start> {} <Caption End>
Question: {}  <|im_end|>
<|im_start|>assistant
The answer is'''

        inputs = []
        answers = []

        if self.all_qa:
            qa_num = []
            for cap, qa in prompts:
                qa_num.append(len(qa))
                for q, a in qa:
                    inputs.append({"prompt": prompt.format(cap, q)})
                    answers.append(a)
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
            generated_text = [out.outputs[0].text for out in outputs]
            temp_rewards = []
            for answer, gt in zip(generated_text, answers):
                temp_rewards.append(parse_easy(answer, gt))
            rewards = []
            for qa_n in qa_num:
                rewards.append(np.mean(temp_rewards[:qa_n]) * 2)
                temp_rewards = temp_rewards[qa_n:]
        else:
            qa_num = self.qa_num
            for cap, qa in prompts:
                for i in range(qa_num):
                    q, a = random.choice(qa)
                    if self.shuffle_qa:
                        q, a = shuffle_options(q, a)
                    inputs.append({"prompt": prompt.format(cap, q)})
                    answers.append(a)
            outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
            generated_text = [out.outputs[0].text for out in outputs]
            rewards = []
            for answer, gt in zip(generated_text, answers):
                rewards.append(parse_easy(answer, gt))
            rewards = (torch.Tensor(rewards).view(-1, qa_num).mean(dim=-1).view(-1) * 2).tolist()

        if inputs:
            print("=================================")
            print("input:", inputs[0])
            print("output:", generated_text[0])
            print("rewards", rewards)
            print("=================================")
        return rewards, generated_text


def run_worker_server(args, rank, gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    torch.cuda.init()
    worker_port = args.worker_base_port + rank
    print(f"Starting worker (local rank {rank}) on GPUs {gpu_ids} at port {worker_port}...")

    reward_model = RewardModelProxy(args)
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    @app.post("/get_reward")
    async def get_reward_endpoint(request: Request):
        data = await request.json()
        t = time.time()
        prompts = data.get('prompts')
        queries = data.get("query")
        labels = data.get("labels")
        rewards, generated_text = reward_model.get_reward(prompts, queries, labels)
        result = {"rewards": rewards, "local_rank": rank}
        if sum(rewards) == 0.:
            print(f"--- Rank {rank} REWARDS ARE 0 ---", flush=True)
            print("Associated Generated Texts:", flush=True)
            print(generated_text, flush=True)
        reward_mean = sum(rewards) / len(rewards)
        print(f"Worker {rank} (GPUs: {gpu_ids}) processed {len(prompts)} samples in {time.time() - t:.3f}s, mean reward: {reward_mean:.3f}", flush=True)
        return JSONResponse(result)

    uvicorn.run(app, host="0.0.0.0", port=worker_port, log_level="info")


def run_master_server(args):
    app = FastAPI()
    num_hosts = len(args.worker_hosts)
    if args.num_workers % num_hosts != 0:
        raise ValueError(
            f"Total workers ({args.num_workers}) must be divisible by number of hosts ({num_hosts}).")
    num_workers_per_host = args.num_workers // num_hosts

    worker_urls = []
    for host in args.worker_hosts:
        for i in range(num_workers_per_host):
            worker_urls.append(f"http://{host}:{args.worker_base_port + i}/get_reward")

    async def send_to_worker(session, url, chunk_data, chunk_index):
        try:
            response = await session.post(url, json=chunk_data, timeout=300)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"Request to {url} (chunk {chunk_index}) failed: {e}")
            return None

    @app.post("/get_reward")
    async def distributed_get_reward(request: Request):
        data = await request.json()
        prompts = data.get('prompts', [])
        queries = data.get("query", [])
        labels = data.get("labels", [])
        num_prompts = len(prompts)
        if num_prompts == 0:
            return JSONResponse({"rewards": []})

        t_start = time.time()
        total_workers = len(worker_urls)
        chunk_size = (num_prompts + total_workers - 1) // total_workers
        chunks = []
        for i in range(total_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_prompts)
            if start_idx >= num_prompts:
                break
            chunks.append({"prompts": prompts[start_idx:end_idx], "query": queries, "labels": labels})

        async with httpx.AsyncClient() as session:
            tasks = [send_to_worker(session, worker_urls[i], chunks[i], i) for i in range(len(chunks))]
            worker_responses = await asyncio.gather(*tasks)

        all_rewards = []
        for i, response in enumerate(worker_responses):
            if response and 'rewards' in response:
                all_rewards.extend(response['rewards'])
            else:
                logger.error(f"Worker for chunk {i} (URL: {worker_urls[i]}) failed. Aborting batch.")
                return JSONResponse(
                    status_code=500,
                    content={"error": f"Worker for chunk {i} failed.", "failed_url": worker_urls[i]})

        print(f"Master: Distributed {num_prompts} samples to {len(chunks)} workers. Total time: {time.time() - t_start:.3f}s", flush=True)
        return JSONResponse({"rewards": all_rewards})

    print(f"Starting master server on http://{args.master_host}:{args.port}")
    print(f"Distributing tasks to {args.num_workers} workers across {num_hosts} hosts.")
    print(f"Worker URLs: {worker_urls}")
    uvicorn.run(app, host=args.master_host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", type=str, required=True, choices=["master", "worker"])
    parser.add_argument("--reward_pretrain", type=str, help="Path to the answerer LLM (e.g. ./models/Qwen2.5-3B-Instruct).")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--shuffle_qa", action="store_true", default=False)
    parser.add_argument("--all_qa", action="store_true", default=False)
    parser.add_argument("--qa_num", type=int, default=8)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--master_host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8889)
    parser.add_argument("--worker_hosts", type=str, nargs='+')
    parser.add_argument("--worker_base_port", type=int, default=8899)
    parser.add_argument("--num_workers", type=int)
    args = parser.parse_args()

    if args.role == "master":
        if not args.worker_hosts or not args.num_workers:
            parser.error("--role 'master' requires --worker_hosts and --num_workers.")
        run_master_server(args)
    elif args.role == "worker":
        mp.set_start_method('spawn', force=True)
        if not args.reward_pretrain:
            parser.error("--role 'worker' requires --reward_pretrain.")
        num_gpus_available = torch.cuda.device_count()
        if num_gpus_available < args.tp:
            raise ValueError(f"Not enough GPUs for TP. Required: {args.tp}, Found: {num_gpus_available}")
        if num_gpus_available % args.tp != 0:
            raise ValueError(f"GPUs ({num_gpus_available}) not divisible by TP ({args.tp}).")
        num_local_workers = num_gpus_available // args.tp
        print(f'Found {num_gpus_available} GPUs. Starting {num_local_workers} worker(s) with TP={args.tp}...')

        gpu_ids = list(range(num_gpus_available))
        gpu_groups = [gpu_ids[i:i + args.tp] for i in range(0, len(gpu_ids), args.tp)]
        processes = []
        for i in range(num_local_workers):
            p = mp.Process(target=run_worker_server, args=(args, i, gpu_groups[i]))
            p.start()
            processes.append(p)
        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            print("Shutting down worker processes...")
            for p in processes:
                p.terminate()
                p.join()
            print("All workers terminated.")
