import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import requests


def load_val(path, limit=None, cap_prompt=None):
    samples = []
    with open(path) as f:
        for line in f:
            if limit is not None and len(samples) >= limit:
                break
            msgs = json.loads(json.loads(line)["message"])
            qa = None
            gen_msgs = []
            for m in msgs:
                if m["role"] == "answer":
                    qa = ast.literal_eval(m["content"])
                else:
                    gen_msgs.append(m)
            assert qa, f"no answer role in line: {line[:120]}"
            if cap_prompt is not None:
                user = next(m for m in gen_msgs if m["role"] == "user")
                for part in user["content"]:
                    if part.get("type") == "text":
                        part["text"] = cap_prompt
                gen_msgs = [user]
            samples.append((gen_msgs, qa))
    return samples


def dump_captions(path, step, ckpt, samples, captions, rewards, accuracies):
    if os.path.exists(path):
        sup = os.path.join(os.path.dirname(path), "superseded")
        os.makedirs(sup, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        moved = os.path.join(sup, f"{os.path.basename(path)}.{stamp}")
        os.replace(path, moved)
        print(f"superseded existing dump -> {moved}", flush=True)

    tmp = path + ".partial"
    n = 0
    with open(tmp, "w") as fh:
        for i, ((gen_msgs, qa), cap) in enumerate(zip(samples, captions)):
            rec = {"step": step, "idx": i, "caption": cap}
            try:
                payload = json.loads(qa[0][0])
                rec["id"] = payload.get("id")
                rec["series_len"] = len(payload.get("true") or [])
            except Exception:
                pass
            for m in gen_msgs:
                for part in m.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "image":
                        rec["image"] = part.get("image")
            if i < len(rewards):
                rec["reward"] = round(float(rewards[i]), 4)
            if i < len(accuracies):
                rec["accuracy"] = round(float(accuracies[i]), 4)
            rec["chars"] = len(cap)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def ensure_processor_config(ckpt, processor_src):
    import shutil
    if os.path.exists(os.path.join(ckpt, "preprocessor_config.json")):
        return
    copied = []
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        src = os.path.join(processor_src, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ckpt, name))
            copied.append(name)
    if not copied:
        raise FileNotFoundError(f"no preprocessor_config.json in {processor_src}")
    print(f"copied {copied} from {processor_src} into {ckpt}", flush=True)


def generate_captions(ckpt, samples, max_tokens=1024, temperature=0.0):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    llm = LLM(model=ckpt, trust_remote_code=True, gpu_memory_utilization=0.85,
              limit_mm_per_prompt={"image": 1})

    inputs = []
    for gen_msgs, _ in samples:
        text = processor.apply_chat_template(gen_msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(gen_msgs)
        inputs.append({"prompt": text, "multi_modal_data": {"image": image_inputs}})

    params = SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(inputs, sampling_params=params)
    return [out.outputs[0].text for out in outputs]


def score(reward_url, captions, samples, batch_size=100, timeout=3600):
    rewards, accuracies, n_options = [], [], None
    for i in range(0, len(captions), batch_size):
        payload = {
            "prompts": [[cap, qa] for cap, (_, qa) in
                        zip(captions[i:i + batch_size], samples[i:i + batch_size])],
            "query": [], "labels": [],
            "full_rotations": True,
        }
        resp = requests.post(reward_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        rewards.extend(body["rewards"])
        accuracies.extend(body.get("accuracies") or [])
        n_options = body.get("n_options", n_options)
    return rewards, (accuracies if len(accuracies) == len(rewards) else []), n_options


def update_best(best_dir, rec, ckpt):
    import shutil
    marker = os.path.join(best_dir, "best.json")
    cur = None
    if os.path.exists(marker):
        with open(marker) as f:
            cur = json.load(f)
    if cur is not None and rec["mean_reward"] <= cur["mean_reward"]:
        return
    os.makedirs(best_dir, exist_ok=True)
    tmp = os.path.join(best_dir, "_incoming")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(ckpt, tmp)
    for d in os.listdir(best_dir):
        if d.startswith("global_step"):
            shutil.rmtree(os.path.join(best_dir, d), ignore_errors=True)
    os.rename(tmp, os.path.join(best_dir, os.path.basename(os.path.normpath(ckpt))))
    with open(marker, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"NEW_BEST step={rec['step']} mean_reward={rec['mean_reward']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir (or base model dir)")
    ap.add_argument("--val", required=True)
    ap.add_argument("--reward-url", required=True)
    ap.add_argument("--score-batch-size", type=int, default=100,
                    help="captions per POST to the reward server")
    ap.add_argument("--score-timeout", type=int, default=3600,
                    help="seconds to wait for one POST; must cover queueing behind the "
                         "training job's own reward requests, not just this batch's work")
    ap.add_argument("--out", required=True, help="jsonl file to append the summary line to")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--cap-prompt", default=None,
                    help="reproduce the trainer's rollout prompt (must match the training "
                         "job's --cap_prompt); omit to keep the dataset's own messages")
    ap.add_argument(
        "--captions-dir", default=None,
        help="where to write per-item greedy captions (default: val_captions/ beside "
        "--out). Set to '' to disable.",
    )
    ap.add_argument("--best-dir", default=None,
                    help="if set, keep a permanent copy of the best-val checkpoint here")
    ap.add_argument("--processor-src",
                    default=str(REPO / "models" / "Qwen2.5-VL-3B-Instruct"),
                    help="base model dir to copy preprocessor_config.json from if the ckpt lacks it")
    args = ap.parse_args()
    ensure_processor_config(args.ckpt, args.processor_src)

    base = os.path.basename(os.path.normpath(args.ckpt))
    m = re.search(r"global_step(\d+)", base) or re.search(r"checkpoint-(\d+)", base)
    step = int(m.group(1)) if m else 0

    samples = load_val(args.val, args.limit, args.cap_prompt)
    t0 = time.time()
    captions = generate_captions(args.ckpt, samples, args.max_tokens, args.temperature)
    t_gen = time.time() - t0
    rewards, accuracies, n_options = score(
        args.reward_url, captions, samples,
        batch_size=args.score_batch_size, timeout=args.score_timeout,
    )
    mean_reward = sum(rewards) / len(rewards)
    accuracy = (
        sum(accuracies) / len(accuracies) if accuracies else mean_reward / 2
    )
    acc_vs_chance = None
    if n_options:
        chance = 1.0 / n_options
        acc_vs_chance = round((accuracy - chance) / (1.0 - chance), 4)

    from collections import Counter

    openings = Counter(" ".join(c.split()[:8]) for c in captions)
    top_opening, top_count = openings.most_common(1)[0]

    rec = {
        "step": step,
        "ckpt": args.ckpt,
        "n_val": len(samples),
        "cap_prompt": args.cap_prompt,
        "mean_reward": round(mean_reward, 4),
        "accuracy": round(accuracy, 4),
        "n_options": n_options,
        "acc_vs_chance": acc_vs_chance,
        "mean_caption_chars": round(sum(len(c) for c in captions) / len(captions), 1),
        "distinct_openings": len(openings),
        "top_opening_rate": round(top_count / len(captions), 4),
        "top_opening": top_opening[:80],
        "temperature": args.temperature,
        "gen_seconds": round(t_gen, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cap_path = None
    try:
        cdir = args.captions_dir
        if cdir is None:
            cdir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "val_captions")
        if cdir:
            os.makedirs(cdir, exist_ok=True)
            cap_path = os.path.join(cdir, f"step{step}.jsonl")
            n = dump_captions(cap_path, step, args.ckpt, samples, captions, rewards, accuracies)
            print(f"wrote {n} captions to {cap_path}", flush=True)
    except Exception as e:
        print(f"WARNING: could not dump captions ({type(e).__name__}: {e})", flush=True)
        cap_path = None
    rec["captions_file"] = cap_path

    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("VAL_RESULT", json.dumps(rec), flush=True)
    if args.best_dir:
        update_best(args.best_dir, rec, args.ckpt)


if __name__ == "__main__":
    main()
