from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT_TEXT

MODEL = "ChengsenWang/ChatTime-1-7B-Chat"

LOW_LIMIT, HIGH_LIMIT, N_TOKENS = -1, 1, 10002
PREC, TIME_SEP, TIME_FLAG, NAN_FLAG = 4, " ", "###", "Nan"


def _centers():
    boundaries = np.linspace(LOW_LIMIT, HIGH_LIMIT, N_TOKENS - 1)
    c = (boundaries[1:] + boundaries[:-1]) / 2
    return boundaries, np.concatenate((c[:1], c, c[-1:]))


def discretize(context: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import MinMaxScaler
    boundaries, centers = _centers()
    scaler = MinMaxScaler()
    scaler.fit(context.reshape(-1, 1))
    scaled = scaler.transform(context.reshape(-1, 1)).reshape(-1) - 0.5
    out = centers[np.digitize(x=scaled, bins=boundaries, right=True)]
    out[np.isnan(context)] = np.nan
    return out


def serialize(context: np.ndarray) -> str:
    s = np.array([f"{TIME_FLAG}{i:.{PREC}f}{TIME_FLAG}" for i in context])
    s[np.isnan(context)] = f"{TIME_FLAG}{NAN_FLAG}{TIME_FLAG}"
    return TIME_SEP.join(s)


ANA_SYS_PROMPT = ("You are a helpful assistant that performs time series analysis. The "
                  "user will provide a sequence and you will respond to the questions "
                  "based on this sequence.")
ANA_INST_PROMPT_TEXT = ("Please answer the following question carefully after analyzing "
                        "the sequence: {}")
TEMPLATE = """{}

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def build_prompt(values) -> str:
    body = TS_CAP_PROMPT_TEXT.split(". ", 1)[1]
    ser = serialize(discretize(np.asarray(values, dtype=np.float64)))
    return TEMPLATE.format(ANA_SYS_PROMPT, ANA_INST_PROMPT_TEXT.format(body), ser, "")


def selftest() -> None:
    _, centers = _centers()
    assert len(centers) == N_TOKENS, len(centers)
    assert abs(centers[len(centers) // 2]) < 1e-3, centers[len(centers) // 2]
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = discretize(x)
    assert abs(d.min() + 0.5) < 1e-3 and abs(d.max() - 0.5) < 1e-3, (d.min(), d.max())
    s = serialize(d)
    assert s.startswith("###-0.5") and s.count("###") == 2 * len(x), s[:80]
    assert not np.isnan(discretize(np.array([7.0, 7.0, 7.0]))).any()
    print("selftest OK:", s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series")
    ap.add_argument("--out")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    selftest()

    rows = [json.loads(l) for l in open(args.series, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    keep, prompts, skipped = [], [], []
    budget = args.max_model_len - args.max_tokens - 16
    for r in rows:
        p = build_prompt(r["series"])
        if len(tok(p)["input_ids"]) > budget:
            skipped.append(r["series_uid"])
            continue
        keep.append(r)
        prompts.append(p)
    if skipped:
        print(f"NOTE: {len(skipped)} series do not fit ChatTime's {args.max_model_len}-token "
              f"context and are skipped, not truncated (e.g. {skipped[:3]})")

    if args.smoke:
        print("=== rendered prompt [0] ===")
        print(prompts[0][:1500])
        keep, prompts = keep[: args.smoke], prompts[: args.smoke]

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True)
    outs = llm.generate(prompts, SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens))
    caps = [o.outputs[0].text.strip() for o in outs]

    if args.smoke:
        for r, c in zip(keep, caps):
            print(f"\n--- {r['series_uid']} (n={len(r['series'])})\n{c}")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_empty = sum(1 for c in caps if not c)
    with open(out, "w", encoding="utf-8") as f:
        for r, c in zip(keep, caps):
            f.write(json.dumps({"id": r["series_uid"], "caption_key": r["series_uid"],
                                "caption": c}, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(caps)} captions, {n_empty} empty, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
