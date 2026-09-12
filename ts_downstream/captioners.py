from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .datasets import DownstreamItem

CaptionFn = Callable[[Sequence[DownstreamItem]], list[str]]


def write_captions_jsonl(path: str | Path, ids: Sequence[str], captions: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for i, c in zip(ids, captions):
            f.write(json.dumps({"id": i, "caption": c}, ensure_ascii=False) + "\n")


def read_captions_jsonl(path: str | Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec["caption"]
    return out


def _trend_word(s: np.ndarray) -> str:
    t = np.arange(len(s))
    slope = float(np.polyfit(t, s, 1)[0]) if len(s) > 1 else 0.0
    scale = float(np.std(s)) + 1e-8
    r = slope * len(s) / scale
    if r > 0.5:
        return "increasing"
    if r < -0.5:
        return "decreasing"
    return "flat"


def _seasonal_hint(s: np.ndarray) -> str:
    s = s - s.mean()
    if len(s) < 8 or np.std(s) < 1e-6:
        return "no clear periodicity"
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-8)
    if len(ac) > 3:
        peak = int(np.argmax(ac[2:]) + 2)
        if ac[peak] > 0.3:
            return f"periodicity around {peak} steps"
    return "no clear periodicity"


def mock_stat_caption(s: np.ndarray) -> str:
    s = np.asarray(s, dtype=np.float64)
    return (
        f"This univariate time series is overall {_trend_word(s)}. "
        f"It starts near {s[0]:.2f} and ends near {s[-1]:.2f}, "
        f"with mean {s.mean():.2f} and standard deviation {s.std():.2f}. "
        f"The minimum {s.min():.2f} occurs around step {int(np.argmin(s))} and the maximum "
        f"{s.max():.2f} around step {int(np.argmax(s))}. It shows {_seasonal_hint(s)}."
    )


class MockStatCaptioner:

    name = "mock"

    def __call__(self, items: Sequence[DownstreamItem]) -> list[str]:
        return [mock_stat_caption(it.series) for it in items]


class VLMCaptioner:

    def __init__(self, ckpt: str, prompt_family: str = "qa",
                 gpu_memory_utilization: float = 0.9, max_tokens: int = 1024,
                 image_root: str | Path = "render", tensor_parallel_size: int = 1):
        self.ckpt = ckpt
        self.prompt_family = prompt_family
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.image_root = Path(image_root)
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None
        self._processor = None

    def _prompt_pair(self) -> tuple[str | None, str]:
        if self.prompt_family == "qa":
            return ("You are an analyst who describes and interprets time series.",
                    "Please describe this image in detail.")
        if self.prompt_family == "decodability":
            from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT
            return (None, TS_CAP_PROMPT)
        raise ValueError(f"unknown prompt_family {self.prompt_family!r}")

    def _lazy_init(self):
        if self._llm is not None:
            return
        from transformers import AutoProcessor
        from vllm import LLM
        self._processor = AutoProcessor.from_pretrained(self.ckpt, trust_remote_code=True)
        self._llm = LLM(model=self.ckpt, trust_remote_code=True,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        limit_mm_per_prompt={"image": 1})

    def __call__(self, items: Sequence[DownstreamItem]) -> list[str]:
        from qwen_vl_utils import process_vision_info
        from vllm import SamplingParams
        from ts_render.render import render_fragment

        self._lazy_init()
        self.image_root.mkdir(parents=True, exist_ok=True)
        sys_prompt, user_prompt = self._prompt_pair()

        prompts = []
        for it in items:
            png = self.image_root / f"{it.id}.png"
            render_fragment(np.asarray(it.series, dtype=np.float64), str(png))
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [
                {"role": "user", "content": [
                    {"type": "image", "image": str(png)},
                    {"type": "text", "text": user_prompt}]}]
            text = self._processor.apply_chat_template(msgs, tokenize=False,
                                                       add_generation_prompt=True)
            image_inputs, _ = process_vision_info(msgs)
            prompts.append({"prompt": text, "multi_modal_data": {"image": image_inputs}})

        outs = self._llm.generate(
            prompts, SamplingParams(n=1, temperature=0.0, max_tokens=self.max_tokens))
        return [o.outputs[0].text.strip() for o in outs]


def build_captioner(kind: str, **kw) -> CaptionFn:
    if kind == "mock":
        return MockStatCaptioner()
    if kind == "vlm":
        return VLMCaptioner(**kw)
    raise ValueError(f"unknown captioner kind {kind!r}")
