from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional

BASE_URL_INTL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-max"


def encode_image(image_path: str | Path) -> str:
    data = Path(image_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


class QwenVLClient:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL_INTL,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai to use the Qwen client") from e

        key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("No API key. Set DASHSCOPE_API_KEY or pass api_key=...")
        self.model = model
        self.max_retries = max_retries
        self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout)

        self._extra_body: Optional[dict] = None
        _et = os.environ.get("ENABLE_THINKING")
        if _et is not None and _et.strip().lower() in ("0", "false", "no", "off"):
            self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def chat_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=self._extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Qwen request failed after retries: {last_err}")

    def chat_vision(
        self,
        prompt: str,
        image_path: str | Path,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Qwen request failed after retries: {last_err}")
