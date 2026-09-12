from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "google/gemma-3-4b-it"


class GemmaVLLocal:

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        dtype: str = "auto",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        attn_implementation: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "Local inference needs torch + transformers. "
                'Install: pip install "torch" "transformers>=4.50" accelerate pillow'
            ) from e

        try:
            from transformers import AutoModelForImageTextToText as _Model
        except ImportError:
            from transformers import Gemma3ForConditionalGeneration as _Model

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        load_kwargs: dict = {"torch_dtype": dtype, "device_map": device_map}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.processor = AutoProcessor.from_pretrained(model_id)
        try:
            self.processor.tokenizer.padding_side = "left"
        except AttributeError:
            pass

        self.model = _Model.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def _build_message(self, prompt, image, system):
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        return messages

    def generate_batch(
        self,
        prompts: list[str],
        image_paths: Optional[list[str | Path | None]] = None,
        system: Optional[str] = None,
    ) -> list[str]:
        import torch

        n = len(prompts)
        if image_paths is None:
            image_paths = [None] * n
        if len(image_paths) != n:
            raise ValueError("prompts and image_paths must have equal length")

        has_img = [p is not None for p in image_paths]
        if any(has_img) and not all(has_img):
            raise ValueError(
                "generate_batch needs a homogeneous batch (all-with-image or "
                "all-without); batch per input setting instead."
            )
        use_images = all(has_img) and n > 0

        from PIL import Image

        texts: list[str] = []
        images: list = []
        for prompt, ip in zip(prompts, image_paths):
            img = Image.open(ip).convert("RGB") if ip is not None else None
            if img is not None:
                images.append(img)
            messages = self._build_message(prompt, img, system)
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        proc_kwargs: dict = {"text": texts, "padding": True, "return_tensors": "pt"}
        if use_images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.model.device)

        gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)

        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        outs = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [o.strip() for o in outs]

    def generate(
        self,
        prompt: str,
        image_path: str | Path | None = None,
        system: Optional[str] = None,
    ) -> str:
        return self.generate_batch([prompt], [image_path], system=system)[0]