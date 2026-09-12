from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import torch
import torch.nn as nn

from .config import ReadoutConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_load_dotenv(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1]
    env = root / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and k not in os.environ:
            os.environ[k] = v


class MockEncoder(nn.Module):

    def __init__(self, vocab: int = 4096, dim: int = 256):
        super().__init__()
        self.vocab = vocab
        self.hidden_size = dim
        self.emb = nn.EmbeddingBag(vocab, dim, mode="mean")

    def _ids(self, text: str) -> list[int]:
        toks = _TOKEN_RE.findall(text.lower()) or ["<empty>"]
        return [int(hashlib.md5(t.encode()).hexdigest(), 16) % self.vocab for t in toks]

    def encode(self, texts: list[str]) -> torch.Tensor:
        dev = self.emb.weight.device
        flat, offsets, off = [], [], 0
        for t in texts:
            ids = self._ids(t)
            offsets.append(off)
            flat.extend(ids)
            off += len(ids)
        input_t = torch.tensor(flat, dtype=torch.long, device=dev)
        offs_t = torch.tensor(offsets, dtype=torch.long, device=dev)
        return self.emb(input_t, offs_t)


class HFEncoder(nn.Module):

    def __init__(self, cfg: ReadoutConfig):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModel, AutoTokenizer

        _maybe_load_dotenv()
        token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model, token=token)
        base = AutoModel.from_pretrained(cfg.encoder_model, token=token,
                                         torch_dtype=torch.float32, trust_remote_code=True)
        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_targets), bias="none",
            task_type=TaskType.FEATURE_EXTRACTION)
        self.model = get_peft_model(base, lora)
        self.hidden_size = base.config.hidden_size
        self.max_text_len = cfg.max_text_len

    def encode(self, texts: list[str]) -> torch.Tensor:
        dev = next(self.model.parameters()).device
        if self.max_text_len and self.max_text_len > 0:
            enc = self.tokenizer(list(texts), padding=True, truncation=True,
                                 max_length=self.max_text_len, return_tensors="pt").to(dev)
        else:
            enc = self.tokenizer(list(texts), padding=True, truncation=False,
                                 return_tensors="pt").to(dev)
        out = self.model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts


def build_encoder(cfg: ReadoutConfig) -> nn.Module:
    if cfg.encoder_model == "mock":
        return MockEncoder(vocab=cfg.mock_vocab, dim=cfg.mock_dim)
    return HFEncoder(cfg)


class ReadoutModel(nn.Module):
    def __init__(self, cfg: ReadoutConfig, out_dim: int):
        super().__init__()
        self.encoder = build_encoder(cfg)
        h = self.encoder.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h, cfg.mlp_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, out_dim))

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.head(self.encoder.encode(texts))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
