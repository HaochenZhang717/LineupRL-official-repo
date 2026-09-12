from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReadoutConfig:

    encoder_model: str = "google/embeddinggemma-300m"
    mlp_hidden: int = 256
    dropout: float = 0.1
    max_text_len: int = 256
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    )
    mock_vocab: int = 4096
    mock_dim: int = 256


@dataclass
class TrainConfig:

    epochs: int = 30
    batch_size: int = 32
    micro_batch_size: Optional[int] = None
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 6
    seed: int = 2020
    device: str = "auto"


@dataclass
class DataConfig:
    task: str = "classification"
    dataset: str = "synthetic"
    lookback: int = 96
    horizon: int = 96
    stride: int = 1
    max_windows: Optional[int] = None
    val_frac: float = 0.2
    seed: int = 2020
    extra: dict = field(default_factory=dict)
