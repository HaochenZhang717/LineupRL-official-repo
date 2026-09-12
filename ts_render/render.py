from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


@dataclass(frozen=True)
class RenderConfig:

    width_in: float = 6.0
    height_in: float = 3.0
    dpi: int = 100
    line_width: float = 1.5
    line_color: str = "#1f77b4"
    show_markers: bool = True
    marker_size: float = 3.0
    n_xticks: int = 6
    n_yticks: int = 5
    grid: bool = True
    grid_alpha: float = 0.3
    y_pad_frac: float = 0.08
    tick_fontsize: float = 9.0


def render_fragment(
    series: Sequence[float],
    out_path: str | Path | None = None,
    config: RenderConfig | None = None,
) -> Path | "plt.Figure":
    cfg = config or RenderConfig()
    y = np.asarray(series, dtype=float).ravel()
    if y.size == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(y)):
        raise ValueError("series contains NaN/Inf; clean it before rendering")

    x = np.arange(y.size)

    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in), dpi=cfg.dpi)
    ax.plot(
        x,
        y,
        color=cfg.line_color,
        linewidth=cfg.line_width,
        marker="o" if cfg.show_markers else None,
        markersize=cfg.marker_size,
        markerfacecolor=cfg.line_color,
        markeredgecolor=cfg.line_color,
    )

    ax.set_xlim(x[0], x[-1] if y.size > 1 else x[0] + 1)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    span = y_max - y_min
    pad = span * cfg.y_pad_frac if span > 0 else (abs(y_max) * cfg.y_pad_frac or 1.0)
    ax.set_ylim(y_min - pad, y_max + pad)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=cfg.n_xticks, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=cfg.n_yticks))
    ax.tick_params(axis="both", labelsize=cfg.tick_fontsize)

    if cfg.grid:
        ax.grid(True, alpha=cfg.grid_alpha, linewidth=0.5)

    ax.set_xlabel("time", fontsize=cfg.tick_fontsize)
    ax.set_ylabel("value", fontsize=cfg.tick_fontsize)

    fig.tight_layout(pad=0.5)

    if out_path is None:
        return fig

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png")
    plt.close(fig)
    return out_path
