from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ts_render.render import RenderConfig

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

MAX_CHANNELS = 8


def render_fragment_multi(
    series_list: Sequence[Sequence[float]],
    out_path: str | Path | None = None,
    config: RenderConfig | None = None,
    show_legend: bool = True,
) -> tuple[Path | "plt.Figure", bool]:
    cfg = config or RenderConfig()
    if not series_list:
        raise ValueError("series_list is empty")

    channels = [np.asarray(s, dtype=float).ravel() for s in series_list]
    for c in channels:
        if c.size == 0:
            raise ValueError("one of the channels is empty")
        if not np.all(np.isfinite(c)):
            raise ValueError("a channel contains NaN/Inf; clean it before rendering")

    truncated = len(channels) > MAX_CHANNELS
    if truncated:
        channels = channels[:MAX_CHANNELS]

    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in), dpi=cfg.dpi)

    max_len = max(c.size for c in channels)
    for i, y in enumerate(channels):
        x = np.arange(y.size)
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(
            x, y,
            color=color,
            linewidth=cfg.line_width,
            marker="o" if cfg.show_markers else None,
            markersize=cfg.marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            label=f"series {i + 1}",
        )

    ax.set_xlim(0, max_len - 1 if max_len > 1 else 1)
    y_min = min(float(np.min(c)) for c in channels)
    y_max = max(float(np.max(c)) for c in channels)
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
    if show_legend and len(channels) > 1:
        ax.legend(loc="upper right", fontsize=cfg.tick_fontsize, framealpha=0.7)

    fig.tight_layout(pad=0.5)

    if out_path is None:
        return fig, truncated

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="png")
    plt.close(fig)
    return out_path, truncated
