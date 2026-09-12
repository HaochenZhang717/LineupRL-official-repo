from __future__ import annotations

from typing import Sequence

CAPTION_SYSTEM = "You are an analyst who describes and interprets time series."

_TASK = (
    "Describe and analyze this time series. Mention whatever you find noteworthy "
    "about its behaviour."
)

_IMAGE_INTRO = (
    "The image is a line chart of a univariate time series "
    "(x-axis = time index, y-axis = value)."
)

_TEXT_INTRO = (
    "Below are the values of a univariate time series, in time order "
    "(time index 0 to {last}):\n{values}"
)

_BOTH_INTRO = (
    "You are given a univariate time series in two forms: a line chart "
    "(x-axis = time index, y-axis = value) AND its underlying values in time order "
    "(time index 0 to {last}):\n{values}"
)

MODES = ("image", "text", "both")


def series_to_text(series: Sequence[float], decimals: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{decimals}f}" for v in series) + "]"


def build_prompt(mode: str, series: Sequence[float] | None = None) -> str:
    if mode == "image":
        return f"{_IMAGE_INTRO}\n\n{_TASK}"
    if mode in ("text", "both"):
        if series is None:
            raise ValueError(f"mode {mode!r} needs the raw series")
        values = series_to_text(series)
        intro = (_TEXT_INTRO if mode == "text" else _BOTH_INTRO).format(
            last=len(series) - 1, values=values
        )
        return f"{intro}\n\n{_TASK}"
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")
