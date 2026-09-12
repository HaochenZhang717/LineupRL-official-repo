from __future__ import annotations

import re

JUDGE_SYSTEM = (
    "You are a careful time-series analyst. You grade written descriptions of time "
    "series against the actual values. You answer with a score and nothing else."
)

JUDGE_TEMPLATE = """Here are the values of one time series, in order:

{series}

Here is a written description of that series:

{caption}

Grade the description on how well it describes THIS series, from 1 to 10.

What to weigh, in order:

1. Accuracy. Every claim it makes -- direction, shape, where features sit, magnitudes --
   must be true of the values above. A confident claim that is wrong is the worst thing a
   description can do and must pull the score below any description that says less.
2. Specificity. A description that would fit many different series is worth little, even
   when nothing in it is false. Ask whether someone holding this description could pick
   this series out from others of broadly similar shape.
3. Completeness. Whether the description covers the movements that matter, not whether it
   covers everything.

Anchors:

  1-2   Mostly wrong, or so generic it would fit almost any series.
  3-4   Broadly right about the overall direction, wrong or silent on everything else.
  5-6   Correct about the main movement, vague about where features occur or how large
        they are.
  7-8   Accurate and specific about the main features and roughly where they occur.
  9-10  Accurate throughout AND specific enough to identify this series among similar
        ones.

Score 1 if the description is empty, or if it could have been written without seeing
these values at all. You are grading the description, not the series: when there is
nothing there to grade, the score is 1, however interesting the values are.

Do not reward length. A short description that is accurate and specific scores higher
than a long one that hedges, repeats itself, or pads with claims you cannot verify.

Reply with the integer score alone, no words, no punctuation."""


def build_judge_prompt(caption: str, series_text: str) -> str:
    return JUDGE_TEMPLATE.format(series=series_text, caption=caption)


_INT = re.compile(r"-?\d+")


def parse_score(reply: str, lo: int = 1, hi: int = 10) -> float | None:
    m = _INT.search(reply or "")
    if not m:
        return None
    try:
        v = int(m.group())
    except ValueError:
        return None
    return float(min(max(v, lo), hi))
