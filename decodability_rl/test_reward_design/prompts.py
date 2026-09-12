from __future__ import annotations


CAPTION_SYSTEM = "You are an analyst who describes and interprets time series."

CAPTION_PROMPT = (
    "This image is a line chart of a time series. "
    "Please describe this time series."
)


DISCRIM_SYSTEM = "You are a careful time-series analyst. You answer with a single letter."
DISCRIM_SYSTEM_TAG = (
    "You are a careful time-series analyst. You answer with a single letter inside "
    "<answer></answer> tags."
)
SYSTEMS = {"letter": DISCRIM_SYSTEM, "tag": DISCRIM_SYSTEM_TAG}

DISCRIM_TEMPLATE = """Below is a written description of ONE time series.

Description:
\"\"\"{caption}\"\"\"

Here are four candidate time series, given as their raw values in time order. Exactly
one of them is the series the description was written about. The other three are
distortions of it: they contain very similar values but in a different temporal
order (for example reordered, time-reversed, or with two segments exchanged), so the
value range alone will not tell them apart -- you have to match *what happens when*.

A) {opt_a}

B) {opt_b}

C) {opt_c}

D) {opt_d}

Which candidate does the description describe? Answer with a single letter: A, B, C, or D."""

DISCRIM_TEMPLATE_NEUTRAL = """Below is a written description of one time series.

Description:
\"\"\"{caption}\"\"\"

Here are four candidate time series, each given as its raw values in time order.

A) {opt_a}

B) {opt_b}

C) {opt_c}

D) {opt_d}

Exactly one of these four is the series that the description was written about.
Which one is it? Answer with a single letter: A, B, C, or D."""

TEMPLATES = {"hinted": DISCRIM_TEMPLATE, "neutral": DISCRIM_TEMPLATE_NEUTRAL}

EMPTY_CAPTION = "(no description available)"


def format_series(values, decimals: int) -> str:
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in values) + "]"


def decimals_for(values) -> int:
    import numpy as np

    scale = float(np.max(np.abs(np.asarray(values, dtype=float))))
    if scale <= 0:
        return 2
    import math

    return int(min(3, max(0, 3 - math.floor(math.log10(scale)))))


OPTION_LETTERS = "ABCDEFGHIJKL"

_NUMBER_WORDS = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve",
}


def letters_for(n_options: int) -> tuple[str, ...]:
    if not 2 <= n_options <= len(OPTION_LETTERS):
        raise ValueError(f"n_options must be in [2, {len(OPTION_LETTERS)}], got {n_options}")
    return tuple(OPTION_LETTERS[:n_options])


def build_neutral_prompt(
    caption: str, options: list[str], answer_format: str = "tag"
) -> str:
    if answer_format not in SYSTEMS:
        raise ValueError(f"answer_format must be one of {sorted(SYSTEMS)}, got {answer_format!r}")
    letters = letters_for(len(options))
    word = _NUMBER_WORDS[len(options)]
    block = "\n\n".join(f"{L}) {opt}" for L, opt in zip(letters, options))
    tail = ", ".join(letters[:-1]) + f", or {letters[-1]}"
    if answer_format == "tag":
        ask = (
            f"Which one is it? Answer with a single letter ({tail}) inside answer tags, "
            "like <answer>LETTER</answer>."
        )
    else:
        ask = f"Which one is it? Answer with a single letter: {tail}."
    return (
        "Below is a written description of one time series.\n"
        "\n"
        "Description:\n"
        f'"""{caption}"""\n'
        "\n"
        f"Here are {word} candidate time series, each given as its raw values in time "
        "order.\n"
        "\n"
        f"{block}\n"
        "\n"
        f"Exactly one of these {word} is the series that the description was written "
        f"about.\n{ask}"
    )


def build_discriminator_prompt(
    caption: str, options: list[str], variant: str = "hinted", answer_format: str = "tag"
) -> str:
    caption = caption.strip() or EMPTY_CAPTION
    if variant == "neutral":
        return build_neutral_prompt(caption, options, answer_format)
    if len(options) != 4:
        raise ValueError(f"variant {variant!r} is fixed at 4 options, got {len(options)}")
    return TEMPLATES[variant].format(
        caption=caption,
        opt_a=options[0],
        opt_b=options[1],
        opt_c=options[2],
        opt_d=options[3],
    )
