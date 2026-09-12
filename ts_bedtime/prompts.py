from __future__ import annotations

from typing import Sequence

LETTERS = "ABCD"


def recognition_question(description: str) -> str:
    return (
        "You are tasked with verifying if the provided description accurately describes "
        "the given time series.\n"
        "Please follow these instructions carefully:\n"
        f"1. Review the description: '{description}'.\n"
        "2. Determine if the description precisely matches the pattern depicted in the "
        "time series.\n"
        "Respond with True if the given description accurately describes the time series.\n"
        "Respond with False if it does not."
    )


def differentiation_question(options: Sequence[str]) -> str:
    if len(options) != len(LETTERS):
        raise ValueError(f"differentiation needs exactly {len(LETTERS)} options, got {len(options)}")
    lines = "\n".join(f"{letter}: {opt}" for letter, opt in zip(LETTERS, options))
    return (
        "Carefully analyze the given time series and choose the single best option that "
        "most accurately describes its pattern.\n"
        "Follow these rules strictly:\n"
        "1. Read all options before deciding.\n"
        "2. Only output the chosen option, highlighted as A, B, C, or D.\n"
        "3. Avoid adding extra text or explanations.\n"
        f"Options:\n{lines}"
    )


EVIDENCE_NOUN_SERIES = "time series"
EVIDENCE_NOUN_CAPTION = "description of the time series"


def _numbered(steps: Sequence[str]) -> str:
    return "\n".join(f"{i + 1}. {step}" for i, step in enumerate(steps))


def recognition_question_native(description: str, evidence: str = "",
                                evidence_noun: str = EVIDENCE_NOUN_SERIES) -> str:
    steps = [f"Review the description: '{description}'."]
    if evidence.strip():
        tail = "" if evidence.rstrip().endswith((".", "!", "?")) else "."
        steps.append(f"Analyze the {evidence_noun}: {evidence.rstrip()}{tail}")
    steps.append("Determine if the description precisely matches the pattern depicted in "
                 "the time series.")
    return (
        "You are tasked with verifying if the provided description accurately describes "
        "the given time series.\n"
        "Please follow these instructions carefully:\n"
        f"{_numbered(steps)}\n"
        "Respond with True if the given description accurately describes the time series.\n"
        "Respond with False if it does not."
    )


def differentiation_question_native(options: Sequence[str], evidence: str = "",
                                    evidence_noun: str = EVIDENCE_NOUN_SERIES) -> str:
    if len(options) != len(LETTERS):
        raise ValueError(f"differentiation needs exactly {len(LETTERS)} options, got {len(options)}")
    lines = "\n".join(f"{letter}: {opt}" for letter, opt in zip(LETTERS, options))
    evidence_block = (f"{evidence_noun.capitalize()}: {evidence}\n"
                      if evidence.strip() else "")
    return (
        "Carefully analyze the given time series and choose the single best option that "
        "most accurately describes its pattern.\n"
        "Follow these rules strictly:\n"
        "1. Read all options before deciding.\n"
        "2. Only output the chosen option, highlighted as A, B, C, or D.\n"
        "3. Avoid adding extra text or explanations.\n"
        f"{evidence_block}Options:\n{lines}"
    )


def native_prompt_builder(task_type: str, description: str, options: Sequence[str] | None):
    if task_type == "recognition":
        return lambda evidence, noun: recognition_question_native(description, evidence, noun)
    if task_type == "differentiation":
        opts = list(options or [])
        return lambda evidence, noun: differentiation_question_native(opts, evidence, noun)
    raise ValueError(f"no native prompt for task_type {task_type!r}")


GENERATION_PROMPT = (
    "You are tasked with generating a textual description of the structural properties "
    "of the provided time series.\n"
    "Please follow these instructions carefully:\n"
    "1. Analyze the given time series.\n"
    "2. Identify and describe the most prominent visual features or patterns observed in "
    "the time series. Consider characteristics such as trends, seasonality, anomalies, or "
    "significant changes.\n"
    "Your response should be a concise textual description of the most pronounced "
    "structural properties of the time series.\n"
    "Avoid including unnecessary details or unrelated commentary."
)

GENERATION_MAX_TOKENS = 150
