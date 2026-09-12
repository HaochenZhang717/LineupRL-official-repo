from __future__ import annotations

import re

from ts_eval.benchmarks.base import QAItem

_LETTER_RE = re.compile(r"\b([A-I])\b")
_WORDLIKE_AI = re.compile(r"^\s+[a-z]")
_TF_RE = re.compile(r"\b(TRUE|FALSE|T|F)\b", re.IGNORECASE)
_ORDER_LETTER_RE = re.compile(r"\b([A-D])\b")
_NUM_RE = re.compile(r"-?\d+\.?\d*")


def extract_first_letter(text: str) -> str | None:
    upper = text.upper()
    for m in _LETTER_RE.finditer(upper):
        letter = m.group(1)
        if letter in ("A", "I") and _WORDLIKE_AI.match(text[m.end():]):
            continue
        return letter
    return None


def letter_exact(prediction: str, item: QAItem) -> float:
    pred = extract_first_letter(prediction)
    gold = item.gold.strip().upper()
    return 1.0 if pred is not None and pred == gold else 0.0


def extract_tf(text: str) -> str | None:
    m = _TF_RE.search(text.upper())
    return m.group(1)[0] if m else None


def tf_exact(prediction: str, item: QAItem) -> float:
    pred = extract_tf(prediction)
    if pred is None:
        return 0.0
    gold = item.gold.strip().upper()[0]
    return 1.0 if pred == gold else 0.0


def letter_flexible(prediction: str, item: QAItem) -> float:
    return 1.0 if item.gold.strip().lower() in prediction.lower() else 0.0


def ordering_exact(prediction: str, item: QAItem) -> float:
    gold_seq = [c.strip().upper() for c in item.gold.split(",") if c.strip()]
    seen: list[str] = []
    for letter in _ORDER_LETTER_RE.findall(prediction.upper()):
        if letter not in seen:
            seen.append(letter)
        if len(seen) == len(gold_seq):
            break
    return 1.0 if seen == gold_seq else 0.0


def extract_numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM_RE.findall(text)]


def numeric_tolerance_value(pred_value: float, gold_value: float) -> float:
    if gold_value == 0:
        return 1.0 if pred_value == 0 else 0.0
    return max(0.0, 1.0 - abs(pred_value - gold_value) / abs(gold_value))


def numeric_tolerance(prediction: str, item: QAItem) -> float:
    try:
        gold_value = float(item.gold)
    except ValueError:
        return 0.0
    pred_nums = extract_numbers(prediction)
    if not pred_nums:
        return 0.0
    return max(numeric_tolerance_value(p, gold_value) for p in pred_nums)


def regex_numeric_extract_tolerance(prediction: str, item: QAItem, tol: float = 0.05) -> float:
    gold_nums = extract_numbers(item.gold)
    pred_nums = extract_numbers(prediction)
    if not gold_nums or not pred_nums:
        return 0.0
    target = gold_nums[-1]
    best = max(numeric_tolerance_value(p, target) for p in pred_nums)
    return 1.0 if best >= (1 - tol) else 0.0


LLM_JUDGE_PROMPT = """You are grading whether a candidate answer correctly states one \
specific reference fact about a time series{topic_clause}.

The candidate may have been prompted with a broader question covering several aspects \
of the series at once, so it may discuss other facts too -- that is fine. Look for \
whether the reference fact appears ANYWHERE in the candidate answer, stated correctly, \
even if surrounded by unrelated content.

Reference fact: {reference}
Candidate answer: {candidate}

Does the candidate answer correctly convey the reference fact somewhere within it? \
Minor wording differences are fine; a missing or contradicted fact is not.
Respond with exactly one word: YES or NO."""


def build_llm_judge_prompt(item: QAItem, prediction: str) -> str:
    topic_clause = f" (specifically its {item.task_type})" if item.task_type else ""
    return LLM_JUDGE_PROMPT.format(topic_clause=topic_clause, reference=item.gold, candidate=prediction)


def parse_llm_judge_verdict(judge_output: str) -> float:
    head = judge_output.strip().upper()[:8]
    return 1.0 if "YES" in head else 0.0


SCORERS = {
    "letter_exact": letter_exact,
    "tf_exact": tf_exact,
    "letter_flexible": letter_flexible,
    "ordering_exact": ordering_exact,
    "numeric_tolerance": numeric_tolerance,
    "regex_numeric_extract_tolerance": regex_numeric_extract_tolerance,
}


_LABEL_PARSERS = {
    "letter_exact": extract_first_letter,
    "tf_exact": extract_tf,
}


_GOLD_NORMALISERS = {
    "letter_exact": lambda g: g.strip().upper(),
    "tf_exact": lambda g: g.strip().upper()[0],
}


def parse_prediction(prediction: str, scoring_type: str) -> str | None:
    parser = _LABEL_PARSERS.get(scoring_type)
    if parser is None:
        raise ValueError(f"{scoring_type!r} does not produce a single-label prediction; "
                         f"expected one of {sorted(_LABEL_PARSERS)}")
    return parser(prediction)


def normalize_gold(gold: str, scoring_type: str) -> str:
    normaliser = _GOLD_NORMALISERS.get(scoring_type)
    if normaliser is None:
        raise ValueError(f"{scoring_type!r} does not produce a single-label prediction; "
                         f"expected one of {sorted(_GOLD_NORMALISERS)}")
    return normaliser(gold)


def is_label_scored(scoring_type: str) -> bool:
    return scoring_type in _LABEL_PARSERS


def score(prediction: str, item: QAItem) -> float:
    if item.scoring_type == "llm_judge":
        raise ValueError(
            "llm_judge scoring requires a live generation call; use "
            "build_llm_judge_prompt/parse_llm_judge_verdict from the harness instead of score()."
        )
    scorer = SCORERS.get(item.scoring_type)
    if scorer is None:
        raise ValueError(f"unknown scoring_type: {item.scoring_type!r}")
    return scorer(prediction, item)
