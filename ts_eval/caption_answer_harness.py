from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from ts_eval import scoring as sc
from ts_eval.benchmarks.base import QAItem
from ts_eval.render_multi import render_fragment_multi
from ts_render.render import render_fragment

CaptionFn = Callable[[list[str]], list[str]]
GenerateFn = Callable[[list[str]], list[str]]

CAPTION_SYSTEM_PROMPT = "You are an analyst who describes and interprets time series."
CAPTION_USER_PROMPT = "Please describe this image in detail."


CAPTION_PROMPT_FAMILIES = ("qa", "decodability", "bedtime_gen")


def caption_prompt_pair(family: str) -> tuple[str | None, str]:
    if family == "qa":
        return CAPTION_SYSTEM_PROMPT, CAPTION_USER_PROMPT
    if family == "decodability":
        from decodability_rl.rl.cap_prompt import TS_CAP_PROMPT
        return None, TS_CAP_PROMPT
    if family == "bedtime_gen":
        from ts_bedtime.prompts import GENERATION_PROMPT
        return None, GENERATION_PROMPT
    raise ValueError(f"unknown prompt family: {family!r}")


@dataclass
class PredictionRecord:
    item_id: str
    source_benchmark: str
    caption: str
    answer_raw: str
    judge_raw: str | None
    score: float
    scoring_type: str
    task_type: str | None
    domain: str | None


def caption_key(item: QAItem) -> str:
    return item.extra.get("caption_key") or item.id


def render_items(items: Sequence[QAItem], image_root: Path) -> list[Path]:
    image_root = Path(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    paths = []
    rendered: dict[str, Path] = {}
    for item in items:
        key = caption_key(item)
        out_path = rendered.get(key)
        if out_path is None:
            out_path = image_root / f"{item.source_benchmark}_{key}.png"
            if item.n_variates > 1:
                render_fragment_multi(item.series, out_path=out_path)
            else:
                render_fragment(item.series, out_path=out_path)
            rendered[key] = out_path
        paths.append(out_path)
    return paths


def caption_messages(image_path: str) -> list[dict]:
    return [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": CAPTION_USER_PROMPT},
        ]},
    ]


_ANSWER_INSTRUCTIONS = {
    "letter_exact": "Respond with only the letter of the correct choice.",
    "letter_flexible": "Respond with the letter and text of the correct choice, e.g. \"B) Decrease\".",
    "tf_exact": "Respond with only True or False.",
    "ordering_exact": "Respond with only the ordered sequence of labels, comma-separated (e.g. 'B,A,C,D').",
    "numeric_tolerance": "Respond with only a numeric value.",
    "regex_numeric_extract_tolerance": "Answer the question in one or two sentences, based only on the caption.",
    "llm_judge": "Answer the question in one or two sentences, based only on the caption.",
}


DEFAULT_EVIDENCE_LABEL = "Description of the time series"
ORACLE_EVIDENCE_LABEL = "Time series values"
DEFAULT_EVIDENCE_NOUN = "description of the time series"
ORACLE_EVIDENCE_NOUN = "time series"
_LABEL_TO_NOUN = {DEFAULT_EVIDENCE_LABEL: DEFAULT_EVIDENCE_NOUN,
                  ORACLE_EVIDENCE_LABEL: ORACLE_EVIDENCE_NOUN}
ANSWER_STYLES = ("caption_qa", "evidence", "native")

_EVIDENCE_HEADER = ("You are answering a question about a time series. "
                    "Use only the information given below.")


def build_answer_messages(item: QAItem, caption: str, style: str = "caption_qa",
                          evidence_label: str | None = None) -> list[dict]:
    instr = _ANSWER_INSTRUCTIONS.get(item.scoring_type)
    if instr is None:
        raise ValueError(f"unsupported scoring_type for answer prompt: {item.scoring_type!r}")

    if style == "caption_qa":
        user_content = (
            "You will be given a caption describing a time series and a question about it. "
            "Answer the question strictly based on the caption, even if the answer may seem "
            "obvious from prior knowledge or the question wording. Ignore any outside knowledge; "
            "do not assume anything the caption does not explicitly or implicitly state.\n\n"
            f"Caption: {caption}\n\n"
            f"Question: {item.question}\n\n"
            f"{instr}"
        )
    elif style == "native" and item.extra.get("native_prompt") is not None:
        noun = _LABEL_TO_NOUN.get(evidence_label or DEFAULT_EVIDENCE_LABEL,
                                  DEFAULT_EVIDENCE_NOUN)
        question = item.extra["native_prompt"](caption or "", noun)
        user_content = f"{question}\n\n{instr}"
    elif style in ("evidence", "native"):
        label = evidence_label or DEFAULT_EVIDENCE_LABEL
        block = f"{label}:\n{caption}\n\n" if caption and caption.strip() else ""
        user_content = f"{_EVIDENCE_HEADER}\n\n{block}Question: {item.question}\n\n{instr}"
    else:
        raise ValueError(f"unknown answer style {style!r}, expected one of {ANSWER_STYLES}")
    return [{"role": "user", "content": user_content}]


def prompt_fingerprint(items: Sequence[QAItem], style: str = "caption_qa", n: int = 20) -> str:
    import hashlib

    sample = sorted(items, key=lambda i: i.id)[:n]
    blob = "\x1e".join(
        build_answer_messages(item, "<CAPTION>", style, DEFAULT_EVIDENCE_LABEL)[0]["content"]
        for item in sample)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _run_in_chunks(fn: Callable[[list[str]], list[str]], xs: list[str], batch_size: int) -> list[str]:
    out: list[str] = []
    for start in range(0, len(xs), batch_size):
        out.extend(fn(xs[start:start + batch_size]))
    if len(out) != len(xs):
        raise RuntimeError(f"generation callable returned {len(out)} outputs for {len(xs)} inputs")
    return out


def caption_items(
    items: Sequence[QAItem],
    image_root: Path,
    generate_captions: CaptionFn,
    batch_size: int = 512,
) -> tuple[list[Path], list[str]]:
    items = list(items)
    image_paths = render_items(items, image_root)

    keys = [caption_key(item) for item in items]
    unique_keys = list(dict.fromkeys(keys))
    path_by_key = dict(zip(keys, (str(p) for p in image_paths)))
    if len(unique_keys) < len(items):
        print(f"captioning {len(unique_keys)} unique charts for {len(items)} items "
              f"(caption_key dedup)", flush=True)
    unique_captions = _run_in_chunks(
        generate_captions, [path_by_key[k] for k in unique_keys], batch_size)
    caption_by_key = dict(zip(unique_keys, unique_captions))
    return image_paths, [caption_by_key[k] for k in keys]


def answer_and_score(
    items: Sequence[QAItem],
    captions: Sequence[str],
    generate_answers: GenerateFn,
    apply_chat_template: Callable[[list[dict]], str],
    judge_fn: GenerateFn | None = None,
    batch_size: int = 512,
    style: str = "caption_qa",
    evidence_labels: Sequence[str | None] | None = None,
) -> list[PredictionRecord]:
    items = list(items)
    captions = list(captions)
    if len(captions) != len(items):
        raise ValueError(f"{len(captions)} captions for {len(items)} items")
    labels = list(evidence_labels) if evidence_labels is not None else [None] * len(items)
    if len(labels) != len(items):
        raise ValueError(f"{len(labels)} evidence labels for {len(items)} items")

    answer_prompts = [
        apply_chat_template(build_answer_messages(item, caption, style, label))
        for item, caption, label in zip(items, captions, labels)
    ]
    answers = _run_in_chunks(generate_answers, answer_prompts, batch_size)

    judge_idx = [i for i, item in enumerate(items) if item.scoring_type == "llm_judge"]
    judge_raw_by_idx: dict[int, str] = {}
    if judge_idx:
        if judge_fn is None:
            raise ValueError(f"{len(judge_idx)} items need llm_judge scoring but judge_fn was not supplied")
        judge_prompts = [sc.build_llm_judge_prompt(items[i], answers[i]) for i in judge_idx]
        judge_outs = _run_in_chunks(judge_fn, judge_prompts, batch_size)
        judge_raw_by_idx = dict(zip(judge_idx, judge_outs))

    records: list[PredictionRecord] = []
    for i, (item, caption, answer) in enumerate(zip(items, captions, answers)):
        if item.scoring_type == "llm_judge":
            judge_raw = judge_raw_by_idx[i]
            score_val = sc.parse_llm_judge_verdict(judge_raw)
        else:
            judge_raw = None
            score_val = sc.score(answer, item)
        records.append(PredictionRecord(
            item_id=item.id,
            source_benchmark=item.source_benchmark,
            caption=caption,
            answer_raw=answer,
            judge_raw=judge_raw,
            score=score_val,
            scoring_type=item.scoring_type,
            task_type=item.task_type,
            domain=item.domain,
        ))
    return records


def run_pipeline1(
    items: Sequence[QAItem],
    image_root: Path,
    generate_captions: CaptionFn,
    generate_answers: GenerateFn,
    apply_chat_template: Callable[[list[dict]], str],
    judge_fn: GenerateFn | None = None,
    batch_size: int = 512,
) -> list[PredictionRecord]:
    items = list(items)
    _, captions = caption_items(items, image_root, generate_captions, batch_size)
    return answer_and_score(items, captions, generate_answers, apply_chat_template, judge_fn, batch_size)
