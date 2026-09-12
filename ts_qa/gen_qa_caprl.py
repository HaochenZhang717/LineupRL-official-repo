from __future__ import annotations

import argparse
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

SYS_PROMPT_V1 = """You are given a line chart of a single univariate time series. Generate TEN multiple-choice questions (with their answers) about THIS specific series.

Hard requirements:
- The ten questions must each probe a DIFFERENT aspect of the curve. Do not ask ten variants of the same thing, and do not always start with "where is the peak" or "what is the overall trend".
- Mix both kinds: some questions about the OVERALL behavior (trend, shape, how it starts vs. ends, which half/third is higher, steeper, or noisier) and some about a SPECIFIC salient event you can clearly see (a prominent peak, dip, surge, or plateau) — when you ask about such an event, locate it by its coarse region and only if it is clearly visible.
- Each question must be SPECIFIC to this curve: refer to the concrete shape you actually see (a particular rise, dip, plateau, spike, oscillation, or where along the series it happens). Avoid generic, template-like questions that could apply to any series.
- Each question must be answerable purely from the curve, have exactly four options, and exactly one correct answer. Make the distractors plausible.
- Do NOT ask for exact numeric values, forecasting, or anything that is not visible in the chart.
- Even when the raw values are provided to you for reference, use them only to read the true shape — do NOT ask for exact y-values or step-precise readings, and only ask about a feature (peak/dip/surge/plateau) if it is genuinely there in the values.
- Write every positional option as a PLAIN NUMERIC time index — a SINGLE index like "t=7" when the answer is intrinsically one point, or an INTERVAL like "t=5–10" when the answer is a region/segment — with NO verbal label (do not write "early part", "middle", etc.; those words are vague and ambiguous). The series has integer time indices 0 to N-1 (N is given to you).
- MATCH the option granularity to what the question actually asks — do NOT force every option into a wide interval when that is the wrong shape:
   • REGION/SEGMENT questions (overall trend, which part is steeper / higher-on-average / noisier, where a rise / dip / plateau spans): use COARSE intervals. Place boundaries where THIS curve actually changes so the four are generally UNEQUAL in width (do NOT just split into four equal quarters), each at least about one fifth of the series wide.
   • SINGLE-POINT questions (where the exact maximum / peak or minimum / trough sits): use SINGLE time indices (or tight windows) as the options. Inflating a one-point answer into a wide interval makes several intervals tie and the question becomes unanswerable — so be truthful and keep these options point-sized.
- CRITICAL — the four options of a positional question must be MUTUALLY EXCLUSIVE (disjoint): no two may overlap or cover the same span/point (adjacent intervals may meet at a single boundary index only). If you cannot fill four disjoint options, make one option a NON-positional choice instead (e.g. "Roughly constant throughout / No single clear region").

Output format — follow it EXACTLY. The example below shows ONLY the format; do NOT copy its wording or topic. Positional options are numeric and disjoint (intervals for region questions, single indices for single-point questions):
#### 1. **<a specific question about this curve>**
   - A) **<Choice A>**
   - B) **<Choice B>**
   - C) **<Choice C>**
   - D) **<Choice D>**

**Answer:** <letter>) <correct option text>
------

Produce questions 1 to 10 in this exact format, each block separated by a line containing only "------". Output nothing else.
"""


SYS_PROMPT_V2 = """Your task is to generate ten multiple-choice questions and their answers about the time series based on the provided line chart.
The questions should be challenging and should not all ask the same kind of thing: ask about where features sit in time, about how large values and changes are, and about the shape of the curve. Your answer should strictly follow the following format:
#### 1. **<a question about WHERE in time some feature of this curve sits>**
   - A) **<a time index or interval, e.g. t=7 or t=12-16>**
   - B) **<another one, not overlapping the others>**
   - C) **<another one>**
   - D) **<another one>**

**Answer:** <letter>) <the correct option>
------
#### 2. **<a question about HOW LARGE a value, a rise or a fall is on this curve, read off the value axis>**
   - A) **<a number on the scale of this chart's value axis>**
   - B) **<another one, far enough apart to be told apart on the chart>**
   - C) **<another one>**
   - D) **<another one>**

**Answer:** <letter>) <the correct option>
------
#### 3. **<a question about the SHAPE of this curve, or comparing two of its segments>**
   - A) **<a short description, e.g. a steady rise throughout>**
   - B) **<another one>**
   - C) **<another one>**
   - D) **<another one>**

**Answer:** <letter>) <the correct option>
------
The three blocks above show ONLY the format and the kinds of questions to mix; do NOT copy their wording, and never reuse their placeholder numbers. You should strictly follow the above format and should not generate irrelevant sentences. All the questions should be answerable from the chart alone.
Produce questions 1 to 10, separating every pair of consecutive blocks with a line containing only "------".
"""

SYS_PROMPTS = {"v1": SYS_PROMPT_V1, "v2": SYS_PROMPT_V2}

SYS_PROMPT = SYS_PROMPT_V1


def build_user_prompt(series=None, with_series: bool = False, version: str = "v1") -> str:
    n = len(series) if series is not None else 0
    if version == "v2":
        reminder = f"The series has N={n} points (time indices 0..{max(n - 1, 0)})."
    else:
        reminder = (f"The series has N={n} points (time indices 0..{max(n - 1, 0)}). "
                    "Write each positional option as a plain numeric time interval only (e.g. "
                    "'t=5–13'), with NO verbal region words. Place the boundaries where this "
                    "curve actually changes so the four intervals are UNEQUAL in width, keep each "
                    "interval coarse (>= about N/5 wide, never a single index), make the four "
                    "intervals non-overlapping, and never reference an exact y-value. Generate the ten questions now.")
    if with_series and series:
        vals = ", ".join(f"{float(v):.3g}" for v in series)
        caveat = ("(Use them for reference; the questions must still be answerable "
                  "from the chart alone.) " if version == "v2" else
                  "(Use them only to read the shape; do NOT turn them into "
                  "value-precise questions.) ")
        return ("Here is the line chart of the time series. For reference, its values "
                f"in time order are:\n[{vals}]\n" + caveat + reminder)
    return "Here is the line chart of the time series. " + reminder

BATCH_SIZE = 50
DEFAULT_WORKERS = 16


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_progress(progress_file: Path) -> int:
    if not progress_file.exists():
        return 0
    try:
        return int(json.loads(progress_file.read_text()).get("index", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _save_progress(progress_file: Path, index: int) -> None:
    progress_file.write_text(json.dumps({"index": index}), encoding="utf-8")


def _append_jsonl(records: List[Dict[str, Any]], outfile: Path) -> None:
    if not records:
        return
    with outfile.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_sample(client, rec: Dict[str, Any], image_root: Optional[Path],
                   temperature: float, max_tokens: int,
                   with_series: bool = False,
                   version: str = "v1") -> Optional[Dict[str, Any]]:
    try:
        img = rec["image_path"]
        if image_root is not None:
            img = image_root / img
        response = client.chat_vision(
            build_user_prompt(rec.get("series"), with_series=with_series, version=version),
            image_path=img, system=SYS_PROMPTS[version],
            temperature=temperature, max_tokens=max_tokens,
        )
        return {
            "id": rec.get("id"),
            "image_path": rec["image_path"],
            "series": rec.get("series"),
            "prompt_version": version,
            "qa_response": response,
        }
    except Exception as exc:
        logging.warning("%s -> %s", rec.get("image_path"), exc)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="CapRL-style VLM QA generation for time series.")
    p.add_argument("--in", dest="in_path", required=True, help="fragments.jsonl from rendering")
    p.add_argument("--image-root", default=None, help="root that image_path is relative to")
    p.add_argument("--out-dir", required=True, help="output folder for part_*.jsonl")
    p.add_argument("--limit", type=int, default=None, help="only process the first N fragments")
    p.add_argument("--with-series", action="store_true",
                   help="also feed raw numeric values as text (default: image-only)")
    p.add_argument("--prompt-version", choices=sorted(SYS_PROMPTS), default="v1",
                   help="v1 = original (positional options only); v2 = relaxed, "
                        "six question families with a coverage quota")

    api = p.add_argument_group("Qwen API")
    api.add_argument("--model", default="qwen-vl-max",
                     help="qwen-vl-max | qwen2.5-vl-72b-instruct | qwen3-vl-plus ...")
    api.add_argument("--base-url", default=None, help="override base_url (default: BASE_URL_INTL)")
    api.add_argument("--api-key", default=None, help="defaults to $DASHSCOPE_API_KEY")
    api.add_argument("--temperature", type=float, default=0.8)
    api.add_argument("--max-tokens", type=int, default=4096,
                     help="max output tokens; 10 MCQs need ~1600+, keep headroom")

    par = p.add_argument_group("parallel / resume (CapRL-style)")
    par.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    par.add_argument("--part-id", type=int, default=0)
    par.add_argument("--all-parts", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    from .qwen_client import BASE_URL_INTL, QwenVLClient
    client = QwenVLClient(
        model=args.model, api_key=args.api_key,
        base_url=args.base_url or BASE_URL_INTL,
    )

    data = _load_jsonl(Path(args.in_path))
    if args.limit:
        data = data[: args.limit]

    part_size = math.ceil(len(data) / args.all_parts)
    start, end = args.part_id * part_size, min((args.part_id + 1) * part_size, len(data))
    data = data[start:end]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part_{args.part_id}.jsonl"
    progress_file = out_dir / f"progress_part_{args.part_id}.json"

    resume = _load_progress(progress_file)
    data = data[resume:]
    image_root = Path(args.image_root) if args.image_root else None
    logging.info("Processing %d fragments (resumed from %d)", len(data), resume)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for idx, rec in enumerate(tqdm(data, desc="gen-qa-caprl")):
            futures.append(ex.submit(process_sample, client, rec, image_root,
                                     args.temperature, args.max_tokens, args.with_series,
                                     args.prompt_version))
            if (idx + 1) % BATCH_SIZE == 0 or idx + 1 == len(data):
                done = [f.result() for f in futures]
                _append_jsonl([d for d in done if d is not None], out_file)
                _save_progress(progress_file, resume + idx + 1)
                futures.clear()

    logging.info("Done. Raw QA responses -> %s", out_file)


if __name__ == "__main__":
    main()
