from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .extract_qa import extract_questions_and_answers
from .gen_qa_caprl import SYS_PROMPTS, build_user_prompt

VERSIONS = ("v1", "v2")

_OPTION = re.compile(r"^\s*-\s*([A-D])\)\s*(.+?)\s*$")
_TIME_OPT = re.compile(r"^\**\s*t\s*=", re.IGNORECASE)
_HAS_NUMBER = re.compile(r"-?\d")
_NUM_FILLER = re.compile(
    r"\b(about|approx\w*|around|roughly|nearly|near|value|values|units?|points?|"
    r"times?|of|a|an|the|to|and|by|is|it|drop|rise|increase|decrease|difference)\b",
    re.IGNORECASE)


def parse_options(question: str) -> List[str]:
    return [m.group(2) for m in (_OPTION.match(ln) for ln in question.splitlines()) if m]


def _is_numeric_opt(opt: str) -> bool:
    if _TIME_OPT.match(opt) or not _HAS_NUMBER.search(opt):
        return False
    rest = _NUM_FILLER.sub(" ", opt.replace("*", ""))
    rest = re.sub(r"[-~≈+%.,;:()\[\]/]|\d", " ", rest)
    return len(rest.split()) <= 1


def classify(question: str) -> str:
    opts = parse_options(question)
    if not opts:
        return "unparsed"
    n_time = sum(bool(_TIME_OPT.match(o)) for o in opts)
    n_num = sum(_is_numeric_opt(o) for o in opts)
    if n_time >= 3:
        return "location (time options)"
    if n_num >= 3:
        return "magnitude/count (numeric options)"
    return "verbal (shape/comparison/variability)"


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fams: collections.Counter = collections.Counter()
    letters: collections.Counter = collections.Counter()
    n_q = n_time_opt = n_opt = 0
    for rec in records:
        for qa in rec["qa_list"]:
            n_q += 1
            fams[classify(qa["question"])] += 1
            letters[qa["answer"]] += 1
            for o in parse_options(qa["question"]):
                n_opt += 1
                n_time_opt += bool(_TIME_OPT.match(o))
    return {
        "n_images": len(records),
        "n_questions": n_q,
        "pct_options_are_time": round(100 * n_time_opt / n_opt, 1) if n_opt else 0.0,
        "families": dict(fams),
        "answer_letters": dict(sorted(letters.items())),
    }


def _mock_response(version: str) -> str:
    if version == "v1":
        blocks = [("Over which interval does the series show the steepest rise?",
                   ["t=0–5", "t=5–12", "t=12–18", "t=18–23"], "B")] * 3
    else:
        blocks = [
            ("Over which interval does the series show the steepest rise?",
             ["t=0–5", "t=5–12", "t=12–18", "t=18–23"], "B"),
            ("Approximately how much does the value fall from its peak to the end?",
             ["about 0.5", "about 1.5", "about 3.5", "about 8.0"], "C"),
            ("Which best describes the overall shape of the curve?",
             ["Monotone rise", "Rise then fall", "A single brief spike", "Flat with noise"], "B"),
        ]
    out = []
    for i, (q, opts, ans) in enumerate(blocks, 1):
        lines = [f"#### {i}. **{q}**"]
        lines += [f"   - {L}) **{o}**" for L, o in zip("ABCD", opts)]
        lines.append("")
        lines.append(f"**Answer:** {ans}) {opts['ABCD'.index(ans)]}")
        out.append("\n".join(lines))
    return "\n------\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description="A/B the QA-generation prompt (v1 vs v2).")
    p.add_argument("--in", dest="in_path", required=True, help="fragments.jsonl")
    p.add_argument("--image-root", default=None)
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--n", type=int, default=5, help="how many charts to compare")
    p.add_argument("--ids", nargs="*", default=None, help="explicit fragment ids (overrides --n)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-series", action="store_true")
    p.add_argument("--model", default="qwen-vl-max")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-key-file", default=None, help="file holding the DashScope key")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--mock", action="store_true", help="skip the API, emit canned responses")
    args = p.parse_args()

    data = [json.loads(ln) for ln in Path(args.in_path).read_text().splitlines() if ln.strip()]
    by_id = {str(r.get("id")): r for r in data}
    if args.ids:
        missing = [i for i in args.ids if i not in by_id]
        if missing:
            raise SystemExit(f"ids not found in {args.in_path}: {missing}")
        picked = [by_id[i] for i in args.ids]
    else:
        picked = random.Random(args.seed).sample(data, args.n)

    client = None
    if not args.mock:
        key: Optional[str] = args.api_key
        if not key and args.api_key_file:
            key = Path(args.api_key_file).read_text().strip()
        from .qwen_client import BASE_URL_INTL, QwenVLClient
        client = QwenVLClient(model=args.model, api_key=key,
                              base_url=args.base_url or BASE_URL_INTL)

    image_root = Path(args.image_root) if args.image_root else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VERSIONS}
    for rec in picked:
        img = rec["image_path"]
        if image_root is not None:
            img = image_root / img
        for v in VERSIONS:
            if args.mock:
                raw = _mock_response(v)
            else:
                raw = client.chat_vision(
                    build_user_prompt(rec.get("series"), with_series=args.with_series, version=v),
                    image_path=img, system=SYS_PROMPTS[v],
                    temperature=args.temperature, max_tokens=args.max_tokens,
                )
            results[v].append({
                "id": rec.get("id"),
                "image_path": rec["image_path"],
                "prompt_version": v,
                "qa_response": raw,
                "qa_list": extract_questions_and_answers(raw),
            })

    for v in VERSIONS:
        with (out_dir / f"qa_{v}.jsonl").open("w", encoding="utf-8") as f:
            for r in results[v]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {v: summarise(results[v]) for v in VERSIONS}
    (out_dir / "summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    lines = ["# QA-generation prompt A/B: v1 (original) vs v2 (relaxed)", "",
             f"Model `{args.model}`, temperature {args.temperature}, "
             f"{len(picked)} charts, identical charts on both sides.", "", "## Summary", "",
             "| Metric | v1 | v2 |", "|---|---:|---:|"]
    lines.append(f"| Questions | {stats['v1']['n_questions']} | {stats['v2']['n_questions']} |")
    lines.append(f"| Share of options that are time indices | {stats['v1']['pct_options_are_time']}% "
                 f"| {stats['v2']['pct_options_are_time']}% |")
    for v in VERSIONS:
        lines += ["", f"**{v} question families**: " + ", ".join(
            f"{k} {n}" for k, n in sorted(stats[v]["families"].items(), key=lambda kv: -kv[1])),
            f"  answer letters: " + ", ".join(f"{k} {n}" for k, n in stats[v]["answer_letters"].items())]

    for i, rec in enumerate(picked):
        lines += ["", f"## `{rec.get('id')}` (N={len(rec.get('series') or [])})", ""]
        for v in VERSIONS:
            lines += [f"### {v}", ""]
            for j, qa in enumerate(results[v][i]["qa_list"], 1):
                lines.append(f"**{j}.** [{classify(qa['question'])}] "
                             f"answer {qa['answer']}")
                lines.append("")
                lines.append("```")
                lines.append(qa["question"])
                lines.append("```")
                lines.append("")
    (out_dir / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n-> {out_dir/'compare.md'}")


if __name__ == "__main__":
    main()
