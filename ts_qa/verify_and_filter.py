from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_OPT_RE = re.compile(r"^\s*-?\s*([A-F])\)\s*(.+?)\s*$")


def parse_question(qtext: str) -> Tuple[str, Dict[str, str]]:
    lines = qtext.splitlines()
    stem_lines, options = [], {}
    for ln in lines:
        m = _OPT_RE.match(ln)
        if m:
            options[m.group(1)] = m.group(2)
        elif not options:
            if ln.strip():
                stem_lines.append(ln.strip())
    return " ".join(stem_lines), options


METRICS = {
    "highest_value":     ("max value in region",        "argmax"),
    "lowest_value":      ("min value in region",        "argmin"),
    "highest_mean":      ("mean of region",             "argmax"),
    "lowest_mean":       ("mean of region",             "argmin"),
    "steepest_rise":     ("largest single up-step",     "argmax"),
    "deepest_dip":       ("largest single down-step",   "argmin"),
    "greatest_increase": ("net change last-first",      "argmax"),
    "greatest_decrease": ("net change last-first",      "argmin"),
    "most_variable":     ("std of step-to-step diffs",  "argmax"),
    "least_variable":    ("std of step-to-step diffs",  "argmin"),
}
METRIC_KW = {
    "highest_value":     ["highest", "maximum", "max ", "peak", "largest", "top", "greatest"],
    "lowest_value":      ["lowest", "minimum", "min ", "smallest", "trough", "bottom"],
    "highest_mean":      ["highest", "maximum", "largest", "greatest", "on average", "overall high"],
    "lowest_mean":       ["lowest", "minimum", "smallest", "bottom"],
    "steepest_rise":     ["rise", "rising", "increase", "increasing", "upward", "surge", "grow", "climb", "ascend", "ascent", "uphill"],
    "deepest_dip":       ["dip", "drop", "decline", "decrease", "decreasing", "downward", "fall", "descent", "plunge", "sink"],
    "greatest_increase": ["increase", "increasing", "rise", "rising", "growth", "gain", "upward"],
    "greatest_decrease": ["decrease", "decline", "drop", "fall", "decreasing", "downward"],
    "most_variable":     ["fluctuat", "volatil", "variab", "noisy", "noise", "choppy", "unstable", "erratic", "oscillat", "jagged"],
    "least_variable":    ["plateau", "flat", "stable", "smooth", "steady", "consistent", "calm", "constant", "level off", "least"],
}
_CMP_TOKENS = ["than", "more ", "less ", "smoother", "similar", "equally", "compared", "both "]
SCOPES = ["whole", "after_peak", "before_peak", "after_trough", "before_trough"]
SHAPES = ["increasing", "decreasing", "flat", "rise_then_fall",
          "fall_then_rise", "rise_with_dips", "oscillating_no_trend"]

PARSE_SYS = f"""You translate a multiple-choice question about a line chart into a machine-computable spec. You do NOT answer the question and you are NOT given the data — only translate.

The series has integer indices 0..n-1. Output STRICT JSON, nothing else.

Decide "kind":
- "region": the options point to WHERE on the curve something is (regions / segments / "after the peak" / halves / thirds / quarters). This is the common case.
- "classification": the options are DESCRIPTIONS of the overall shape of a stretch (e.g. "rises then falls", "flat with noise").
- "none": cannot be turned into either (subjective, needs outside knowledge, etc.).

Mapping hints (respect the DIRECTION of the wording):
- "plateau" / "flat" / "smooth" / "stable" / "steady" / "consistent" -> least_variable
- "decline" / "drop" / "decrease" / "fall" / "downward" -> deepest_dip (or greatest_decrease)
- "rise" / "increase" / "surge" / "upward" / "grow" -> steepest_rise (or greatest_increase)
- "fluctuation" / "volatility" / "noisy" / "choppy" -> most_variable
- If the options are RELATIONAL sentences comparing two named parts (e.g. "the early part is more variable", "both are similar", "the late part is smoother"), this is NOT a region question -> set kind="none".

For kind="region", output:
  "metric": one of {list(METRICS)}  (pick the one whose wording matches; each already implies which extreme wins)
  "scope":  one of {SCOPES}         (where the question restricts attention; "after the peak" -> after_peak, etc.)
  "uniqueness_feature": one of ["none","dip","spike","peak","plateau"]  -> set to the feature ONLY if the stem claims it is the *only/single* such feature; else "none"
  "options": list of {{"label","lo","hi"}} where lo<hi are FRACTIONS in [0,1] of the scope window.
     Map words to fractions: first/early third -> 0..0.33, middle -> 0.33..0.67, late/last third -> 0.67..1.0;
     first half -> 0..0.5, second half -> 0.5..1.0; first quarter -> 0..0.25; final third -> 0.67..1.0;
     within a scope like after_peak: "immediately after" -> 0..0.3, "midway" -> 0.3..0.6, "near the end" -> 0.7..1.0.
     If an option names an explicit index (e.g. "time 30"), still give your best fractional interval.

For kind="classification", output:
  "window": {{"lo","hi"}} fractions of the whole series the question asks about (whole -> 0..1; "first half" -> 0..0.5)
  "shape_options": list of {{"label","shape"}} where shape is one of {SHAPES}.

Output only JSON like:
{{"computable":true,"kind":"region","metric":"deepest_dip","scope":"whole","uniqueness_feature":"none","options":[{{"label":"A","lo":0.0,"hi":0.33}}, ...]}}
"""


def _strip_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    a, b = text.find("{"), text.rfind("}")
    return text[a:b + 1] if a >= 0 and b > a else text


def llm_parse(client, stem: str, options: Dict[str, str], n: int) -> Optional[dict]:
    opts = "\n".join(f"  - {k}) {v}" for k, v in options.items())
    prompt = f"n = {n}\nQuestion: {stem}\nOptions:\n{opts}\n\nReturn the JSON spec."
    try:
        raw = client.chat_text(prompt, system=PARSE_SYS)
        return json.loads(_strip_json(raw))
    except Exception as exc:
        logging.warning("parse failed: %s", exc)
        return None


def _window(series: np.ndarray, scope: str) -> Tuple[int, int]:
    n = len(series)
    if scope == "after_peak":   return int(np.argmax(series)), n
    if scope == "before_peak":  return 0, int(np.argmax(series)) + 1
    if scope == "after_trough": return int(np.argmin(series)), n
    if scope == "before_trough":return 0, int(np.argmin(series)) + 1
    return 0, n


def _region_idx(a: int, b: int, lo: float, hi: float) -> Tuple[int, int]:
    span = b - a
    i0 = a + int(round(lo * (span - 1)))
    i1 = a + int(round(hi * (span - 1)))
    return max(a, min(i0, b - 1)), max(a, min(i1, b - 1))


def _score(series: np.ndarray, i0: int, i1: int, metric: str) -> Optional[float]:
    seg = series[i0:i1 + 1]
    base = METRICS[metric][0]
    if "single" in base and len(seg) < 2:
        return None
    if len(seg) == 0:
        return None
    d = np.diff(seg)
    if metric in ("highest_value", "lowest_value"):
        return float(seg.max() if metric == "highest_value" else seg.min())
    if metric in ("highest_mean", "lowest_mean"):
        return float(seg.mean())
    if metric == "steepest_rise":      return float(d.max())
    if metric == "deepest_dip":        return float(d.min())
    if metric in ("greatest_increase", "greatest_decrease"):
        return float(seg[-1] - seg[0])
    if metric in ("most_variable", "least_variable"):
        return float(d.std()) if len(d) else None
    return None


def _overlap(r1: Tuple[int, int], r2: Tuple[int, int]) -> bool:
    inter = min(r1[1], r2[1]) - max(r1[0], r2[0]) + 1
    return inter >= 2


def eval_region(series: np.ndarray, spec: dict) -> Optional[dict]:
    metric, scope = spec.get("metric"), spec.get("scope", "whole")
    if metric not in METRICS:
        return None
    a, b = _window(series, scope)
    if b - a < 2:
        return None
    arg = METRICS[metric][1]
    scores: Dict[str, float] = {}
    spans: Dict[str, Tuple[int, int]] = {}
    for o in spec.get("options", []):
        lab = o.get("label")
        try:
            i0, i1 = _region_idx(a, b, float(o["lo"]), float(o["hi"]))
        except (KeyError, TypeError, ValueError):
            continue
        s = _score(series, i0, i1, metric)
        if s is not None:
            scores[lab] = s
            spans[lab] = (i0, i1)
    if len(scores) < 2:
        return None

    better = (lambda x, y: x > y) if arg == "argmax" else (lambda x, y: x < y)
    best = max(scores.values()) if arg == "argmax" else min(scores.values())
    eps = 1e-6 * (abs(best) + 1)
    tied = [l for l, v in scores.items() if abs(v - best) <= eps]
    winner = min(tied, key=lambda l: spans[l][1] - spans[l][0])

    rivals = [v for l, v in scores.items() if l != winner and not _overlap(spans[l], spans[winner])]
    spread = max(scores.values()) - min(scores.values())
    margin = 1.0 if not rivals else abs(best - (max(rivals) if arg == "argmax" else min(rivals))) / (spread + 1e-9)
    return {"winner": winner, "margin": margin, "scores": {k: round(v, 3) for k, v in scores.items()}}


def classify_shape(seg: np.ndarray) -> Tuple[str, float]:
    n = len(seg)
    if n < 3:
        return "flat", 0.0
    rng = float(seg.max() - seg.min()) + 1e-9
    net = float(seg[-1] - seg[0]) / rng
    d = np.diff(seg)
    up = float(np.mean(d > 0))
    amax, amin = int(np.argmax(seg)) / (n - 1), int(np.argmin(seg)) / (n - 1)
    std_dt = float((seg - np.poly1d(np.polyfit(np.arange(n), seg, 1))(np.arange(n))).std()) / rng
    n_dips = _count_feature(seg, "dip")
    if rng < 0.15 * (abs(float(seg.mean())) + 1e-9) or (abs(net) < 0.15 and std_dt < 0.1):
        return "flat", 0.7
    if 0.3 < amax < 0.85 and seg[0] < seg.max() * 0.9 and seg[-1] < seg.max() * 0.9:
        return "rise_then_fall", 0.7
    if 0.3 < amin < 0.85 and seg[0] > seg.min() and seg[-1] > seg.min():
        return "fall_then_rise", 0.6
    if net > 0.4:
        return ("rise_with_dips", 0.6) if n_dips >= 2 else ("increasing", 0.7)
    if net < -0.4:
        return "decreasing", 0.7
    return "oscillating_no_trend", 0.5


def eval_classification(series: np.ndarray, spec: dict) -> Optional[dict]:
    w = spec.get("window", {"lo": 0.0, "hi": 1.0})
    i0, i1 = _region_idx(0, len(series), float(w.get("lo", 0)), float(w.get("hi", 1)))
    label, conf = classify_shape(series[i0:i1 + 1])
    if conf < 0.6:
        return None
    matches = [o["label"] for o in spec.get("shape_options", []) if o.get("shape") == label]
    if len(matches) != 1:
        return None
    return {"winner": matches[0], "margin": conf, "scores": {"shape": label}}


def _count_feature(seg: np.ndarray, feature: str) -> int:
    if len(seg) < 3:
        return 0
    rng = float(seg.max() - seg.min()) + 1e-9
    prom = 0.12 * rng
    cnt = 0
    for i in range(1, len(seg) - 1):
        if feature in ("dip", "plateau") and seg[i] < seg[i - 1] and seg[i] <= seg[i + 1]:
            if min(seg[i - 1], seg[i + 1]) - seg[i] >= prom:
                cnt += 1
        if feature in ("spike", "peak") and seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]:
            if seg[i] - max(seg[i - 1], seg[i + 1]) >= prom:
                cnt += 1
    return cnt


def _keyword_ok(stem: str, metric: str) -> bool:
    s = stem.lower()
    return any(kw in s for kw in METRIC_KW.get(metric, []))


def _is_comparison(options: Dict[str, str]) -> bool:
    hits = sum(any(t in v.lower() for t in _CMP_TOKENS) for v in options.values())
    return hits >= 2


_UNIQUENESS_RE = re.compile(r"\b(only|sole|solely|exactly one|one and only|unique)\b", re.I)


def _claims_uniqueness(stem: str) -> bool:
    return bool(_UNIQUENESS_RE.search(stem or ""))


def decide(spec: dict, series: np.ndarray, vlm_label: str, margin_thr: float,
           stem: str = "", options: Optional[Dict[str, str]] = None) -> Tuple[str, Optional[str], str]:
    if not spec or not spec.get("computable", True) or spec.get("kind") == "none":
        return "DROP", None, "non-computable"

    uf = spec.get("uniqueness_feature", "none")
    if uf and uf != "none" and _claims_uniqueness(stem) and _count_feature(series, uf) > 1:
        return "DROP", None, f"false premise: '{uf}' occurs >1 time"

    if spec.get("kind") == "region":
        if options and _is_comparison(options):
            return "DROP", None, "comparison-style options (not a region question)"
        if not _keyword_ok(stem, spec.get("metric", "")):
            return "DROP", None, f"metric '{spec.get('metric')}' not supported by stem wording"

    if spec.get("kind") == "classification":
        res = eval_classification(series, spec)
        if res is None:
            return "DROP", None, "shape ambiguous / no unique matching option"
    else:
        res = eval_region(series, spec)
        if res is None:
            return "DROP", None, "metric not computable / <2 valid options"
        if res["margin"] < margin_thr:
            return "DROP", None, f"near-tie (margin {res['margin']:.2f} < {margin_thr})"

    truth = res["winner"]
    if truth == vlm_label:
        return "KEEP", truth, f"verified (margin {res['margin']:.2f})"
    return "CORRECT", truth, f"label {vlm_label}->{truth} (margin {res['margin']:.2f})"


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-parse + code-compute verification/correction of TS MCQs.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True, help="corrected survivors jsonl")
    p.add_argument("--report", default=None, help="markdown per-question report")
    p.add_argument("--parser-model", default="qwen-max", help="text LLM for parsing (e.g. qwen-max)")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--margin", type=float, default=0.15, help="min top1-vs-top2 relative gap to keep")
    p.add_argument("--min-per-image", type=int, default=2, help="drop image if fewer survive")
    p.add_argument("--on-correct", choices=["overwrite", "drop"], default="overwrite",
                   help="when code-truth != VLM label: overwrite label or drop the question")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    from .qwen_client import BASE_URL_INTL, QwenVLClient
    client = QwenVLClient(model=args.parser_model, api_key=args.api_key,
                          base_url=args.base_url or BASE_URL_INTL)

    records = [json.loads(l) for l in Path(args.in_path).read_text().splitlines() if l.strip()]
    out_records, report = [], []
    stats = {"KEEP": 0, "CORRECT": 0, "DROP": 0}

    for rec in records:
        series = np.asarray(rec["series"], dtype=float)
        survivors, rep_qs = [], []
        for qa in rec["qa_list"]:
            stem, options = parse_question(qa["question"])
            vlm = (qa.get("answer") or "").strip()[:1]
            spec = llm_parse(client, stem, options, len(series))
            decision, truth, reason = decide(spec or {}, series, vlm, args.margin, stem, options)
            if decision == "CORRECT" and args.on_correct == "drop":
                decision, reason = "DROP", "corrected-but-on-correct=drop: " + reason
            stats[decision] += 1
            rep_qs.append((stem, vlm, truth, decision, reason,
                           spec.get("metric") or spec.get("kind") if spec else None))
            if decision in ("KEEP", "CORRECT"):
                q2 = dict(qa)
                q2["answer"] = truth
                q2["verify"] = {"decision": decision, "metric": spec.get("metric") if spec else None,
                                "reason": reason}
                survivors.append(q2)
        if len(survivors) >= args.min_per_image:
            out_records.append({**{k: rec[k] for k in ("id", "image_path", "series") if k in rec},
                                "qa_list": survivors})
        report.append((rec.get("image_path"), len(survivors), rep_qs))

    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n")
    logging.info("decisions: %s | images kept %d/%d -> %s",
                 stats, len(out_records), len(records), args.out)

    if args.report:
        lines = [f"# Verify & correct report\n",
                 f"> parser=`{args.parser_model}`  margin≥{args.margin}  on_correct={args.on_correct}\n",
                 f"**KEEP {stats['KEEP']} · CORRECT {stats['CORRECT']} · DROP {stats['DROP']}** "
                 f"| images kept {len(out_records)}/{len(records)}\n"]
        for img, nkeep, qs in report:
            lines.append(f"\n## {img}  (survivors: {nkeep})")
            for stem, vlm, truth, dec, reason, mk in qs:
                mark = {"KEEP": "✅", "CORRECT": "✏️", "DROP": "🗑️"}[dec]
                lines.append(f"- {mark} **{dec}** [{mk}] VLM={vlm} truth={truth} — {reason}\n    - {stem}")
        Path(args.report).write_text("\n".join(lines) + "\n")
        logging.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
