from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from .verify_and_filter import (
    METRICS, METRIC_KW, SCOPES, SHAPES,
    _window, _score, _overlap, _is_comparison, _keyword_ok,
    classify_shape, _strip_json, parse_question,
)


class SeriesTools:

    def __init__(self, series: np.ndarray) -> None:
        self.s = np.asarray(series, dtype=float)

    def series_info(self) -> dict:
        s = self.s
        return {"n": int(len(s)), "argmax": int(np.argmax(s)), "argmin": int(np.argmin(s)),
                "vmin": round(float(s.min()), 4), "vmax": round(float(s.max()), 4),
                "range": round(float(s.max() - s.min()), 4), "mean": round(float(s.mean()), 4)}

    def locate_window(self, scope: str) -> dict:
        a, b = _window(self.s, scope)
        return {"start": int(a), "end_exclusive": int(b), "i0": int(a), "i1": int(b - 1)}

    def value_at(self, i: int) -> dict:
        i = int(max(0, min(i, len(self.s) - 1)))
        return {"index": i, "value": round(float(self.s[i]), 4)}

    def segment_stats(self, i0: int, i1: int) -> dict:
        i0, i1 = self._clip(i0, i1)
        seg = self.s[i0:i1 + 1]
        d = np.diff(seg) if len(seg) >= 2 else np.array([0.0])
        return {"i0": i0, "i1": i1, "n": int(len(seg)),
                "mean": round(float(seg.mean()), 4),
                "vmin": round(float(seg.min()), 4), "vmax": round(float(seg.max()), 4),
                "argmin": int(i0 + np.argmin(seg)), "argmax": int(i0 + np.argmax(seg)),
                "net_last_minus_first": round(float(seg[-1] - seg[0]), 4),
                "max_up_step": round(float(d.max()), 4), "max_down_step": round(float(d.min()), 4),
                "std_step": round(float(d.std()), 4)}

    def residual_stats(self, i0: int, i1: int) -> dict:
        i0, i1 = self._clip(i0, i1)
        seg = self.s[i0:i1 + 1]
        if len(seg) < 3:
            return {"i0": i0, "i1": i1, "error": "segment too short to detrend"}
        x = np.arange(len(seg))
        resid = seg - np.poly1d(np.polyfit(x, seg, 1))(x)
        return {"i0": i0, "i1": i1,
                "resid_min": round(float(resid.min()), 4), "resid_max": round(float(resid.max()), 4),
                "resid_std": round(float(resid.std()), 4),
                "resid_argmin": int(i0 + np.argmin(resid)), "resid_argmax": int(i0 + np.argmax(resid))}

    def find_features(self, kind: str, i0: Optional[int] = None, i1: Optional[int] = None,
                      min_prominence_frac: float = 0.10) -> dict:
        a, b = self._clip(i0 if i0 is not None else 0, i1 if i1 is not None else len(self.s) - 1)
        seg = self.s[a:b + 1]
        rng = float(seg.max() - seg.min()) + 1e-12
        thr = min_prominence_frac * rng
        out = []
        for k, prom in self._topo_prominence(seg, kind):
            if prom >= thr:
                out.append({"index": int(a + k), "value": round(float(seg[k]), 4),
                            "prominence": round(float(prom), 4),
                            "prominence_frac": round(float(prom) / rng, 3)})
        return {"kind": kind, "i0": a, "i1": b, "range": round(rng, 4),
                "count": len(out), "features": out}

    def flat_run(self, i0: Optional[int] = None, i1: Optional[int] = None,
                 tol_frac: float = 0.05) -> dict:
        a, b = self._clip(i0 if i0 is not None else 0, i1 if i1 is not None else len(self.s) - 1)
        seg = self.s[a:b + 1]
        rng = float(seg.max() - seg.min()) + 1e-12
        tol = tol_frac * rng
        best = cur = 1
        bstart = start = 0
        for i in range(1, len(seg)):
            if abs(seg[i] - seg[i - 1]) <= tol:
                cur += 1
            else:
                if cur > best:
                    best, bstart = cur, start
                cur, start = 1, i
        if cur > best:
            best, bstart = cur, start
        return {"i0": a, "i1": b, "tol": round(tol, 5), "longest_flat_len": int(best),
                "run_i0": int(a + bstart), "run_i1": int(a + bstart + best - 1),
                "frac_of_window": round(best / max(1, len(seg)), 2)}

    def _clip(self, i0: int, i1: int) -> Tuple[int, int]:
        n = len(self.s)
        i0 = int(max(0, min(int(i0), n - 1)))
        i1 = int(max(0, min(int(i1), n - 1)))
        return (i0, i1) if i0 <= i1 else (i1, i0)

    @staticmethod
    def _topo_prominence(seg: np.ndarray, kind: str) -> List[Tuple[int, float]]:
        n = len(seg)
        if n < 3:
            return []
        s = seg if kind == "peak" else -seg
        res = []
        for i in range(1, n - 1):
            if s[i] >= s[i - 1] and s[i] >= s[i + 1] and (s[i] > s[i - 1] or s[i] > s[i + 1]):
                lv = s[i]
                for j in range(i - 1, -1, -1):
                    lv = min(lv, s[j])
                    if j > 0 and s[j] < s[j - 1]:
                        break
                rv = s[i]
                for j in range(i + 1, n):
                    rv = min(rv, s[j])
                    if j < n - 1 and s[j] < s[j + 1]:
                        break
                prom = s[i] - max(lv, rv)
                if prom > 0:
                    res.append((i, float(prom)))
        return res


_TOOLS = ("series_info", "locate_window", "value_at", "segment_stats",
          "residual_stats", "find_features", "flat_run")


AGENT_SYS = f"""You convert ONE multiple-choice question about a line chart into a machine-computable spec, by PROBING the real series with tools. You do NOT answer the question; a separate program computes the winner from your spec.

The series has integer indices 0..n-1. Call series_info first to learn n.

You may call these tools (one per step), passing INCLUSIVE index ranges [i0,i1]:
- series_info(): n, argmax, argmin, range, mean
- locate_window(scope): one of {SCOPES} -> the index window the scope refers to
- value_at(i)
- segment_stats(i0,i1): mean/min/max/net/max_up_step/max_down_step/std_step
- residual_stats(i0,i1): linear-detrended residual min/max/std (use for "below/around the overall trend")
- find_features(kind,i0,i1): kind="peak"|"trough"; features ranked, prominence as a fraction of the window range (use to LOCATE the 1st/2nd/Nth peak/trough, or to check a claimed unique spike/dip; if count==0 the claimed feature is absent)
- flat_run(i0,i1): longest near-flat run (use for "longest plateau / most stable stretch")

Protocol — output STRICT JSON each turn, nothing else, one of:
  {{"thought":"...","action":{{"tool":"segment_stats","args":{{"i0":0,"i1":7}}}}}}
  {{"thought":"...","final_spec":{{...}}}}

final_spec MUST be one of:
- region question (options point to WHERE on the curve something is):
  {{"kind":"region","metric":<one of {list(METRICS)}>,
    "options":[{{"label":"A","i0":0,"i1":7}}, ...],   // REAL inclusive index ranges you grounded
    "none_option": <label or null>}}                  // label meaning "none/all-equal/does-not-occur", else null
- classification question (options DESCRIBE the shape of a stretch):
  {{"kind":"classification","window":{{"i0":0,"i1":23}},
    "shape_options":[{{"label":"A","shape":<one of {SHAPES}>}}, ...]}}
- not computable: {{"kind":"none","reason":"..."}}
  Use kind="none" when the question is subjective, needs outside knowledge, the options are
  RELATIONAL sentences comparing two named parts, OR your probing shows the feature the
  question asks about does NOT exist in the data (e.g. it asks where the biggest drop/peak is
  but the series is essentially flat with no real move).

Rules:
- Pick the metric whose wording matches the stem; each metric already implies which extreme wins. Do NOT judge magnitudes yourself.
- Ground every option to the index range it refers to (first/early third -> first ~third of the relevant window; "after the 2nd peak" -> find_features then i0=that peak's index). Options may be anchored to DIFFERENT features.
- Keep probing short (a handful of calls). Emit final_spec once every option maps to indices and you know the metric.
"""

AGENT_MAX_STEPS = 8


def _extract_action(raw: str) -> Optional[dict]:
    try:
        return json.loads(_strip_json(raw))
    except Exception:
        return None


def agent_ground(client, stem: str, options: Dict[str, str], tools: SeriesTools,
                 max_steps: int = AGENT_MAX_STEPS, trace: Optional[list] = None,
                 max_tokens: int = 700) -> Optional[dict]:
    opts = "\n".join(f"  - {k}) {v}" for k, v in options.items())
    convo = [f"n = {tools.series_info()['n']}\nQuestion: {stem}\nOptions:\n{opts}\n\nBegin. Call series_info first."]
    for step in range(max_steps):
        raw = client.chat_text("\n".join(convo), system=AGENT_SYS, temperature=0.0, max_tokens=max_tokens)
        act = _extract_action(raw)
        ev = {"step": step, "raw": raw.strip()[:2000], "parsed_json": act is not None}
        if not act:
            ev["event"] = "invalid_json"
            if trace is not None:
                trace.append(ev)
            convo.append("[your output was not valid JSON; re-emit a single JSON object]")
            continue
        if "final_spec" in act:
            ev["event"] = "final_spec"
            if trace is not None:
                trace.append(ev)
            return act["final_spec"]
        action = act.get("action") or {}
        tool, targs = action.get("tool"), action.get("args", {}) or {}
        if tool not in _TOOLS:
            ev["event"] = "unknown_tool"; ev["tool"] = tool
            if trace is not None:
                trace.append(ev)
            convo.append(f"OBSERVATION: unknown tool {tool!r}. Valid: {list(_TOOLS)}")
            continue
        try:
            obs = getattr(tools, tool)(**targs)
        except Exception as e:
            obs = {"error": f"{type(e).__name__}: {e}"}
        ev["event"] = "tool_call"; ev["tool"] = tool; ev["args"] = targs; ev["observation"] = obs
        if trace is not None:
            trace.append(ev)
        convo.append(json.dumps(act))
        convo.append("OBSERVATION: " + json.dumps(obs))
    if trace is not None:
        trace.append({"step": max_steps, "event": "exhausted_no_final_spec"})
    return None


def eval_region_idx(series: np.ndarray, spec: dict) -> Optional[dict]:
    metric = spec.get("metric")
    if metric not in METRICS:
        return None
    arg = METRICS[metric][1]
    scores: Dict[str, float] = {}
    spans: Dict[str, Tuple[int, int]] = {}
    for o in spec.get("options", []):
        lab = o.get("label")
        try:
            i0, i1 = int(o["i0"]), int(o["i1"])
        except (KeyError, TypeError, ValueError):
            continue
        i0, i1 = max(0, min(i0, len(series) - 1)), max(0, min(i1, len(series) - 1))
        if i0 > i1:
            i0, i1 = i1, i0
        s = _score(series, i0, i1, metric)
        if s is not None:
            scores[lab] = s
            spans[lab] = (i0, i1)
    if len(scores) < 2:
        return None
    best = max(scores.values()) if arg == "argmax" else min(scores.values())
    eps = 1e-6 * (abs(best) + 1)
    tied = [l for l, v in scores.items() if abs(v - best) <= eps]
    winner = min(tied, key=lambda l: spans[l][1] - spans[l][0])
    rivals = [v for l, v in scores.items() if l != winner and not _overlap(spans[l], spans[winner])]
    spread = max(scores.values()) - min(scores.values())
    margin = 1.0 if not rivals else abs(best - (max(rivals) if arg == "argmax" else min(rivals))) / (spread + 1e-9)
    return {"winner": winner, "margin": margin, "scores": {k: round(v, 4) for k, v in scores.items()}}


def eval_classification_idx(series: np.ndarray, spec: dict) -> Optional[dict]:
    w = spec.get("window", {})
    try:
        i0, i1 = int(w["i0"]), int(w["i1"])
    except (KeyError, TypeError, ValueError):
        i0, i1 = 0, len(series) - 1
    label, conf = classify_shape(series[i0:i1 + 1])
    if conf < 0.6:
        return None
    matches = [o["label"] for o in spec.get("shape_options", []) if o.get("shape") == label]
    if len(matches) != 1:
        return None
    return {"winner": matches[0], "margin": conf, "scores": {"shape": label}}


def adjudicate(spec: Optional[dict], series: np.ndarray, vlm_label: str,
               stem: str, options: Dict[str, str], margin_thr: float = 0.15
               ) -> Tuple[str, Optional[str], str]:
    if not spec or spec.get("kind") == "none":
        return "DROP", None, "non-computable: " + (spec or {}).get("reason", "agent->none")

    kind = spec.get("kind")
    if kind == "classification":
        res = eval_classification_idx(series, spec)
        if res is None:
            return "DROP", None, "shape ambiguous / no unique matching option"
    else:
        if options and _is_comparison(options):
            return "DROP", None, "comparison-style options (not a region question)"
        if not _keyword_ok(stem, spec.get("metric", "")):
            return "DROP", None, f"metric '{spec.get('metric')}' not supported by stem wording"
        res = eval_region_idx(series, spec)
        if res is None:
            return "DROP", None, "metric not computable / <2 valid options"
        if res["margin"] < margin_thr:
            none_opt = spec.get("none_option")
            if none_opt:
                truth = str(none_opt)[:1]
                dec = "KEEP" if truth == vlm_label else "CORRECT"
                return dec, truth, f"no clear region (margin {res['margin']:.2f}) -> none-option {truth}"
            return "DROP", None, f"near-tie (margin {res['margin']:.2f} < {margin_thr})"

    truth = res["winner"]
    tag = f"margin {res['margin']:.2f}"
    if truth == vlm_label:
        return "KEEP", truth, f"verified ({tag})"
    return "CORRECT", truth, f"{vlm_label}->{truth} ({tag})"


def _verify_one(client, series: np.ndarray, stem: str, options: Dict[str, str],
                vlm: str, margin_thr: float, want_trace: bool = False,
                max_tokens: int = 700) -> tuple:
    tools = SeriesTools(series)
    trace: Optional[list] = [] if want_trace else None
    err = None
    try:
        spec = agent_ground(client, stem, options, tools, trace=trace, max_tokens=max_tokens)
    except Exception as e:
        spec = None
        err = f"{type(e).__name__}: {e}"
        logging.warning("agent failed: %s", e)
    decision, truth, reason = adjudicate(spec, series, vlm, stem, options, margin_thr)
    kind = (spec or {}).get("kind"); metric = (spec or {}).get("metric")
    return decision, truth, reason, kind, metric, spec, trace, err


def main() -> None:
    p = argparse.ArgumentParser(description="Agent-grounded verify/correct of TS MCQs.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", default=None, help="corrected survivors jsonl (optional)")
    p.add_argument("--report", default=None, help="markdown per-question report")
    p.add_argument("--trace-out", default=None,
                   help="jsonl: one record per question with the full ReAct trace (debug)")
    p.add_argument("--agent-model", default="deepseek-v4-pro")
    p.add_argument("--base-url", default="https://api.deepseek.com")
    p.add_argument("--api-key", default=None)
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=None, help="only first N images (debug)")
    p.add_argument("--shard", default=None,
                   help="process only shard i of N images, format 'i/N' (i in 0..N-1); "
                        "strided so each shard gets a balanced interleaved subset")
    p.add_argument("--workers", type=int, default=8,
                   help="concurrent questions (each is an independent agent loop)")
    p.add_argument("--max-tokens", type=int, default=700,
                   help="max output tokens per ReAct step (raise for verbose models that "
                        "get truncated mid-JSON, e.g. Qwen3.6)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    from .qwen_client import QwenVLClient
    client = QwenVLClient(model=args.agent_model, api_key=args.api_key, base_url=args.base_url)

    records = [json.loads(l) for l in open(args.in_path) if l.strip()]
    if args.limit:
        records = records[: args.limit]
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if not (0 <= si < sn):
            raise SystemExit(f"--shard {args.shard}: need 0 <= i < N")
        records = records[si::sn]
        logging.info("shard %d/%d -> %d images", si, sn, len(records))
    stats = {"KEEP": 0, "CORRECT": 0, "DROP": 0}
    report: list = []
    out_records: list = []

    tasks = []
    for ii, rec in enumerate(records):
        series = np.asarray(rec["series"], dtype=float)
        for qi, qa in enumerate(rec["qa_list"]):
            stem, options = parse_question(qa["question"])
            vlm = (qa.get("answer") or "").strip()[:1]
            tasks.append((ii, qi, series, qa, stem, options, vlm))

    want_trace = bool(args.trace_out)
    trace_records: list = []
    results: Dict[tuple, tuple] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut2task = {ex.submit(_verify_one, client, t[2], t[4], t[5], t[6], args.margin,
                              want_trace, args.max_tokens): t
                    for t in tasks}
        for fut in as_completed(fut2task):
            ii, qi, series, qa, stem, options, vlm = fut2task[fut]
            decision, truth, reason, kind, metric, spec, trace, err = fut.result()
            results[(ii, qi)] = (decision, truth, reason, kind, metric)
            if want_trace:
                s = series
                trace_records.append({
                    "image_path": records[ii].get("image_path") or records[ii].get("id"),
                    "qi": qi, "stem": stem, "options": options, "vlm": vlm,
                    "decision": decision, "truth": truth, "reason": reason,
                    "kind": kind, "metric": metric, "spec": spec, "error": err,
                    "n_steps": (len(trace) if trace else 0),
                    "series_stats": {"n": int(len(s)), "min": float(s.min()), "max": float(s.max()),
                                     "argmin": int(s.argmin()), "argmax": int(s.argmax()),
                                     "mean": round(float(s.mean()), 3)},
                    "trace": trace,
                })
            stats[decision] += 1
            done += 1
            logging.info("(%d/%d) [%s] VLM=%s truth=%s %-22s | %s", done, len(tasks), decision,
                         "?", truth, f"({kind}/{metric})", stem[:60])

    for ii, rec in enumerate(records):
        survivors, rep_qs = [], []
        for qi, qa in enumerate(rec["qa_list"]):
            decision, truth, reason, kind, metric = results[(ii, qi)]
            stem, _ = parse_question(qa["question"])
            vlm = (qa.get("answer") or "").strip()[:1]
            rep_qs.append((stem, vlm, truth, decision, reason, kind, metric))
            if decision in ("KEEP", "CORRECT"):
                q2 = dict(qa); q2["answer"] = truth
                q2["verify"] = {"decision": decision, "kind": kind, "metric": metric, "reason": reason}
                survivors.append(q2)
        out_records.append({**{k: rec[k] for k in ("id", "image_path", "series", "source") if k in rec},
                            "qa_list": survivors})
        report.append((rec.get("image_path") or rec.get("id"), len(rec["qa_list"]), len(survivors), rep_qs))

    logging.info("decisions: %s", stats)
    if args.trace_out:
        trace_records.sort(key=lambda r: (r["image_path"], r["qi"]))
        open(args.trace_out, "w").write(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in trace_records) + "\n")
        logging.info("trace -> %s (%d questions)", args.trace_out, len(trace_records))
    if args.out:
        open(args.out, "w").write("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n")
    if args.report:
        lines = [f"# Agent verify report  (model={args.agent_model}, margin≥{args.margin})\n",
                 f"**KEEP {stats['KEEP']} · CORRECT {stats['CORRECT']} · DROP {stats['DROP']}**\n"]
        for img, ntot, nkeep, qs in report:
            lines.append(f"\n## {img}  ({nkeep}/{ntot} survive)")
            for stem, vlm, truth, dec, reason, kind, metric in qs:
                mark = {"KEEP": "✅", "CORRECT": "✏️", "DROP": "🗑️"}[dec]
                lines.append(f"- {mark} **{dec}** [{kind}/{metric}] VLM={vlm} truth={truth} — {reason}\n    - {stem}")
        open(args.report, "w").write("\n".join(lines) + "\n")
        logging.info("report -> %s", args.report)


if __name__ == "__main__":
    main()
