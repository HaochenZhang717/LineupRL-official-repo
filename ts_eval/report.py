from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from ts_eval import scoring as sc
from ts_eval.caption_answer_harness import PredictionRecord


def unparsed_rate(records: list[PredictionRecord]) -> float | None:
    labelled = [r for r in records if sc.is_label_scored(r.scoring_type)]
    if not labelled:
        return None
    return sum(sc.parse_prediction(r.answer_raw, r.scoring_type) is None
               for r in labelled) / len(labelled)


def aggregate(records: list[PredictionRecord]) -> dict:
    def _bucket(key_fn) -> dict:
        buckets: dict[str, list[PredictionRecord]] = defaultdict(list)
        for r in records:
            buckets[key_fn(r) or "unknown"].append(r)
        out = {}
        for k, rs in sorted(buckets.items()):
            entry = {"n": len(rs), "accuracy": sum(r.score for r in rs) / len(rs)}
            rate = unparsed_rate(rs)
            if rate is not None:
                entry["unparsed_rate"] = rate
            out[k] = entry
        return out

    report = {
        "n_items": len(records),
        "overall_accuracy": (sum(r.score for r in records) / len(records)) if records else 0.0,
        "by_domain": _bucket(lambda r: r.domain),
        "by_task_type": _bucket(lambda r: r.task_type),
        "by_scoring_type": _bucket(lambda r: r.scoring_type),
    }
    rate = unparsed_rate(records)
    if rate is not None:
        report["unparsed_rate"] = rate
    return report


def _markdown_table(rows: dict[str, dict]) -> list[str]:
    show_unparsed = any("unparsed_rate" in v for v in rows.values())
    header = "| key | n | accuracy |" + (" unparsed |" if show_unparsed else "")
    lines = [header, "|" + "---|" * (3 + int(show_unparsed))]
    for k, v in rows.items():
        row = f"| {k} | {v['n']} | {v['accuracy']:.4f} |"
        if show_unparsed:
            row += (f" {v['unparsed_rate']:.4f} |" if "unparsed_rate" in v else " - |")
        lines.append(row)
    return lines


def write_report(records: list[PredictionRecord], out_dir: Path, run_name: str,
                 provenance: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    report = aggregate(records)
    report["run_name"] = run_name
    if provenance:
        report["provenance"] = provenance
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# {run_name}", "",
        f"- n_items: {report['n_items']}",
        f"- overall_accuracy: {report['overall_accuracy']:.4f}", "",
        "## By domain", "", *_markdown_table(report["by_domain"]), "",
        "## By task_type", "", *_markdown_table(report["by_task_type"]), "",
        "## By scoring_type", "", *_markdown_table(report["by_scoring_type"]),
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
