from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ts_eval.scoring import normalize_gold, parse_prediction

GATE = 0.10
CHANCE = {"recognition": 0.5, "differentiation": 0.25}
LEGS = ("L0", "L2")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _predicted_label(record: dict) -> str | None:
    return parse_prediction(record["answer_raw"], record["scoring_type"])


def _weighted_f1(golds: list[str], preds: list[str | None]) -> float:
    labels = sorted(set(golds))
    preds = [p if p is not None else "<unparsed>" for p in preds]
    total = len(golds)
    score = 0.0
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        support = sum(g == label for g in golds)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        score += f1 * support / total
    return score


def load_provenance(root: Path, run: str) -> dict:
    path = root / run / "report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("provenance", {})


def comparable(a: dict, b: dict) -> bool:
    fa, fb = a.get("prompt_fingerprint"), b.get("prompt_fingerprint")
    return bool(fa) and fa == fb


def load_run(root: Path, run: str, gold_by_id: dict[str, str]) -> dict[tuple[str, str], dict]:
    path = root / run / "predictions.jsonl"
    if not path.exists():
        return {}
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in _read_jsonl(path):
        cells[(record["domain"], record["task_type"])].append(record)

    out = {}
    for key, records in cells.items():
        preds = [_predicted_label(r) for r in records]
        golds = [normalize_gold(gold_by_id[r["item_id"]], r["scoring_type"]) for r in records]
        out[key] = {
            "n": len(records),
            "accuracy": sum(r["score"] for r in records) / len(records),
            "weighted_f1": _weighted_f1(golds, preds),
            "unparsed_rate": sum(p is None for p in preds) / len(records),
        }
    return out


def information_recovery(l0: float, l1: float, l2: float) -> float | None:
    spread = l2 - l0
    return None if spread <= 0 else (l1 - l0) / spread


def build(root: Path, arms: list[str], items_path: Path) -> dict:
    gold_by_id = {row["item_id"]: row["gold"] for row in _read_jsonl(items_path)}
    runs = {name: load_run(root, name, gold_by_id) for name in (*LEGS, *arms)}
    missing = [name for name, cells in runs.items() if not cells]
    keys = sorted({k for cells in runs.values() for k in cells})

    prov = {name: load_provenance(root, name) for name in (*LEGS, *arms)}
    incomparable = sorted(
        arm for arm in arms
        if runs.get(arm) and not (comparable(prov.get(arm, {}), prov.get("L0", {}))
                                  and comparable(prov.get(arm, {}), prov.get("L2", {}))))

    table = []
    for dataset, task in keys:
        cell = {"dataset": dataset, "task": task, "chance": CHANCE.get(task),
                "n": next(runs[r][(dataset, task)]["n"] for r in runs
                          if (dataset, task) in runs[r])}
        for name, cells in runs.items():
            if (dataset, task) in cells:
                cell[name] = cells[(dataset, task)]
        l0 = cell.get("L0", {}).get("accuracy")
        l2 = cell.get("L2", {}).get("accuracy")
        if l0 is not None and l2 is not None:
            cell["oracle_spread"] = l2 - l0
            cell["admitted"] = (l2 - l0) >= GATE
            for arm in arms:
                if arm in cell and arm not in incomparable:
                    cell[f"IR_{arm}"] = information_recovery(l0, cell[arm]["accuracy"], l2)
        table.append(cell)

    return {"root": str(root), "gate": GATE, "arms": arms, "missing_runs": missing,
            "incomparable_arms": incomparable, "provenance": prov, "cells": table}


def build_direct(root: Path, arms: list[str], items_path: Path) -> dict:
    gold_by_id = {row["item_id"]: row["gold"] for row in _read_jsonl(items_path)}
    names = [n for arm in arms for n in (arm, f"{arm}_noimg")]
    runs = {name: load_run(root, name, gold_by_id) for name in names}
    missing = [name for name, cells in runs.items() if not cells]
    keys = sorted({k for cells in runs.values() for k in cells})

    table = []
    for dataset, task in keys:
        cell = {"dataset": dataset, "task": task, "chance": CHANCE.get(task),
                "n": next(runs[r][(dataset, task)]["n"] for r in runs
                          if (dataset, task) in runs[r])}
        for name, cells in runs.items():
            if (dataset, task) in cells:
                cell[name] = cells[(dataset, task)]
        for arm in arms:
            floor = cell.get(f"{arm}_noimg", {}).get("accuracy")
            acc = cell.get(arm, {}).get("accuracy")
            if floor is not None and acc is not None:
                cell[f"lift_{arm}"] = acc - floor
        table.append(cell)

    return {"root": str(root), "protocol": "direct", "arms": arms,
            "missing_runs": missing, "cells": table}


def to_markdown_direct(summary: dict) -> str:
    arms = summary["arms"]
    columns = ["dataset", "task", "n", "chance",
               *(f"{a} (floor)" for a in arms), *arms, *(f"lift {a}" for a in arms),
               *(f"unparsed {a}" for a in arms)]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for cell in summary["cells"]:
        def fmt(name, key="accuracy", spec=".4f", sign=""):
            entry = cell.get(name)
            return format(entry[key], sign + spec) if entry and key in entry else "-"

        def lift(a):
            v = cell.get(f"lift_{a}")
            return "-" if v is None else f"{v:+.4f}"

        row = [cell["dataset"], cell["task"], str(cell["n"]), f"{cell['chance']:.2f}",
               *(fmt(f"{a}_noimg") for a in arms), *(fmt(a) for a in arms),
               *(lift(a) for a in arms),
               *(fmt(a, "unparsed_rate") for a in arms)]
        lines.append("| " + " | ".join(row) + " |")
    note = ("\nfloor = same model, same question, chart withheld (`NO_IMAGE=1`).\n"
            "**Read `unparsed` before accuracy**: the RL and SFT arms are captioners being "
            "asked for a bare letter, so a low score can mean 'answered wrongly' or "
            "'stopped answering in the required format', and only this column separates "
            "them.\n")
    return "\n".join(lines) + "\n" + note


def to_markdown(summary: dict) -> str:
    arms = summary["arms"]
    columns = ["dataset", "task", "n", "chance", "L0", "L2", "spread",
               *arms, *(f"IR {a}" for a in arms)]
    head = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines = []
    for cell in summary["cells"]:
        def acc(name):
            return f"{cell[name]['accuracy']:.4f}" if name in cell else "-"

        def ir(name):
            value = cell.get(f"IR_{name}")
            return "-" if value is None else f"{value:+.3f}"

        flag = "" if cell.get("admitted", True) else " ⚠️"
        spread = f"{cell['oracle_spread']:+.4f}{flag}" if "oracle_spread" in cell else "-"
        cells = [cell["dataset"], cell["task"], str(cell["n"]), f"{cell['chance']:.2f}",
                 acc("L0"), acc("L2"), spread,
                 *(acc(a) for a in arms), *(ir(a) for a in arms)]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\n⚠️ = `L2 - L0 < %.2f`: the oracle barely beats the option-text floor, so "
            "this cell cannot measure a caption.\n" % summary["gate"])
    if summary.get("incomparable_arms"):
        note += (f"\n**IR withheld for {', '.join(summary['incomparable_arms'])}**: these runs "
                 f"and the L0/L2 legs did not send the answerer the same question wording "
                 f"(differing prompt fingerprints), so their difference is not "
                 f"attributable to the caption. Re-run the legs to match.\n")
    return "\n".join(head + lines) + "\n" + note


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/eval_protocol/bedtime",
                    help="protocol A (caption-mediated) results root")
    ap.add_argument("--direct-root", default="results/eval_protocol/bedtime_direct/bedtime",
                    help="protocol B (direct VQA) results root; skipped when absent")
    ap.add_argument("--arms", default="base,decod550,sft")
    ap.add_argument("--items", default="bench_data/bedtime/items_sbert.jsonl")
    ap.add_argument("--out", default=None, help="defaults to <root>/summary.json")
    args = ap.parse_args()

    root = Path(args.root)
    arms = [a for a in args.arms.split(",") if a]
    items = Path(args.items)

    summary = build(root, arms, items)
    if summary["missing_runs"]:
        print(f"warning: no predictions.jsonl for {summary['missing_runs']}", flush=True)
    sections = ["## Protocol A — caption-mediated (VLM captions, frozen 14B answers)", "",
                to_markdown(summary)]

    direct_root = Path(args.direct_root)
    if direct_root.exists():
        direct = build_direct(direct_root, arms, items)
        summary["direct"] = direct
        if direct["missing_runs"]:
            print(f"warning: no direct predictions for {direct['missing_runs']}", flush=True)
        sections += ["## Protocol B — direct VQA (the VLM sees the chart and answers)", "",
                     to_markdown_direct(direct)]
    else:
        print(f"note: {direct_root} does not exist, protocol B omitted", flush=True)

    out_path = Path(args.out) if args.out else root / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    markdown = "\n".join(sections)
    out_path.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
