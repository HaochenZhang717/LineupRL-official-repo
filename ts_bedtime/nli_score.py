from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_MODEL = "tasksource/deberta-base-long-nli"
DEFAULT_SERIES = "bench_data/bedtime/series.jsonl"
ENTAILMENT_LABEL = "entailment"


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_captions_by_series(path: Path, series_uids: set[str]) -> dict[str, str]:
    by_series: dict[str, str] = {}
    for row in _read_jsonl(path):
        uid = row.get("caption_key")
        if uid is None:
            raise ValueError(
                f"{path} has no caption_key column -- it predates that change; re-run the "
                f"caption step, or fold it by item id yourself")
        if uid not in series_uids:
            continue
        previous = by_series.setdefault(uid, row["caption"])
        if previous != row["caption"]:
            raise ValueError(
                f"{uid} has two different captions in {path}; caption_key dedup is broken")
    return by_series


def entailment_scores(pairs: list[tuple[str, str]], model_name: str, device: str | None,
                      batch_size: int, max_length: int) -> list[float]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    label2id = {v.lower(): k for k, v in model.config.id2label.items()}
    if ENTAILMENT_LABEL not in label2id:
        raise ValueError(f"{model_name} has labels {model.config.id2label}, no entailment head")
    idx = label2id[ENTAILMENT_LABEL]

    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                      max_length=max_length, padding=True, return_tensors="pt").to(device)
            probs = model(**enc).logits.softmax(-1)[:, idx]
            out.extend(probs.tolist())
    return out


def summarise(records: list[dict], threshold: float) -> dict:
    def rates(rows: list[dict]) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0}
        fwd = [r["p_gen_entails_gt"] >= threshold for r in rows]
        bwd = [r["p_gt_entails_gen"] >= threshold for r in rows]
        return {
            "n": n,
            "gen_entails_gt": sum(fwd) / n,
            "gt_entails_gen": sum(bwd) / n,
            "bidirectional": sum(f and b for f, b in zip(fwd, bwd)) / n,
            "mean_p_gen_entails_gt": sum(r["p_gen_entails_gt"] for r in rows) / n,
            "mean_p_gt_entails_gen": sum(r["p_gt_entails_gen"] for r in rows) / n,
            "mean_caption_chars": sum(len(r["caption"]) for r in rows) / n,
            "mean_gt_chars": sum(len(r["ground_truth"]) for r in rows) / n,
        }

    by_dataset = defaultdict(list)
    for record in records:
        by_dataset[record["dataset"]].append(record)
    return {
        "threshold": threshold,
        "overall": rates(records),
        "by_dataset": {name: rates(rows) for name, rows in sorted(by_dataset.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True, help="captions.jsonl from a caption pass")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--variant", default="deployed", choices=["deployed", "gen150"],
                    help="which caption pass this is: our TS_CAP_PROMPT captions, or "
                         "BEDTime's own task-3 prompt capped at 150 tokens")
    ap.add_argument("--series", default=DEFAULT_SERIES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1280,
                    help="the checkpoint's max_position_embeddings. Our captions run to "
                         "~760 chars (decod550) and ~2300 (teacher), so a shorter window "
                         "would truncate the premise and silently depress entailment")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="P(entailment) above which a pair counts as entailed")
    ap.add_argument("--controls", action="store_true",
                    help="also score the empty-description floor and the ground-truth-"
                         "against-itself ceiling, on the same pairs")
    args = ap.parse_args()

    series_rows = _read_jsonl(Path(args.series))
    series_by_uid = {r["series_uid"]: r for r in series_rows}
    captions = load_captions_by_series(Path(args.captions), set(series_by_uid))
    print(f"{len(captions)} captions covering {len(series_by_uid)} series", flush=True)

    records: list[dict] = []
    for uid, caption in sorted(captions.items()):
        row = series_by_uid[uid]
        for annotation in row["annotations"]:
            records.append({"series_uid": uid, "dataset": row["dataset"], "cls": row.get("cls"),
                            "caption": caption, "ground_truth": annotation})

    forward = [(r["caption"], r["ground_truth"]) for r in records]
    backward = [(r["ground_truth"], r["caption"]) for r in records]
    print(f"scoring {len(forward)} pairs in each direction with {args.model}", flush=True)
    p_fwd = entailment_scores(forward, args.model, args.device, args.batch_size, args.max_length)
    p_bwd = entailment_scores(backward, args.model, args.device, args.batch_size, args.max_length)
    for record, pf, pb in zip(records, p_fwd, p_bwd):
        record["p_gen_entails_gt"] = pf
        record["p_gt_entails_gen"] = pb

    result = {"run_name": args.run_name, "variant": args.variant, "model": args.model,
              "captions": str(args.captions), **summarise(records, args.threshold)}

    if args.controls:
        controls = {}
        for name, caption_of in (("empty", lambda r: ""),
                                 ("oracle", lambda r: r["ground_truth"])):
            ctrl = [{**r, "caption": caption_of(r)} for r in records]
            cf = entailment_scores([(c["caption"], c["ground_truth"]) for c in ctrl],
                                   args.model, args.device, args.batch_size, args.max_length)
            cb = entailment_scores([(c["ground_truth"], c["caption"]) for c in ctrl],
                                   args.model, args.device, args.batch_size, args.max_length)
            for c, pf, pb in zip(ctrl, cf, cb):
                c["p_gen_entails_gt"], c["p_gt_entails_gen"] = pf, pb
            controls[name] = summarise(ctrl, args.threshold)["overall"]
        result["controls"] = controls

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    with open(out_path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({k: v for k, v in record.items() if k != "caption"},
                               ensure_ascii=False) + "\n")
    print(json.dumps(result["overall"], indent=1), flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
