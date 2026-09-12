from __future__ import annotations

import argparse
import json
from pathlib import Path

from ts_eval.benchmarks.base import QAItem
from ts_eval.caption_answer_harness import ORACLE_EVIDENCE_LABEL
from ts_eval.cli_common import add_benchmark_args, load_items

MODES = ("none", "oracle_series")
MODE_EVIDENCE_LABEL = {"none": None, "oracle_series": ORACLE_EVIDENCE_LABEL}


def reference_caption(item: QAItem, mode: str, precision: int = 4) -> str:
    if mode == "none":
        return ""
    if mode == "oracle_series":
        series = item.series
        if item.n_variates > 1:
            return "\n".join(
                f"Series {i + 1}: " + ", ".join(f"{float(v):.{precision}g}" for v in variate)
                for i, variate in enumerate(series))
        return ", ".join(f"{float(v):.{precision}g}" for v in series)
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_benchmark_args(ap)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--captions-out", required=True)
    ap.add_argument("--precision", type=int, default=4,
                    help="significant digits per value in oracle_series mode")
    args = ap.parse_args()

    items = load_items(args.benchmark, args.limit, args.i_accept_unclear_license, args.hf_token)
    out_path = Path(args.captions_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({"id": item.id,
                                "caption_key": item.extra.get("caption_key") or item.id,
                                "evidence_label": MODE_EVIDENCE_LABEL[args.mode],
                                "caption": reference_caption(item, args.mode, args.precision)},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} {args.mode} captions to {out_path}", flush=True)


if __name__ == "__main__":
    main()
