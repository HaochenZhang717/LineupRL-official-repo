from __future__ import annotations

import os

from ts_eval.benchmarks import REGISTRY
from ts_eval.caption_answer_harness import _ANSWER_INSTRUCTIONS


def load_items(benchmark: str, limit: int | None, i_accept_unclear_license: bool, hf_token: str | None):
    module = REGISTRY[benchmark]
    kwargs = {"limit": limit}
    if benchmark == "tsrbench":
        kwargs["i_accept_unclear_license"] = i_accept_unclear_license
    if benchmark == "time_mqa":
        kwargs["hf_token"] = hf_token or os.environ.get("HF_TOKEN")
    items = list(module.load(**kwargs))

    supported = set(_ANSWER_INSTRUCTIONS)
    skipped = [i for i in items if i.scoring_type not in supported]
    items = [i for i in items if i.scoring_type in supported]
    if skipped:
        skipped_types = sorted({i.scoring_type for i in skipped})
        print(f"skipping {len(skipped)}/{len(skipped) + len(items)} items with unsupported "
              f"scoring_type(s) not in {sorted(supported)}: {skipped_types}", flush=True)
    return items


def add_benchmark_args(ap) -> None:
    ap.add_argument("--benchmark", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--i-accept-unclear-license", action="store_true",
                    help="required for --benchmark tsrbench (no confirmed license, see ts_eval/benchmarks/tsrbench.py)")
    ap.add_argument("--hf-token", default=None, help="for --benchmark time_mqa (gated dataset)")
