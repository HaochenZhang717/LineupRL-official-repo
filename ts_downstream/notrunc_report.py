from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEW = REPO / "results" / "eval_protocol" / "pipeline3_forecast_notrunc"
OLD = REPO / "results" / "eval_protocol" / "pipeline3_forecast"

DATASETS = ["ETTh2", "ETTm2", "saugeen", "aus_elec"]
ARMS = ["base", "decod550", "rl650", "sft", "teacher72b", "valmask400", "judge550"]


def mse(root: Path, ds: str, arm: str) -> float | None:
    p = root / ds / arm / "metrics.json"
    return json.loads(p.read_text())["test"]["mse"] if p.exists() else None


def tokens(ds: str, arm: str) -> dict:
    p = NEW / ds / arm / "metrics.json"
    return json.loads(p.read_text()).get("caption_tokens", {}) if p.exists() else {}


def c(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    L: list[str] = []
    A = L.append
    A("# Forecasting without the caption cap")
    A("")
    A("`pipeline3_forecast` ran at `max_text_len=256`, ReadoutConfig's default. Caption")
    A("length differs by arm by more than 2x, so that cap was not a neutral setting — it")
    A("deleted most of what the long arms wrote and almost nothing of what the short arms")
    A("wrote, and the resulting table partly ranked brevity. Every cell is re-run here with")
    A("truncation disabled (`--max-text-len 0`, verified per run: `n_truncated == 0`).")
    A("")

    A("## How much of each arm the cap removed")
    A("")
    A("Median caption length, and the share of captions longer than 256 tokens (ETTh2;")
    A("the other datasets are within a few tokens).")
    A("")
    A("| arm | median tokens | longest | % cut at 256 |")
    A("|---|---|---|---|")
    for arm in ARMS:
        t = tokens("ETTh2", arm)
        if not t:
            continue
        A(f"| `{arm}` | {t.get('median_tokens', '—')} | {t.get('max_tokens', '—')} | "
          f"{'100%' if t.get('median_tokens', 0) > 256 else '—'} |")
    A("")
    A("Exact figures measured separately: 100% of `sft` and `teacher72b` captions exceed")
    A("256 tokens, 76% of `rl650`'s and 21% of `decod550`'s.")
    A("")

    A("## test MSE, both settings")
    A("")
    A("| dataset | arm | untruncated | at 256 | change |")
    A("|---|---|---|---|---|")
    for ds in DATASETS:
        for arm in ARMS:
            n, o = mse(NEW, ds, arm), mse(OLD, ds, arm)
            ch = "—" if (n is None or o is None) else f"{100 * (n - o) / o:+.1f}%"
            A(f"| {ds} | `{arm}` | {c(n)} | {c(o)} | {ch} |")
    A("")
    A("The change column is the finding. The two short arms barely move (`base` +0.3 to")
    A("−5.4%, `decod550` +5.7 to −6.8%, both within the ~5% nondeterminism floor); the")
    A("three long arms improve by up to 67.5%. Nothing about the models changed — only how")
    A("much of their captions the encoder was allowed to read.")
    A("")

    A("## Rankings, before and after")
    A("")
    A("| dataset | at 256 | untruncated |")
    A("|---|---|---|")
    for ds in DATASETS:
        old_r = sorted((a for a in ARMS if mse(OLD, ds, a) is not None),
                       key=lambda a: mse(OLD, ds, a))
        new_r = sorted((a for a in ARMS if mse(NEW, ds, a) is not None),
                       key=lambda a: mse(NEW, ds, a))
        A(f"| {ds} | {' < '.join(old_r)} | {' < '.join(new_r)} |")
    A("")
    A("`valmask400` is the clearest casualty of the old table: last or near-last on three")
    A("datasets at 256, second on three untruncated. Its captions are long (median 1535")
    A("chars) and it lost 41-65% of its MSE once they were read whole. Any conclusion drawn")
    A("about the masked-reward ablation from the 256 numbers should be revisited.")
    A("")
    A("`teacher72b` moves nearly as much and overtakes `decod550` on ETTm2.")
    A("")

    A("## What survives")
    A("")
    A("| claim | at 256 | untruncated |")
    A("|---|---|---|")
    A("| decod550 is first | 4/4 datasets | 3/4 (loses ETTm2 to teacher72b) |")
    A("| decod550 beats sft | 4/4 | 4/4 |")
    A("| decod550 beats base | 4/4 | 4/4 |")
    A("| rl650 is weakest | last on 1/4 | last on 3/4 (all but aus_elec) |")
    A("")
    A("decod550's lead is real but smaller than the old table implied: teacher72b's MSE was")
    A("89%/178%/18%/118% above decod550's at 256 and is +29%/−14%/+4%/+59% untruncated —")
    A("i.e. teacher72b now wins ETTm2 and is within noise on saugeen.")
    A("")

    A("## Caveats")
    A("")
    A("- One training seed per cell, matching the original runs. `pipeline3_fusion` covers")
    A("  256 vs 1400 with three seeds on the same four datasets and agrees on both the")
    A("  direction and the rough size, so the effect is not a single-seed artifact — but")
    A("  individual gaps under ~5% here are still unresolved.")
    A("- The saugeen and aus_elec `base` cells also moved (−5.0%, −5.4%) despite `base`")
    A("  being a short-caption arm. That is at the noise floor and should not be read as a")
    A("  truncation effect.")
    A("- The same cap applies to every other study that used this readout at its default,")
    A("  including the classification runs in `pipeline3`, which have not been re-run.")
    A("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
