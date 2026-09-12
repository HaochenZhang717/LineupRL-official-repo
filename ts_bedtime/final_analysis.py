from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("results/eval_protocol")
ARMS = ("base", "decod550", "sft")
GEN_ARMS = ("base", "sft", "rl650", "decod550", "valmask400")
TASKS = ("differentiation", "recognition")

N_TESTS = 16
ALPHA = 0.05


LEN_EDGES = (600, 900, 1300, 1800)
LEN_LABELS = ("<600", "600-900", "900-1300", "1300-1800", ">1800")


def task3_length_control() -> None:
    print("### Task 3 is confounded with caption length -- the control\n")
    print("`gen→gt` at matched caption length, deployed variant. Blank = fewer than 100 "
          "items in that bin for that arm, so the arms genuinely do not overlap there.\n")
    print("| arm | " + " | ".join(f"{l} chars" for l in LEN_LABELS) + " |")
    print("|---" * (len(LEN_LABELS) + 1) + "|")
    for arm in GEN_ARMS:
        cap_path = ROOT / f"bedtime/{arm}/captions.jsonl"
        gen_path = ROOT / f"bedtime/{arm}/generation_deployed.jsonl"
        if not (cap_path.exists() and gen_path.exists()):
            continue
        lens: dict[str, int] = {}
        with open(cap_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                lens.setdefault(r["caption_key"], len(r["caption"]))
        buckets: list[list[float]] = [[] for _ in LEN_LABELS]
        with open(gen_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n = lens.get(r["series_uid"])
                if n is None:
                    continue
                i = sum(n >= e for e in LEN_EDGES)
                buckets[i].append(float(r["p_gen_entails_gt"] >= 0.5))
        cells = [f"{sum(b) / len(b):.3f} (n={len(b)})" if len(b) >= 100 else "—"
                 for b in buckets]
        print(f"| `{arm}` | " + " | ".join(cells) + " |")
    print()
    print("The entailment rate falls with caption length inside every arm (r = -0.07 to "
          "-0.12), which is why this control exists. But the bins are NOT evidence that "
          "length causes the gap: each arm covers a different subset of series at a given "
          "length, so the cells are not comparable across rows. `ts_bedtime.nli_length_probe`"
          " settles it by intervening on length instead of conditioning on it -- see the "
          "next section.\n")


def task3_paired() -> None:
    print("### Task 3, decod550 vs valmask400 -- paired, and length ruled out\n")
    pairs: dict[str, dict] = {}
    for arm in ("decod550", "valmask400", "base"):
        p = ROOT / f"bedtime/{arm}/generation_deployed.jsonl"
        if not p.exists():
            return
        d = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                d[(r["series_uid"], r["ground_truth"])] = r["p_gen_entails_gt"] >= 0.5
        pairs[arm] = d

    print("| comparison | wins A | wins B | McNemar p | series A ahead | series B ahead |")
    print("|---|---|---|---|---|---|")
    for a_name, b_name in (("decod550", "valmask400"), ("decod550", "base"),
                           ("valmask400", "base")):
        a, b = pairs[a_name], pairs[b_name]
        keys = [k for k in a if k in b]
        n01 = sum(1 for k in keys if a[k] and not b[k])
        n10 = sum(1 for k in keys if b[k] and not a[k])
        chi = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
        p_val = math.erfc(math.sqrt(chi / 2))
        by_series: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for uid, gt in keys:
            by_series[uid].append((a[(uid, gt)], b[(uid, gt)]))
        ahead_a = sum(1 for v in by_series.values()
                      if sum(x for x, _ in v) > sum(y for _, y in v))
        ahead_b = sum(1 for v in by_series.values()
                      if sum(y for _, y in v) > sum(x for x, _ in v))
        print(f"| {a_name} vs {b_name} | {n01} | {n10} | {p_val:.1e} | {ahead_a} | {ahead_b} |")
    print()
    print("**Length is not the explanation, and this was checked by intervention rather "
          "than by conditioning** (`python -m ts_bedtime.nli_length_probe`, n=250). "
          "Truncating valmask400's captions to decod550's length does not raise its score, "
          "it lowers it by 0.036 -- the deleted text was carrying content it had been "
          "credited for. Padding decod550's captions out to valmask400's length costs it "
          "0.024. The gap between the two survives both, so decod550 really does cover "
          "these crowd descriptions more often.\n")
    print("**But do not over-read the magnitude.** The same probe shows the scorer is not "
          "a faithful entailment engine: repeating a caption verbatim adds no information "
          "and therefore cannot change an entailment, yet it collapses the rate from 0.227 "
          "to 0.043. Appending irrelevant boilerplate, by contrast, changes nothing. The "
          "ordering is trustworthy; the numbers are a similarity score wearing an "
          "entailment label.\n")


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["item_id"]] = (r["score"], r["task_type"], r["domain"])
    return out


def accuracy(run: dict, task: str) -> float | None:
    vals = [s for s, t, _ in run.values() if t == task]
    return sum(vals) / len(vals) if vals else None


def mcnemar(a: dict, b: dict, task: str) -> tuple[int, int, float]:
    n10 = n01 = 0
    for key, (sa, ta, _) in a.items():
        if ta != task or key not in b:
            continue
        sb = b[key][0]
        if sa == 1 and sb == 0:
            n10 += 1
        elif sa == 0 and sb == 1:
            n01 += 1
    if n01 + n10 == 0:
        return n10, n01, 1.0
    z = (n01 - n10) / math.sqrt(n01 + n10)
    return n10, n01, math.erfc(abs(z) / math.sqrt(2))


def stars(p: float) -> str:
    if p < ALPHA / N_TESTS / 100:
        return "***"
    if p < ALPHA / N_TESTS / 10:
        return "**"
    if p < ALPHA / N_TESTS:
        return "*"
    return "n.s."


PROTOCOLS = {
    "A (caption-mediated)": {
        "sbert": ROOT / "bedtime",
        "euclid": ROOT / "bedtime_euclid/bedtime",
    },
    "B (direct VQA)": {
        "sbert": ROOT / "bedtime_direct/bedtime",
        "euclid": ROOT / "bedtime_direct_euclid/bedtime",
    },
}


def main() -> None:
    print(f"# BEDTime — all results\n")
    print(f"Significance vs `base`, McNemar paired test, Bonferroni-corrected for "
          f"{N_TESTS} tests (`*` = p < {ALPHA / N_TESTS:.4f}).\n")

    for protocol, roots in PROTOCOLS.items():
        print(f"## Protocol {protocol}\n")
        print("| distractor | task | base | decod550 | sft | decod550 vs base | sft vs base |")
        print("|---|---|---|---|---|---|---|")
        for strategy, root in roots.items():
            runs = {a: load(root / a / "predictions.jsonl") for a in ARMS}
            if not all(runs.values()):
                missing = [a for a in ARMS if not runs[a]]
                print(f"| {strategy} | *(missing: {', '.join(missing)})* | | | | | |")
                continue
            for task in TASKS:
                accs = {a: accuracy(runs[a], task) for a in ARMS}
                cells = [strategy, task] + [f"{accs[a]:.4f}" for a in ARMS]
                for arm in ("decod550", "sft"):
                    n10, n01, p = mcnemar(runs["base"], runs[arm], task)
                    delta = accs[arm] - accs["base"]
                    cells.append(f"{delta:+.4f} p={p:.1e} {stars(p)}")
                print("| " + " | ".join(cells) + " |")
        print()

    print("## Task 3 — open generation (NLI entailment)\n")
    print("| variant | arm | gen→gt | gt→gen | bidirectional | mean caption chars |")
    print("|---|---|---|---|---|---|")
    for variant in ("deployed", "gen150"):
        for arm in GEN_ARMS:
            p = ROOT / f"bedtime/{arm}/generation_{variant}.json"
            if not p.exists():
                continue
            o = json.loads(p.read_text())["overall"]
            print(f"| {variant} | {arm} | {o['gen_entails_gt']:.4f} | {o['gt_entails_gen']:.4f} "
                  f"| {o['bidirectional']:.4f} | {o['mean_caption_chars']:.0f} |")
    ctrl_path = ROOT / "bedtime/base/generation_deployed.json"
    if ctrl_path.exists():
        ctrl = json.loads(ctrl_path.read_text()).get("controls", {})
        for name, v in ctrl.items():
            print(f"| *control* | {name} | {v['gen_entails_gt']:.4f} | {v['gt_entails_gen']:.4f} "
                  f"| {v['bidirectional']:.4f} | — |")
    print()

    task3_length_control()
    task3_paired()

    print("## Protocol A with a different reader (home-field check)\n")
    print("The 14B answerer is decod550's own training-time discriminator. Same captions, "
          "same items, re-answered by Qwen2.5-3B-Instruct.\n")
    print("| answerer | task | base | decod550 | sft | decod550 vs base | sft vs base |")
    print("|---|---|---|---|---|---|---|")
    for label, root in [("14B", ROOT / "bedtime"), ("3B", ROOT / "bedtime_3b/bedtime")]:
        runs = {a: load(root / a / "predictions.jsonl") for a in ARMS}
        if not all(runs.values()):
            print(f"| {label} | *(incomplete)* | | | | | |")
            continue
        for task in TASKS:
            accs = {a: accuracy(runs[a], task) for a in ARMS}
            cells = [label, task] + [f"{accs[a]:.4f}" for a in ARMS]
            for arm in ("decod550", "sft"):
                _, _, p = mcnemar(runs["base"], runs[arm], task)
                cells.append(f"{accs[arm] - accs['base']:+.4f} p={p:.1e} {stars(p)}")
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
