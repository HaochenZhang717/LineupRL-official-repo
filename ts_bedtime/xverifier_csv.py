from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XV = REPO / "results" / "eval_protocol" / "xverifier"
BENCHES = ("bedtime", "cats")
READER_ORDER = ["qwen25-14b", "llama31-8b", "gemma3-12b", "mistral-nemo-12b", "phi4-14b", "glm4-9b", "qwen3-14b", "qwen35-9b"]
ARM_ORDER = ["base", "sft", "rl650", "judge550", "decod550", "teacher72b",
             "qwen3vl-8b", "internvl3-14b", "phi35-mini", "qwen25-7b", "qwen25-14b", "chatts-14b",
             "neg6", "negrand", "rm3b", "rm7b", "neg8"]


def parse_name(stem: str) -> tuple[str, str, str]:
    order = "rotation"
    if stem.endswith("_random"):
        order, stem = "random", stem[: -len("_random")]
    control = "none"
    if stem == "empty":
        return "", "empty", order
    if stem.endswith("_mismatch"):
        control, stem = "mismatch", stem[: -len("_mismatch")]
    if stem.endswith("_valmask") or stem.endswith("_nummask"):
        control, stem = stem.rsplit("_", 1)[1], stem.rsplit("_", 1)[0]
    return stem, control, order


def main() -> None:
    rows = []
    for bench in BENCHES:
        root = XV / bench
        if not root.exists():
            continue
        for rd in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith("_smoke")):
            for f in sorted(rd.glob("*.json")):
                d = json.loads(f.read_text())
                arm, control, order = parse_name(f.stem)
                row = {"bench": bench, "reader": rd.name, "arm": arm, "control": control, "order": order,
                       "rotations": d.get("rotations"), "n_series": d.get("A", d.get("B", {})).get("n_series"),
                       "acc_c2s": d.get("A", {}).get("accuracy"), "acc_s2c": d.get("B", {}).get("accuracy"),
                       "unparsed_rate": d.get("unparsed_rate"), "reader_model": d.get("reader"),
                       "max_tokens": d.get("max_tokens"), "donor_in_candidates": d.get("mismatch_donor_in_candidates")}
                for m, key in (("A", "c2s"), ("B", "s2c")):
                    for pos, v in d.get(m, {}).get("by_gold_position", {}).items():
                        row[f"{key}_gold_{pos}"] = v
                rows.append(row)
    if not rows:
        raise SystemExit("no xverifier json found")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("bench", "reader", "arm", "control", "order"), k))
    lead = ["bench", "reader", "arm", "control", "order", "rotations", "n_series", "acc_c2s", "acc_s2c",
            "unparsed_rate", "reader_model", "max_tokens", "donor_in_candidates"]
    fields = lead + [k for k in fields if k not in lead]
    rows.sort(key=lambda r: (r["bench"], READER_ORDER.index(r["reader"]) if r["reader"] in READER_ORDER else 99,
                             ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 99, r["control"], r["order"]))
    with (XV / "xverifier_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    main = [r for r in rows if r["control"] == "none" and r["order"] == "rotation"]
    readers = [r for r in READER_ORDER if any(x["reader"] == r for x in main)]
    arms = [a for a in ARM_ORDER if any(x["arm"] == a for x in main)]
    cols = ["arm"] + [f"{b}/{r}/{m}" for b in BENCHES for r in readers for m in ("c2s", "s2c")]
    with (XV / "xverifier_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for a in arms:
            line = [a]
            for b in BENCHES:
                for r in readers:
                    hit = next((x for x in main if x["bench"] == b and x["reader"] == r and x["arm"] == a), None)
                    line += [f"{hit['acc_c2s']:.4f}" if hit and hit["acc_c2s"] is not None else "",
                             f"{hit['acc_s2c']:.4f}" if hit and hit["acc_s2c"] is not None else ""]
            w.writerow(line)

    with (XV / "xverifier_controls.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bench", "reader", "arm", "empty_c2s", "empty_s2c", "rotation_c2s", "rotation_s2c",
                    "random_c2s", "random_s2c", "mismatch_c2s", "mismatch_s2c", "mismatch_random_c2s",
                    "valmask_c2s", "nummask_c2s"])
        for b in BENCHES:
            for r in readers:
                sub = [x for x in rows if x["bench"] == b and x["reader"] == r]
                emp = next((x for x in sub if x["control"] == "empty" and x["order"] == "rotation"), None)
                for a in arms:
                    def pick(control, order):
                        return next((x for x in sub if x["arm"] == a and x["control"] == control and x["order"] == order), None)
                    rot, rnd, mm, mmr = pick("none", "rotation"), pick("none", "random"), pick("mismatch", "rotation"), pick("mismatch", "random")
                    vm, nm = pick("valmask", "rotation"), pick("nummask", "rotation")
                    if rot is None:
                        continue
                    g = lambda x, k: (f"{x[k]:.4f}" if x and x.get(k) is not None else "")
                    w.writerow([b, r, a, g(emp, "acc_c2s"), g(emp, "acc_s2c"), g(rot, "acc_c2s"), g(rot, "acc_s2c"),
                                g(rnd, "acc_c2s"), g(rnd, "acc_s2c"), g(mm, "acc_c2s"), g(mm, "acc_s2c"),
                                g(mmr, "acc_c2s"), g(vm, "acc_c2s"), g(nm, "acc_c2s")])
    print(f"{len(rows)} cells -> {XV / 'xverifier_results.csv'}; table {len(arms)} arms x {len(readers)} readers")


if __name__ == "__main__":
    main()
