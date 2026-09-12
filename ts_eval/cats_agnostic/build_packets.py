import argparse
import json
import os
import pathlib

CT = os.environ.get("CATS_TEST_DATA", "bench_data/CaTS_Datasets/test_data")
REPO = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--source", choices=("synth", "human"), default="synth")
    ap.add_argument("--with-metadata", action="store_true")
    args = ap.parse_args()

    ids = [json.loads(l)["id"] for l in open(REPO / "results/eval_protocol/cats_mcq/captions_base.jsonl")]
    assert len(ids) == len(set(ids)) == 2969
    folder = "synth_gt_captions" if args.source == "synth" else "human_rewritten_captions"
    series_len = {json.loads(l)["series_uid"]: len(json.loads(l)["series"])
                  for l in open(REPO / "bench_data/cats_bench/series.jsonl")}
    n = 0
    with open(args.out, "w") as f:
        for i in ids:
            p = f"{CT}/{folder}/{i}.txt"
            if not os.path.exists(p):
                if args.source == "human":
                    continue
                raise SystemExit(f"missing {p}")
            row = {"id": i, "gold": open(p).read().strip()}
            if args.with_metadata:
                m = json.load(open(f"{CT}/metadata/{i}.json"))
                row["meta"] = {"start_year": m.get("start year of this series"),
                               "end_year": m.get("end year of this series"),
                               "frequency": m.get("sampling frequency"),
                               "n_points": series_len[i]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} packets ({args.source}) -> {args.out}")


if __name__ == "__main__":
    main()
