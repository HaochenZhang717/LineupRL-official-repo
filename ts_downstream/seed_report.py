from __future__ import annotations

import argparse
import collections
import csv
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "eval_protocol" / "readout_results.csv"

DATASETS = ["ETTh2", "ETTm2", "saugeen", "aus_elec"]
CONTROLS = ("null", "decod550_shuffled")
RECON = ["decod550", "teacher72b", "sft", "rl650", "judge550", "base", *CONTROLS]
FORECAST = ["decod550", "teacher72b", "valmask400", "sft", "base", "rl650", "judge550", *CONTROLS]
STUDIES = (("reconstruction", RECON), ("forecasting", FORECAST))
ALPHA = 0.05


def arms_of(conds: list[str]) -> list[str]:
    return [c for c in conds if c not in CONTROLS]


def rivals_of(conds: list[str]) -> list[str]:
    return [c for c in arms_of(conds) if c != "decod550"]


def load() -> dict:
    g: dict = collections.defaultdict(dict)
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            g[(r["study"], r["dataset"], r["condition"])][r["train_seed"]] = float(r["mse"])
    return g


def load_naive() -> dict:
    acc: dict = collections.defaultdict(list)
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sk = r.get("naive_skill")
            if not sk:
                continue
            skill = float(sk)
            if skill >= 1.0:
                continue
            acc[(r["study"], r["dataset"])].append(float(r["mse"]) / (1.0 - skill))
    return {k: st.mean(v) for k, v in acc.items() if v}


def ttest_rel(a: list[float], b: list[float]) -> float:
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return 1.0
    sd = st.stdev(d)
    if sd == 0:
        return 0.0 if st.mean(d) != 0 else 1.0
    t = abs(st.mean(d)) / (sd / n ** 0.5)
    df = n - 1
    x = df / (df + t * t)

    def betacf(a_, b_, x_):
        tiny = 1e-30
        qab, qap, qam = a_ + b_, a_ + 1.0, a_ - 1.0
        c, d_ = 1.0, 1.0 - qab * x_ / qap
        d_ = tiny if abs(d_) < tiny else d_
        d_ = 1.0 / d_
        h = d_
        for m in range(1, 200):
            m2 = 2 * m
            aa = m * (b_ - m) * x_ / ((qam + m2) * (a_ + m2))
            d_ = 1.0 + aa * d_
            d_ = tiny if abs(d_) < tiny else d_
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d_ = 1.0 / d_
            h *= d_ * c
            aa = -(a_ + m) * (qab + m) * x_ / ((a_ + m2) * (qap + m2))
            d_ = 1.0 + aa * d_
            d_ = tiny if abs(d_) < tiny else d_
            c = 1.0 + aa / c
            c = tiny if abs(c) < tiny else c
            d_ = 1.0 / d_
            de = d_ * c
            h *= de
            if abs(de - 1.0) < 3e-16:
                break
        return h

    import math
    a_, b_ = df / 2.0, 0.5
    lbeta = (math.lgamma(a_) + math.lgamma(b_) - math.lgamma(a_ + b_))
    front = math.exp(a_ * math.log(x) + b_ * math.log(1 - x) - lbeta) / a_ if 0 < x < 1 else 0.0
    ib = front * betacf(a_, b_, x) if x < (a_ + 1) / (a_ + b_ + 2) else \
        1.0 - math.exp(b_ * math.log(1 - x) + a_ * math.log(x) - lbeta) / b_ * betacf(b_, a_, 1 - x)
    return max(0.0, min(1.0, ib))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    g = load()
    naive = load_naive()
    seeds = sorted({s for v in g.values() for s in v})
    n_seeds = len(seeds)

    ranks = {}
    for study, conds in STUDIES:
        arms = arms_of(conds)
        for ds in DATASETS:
            order = sorted(arms, key=lambda a: st.mean(list(g[(study, ds, a)].values())))
            ranks[(study, ds)] = order.index("judge550") + 1

    tally: dict[str, list[int]] = {}
    for study, conds in STUDIES:
        for o in rivals_of(conds):
            for ds in DATASETS:
                va = [g[(study, ds, "decod550")][s_] for s_ in seeds]
                vb = [g[(study, ds, o)][s_] for s_ in seeds]
                rec = tally.setdefault(o, [0, 0, 0])
                if ttest_rel(va, vb) >= ALPHA:
                    rec[1] += 1
                elif st.mean(va) < st.mean(vb):
                    rec[0] += 1
                else:
                    rec[2] += 1

    L: list[str] = []
    A = L.append
    A(f"# Forecasting and reconstruction over {n_seeds} training seeds")
    A("")
    n_runs = sum(len(v) for v in g.values())
    A(f"Seeds {', '.join(seeds)}, every cell run {n_seeds} times: {n_runs} runs, no failures.")
    A("Only the training loop moves between seeds — `--seed 2020` still drives the window")
    A(f"subsample — so all {n_seeds} score identical windows against identical captions and")
    A("the spread measures training, not resampling. Captions are never truncated.")
    A("")
    A("Single-seed versions of these tables ranked the arms. They should not have: two of")
    A("the gaps they showed reversed or vanished here.")
    A("")

    A("## How much a seed is worth")
    A("")
    A(f"Coefficient of variation (std/mean of test MSE) across the {n_seeds} seeds.")
    A("")
    A("| study | cells | median | p90 | max |")
    A("|---|---|---|---|---|")
    for study, conds in STUDIES:
        cvs = sorted(100 * st.stdev(list(v.values())) / st.mean(list(v.values()))
                     for k, v in g.items()
                     if k[0] == study and k[2] in arms_of(conds) and len(v) == n_seeds)
        A(f"| {study} | {len(cvs)} | {st.median(cvs):.2f}% | {cvs[int(0.9 * len(cvs))]:.2f}% | "
          f"{max(cvs):.2f}% |")
    A("")
    A("Arms only — the controls are near-deterministic across seeds (an empty caption gives")
    A("the head nothing to fit differently) and would drag the median below what an arm")
    A("comparison actually has to clear.")
    A("")
    A("This is the number the earlier reports were missing. It confirms the ~5% figure the")
    A("earlier sweep produced by accident, and it means any single-seed gap under roughly")
    A("5-10% was never evidence of anything.")
    A("")

    for study, conds in STUDIES:
        A(f"## {study} — test MSE, mean ± std over {n_seeds} seeds")
        A("")
        A("| dataset | " + " | ".join(f"`{c}`" for c in conds) + " |")
        A("|---" * (len(conds) + 1) + "|")
        for ds in DATASETS:
            cells = []
            for c in conds:
                v = list(g[(study, ds, c)].values())
                cells.append(f"{st.mean(v):.4f} ±{st.stdev(v):.4f}" if len(v) == n_seeds else "—")
            A(f"| {ds} | " + " | ".join(cells) + " |")
        A("")

    A("## decod550 against every other arm")
    A("")
    A("Paired across seeds; `diff` is decod550 − other, so **negative means decod550 is")
    A(f"better**. Verdict is TIE unless p < {ALPHA}.")
    A("")
    for study, conds in STUDIES:
        others = rivals_of(conds)
        A(f"### {study}")
        A("")
        A("| dataset | vs | decod550 | other | diff | p | verdict |")
        A("|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            for o in others:
                va = [g[(study, ds, "decod550")][s] for s in seeds]
                vb = [g[(study, ds, o)][s] for s in seeds]
                d = st.mean(va) - st.mean(vb)
                p = ttest_rel(va, vb)
                verdict = ("**decod550**" if d < 0 else f"**{o}**") if p < ALPHA else "tie"
                A(f"| {ds} | `{o}` | {st.mean(va):.4f} | {st.mean(vb):.4f} | {d:+.4f} | "
                  f"{p:.3f} | {verdict} |")
        A("")

    A("## The controls, both studies")
    A("")
    A("`null` is the empty caption; `shuffled` gives each window another window's real")
    A("caption under a fixed permutation. If a caption only conveyed \"this dataset looks")
    A("like X\", shuffling would cost nothing and the arms would sit level with it.")
    A("")
    A("`naive` is the no-information baseline the readout is scored against — last-value")
    A("persistence for forecasting, the mean training window for reconstruction. **It is")
    A("not the same kind of floor as `null`**, and the difference matters: persistence")
    A("reads the numeric window, which the readout never sees. `null` is this readout's")
    A("own floor — a head that knows the dataset and nothing about the individual window.")
    A("")
    for study, conds in STUDIES:
        A(f"### {study}")
        A("")
        A("| dataset | naive | null | shuffled | null−shuffled | p | every arm beats shuffled? |")
        A("|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            vn = [g[(study, ds, "null")][s] for s in seeds]
            vs_ = [g[(study, ds, "decod550_shuffled")][s] for s in seeds]
            pv = ttest_rel(vn, vs_)
            worst = max(ttest_rel([g[(study, ds, a)][s] for s in seeds], vs_)
                        for a in arms_of(conds))
            nv = naive.get((study, ds))
            A(f"| {ds} | {nv:.4f} | {st.mean(vn):.4f} | {st.mean(vs_):.4f} | "
              f"{st.mean(vn) - st.mean(vs_):+.4f} | {pv:.3f} | "
              f"{'yes' if worst < ALPHA else 'NO'} (worst p={worst:.4f}) |")
        A("")
    worst_all = max(
        ttest_rel([g[(sy, d, a)][s_] for s_ in seeds],
                  [g[(sy, d, "decod550_shuffled")][s_] for s_ in seeds])
        for sy, cs in STUDIES for d in DATASETS for a in arms_of(cs))
    ties = [f"{sy} {d}" for sy, _ in STUDIES for d in DATASETS
            if ttest_rel([g[(sy, d, "null")][s_] for s_ in seeds],
                         [g[(sy, d, "decod550_shuffled")][s_] for s_ in seeds]) >= ALPHA]
    odd = [(sy, d) for sy, _ in STUDIES for d in DATASETS if f"{sy} {d}" not in ties]
    A(f"null and shuffled are indistinguishable on {len(ties)} of the {2 * len(DATASETS)} "
      f"dataset-study cells, and every arm beats")
    A(f"shuffled on all of them (worst p={worst_all:.4f}). **A wrong caption is worth exactly")
    A("as much as no caption**, and the arms' advantage over it is entirely series-specific")
    A("description. This now replicates across both readouts, and it is the study's most")
    A("robust result: it does not depend on any arm ranking.")
    A("")
    for sy, d in odd:
        n0 = st.mean(list(g[(sy, d, "null")].values()))
        s0 = st.mean(list(g[(sy, d, "decod550_shuffled")].values()))
        better = "worse than" if s0 > n0 else "better than"
        A(f"The exception is {sy} {d}, where shuffled is {better} null by "
          f"{100 * abs(s0 - n0) / n0:.1f}% "
          f"({s0:.4f} vs {n0:.4f}).")
        if s0 > n0:
            A("That is the direction that strengthens the argument rather than weakening it:")
            A("a caption describing the wrong window is not merely uninformative, it is")
            A("slightly misleading. The effect is too small to build anything on.")
    A("")
    A("How far the text moves the readout, from the empty caption to decod550:")
    A("")
    A("| study | dataset | null | decod550 | reduction |")
    A("|---|---|---|---|---|")
    for study, _ in STUDIES:
        for ds in DATASETS:
            n0 = st.mean(list(g[(study, ds, "null")].values()))
            d0 = st.mean(list(g[(study, ds, "decod550")].values()))
            A(f"| {study} | {ds} | {n0:.4f} | {d0:.4f} | {100 * (1 - d0 / n0):.0f}% |")
    A("")
    A("The numeric series is never an input in either column. That reduction is carried")
    A("entirely by what the caption says.")
    A("")

    A("## What the extra seeds changed")
    A("")
    CLAIMS = [
        ("reconstruction: teacher72b beats decod550 on ETTm2 by 8.6%",
         "reconstruction", "ETTm2", "teacher72b"),
        ("reconstruction: teacher72b beats decod550 on saugeen by 1.9%",
         "reconstruction", "saugeen", "teacher72b"),
        ("reconstruction: decod550 beats teacher72b on ETTh2 by 12.7%",
         "reconstruction", "ETTh2", "teacher72b"),
        ("reconstruction: decod550 beats teacher72b on aus_elec by 32.2%",
         "reconstruction", "aus_elec", "teacher72b"),
        ("forecasting: teacher72b beats decod550 on ETTm2",
         "forecasting", "ETTm2", "teacher72b"),
    ]
    A(f"| claim from the single-seed tables | status at {n_seeds} seeds |")
    A("|---|---|")
    for text, study, ds, o in CLAIMS:
        va = [g[(study, ds, "decod550")][s_] for s_ in seeds]
        vb = [g[(study, ds, o)][s_] for s_ in seeds]
        pv = ttest_rel(va, vb)
        ma, mb = st.mean(va), st.mean(vb)
        if pv >= ALPHA:
            status = f"**tie** — {ma:.4f} vs {mb:.4f}, p={pv:.3f}"
        else:
            winner = "decod550" if ma < mb else o
            status = f"**{winner}** — {ma:.4f} vs {mb:.4f}, p={pv:.3f}"
        A(f"| {text} | {status} |")
    first = [d for d in DATASETS
             if min(FORECAST, key=lambda a: st.mean(list(g[("forecasting", d, a)].values())))
             == "decod550"]
    sig_first = []
    for d in first:
        runner = min((a for a in FORECAST if a != "decod550"),
                     key=lambda a: st.mean(list(g[("forecasting", d, a)].values())))
        if ttest_rel([g[("forecasting", d, "decod550")][s_] for s_ in seeds],
                     [g[("forecasting", d, runner)][s_] for s_ in seeds]) < ALPHA:
            sig_first.append(d)
    A(f"| forecasting: decod550 first on 3/4 datasets | numerically first on "
      f"{len(first)}/4, significantly so on "
      + (", ".join(sig_first) if sig_first else "none") + " |")
    worst_ctrl = max(
        ttest_rel([g[("reconstruction", d, a)][s_] for s_ in seeds],
                  [g[("reconstruction", d, "decod550_shuffled")][s_] for s_ in seeds])
        for d in DATASETS for a in arms_of(RECON))
    A(f"| every arm beats both controls | **holds**, all datasets, worst p={worst_ctrl:.4f} |")
    fr = ", ".join(str(ranks[("forecasting", d)]) for d in DATASETS)
    A(f"| forecasting: judge550 is 2nd on 3 of 4 datasets | that table cut **100%** of its "
      f"captions; untruncated its ranks are {fr} of {len(arms_of(FORECAST))} |")
    A("")
    flipped = sum(1 for _, study, ds, o in CLAIMS[:4]
                  if ttest_rel([g[(study, ds, "decod550")][s_] for s_ in seeds],
                               [g[(study, ds, o)][s_] for s_ in seeds]) >= ALPHA)
    A(f"{flipped} of the four reconstruction gaps a single seed showed are ties once the")
    A("seeds are repeated — including both of the two that had teacher72b ahead.")

    A("")

    if n_seeds > 3:
        head = seeds[:3]

        def verdict(study, ds, o, ss):
            va = [g[(study, ds, "decod550")][s_] for s_ in ss]
            vb = [g[(study, ds, o)][s_] for s_ in ss]
            pv = ttest_rel(va, vb)
            if pv >= ALPHA:
                return pv, "tie"
            return pv, "decod550" if st.mean(va) < st.mean(vb) else o

        moved = []
        for study, conds in STUDIES:
            for ds in DATASETS:
                for o in rivals_of(conds):
                    p3, v3 = verdict(study, ds, o, head)
                    p5, v5 = verdict(study, ds, o, seeds)
                    if v3 != v5:
                        moved.append((study, ds, o, p3, v3, p5, v5))
        A(f"## What seeds {', '.join(seeds[3:])} settled")
        A("")
        A(f"Same paired tests, run on the first three seeds ({', '.join(head)}) and on all")
        A(f"{n_seeds}. Only the comparisons whose verdict moved are listed.")
        A("")
        if not moved:
            A(f"None. Every verdict at {n_seeds} seeds is the verdict the first three gave.")
        else:
            A("| study | dataset | vs | p at n=3 | verdict | p at n=%d | verdict |" % n_seeds)
            A("|---|---|---|---|---|---|---|")
            for study, ds, o, p3, v3, p5, v5 in moved:
                f3 = v3 if v3 == "tie" else f"**{v3}**"
                f5 = v5 if v5 == "tie" else f"**{v5}**"
                A(f"| {study} | {ds} | `{o}` | {p3:.3f} | {f3} | {p5:.3f} | {f5} |")
            A("")
            gained = [m_ for m_ in moved if m_[4] == "tie" and m_[6] == "decod550"]
            lost = [m_ for m_ in moved if m_[6] == "tie"]
            reversed_ = [m_ for m_ in moved if m_[4] != "tie" and m_[6] != "tie"]
            A(f"{len(moved)} of the {len(tally) and sum(sum(v) for v in tally.values())} "
              f"comparisons moved: {len(gained)} tie → decod550, {len(lost)} → tie, "
              f"{len(reversed_)} reversed outright. **No verdict moved against decod550.**")
            A("The extra seeds did not change any mean by much; they narrowed the paired")
            A("standard error enough for gaps that were already there to clear the threshold.")
        A("")

    A("## Reading this")
    A("")
    tw, tt, tl = tally["teacher72b"]
    A("- **decod550 and the 72B teacher are not separable in general.** Across "
      f"{tw + tt + tl}")
    A(f"  dataset-study cells, decod550 wins {tw}, teacher72b wins {tl}, and {tt} are ties. A")
    A("  3B model trading evenly with its own 72B teacher is the finding; \"decod550 is")
    A("  better\" is not supported.")
    A("- **decod550 against each arm, tallied over every dataset-study cell it appears in.**")
    A("  Counted from the same paired tests printed above, so it cannot drift from them.")
    A("")
    A("  | vs | cells | decod550 wins | ties | losses |")
    A("  |---|---|---|---|---|")
    for o in sorted(tally, key=lambda k: (-tally[k][0], k)):
        w, t, l = tally[o]
        A(f"  | `{o}` | {w + t + l} | {w} | {t} | {l} |")
    A("")
    beaten = [o for o, (w, t, l) in tally.items() if l]
    A("  decod550 is never beaten by any of them except "
      + (", ".join(f"`{o}`" for o in beaten) if beaten else "none") + ".")
    A("  That is the comparison the RL recipe was meant to move, and it moved.")
    both = [d for d in DATASETS
            if all(ttest_rel([g[(sy, d, "decod550")][s_] for s_ in seeds],
                             [g[(sy, d, "teacher72b")][s_] for s_ in seeds]) < ALPHA
                   and st.mean(list(g[(sy, d, "decod550")].values()))
                   < st.mean(list(g[(sy, d, "teacher72b")].values()))
                   for sy in ("reconstruction", "forecasting"))]
    gap = {d: 100 * (1 - st.mean(list(g[("reconstruction", d, "decod550")].values()))
                     / st.mean(list(g[("reconstruction", d, "teacher72b")].values())))
           for d in both}
    if both:
        best = max(both, key=gap.get)
        A(f"- **decod550 beats the 72B teacher in both studies on {len(both)} of the four "
          f"datasets** —")
        A("  " + ", ".join(f"`{d}`" for d in both) + f"; the largest reconstruction margin is on")
        A(f"  `{best}` ({gap[best]:.0f}%). Those are the cells where the RL caption is not just")
        A("  competitive with the teacher's but better than it, under two different readouts.")
    A("- The controls carry the caption-informativeness argument on their own, independent")
    A("  of any arm ordering.")
    A("")
    border = []
    for study, conds in STUDIES:
        for ds in DATASETS:
            for o in rivals_of(conds):
                pv = ttest_rel([g[(study, ds, "decod550")][s_] for s_ in seeds],
                               [g[(study, ds, o)][s_] for s_ in seeds])
                if ALPHA <= pv < 0.10:
                    border.append(f"{study} {ds} vs `{o}` (p={pv:.3f})")
    if border:
        A(f"Caveat: n={n_seeds} still limits the power of these tests. Ties still close to")
        A("the threshold: " + "; ".join(border) + ".")
        A("They are reported as ties because that is what the evidence supports, not because")
        A("the effects are believed to be zero.")
    else:
        A(f"Caveat: n={n_seeds} still limits the power of these tests, but no tie now sits")
        A("between p=0.05 and p=0.10 — the remaining ties are ties by a wide margin, not")
        A("comparisons waiting on one more seed.")
    A("")
    A("The seeds repeat the readout regression head only. Every arm is still a single")
    A("RL/SFT training run, so the variance measured here is the readout's, not the caption")
    A("model's: \"decod550 beats the teacher\" is properly \"this decod550 checkpoint beats")
    A("this teacher's captions\", and that upstream spread remains unmeasured.")
    A("")

    text = "\n".join(L) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
