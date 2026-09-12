import re

from ts_eval.benchmarks import timeseries_exam

_GOLD_RE = re.compile(r"^[A-Z]\) .+")


def test_timeseries_exam_load_full_split():
    items = list(timeseries_exam.load(split="test"))
    assert len(items) == 746, f"expected 746 QAItems, got {len(items)} (some rows may have been skipped)"

    saw_single = False
    saw_multi = False
    for item in items:
        assert _GOLD_RE.match(item.gold), f"gold {item.gold!r} doesn't match '<letter>) ...' pattern"
        assert item.n_variates in (1, 2)
        if item.n_variates == 1:
            saw_single = True
        else:
            saw_multi = True
        assert item.source_benchmark == "timeseries_exam"

    assert saw_single, "expected at least one single-series (ts) row"
    assert saw_multi, "expected at least one two-series (ts1/ts2) row"
