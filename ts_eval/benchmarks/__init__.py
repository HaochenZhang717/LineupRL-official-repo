from . import (bedtime, cats_bench, chatts_eval, rl_valset, time_mqa, timeseries_exam, ts_skill,
               tsaqa, tsrbench)
from .base import BenchmarkAdapter, QAItem

REGISTRY = {
    tsaqa.NAME: tsaqa,
    timeseries_exam.NAME: timeseries_exam,
    ts_skill.NAME: ts_skill,
    tsrbench.NAME: tsrbench,
    time_mqa.NAME: time_mqa,
    chatts_eval.NAME: chatts_eval,
    cats_bench.NAME: cats_bench,
    rl_valset.NAME: rl_valset,
    bedtime.NAME: bedtime,
}

__all__ = ["BenchmarkAdapter", "QAItem", "REGISTRY"]
