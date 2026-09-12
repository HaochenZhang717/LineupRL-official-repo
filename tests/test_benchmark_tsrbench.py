import pytest

from ts_eval.benchmarks import tsrbench
from ts_eval.scoring import SCORERS

_VALID_SCORING_TYPES = set(SCORERS) | {"llm_judge"}


def test_load_without_license_accept_raises():
    with pytest.raises(RuntimeError):
        list(tsrbench.load())


def test_load_with_license_accept_succeeds():
    items = list(tsrbench.load(i_accept_unclear_license=True, limit=15))
    assert len(items) == 15
    for item in items:
        assert item.scoring_type in _VALID_SCORING_TYPES
        assert item.gold
        assert item.question
        assert len(item.series) > 0
        assert item.source_benchmark == "tsrbench"
        assert item.task_type


def test_load_covers_multiple_task_files():
    items = list(tsrbench.load(i_accept_unclear_license=True, limit=400))
    task_types = {item.task_type for item in items}
    assert len(task_types) > 1
