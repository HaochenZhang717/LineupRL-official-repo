from ts_eval.answer_step import filter_items_with_captions
from ts_eval.benchmarks.base import QAItem


def _item(id_):
    return QAItem(id=id_, series=[1.0, 2.0], question="q", gold="A", scoring_type="letter_exact")


def test_no_missing_returns_all_items_unchanged():
    items = [_item("1"), _item("2")]
    captions = {"1": "cap1", "2": "cap2"}
    assert filter_items_with_captions(items, captions) == items


def test_small_fraction_missing_is_dropped_not_raised():
    items = [_item(str(i)) for i in range(100)]
    captions = {str(i): "cap" for i in range(99)}
    kept = filter_items_with_captions(items, captions)
    assert len(kept) == 99
    assert "99" not in {item.id for item in kept}


def test_large_fraction_missing_raises():
    items = [_item(str(i)) for i in range(100)]
    captions = {str(i): "cap" for i in range(50)}
    try:
        filter_items_with_captions(items, captions)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "systemic id mismatch" in str(e)


def test_tolerance_boundary_is_configurable():
    items = [_item(str(i)) for i in range(10)]
    captions = {str(i): "cap" for i in range(9)}
    try:
        filter_items_with_captions(items, captions, tolerance=0.02)
        assert False, "expected ValueError at 2% tolerance"
    except ValueError:
        pass
    kept = filter_items_with_captions(items, captions, tolerance=0.5)
    assert len(kept) == 9
