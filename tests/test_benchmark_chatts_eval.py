from collections import Counter

from ts_eval.benchmarks import chatts_eval

_ELIGIBLE = {"trend", "season", "noise", "local", "causal"}


def test_load_limit_succeeds_and_has_expected_shape():
    items = list(chatts_eval.load(limit=30))
    assert len(items) == 30
    for item in items:
        assert item.task_type in _ELIGIBLE
        assert item.scoring_type == "llm_judge"
        assert len(item.series) > 0
        assert item.gold
        assert item.source_benchmark == "chatts_eval"


def test_full_sub_question_counts_by_ability_type():
    items = list(chatts_eval.load(limit=None))
    counts = Counter(item.task_type for item in items)
    print("\nchatts_eval closed-form-eligible sub-question counts:")
    for ability in sorted(_ELIGIBLE):
        print(f"  {ability}: {counts.get(ability, 0)}")
    total = sum(counts.values())
    print(f"  TOTAL: {total}")

    assert set(counts) <= _ELIGIBLE
    assert total > 0
    for item in items:
        assert item.task_type not in {
            "local-inductive",
            "local-cluster-inductive",
            "local-correlation-inductive",
            "shape-cluster-inductive",
            "shape-correlation-inductive",
            "deductive",
            "MCQ2",
        }
