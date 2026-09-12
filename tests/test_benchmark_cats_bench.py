import json

from ts_eval.benchmarks import cats_bench

_KNOWN_TASK_TYPES = {
    "caption_retrieval_perturbed",
    "plot_retrieval_same_domain",
    "ts_retrieval_perturbed",
    "ts_comparison_amplitude",
    "ts_comparison_mean",
    "ts_comparison_volatility",
    "ts_comparison_peak_earlier",
    "temporal_matching",
}


def test_load_limit_succeeds_and_has_expected_shape():
    items = list(cats_bench.load(limit=30))
    assert len(items) == 30
    for item in items:
        assert item.scoring_type == "letter_exact"
        assert item.gold.strip().upper() in {"A", "B", "C", "D"}
        assert item.task_type in _KNOWN_TASK_TYPES


def test_full_raw_row_count_is_920():
    cats_bench._ensure_extracted()

    tasks_path = cats_bench._EXTRACT_DIR / "QA_hard_small" / "tasks.json"
    with open(tasks_path) as f:
        pool1_count = len(json.load(f))

    pool2_files = [
        cats_bench._EXTRACT_DIR / "QA" / "temporal_matching" / "mcqs_final.jsonl",
        cats_bench._EXTRACT_DIR / "QA" / "temporal_matching" / "mcqs_mid2_short_len6.jsonl",
    ]
    pool2_count = 0
    for path in pool2_files:
        with open(path) as f:
            pool2_count += sum(1 for line in f if line.strip())

    assert pool1_count == 460
    assert pool2_count == 460
    assert pool1_count + pool2_count == 920
