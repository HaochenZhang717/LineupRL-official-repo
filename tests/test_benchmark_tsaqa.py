from ts_eval.benchmarks import tsaqa


def test_tsaqa_test_split_has_42000_rows():
    from datasets import load_dataset

    ds = load_dataset("TSAQA/TSAQA-Benchmark", split="test", cache_dir="bench_data/tsaqa")
    assert len(ds) == 42000


def test_tsaqa_load_small_slice():
    items = list(tsaqa.load(split="test", limit=20))
    assert len(items) > 0
    for item in items:
        assert item.scoring_type in ("letter_exact", "tf_exact", "ordering_exact")
        assert isinstance(item.series, list)
        assert len(item.series) > 0
        assert all(isinstance(x, (int, float)) for x in item.series)
        assert item.gold
        assert item.source_benchmark == "tsaqa"
