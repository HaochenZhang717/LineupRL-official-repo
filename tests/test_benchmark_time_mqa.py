from __future__ import annotations

import pytest

from ts_eval.benchmarks import time_mqa

REAL_ACCESS_AVAILABLE = False


def test_missing_token_raises_helpful_error(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        list(time_mqa.load())

    message = str(exc_info.value).lower()
    assert "huggingface.co" in message
    assert "accept" in message
    assert "hf_token" in message


@pytest.mark.skip(reason="Time-MQA is gated, no HF_TOKEN available")
def test_real_data_smoke():
    items = list(time_mqa.load(split="test", limit=20))
    assert len(items) > 0
    for item in items:
        assert item.scoring_type in ("letter_exact", "tf_exact")
        assert isinstance(item.series, list)
        assert len(item.series) > 0
        assert all(isinstance(x, (int, float)) for x in item.series)
        assert item.gold
        assert item.question
        assert item.source_benchmark == "time_mqa"
        assert item.task_type in ("mcq", "tf")
