from __future__ import annotations

from ts_downstream.config import ReadoutConfig, TrainConfig
from ts_downstream.captioners import MockStatCaptioner, read_captions_jsonl, write_captions_jsonl
from ts_downstream.datasets import make_synthetic, load_dataset
from ts_downstream.readout_step import train_readout


def _caps(items):
    caps = MockStatCaptioner()(items)
    return dict(zip((it.id for it in items), caps))


def test_classification_learns():
    items = make_synthetic(task="classification", n=240, seed=0)
    captions = _caps(items)
    rcfg = ReadoutConfig(encoder_model="mock", mock_dim=64)
    tcfg = TrainConfig(epochs=25, batch_size=32, lr=1e-2, patience=8, device="cpu")
    res = train_readout(items, captions, "classification", rcfg, tcfg)
    assert res["test"]["accuracy"] > 0.6, res["test"]
    assert res["trainable_params"] > 0


def test_forecasting_runs():
    items = make_synthetic(task="forecasting", n=180, horizon=24, seed=1)
    captions = _caps(items)
    rcfg = ReadoutConfig(encoder_model="mock", mock_dim=64)
    tcfg = TrainConfig(epochs=15, batch_size=32, lr=1e-2, patience=6, device="cpu")
    res = train_readout(items, captions, "forecasting", rcfg, tcfg)
    assert "mse" in res["test"] and res["test"]["mse"] >= 0.0
    assert "naive_skill" in res["test"]


def test_captions_jsonl_roundtrip(tmp_path):
    items = make_synthetic(task="classification", n=10, seed=2)
    caps = MockStatCaptioner()(items)
    p = tmp_path / "captions.jsonl"
    write_captions_jsonl(p, [it.id for it in items], caps)
    back = read_captions_jsonl(p)
    assert len(back) == 10
    assert back[items[0].id] == caps[0]


def test_dataset_dispatch():
    items = load_dataset("synthetic", "classification")
    assert items and items[0].task == "classification"
    splits = {it.split for it in items}
    assert {"train", "val", "test"} <= splits


if __name__ == "__main__":
    test_classification_learns()
    test_forecasting_runs()
    test_dataset_dispatch()
    print("smoke OK")
