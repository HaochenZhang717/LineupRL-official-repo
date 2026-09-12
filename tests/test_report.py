import json
from pathlib import Path

from ts_eval.caption_answer_harness import PredictionRecord
from ts_eval.report import aggregate, write_report


def _rec(item_id, score, domain="d1", task_type="t1", scoring_type="letter_exact"):
    return PredictionRecord(
        item_id=item_id, source_benchmark="fake", caption="cap", answer_raw="ans",
        judge_raw=None, score=score, scoring_type=scoring_type, task_type=task_type, domain=domain,
    )


def test_aggregate_overall_and_stratified():
    records = [
        _rec("1", 1.0, domain="a", task_type="x"),
        _rec("2", 0.0, domain="a", task_type="y"),
        _rec("3", 1.0, domain="b", task_type="x"),
    ]
    report = aggregate(records)
    assert report["n_items"] == 3
    assert abs(report["overall_accuracy"] - (2 / 3)) < 1e-9
    assert report["by_domain"]["a"]["n"] == 2
    assert abs(report["by_domain"]["a"]["accuracy"] - 0.5) < 1e-9
    assert report["by_domain"]["b"]["n"] == 1
    assert report["by_task_type"]["x"]["n"] == 2


def test_aggregate_empty_records_no_crash():
    report = aggregate([])
    assert report["n_items"] == 0
    assert report["overall_accuracy"] == 0.0


def test_write_report_creates_expected_files(tmp_path):
    records = [_rec("1", 1.0), _rec("2", 0.0)]
    out_dir = tmp_path / "out"
    report = write_report(records, out_dir, run_name="test_run")

    assert (out_dir / "predictions.jsonl").exists()
    lines = (out_dir / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["item_id"] == "1"

    assert (out_dir / "report.json").exists()
    saved = json.loads((out_dir / "report.json").read_text())
    assert saved["run_name"] == "test_run"
    assert saved["overall_accuracy"] == report["overall_accuracy"]

    md = (out_dir / "report.md").read_text()
    assert "test_run" in md
    assert "overall_accuracy" in md
