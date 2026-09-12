import json

from ts_eval.benchmarks import ts_skill


def test_load_limit_yields_expected_count():
    items = list(ts_skill.load(limit=10))
    assert len(items) == 10


def test_series_are_nonempty_floats():
    items = list(ts_skill.load(limit=5))
    for item in items:
        assert len(item.series) > 0
        assert all(isinstance(v, float) for v in item.series)


def test_gold_is_nontrivial_sentence():
    items = list(ts_skill.load(limit=5))
    for item in items:
        assert isinstance(item.gold, str)
        assert len(item.gold) > 10


def test_scoring_type_is_regex_numeric_extract_tolerance():
    items = list(ts_skill.load(limit=10))
    assert len(items) == 10
    for item in items:
        assert item.scoring_type == "regex_numeric_extract_tolerance"


def test_license_carries_unspecified_warning():
    assert "UNSPECIFIED" in ts_skill.LICENSE


def test_ids_stable_across_runs_even_if_a_different_row_fails_each_time(tmp_path, monkeypatch):
    rows = []
    for i in range(5):
        rows.append({
            "ts_file": f"f{i}.h5", "channel_idx": 0,
            "question": f"q{i}", "answer": f"a{i} with a number 42",
            "sk_type": "SK1", "category": "cat", "sk_level": "L1",
            "cluster": "c", "metric": "m",
        })
    qa_path = tmp_path / "ts_skill_qa.jsonl"
    with open(qa_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(ts_skill, "_download_qa_jsonl", lambda: qa_path)

    def make_reader(fail_line_idx):
        def _read_channel(ts_file, channel_idx):
            idx = int(ts_file[1:].split(".")[0])
            if idx == fail_line_idx:
                raise RuntimeError("simulated transient download failure")
            return [1.0, 2.0, 3.0]
        return _read_channel

    monkeypatch.setattr(ts_skill, "_read_channel", make_reader(fail_line_idx=1))
    items_run_a = {item.id: item for item in ts_skill.load()}

    monkeypatch.setattr(ts_skill, "_read_channel", make_reader(fail_line_idx=3))
    items_run_b = {item.id: item for item in ts_skill.load()}

    common_surviving_rows = {"f0.h5", "f2.h5", "f4.h5"}
    for ts_file in common_surviving_rows:
        matches_a = [iid for iid in items_run_a if iid.startswith(f"{ts_file}::")]
        matches_b = [iid for iid in items_run_b if iid.startswith(f"{ts_file}::")]
        assert matches_a == matches_b, f"id for {ts_file} diverged between runs: {matches_a} vs {matches_b}"
