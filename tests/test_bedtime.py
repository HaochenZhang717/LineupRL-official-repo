from __future__ import annotations

import json
from pathlib import Path

import pytest

from ts_bedtime.data import SeriesRow, parse_series, sample_rows, synthetic_rows
from ts_bedtime.negatives import build_negatives
from ts_bedtime.prepare import build_items
from ts_bedtime.prompts import LETTERS, differentiation_question, recognition_question

DATA_DIR = Path("bench_data/bedtime")
needs_real_data = pytest.mark.skipif(
    not (DATA_DIR / "items_sbert.jsonl").exists(),
    reason="run `python -m ts_bedtime.prepare` first")


def _rows(dataset="truce_stock", n=16):
    return synthetic_rows(dataset, n, seed=0, classed=dataset in ("sushi", "taxosynth"))


def _built(dataset="truce_stock", n=16, strategy="euclid", seed=2020):
    rows = _rows(dataset, n)
    negatives = build_negatives(rows, strategy, top_n=len(LETTERS) - 1)
    return rows, build_items(rows, negatives, seed)


def test_parse_series_handles_bare_nan():
    values = parse_series("[1.0, nan, 3.5]")
    assert len(values) == 3
    assert values[0] == 1.0 and values[2] == 3.5
    assert values[1] != values[1]


def test_parse_series_plain_list():
    assert parse_series("[8, 8, 6, 5]") == [8.0, 8.0, 6.0, 5.0]


def test_sample_rows_is_deterministic_and_order_independent():
    rows = _rows(n=40)
    a = [r.series_uid for r in sample_rows(rows, 10, seed=2020)]
    b = [r.series_uid for r in sample_rows(list(reversed(rows)), 10, seed=2020)]
    assert a == b, "sample must not depend on input order"
    assert len(set(a)) == 10


@pytest.mark.parametrize("strategy", ["euclid"])
def test_distractors_exclude_own_series_annotations(strategy):
    rows = _rows(n=16)
    negatives = build_negatives(rows, strategy, top_n=3)
    for row in rows:
        for annotation in row.annotations:
            picked = negatives[(row.series_uid, annotation)]
            assert len(picked) == 3
            assert len(set(picked)) == 3, "options must be distinct"
            assert not (set(picked) & set(row.annotations))


def test_distractors_respect_cross_class_constraint():
    rows = _rows("taxosynth", n=16)
    by_text = {a: r.cls for r in rows for a in r.annotations}
    negatives = build_negatives(rows, "euclid", top_n=3)
    for row in rows:
        for annotation in row.annotations:
            for distractor in negatives[(row.series_uid, annotation)]:
                assert by_text[distractor] != row.cls


def test_build_negatives_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_negatives(_rows(n=4), "dtw")


def test_recognition_is_exactly_balanced():
    _, items = _built()
    rec = [i for i in items if i["task_type"] == "recognition"]
    assert rec and sum(i["gold"] == "True" for i in rec) == sum(i["gold"] == "False" for i in rec)


def test_differentiation_gold_points_at_the_true_description():
    _, items = _built()
    for item in (i for i in items if i["task_type"] == "differentiation"):
        assert len(item["options"]) == len(LETTERS)
        assert len(set(item["options"])) == len(LETTERS)
        assert item["options"][LETTERS.index(item["gold"])] == item["description"]


def test_item_ids_are_unique():
    _, items = _built()
    ids = [i["item_id"] for i in items]
    assert len(ids) == len(set(ids))


def test_items_are_reproducible_for_a_fixed_seed():
    a = _built()[1]
    b = _built()[1]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_option_shuffle_is_keyed_per_item_not_by_a_walking_rng():
    rows = _rows(n=16)
    negatives = build_negatives(rows, "euclid", top_n=3)
    full = {i["item_id"]: i.get("options") for i in build_items(rows, negatives, 2020)}
    partial = {i["item_id"]: i.get("options")
               for i in build_items(rows[1:], negatives, 2020)}
    shared = set(full) & set(partial)
    assert shared
    assert all(full[k] == partial[k] for k in shared)


def test_recognition_prompt_keeps_the_papers_decision_criterion():
    q = recognition_question("rises sharply at the end")
    assert "rises sharply at the end" in q
    assert "Respond with True if the given description accurately describes" in q
    assert "Respond with False if it does not." in q


def test_prompts_never_splice_the_series():
    q = recognition_question("rises sharply at the end")
    d = differentiation_question(["a", "b", "c", "d"])
    for prompt in (q, d):
        assert "{series}" not in prompt
        assert "Analyze the time series:" not in prompt
    for letter in LETTERS:
        assert f"{letter}: " in d


def test_differentiation_question_requires_four_options():
    with pytest.raises(ValueError):
        differentiation_question(["a", "b"])


def test_caption_items_captions_each_chart_once(tmp_path):
    from ts_eval.benchmarks.base import QAItem
    from ts_eval.caption_answer_harness import caption_items

    def make(item_id, key, series):
        return QAItem(id=item_id, series=series, question="q", gold="True",
                      scoring_type="tf_exact", source_benchmark="fake",
                      extra={"caption_key": key})

    items = [make("i1", "s1", [1, 2, 3]), make("i2", "s1", [1, 2, 3]),
             make("i3", "s2", [3, 2, 1])]
    calls = []

    def generate(paths):
        calls.extend(paths)
        return [f"cap:{Path(p).stem}" for p in paths]

    paths, captions = caption_items(items, tmp_path, generate)
    assert len(calls) == 2, "two unique charts, two generation inputs"
    assert len(captions) == len(items) == len(paths), "still one result per item"
    assert captions[0] == captions[1] != captions[2]
    assert paths[0] == paths[1] != paths[2]


def test_caption_items_without_caption_key_is_unchanged(tmp_path):
    from ts_eval.benchmarks.base import QAItem
    from ts_eval.caption_answer_harness import caption_items

    items = [QAItem(id=f"i{n}", series=[1, 2, 3], question="q", gold="True",
                    scoring_type="tf_exact", source_benchmark="fake") for n in range(3)]
    calls = []

    def generate(paths):
        calls.extend(paths)
        return [f"cap{i}" for i in range(len(paths))]

    paths, captions = caption_items(items, tmp_path, generate)
    assert len(calls) == 3
    assert [p.name for p in paths] == ["fake_i0.png", "fake_i1.png", "fake_i2.png"]


def _tf_item():
    from ts_eval.benchmarks.base import QAItem

    return QAItem(id="i", series=[1, 2, 3], question="Does it rise?", gold="True",
                  scoring_type="tf_exact", source_benchmark="fake")


def test_caption_qa_style_is_byte_identical_to_the_pre_change_prompt():
    from ts_eval.caption_answer_harness import build_answer_messages

    expected = (
        "You will be given a caption describing a time series and a question about it. "
        "Answer the question strictly based on the caption, even if the answer may seem "
        "obvious from prior knowledge or the question wording. Ignore any outside knowledge; "
        "do not assume anything the caption does not explicitly or implicitly state.\n\n"
        "Caption: it goes up\n\n"
        "Question: Does it rise?\n\n"
        "Respond with only True or False."
    )
    assert build_answer_messages(_tf_item(), "it goes up")[0]["content"] == expected
    assert build_answer_messages(_tf_item(), "it goes up", "caption_qa")[0]["content"] == expected


def test_evidence_style_labels_the_block_and_drops_the_anti_inference_clause():
    from ts_eval.caption_answer_harness import (DEFAULT_EVIDENCE_LABEL,
                                                ORACLE_EVIDENCE_LABEL, build_answer_messages)

    oracle = build_answer_messages(_tf_item(), "1, 2, 3", "evidence",
                                   ORACLE_EVIDENCE_LABEL)[0]["content"]
    assert f"{ORACLE_EVIDENCE_LABEL}:\n1, 2, 3" in oracle
    assert "Caption:" not in oracle
    assert "do not assume anything" not in oracle

    l1 = build_answer_messages(_tf_item(), "it goes up", "evidence")[0]["content"]
    assert f"{DEFAULT_EVIDENCE_LABEL}:\nit goes up" in l1


def test_evidence_style_floor_leg_has_no_evidence_block_at_all():
    from ts_eval.caption_answer_harness import build_answer_messages

    floor = build_answer_messages(_tf_item(), "", "evidence")[0]["content"]
    assert "Description of the time series" not in floor
    assert "Time series values" not in floor
    assert "Question: Does it rise?" in floor
    assert build_answer_messages(_tf_item(), "   ", "evidence")[0]["content"] == floor


@needs_real_data
def test_native_style_reproduces_the_papers_numbered_shape():
    from ts_eval.benchmarks import bedtime
    from ts_eval.caption_answer_harness import ORACLE_EVIDENCE_LABEL, build_answer_messages

    item = next(i for i in bedtime.load(limit=6) if i.task_type == "recognition")
    got = build_answer_messages(item, "1, 2, 3", "native", ORACLE_EVIDENCE_LABEL)[0]["content"]
    assert "2. Analyze the time series: 1, 2, 3." in got
    assert "3. Determine if the description precisely matches" in got
    assert "Time series values:" not in got, "native must not use the detached block"


def test_native_style_renumbers_when_the_evidence_step_is_dropped():
    from ts_bedtime.prompts import recognition_question_native

    floor = recognition_question_native("it rises", evidence="")
    assert "1. Review the description" in floor
    assert "2. Determine if the description" in floor
    assert "Analyze the" not in floor, "no 'evidence not provided' line -- that is a signal"


def test_native_style_does_not_double_the_sentence_period():
    from ts_bedtime.prompts import recognition_question_native

    assert "sharply.." not in recognition_question_native("x", "It falls sharply.", "caption")
    assert "1, 2, 3." in recognition_question_native("x", "1, 2, 3", "time series")


def test_native_style_falls_back_for_adapters_without_a_native_prompt():
    from ts_eval.caption_answer_harness import build_answer_messages

    got = build_answer_messages(_tf_item(), "it goes up", "native")[0]["content"]
    assert "Description of the time series:\nit goes up" in got


def test_unknown_answer_style_raises():
    from ts_eval.caption_answer_harness import build_answer_messages

    with pytest.raises(ValueError, match="unknown answer style"):
        build_answer_messages(_tf_item(), "x", "freeform")


def test_answer_and_score_threads_per_item_evidence_labels():
    from ts_eval.caption_answer_harness import answer_and_score

    items = [_tf_item(), _tf_item()]
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        return ["True"] * len(prompts)

    answer_and_score(items, ["1, 2, 3", "it goes up"], generate, lambda m: m[0]["content"],
                     style="evidence", evidence_labels=["Time series values", None])
    assert "Time series values:\n1, 2, 3" in seen[0]
    assert "Description of the time series:\nit goes up" in seen[1]

    with pytest.raises(ValueError, match="evidence labels"):
        answer_and_score(items, ["a", "b"], generate, lambda m: m[0]["content"],
                         style="evidence", evidence_labels=["only-one"])


def test_reference_captions():
    from ts_eval.benchmarks.base import QAItem
    from ts_eval.make_reference_captions import reference_caption

    item = QAItem(id="i", series=[1.0, 2.5, 3.25], question="q", gold="True",
                  scoring_type="tf_exact", source_benchmark="fake")
    assert reference_caption(item, "none") == ""
    assert reference_caption(item, "oracle_series") == "1, 2.5, 3.25"
    with pytest.raises(ValueError):
        reference_caption(item, "l3")


@pytest.mark.parametrize("golds,preds", [
    (["True", "False", "True", "False", "True"], ["True", "False", "False", "False", None]),
    (["A", "B", "C", "D", "A", "C"], ["A", "B", "B", "D", "C", None]),
])
def test_weighted_f1_matches_sklearn(golds, preds):
    from sklearn.metrics import f1_score

    from ts_bedtime.report import _weighted_f1

    expected = f1_score(golds, [p if p else "<unparsed>" for p in preds],
                        average="weighted", labels=sorted(set(golds)), zero_division=0)
    assert _weighted_f1(golds, preds) == pytest.approx(expected)


def test_predicted_label_recovers_the_class_and_flags_garbage():
    from ts_bedtime.report import _predicted_label

    assert _predicted_label({"scoring_type": "tf_exact", "answer_raw": " True."}) == "T"
    assert _predicted_label({"scoring_type": "tf_exact", "answer_raw": "FALSE"}) == "F"
    assert _predicted_label({"scoring_type": "tf_exact", "answer_raw": "maybe"}) is None
    assert _predicted_label({"scoring_type": "letter_exact", "answer_raw": "B) up"}) == "B"


def test_gold_and_prediction_land_in_the_same_space():
    from ts_eval.scoring import normalize_gold, parse_prediction

    for scoring_type, raw, gold in [("tf_exact", "True", "True"),
                                    ("tf_exact", "False.", "False"),
                                    ("letter_exact", "C", "C")]:
        assert parse_prediction(raw, scoring_type) == normalize_gold(gold, scoring_type)
    for fn in (lambda: parse_prediction("1.0", "numeric_tolerance"),
               lambda: normalize_gold("1.0", "numeric_tolerance")):
        with pytest.raises(ValueError):
            fn()


def test_weighted_f1_is_not_silently_zero_on_real_tf_records():
    from ts_bedtime.report import _weighted_f1
    from ts_eval.scoring import normalize_gold, parse_prediction

    raws = ["True", "False", "True", "False"]
    golds = [normalize_gold(g, "tf_exact") for g in ["True", "False", "True", "True"]]
    preds = [parse_prediction(r, "tf_exact") for r in raws]
    assert _weighted_f1(golds, preds) > 0.5


def _fake_run(root, name, records):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(item_id, answer, score, task="differentiation", domain="sushi"):
    return {"item_id": item_id, "answer_raw": answer, "score": score,
            "scoring_type": "letter_exact" if task == "differentiation" else "tf_exact",
            "task_type": task, "domain": domain}


def test_build_direct_uses_a_per_arm_floor_and_reports_lift(tmp_path):
    from ts_bedtime.report import build_direct

    items = tmp_path / "items.jsonl"
    with open(items, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"item_id": f"i{i}", "gold": "A"}) + "\n")

    root = tmp_path / "direct"
    _fake_run(root, "base", [_rec(f"i{i}", "A", 1.0) for i in range(4)])
    _fake_run(root, "base_noimg", [_rec(f"i{i}", "A", float(i < 1)) for i in range(4)])

    summary = build_direct(root, ["base"], items)
    cell = summary["cells"][0]
    assert cell["base"]["accuracy"] == pytest.approx(1.0)
    assert cell["base_noimg"]["accuracy"] == pytest.approx(0.25)
    assert cell["lift_base"] == pytest.approx(0.75)
    assert "L0" not in cell and "L2" not in cell


def test_build_direct_surfaces_unparsed_answers(tmp_path):
    from ts_bedtime.report import build_direct, to_markdown_direct

    items = tmp_path / "items.jsonl"
    with open(items, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({"item_id": f"i{i}", "gold": "A"}) + "\n")

    root = tmp_path / "direct"
    prose = "The series shows a gradual upward movement over time"
    _fake_run(root, "decod550", [_rec("i0", "A", 1.0)] + [_rec(f"i{i}", prose, 0.0)
                                                          for i in range(1, 4)])
    _fake_run(root, "decod550_noimg", [_rec(f"i{i}", "A", 0.25) for i in range(4)])

    summary = build_direct(root, ["decod550"], items)
    cell = summary["cells"][0]
    assert cell["decod550"]["unparsed_rate"] == pytest.approx(0.75)
    md = to_markdown_direct(summary)
    assert "unparsed decod550" in md and "0.7500" in md


@pytest.mark.parametrize("text,expected", [
    ("The time series shows a gradual upward movement", None),
    ("I think the pattern is decreasing", None),
    ("This chart has a clear downward trend", None),
    ("A gradual rise, so the answer is C", "C"),
    ("A", "A"), ("(A)", "A"), ("A) Increase", "A"), ("Answer: A. The series rises", "A"),
    ("The answer is A", "A"), ("I", "I"), ("B", "B"), ("B) Decrease", "B"), ("D.", "D"),
])
def test_letter_extraction_ignores_the_english_article_and_pronoun(text, expected):
    from ts_eval.scoring import extract_first_letter

    assert extract_first_letter(text) == expected


@needs_real_data
def test_prompt_fingerprint_tracks_wording_not_the_evidence_itself():
    from ts_eval.benchmarks import bedtime
    from ts_eval.caption_answer_harness import prompt_fingerprint

    items = list(bedtime.load(limit=40))
    base = prompt_fingerprint(items, "evidence")
    assert base == prompt_fingerprint(items, "evidence"), "must be deterministic"
    assert base != prompt_fingerprint(items, "native"), "style change must show"
    assert base != prompt_fingerprint(items, "caption_qa")


@needs_real_data
def test_l1_and_l2_legs_are_comparable_despite_different_evidence_labels():
    from ts_eval.benchmarks import bedtime
    from ts_eval.caption_answer_harness import prompt_fingerprint
    from ts_bedtime.report import comparable

    items = list(bedtime.load(limit=40))
    fp = prompt_fingerprint(items, "evidence")
    assert comparable({"prompt_fingerprint": fp}, {"prompt_fingerprint": fp})


def test_runs_with_different_prompt_wording_are_not_comparable():
    from ts_bedtime.report import comparable

    assert comparable({"prompt_fingerprint": "abc"}, {"prompt_fingerprint": "abc"})
    assert not comparable({"prompt_fingerprint": "abc"}, {"prompt_fingerprint": "xyz"})
    assert not comparable({}, {"prompt_fingerprint": "abc"})
    assert not comparable({}, {})


def test_information_recovery_is_none_when_the_oracle_cannot_beat_the_floor():
    from ts_bedtime.report import information_recovery

    assert information_recovery(0.5, 0.7, 0.9) == pytest.approx(0.5)
    assert information_recovery(0.5, 0.7, 0.5) is None
    assert information_recovery(0.6, 0.7, 0.5) is None


def test_caption_folding_requires_the_caption_key_column(tmp_path):
    from ts_bedtime.nli_score import load_captions_by_series

    path = tmp_path / "captions.jsonl"
    path.write_text(json.dumps({"id": "x", "caption": "c"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no caption_key column"):
        load_captions_by_series(path, {"s1"})


def test_caption_folding_collapses_the_six_items_per_series(tmp_path):
    from ts_bedtime.nli_score import load_captions_by_series

    path = tmp_path / "captions.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({"id": f"i{i}", "caption_key": "s1", "caption": "c1"}) + "\n")
        f.write(json.dumps({"id": "j", "caption_key": "s2", "caption": "c2"}) + "\n")
    assert load_captions_by_series(path, {"s1", "s2"}) == {"s1": "c1", "s2": "c2"}


def test_caption_folding_refuses_to_pick_between_diverging_captions(tmp_path):
    from ts_bedtime.nli_score import load_captions_by_series

    path = tmp_path / "captions.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "a", "caption_key": "s1", "caption": "one"}) + "\n")
        f.write(json.dumps({"id": "b", "caption_key": "s1", "caption": "two"}) + "\n")
    with pytest.raises(ValueError, match="dedup is broken"):
        load_captions_by_series(path, {"s1"})


def test_nli_summary_counts_both_directions_and_strict_equivalence():
    from ts_bedtime.nli_score import summarise

    records = [
        {"dataset": "d", "caption": "cap", "ground_truth": "gt",
         "p_gen_entails_gt": 0.9, "p_gt_entails_gen": 0.9},
        {"dataset": "d", "caption": "cap", "ground_truth": "gt",
         "p_gen_entails_gt": 0.9, "p_gt_entails_gen": 0.1},
        {"dataset": "d", "caption": "cap", "ground_truth": "gt",
         "p_gen_entails_gt": 0.1, "p_gt_entails_gen": 0.1},
    ]
    got = summarise(records, threshold=0.5)["overall"]
    assert got["gen_entails_gt"] == pytest.approx(2 / 3)
    assert got["gt_entails_gen"] == pytest.approx(1 / 3)
    assert got["bidirectional"] == pytest.approx(1 / 3)


@needs_real_data
def test_real_items_load_through_the_adapter():
    from ts_eval.benchmarks import bedtime

    items = list(bedtime.load())
    assert len(items) > 10_000
    keys = {i.extra["caption_key"] for i in items}
    assert len(keys) == 2000, "500 series x 4 datasets"
    assert all(i.scoring_type in ("tf_exact", "letter_exact") for i in items)
    rec = [i for i in items if i.task_type == "recognition"]
    assert sum(i.gold == "True" for i in rec) == sum(i.gold == "False" for i in rec)


@needs_real_data
def test_real_series_are_finite_and_renderable():
    with open(DATA_DIR / "series.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 2000
    for row in rows:
        assert row["series"], row["series_uid"]
        assert all(v == v and abs(v) != float("inf") for v in row["series"]), row["series_uid"]
