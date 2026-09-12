from pathlib import Path

from ts_eval import scoring as sc
from ts_eval.benchmarks.base import QAItem
from ts_eval.caption_answer_harness import (
    _ANSWER_INSTRUCTIONS,
    build_answer_messages,
    render_items,
    run_pipeline1,
)


def test_every_scoring_type_has_an_answer_instruction():
    all_scoring_types = set(sc.SCORERS) | {"llm_judge"}
    assert all_scoring_types <= set(_ANSWER_INSTRUCTIONS)


def _mk_item(id_, scoring_type, gold, series=None, question="Q?"):
    return QAItem(
        id=id_, series=series or [1.0, 2.0, 3.0, 4.0], question=question,
        gold=gold, scoring_type=scoring_type, task_type="t", domain="d",
        source_benchmark="fake",
    )


def test_render_items_writes_one_png_per_item(tmp_path):
    items = [
        _mk_item("1", "letter_exact", "A"),
        _mk_item("2", "letter_exact", "B", series=[[1.0, 2.0], [3.0, 4.0]]),
    ]
    paths = render_items(items, tmp_path)
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_build_answer_messages_unsupported_scoring_type_raises():
    item = _mk_item("1", "something_new", "A")
    try:
        build_answer_messages(item, "a caption")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_pipeline1_end_to_end_with_mocked_models(tmp_path):
    items = [
        _mk_item("1", "letter_exact", "C"),
        _mk_item("2", "tf_exact", "T"),
        _mk_item("3", "llm_judge", "the series is stationary"),
    ]

    def fake_generate_captions(image_paths):
        assert 0 < len(image_paths) <= len(items)
        return ["the series rises then falls" for _ in image_paths]

    def fake_generate_answers(prompts):
        out = []
        for p in prompts:
            if "letter of the correct choice" in p:
                out.append("The answer is C.")
            elif "True or False" in p:
                out.append("True.")
            else:
                out.append("it stays flat with no trend")
        return out

    def fake_judge_fn(prompts):
        return ["YES, they match." for _ in prompts]

    def fake_apply_chat_template(messages):
        return messages[0]["content"]

    records = run_pipeline1(
        items=items,
        image_root=tmp_path / "render",
        generate_captions=fake_generate_captions,
        generate_answers=fake_generate_answers,
        apply_chat_template=fake_apply_chat_template,
        judge_fn=fake_judge_fn,
        batch_size=2,
    )

    assert len(records) == 3
    by_id = {r.item_id: r for r in records}
    assert by_id["1"].score == 1.0
    assert by_id["2"].score == 1.0
    assert by_id["3"].score == 1.0
    assert by_id["3"].judge_raw is not None
    assert by_id["1"].judge_raw is None


def test_run_pipeline1_missing_judge_fn_raises(tmp_path):
    items = [_mk_item("1", "llm_judge", "gold text")]
    try:
        run_pipeline1(
            items=items, image_root=tmp_path / "render",
            generate_captions=lambda ps: ["cap" for _ in ps],
            generate_answers=lambda ps: ["ans" for _ in ps],
            apply_chat_template=lambda m: m[0]["content"],
            judge_fn=None,
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_run_pipeline1_mismatched_generation_count_raises(tmp_path):
    items = [_mk_item("1", "letter_exact", "A"), _mk_item("2", "letter_exact", "B")]
    try:
        run_pipeline1(
            items=items, image_root=tmp_path / "render",
            generate_captions=lambda ps: ["only one caption"],
            generate_answers=lambda ps: ["ans" for _ in ps],
            apply_chat_template=lambda m: m[0]["content"],
        )
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
