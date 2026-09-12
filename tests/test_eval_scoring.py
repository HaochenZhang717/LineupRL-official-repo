from ts_eval.benchmarks.base import QAItem
from ts_eval.scoring import (
    build_llm_judge_prompt,
    letter_exact,
    letter_flexible,
    ordering_exact,
    parse_llm_judge_verdict,
    regex_numeric_extract_tolerance,
    score,
    tf_exact,
)


def _item(**kw):
    base = dict(id="t1", series=[1.0, 2.0, 3.0], question="q", gold="A", scoring_type="letter_exact")
    base.update(kw)
    return QAItem(**base)


def test_letter_exact_matches_parse_easy_style():
    item = _item(gold="C")
    assert letter_exact("The answer is C.", item) == 1.0
    assert letter_exact("The answer is D.", item) == 0.0
    assert letter_exact("no letter here", item) == 0.0


def test_tf_exact():
    item = _item(gold="T", scoring_type="tf_exact")
    assert tf_exact("True, because the series rises.", item) == 1.0
    assert tf_exact("False.", item) == 0.0


def test_letter_flexible_substring_matches_timeseriesexam():
    item = _item(gold="B) Decrease", scoring_type="letter_flexible")
    assert letter_flexible("Based on the trend, B) Decrease is correct.", item) == 1.0
    assert letter_flexible("A) Increase seems right.", item) == 0.0


def test_ordering_exact_permutation():
    item = _item(gold="B,A,C,D", scoring_type="ordering_exact")
    assert ordering_exact("The correct order is B, A, C, D.", item) == 1.0
    assert ordering_exact("The correct order is A, B, C, D.", item) == 0.0


def test_numeric_tolerance_dispatch_via_score():
    item = _item(gold="100", scoring_type="numeric_tolerance")
    assert score("about 100", item) == 1.0
    assert score("about 90", item) == 0.9
    assert score("no numbers", item) == 0.0


def test_regex_numeric_extract_tolerance_freeform():
    item = _item(
        gold="The highest average vehicle count occurred at 2005-11-07, with an average of 1486.04.",
        scoring_type="regex_numeric_extract_tolerance",
    )
    assert regex_numeric_extract_tolerance("The peak value was about 1486.0.", item) == 1.0
    assert regex_numeric_extract_tolerance("The peak value was about 200.", item) == 0.0


def test_llm_judge_prompt_and_verdict_parsing():
    item = _item(gold="the series is stationary", scoring_type="llm_judge")
    prompt = build_llm_judge_prompt(item, "it stays flat with no trend")
    assert "stationary" in prompt
    assert parse_llm_judge_verdict("YES, they match.") == 1.0
    assert parse_llm_judge_verdict("No, they differ.") == 0.0


def test_llm_judge_prompt_tolerates_multi_topic_candidate_and_omits_raw_question():
    item = _item(gold="noisy, std around 1.5", scoring_type="llm_judge",
                 task_type="noise", question="1. ...\n2. ...\n3. ...")
    prompt = build_llm_judge_prompt(item, "1. noisy, std ~1.5\n2. shake at point 125\n3. steady")
    assert "1. ...\n2. ...\n3. ..." not in prompt
    assert "noise" in prompt
    assert "may discuss other facts" in prompt or "ANYWHERE" in prompt


def test_score_rejects_llm_judge_without_generation():
    item = _item(gold="x", scoring_type="llm_judge")
    try:
        score("anything", item)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_multivariate_n_variates_detected():
    item = _item(series=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert item.n_variates == 3
    item_uni = _item(series=[1.0, 2.0, 3.0])
    assert item_uni.n_variates == 1
