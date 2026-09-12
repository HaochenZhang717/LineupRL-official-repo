from __future__ import annotations

from ts_qa.extract_qa import extract_questions_and_answers

RAW = """#### 1. **In which part of the series does the highest value occur?**
   - A) The early part
   - B) The middle part
   - C) The late part
   - D) Roughly constant

**Answer:** C) The late part
------
#### 2. **What is the overall trend?**
   - A) Increasing overall
   - B) Decreasing overall
   - C) Roughly flat
   - D) First rising then falling

**Answer:** B) Decreasing overall
------
"""


def test_parses_two_questions():
    qas = extract_questions_and_answers(RAW)
    assert len(qas) == 2
    assert qas[0]["answer"] == "C"
    assert qas[1]["answer"] == "B"


def test_question_keeps_options_and_strips_markers():
    qas = extract_questions_and_answers(RAW)
    q0 = qas[0]["question"]
    assert q0.startswith("In which part")
    assert "#### 1." not in q0 and "**" not in q0
    assert "- A) The early part" in q0
    assert "- C) The late part" in q0


def test_drops_malformed_block_without_options():
    bad = "#### 1. **A vague question with no options?**\n\n**Answer:** A) something\n------"
    assert extract_questions_and_answers(bad) == []


def test_drops_block_without_answer_letter():
    bad = ("#### 1. **Q?**\n   - A) x\n   - B) y\n\n"
           "**Answer:** none of the above\n------")
    assert extract_questions_and_answers(bad) == []


def test_handles_answer_on_last_line_without_bold():
    raw = ("#### 1. **Q?**\n   - A) x\n   - B) y\n   - C) z\n   - D) w\n"
           "Answer: D) w\n------")
    qas = extract_questions_and_answers(raw)
    assert len(qas) == 1 and qas[0]["answer"] == "D"
