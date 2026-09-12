from decodability_rl.rl.llm_mask import (
    count_numbers,
    mask_all_numbers,
    mask_one,
    numeric_density,
    parse_reply,
    split_for_masking,
    token_budget,
)
from decodability_rl.rl.llm_mask import _BOUNDARY_RE


def test_one_line_per_number():
    assert parse_reply("20 1\n21.2 0") == (["20", "21.2"], [1, 0])


def test_decimals_survive_the_parser():
    nums, flags = parse_reply("21.2 1\n0.004 1")
    assert nums == ["21.2", "0.004"] and flags == [1, 1]


def test_formatting_noise_is_tolerated():
    assert parse_reply("1. 21.2 1\n2. 7 0") == (["21.2", "7"], [1, 0])
    assert parse_reply("21.2: 1\n7: 0") == (["21.2", "7"], [1, 0])
    assert parse_reply("Sure:\n21.2 1\n7 0\nDone.") == (["21.2", "7"], [1, 0])


def test_rejects_a_reply_with_no_usable_line():
    assert parse_reply("20\n30") is None
    assert parse_reply("") is None
    assert parse_reply("I cannot help with that.") is None


def test_repeated_literal_is_told_apart_by_position():
    got, unnamed = mask_one("peaks at 20 around step 20.", "20 1\n20 0")
    assert got == "peaks at <val> around step 20." and unnamed == 0


def test_sentence_final_value_is_masked():
    assert mask_one("ends at 3300.", "3300 1")[0] == "ends at <val>."


def test_longer_number_is_not_hit():
    got, _ = mask_one("13300 and 3300 and 3300.5", "13300 0\n3300 1\n3300.5 0")
    assert got == "13300 and <val> and 3300.5"


def test_time_and_structure_survive_when_flagged_zero():
    got, unnamed = mask_one("1. **Peak**: rises to 3700 by the 20th time step.",
                            "1 0\n3700 1\n20 0")
    assert got == "1. **Peak**: rises to <val> by the 20th time step."
    assert unnamed == 0


def test_numbers_the_reader_never_named_are_left_alone():
    got, unnamed = mask_one("rises to 3700 by step 20, then 2600 by step 40.",
                            "3700 1\n20 0")
    assert got == "rises to <val> by step 20, then 2600 by step 40."
    assert unnamed == 2


def test_an_unusable_reply_leaves_the_chunk_untouched():
    got, unnamed = mask_one("rises to 3700 by step 20.", "I cannot help.")
    assert got == "rises to 3700 by step 20." and unnamed == 2


def test_a_stray_line_costs_only_itself():
    got, unnamed = mask_one("rises to 3700 by step 20, then 2600 by step 40.",
                            "3700 1\n9999 1\n20 0\n2600 1\n40 0")
    assert got == "rises to <val> by step 20, then <val> by step 40."
    assert unnamed == 0


def test_a_chunk_with_no_numbers_is_untouched():
    got, unnamed = mask_one("The series then flattens out.", "")
    assert got == "The series then flattens out." and unnamed == 0


def test_budget_scales_with_the_number_count():
    few, many = "rises to 5", " ".join(str(i) for i in range(60))
    assert token_budget(many) > token_budget(few)
    assert token_budget(many) >= 96 + 8 * count_numbers(many) - 1


def test_failsafe_hides_every_number():
    assert mask_all_numbers("peaks at 3700 by step 20.") == "peaks at <val> by step <val>."


def test_numeric_density():
    assert numeric_density("") == 0.0
    assert numeric_density("rises to 3450 at step 2") > 0.0
    assert numeric_density(mask_all_numbers("rises to 3450")) == 0.0


def test_chunking_is_lossless_and_bounded():
    cap = ("Starts at 3300. Rises to 3700 by step 20. Falls to 2600 by step 40.\n"
           "Then 2700, 2800, 2900 across steps 50 to 60. Ends at 3100 by step 96.")
    chunks = split_for_masking(cap, max_numbers=5)
    assert "".join(chunks) == cap
    assert len(chunks) > 1
    assert all(count_numbers(c) <= 5 or _BOUNDARY_RE.search(c) is None for c in chunks)


def test_a_range_on_one_line_is_two_numbers():
    assert parse_reply("9.5 to 10.2 1") == (["9.5", "10.2"], [1, 1])
    assert parse_reply("18-21 0") == (["18", "21"], [0, 0])
    got, unnamed = mask_one("values around 9.5 to 10.2 by time steps 18-21.",
                            "9.5 to 10.2 1\n18-21 0")
    assert got == "values around <val> to <val> by time steps 18-21."
    assert unnamed == 0


def test_prompt_states_how_many_numbers_there_are():
    from decodability_rl.rl.llm_mask import build_prompt

    p = build_prompt("rises to 3700 by step 20, then 2600.")
    assert "exactly 3 numbers" in p
    assert "exactly 3 lines" in p


def test_reader_answers_out_of_order():
    ch = ("reaching a high of about 17.5 by step 12. There's a period from steps 23 to 25 "
          "where the value remains relatively low, around 9.3.")
    got, unnamed = mask_one(ch, "17.5 1\n12 0\n9.3 1\n23 0\n25 0")
    assert "17.5" not in got and "9.3" not in got
    assert "step 12" in got and "steps 23 to 25" in got
    assert unnamed == 0


def test_a_genuinely_missing_number_is_still_counted():
    got, unnamed = mask_one("rises to 3700 by step 20.", "3700 1")
    assert got == "rises to <val> by step 20." and unnamed == 1


def test_reader_rewrites_what_it_copies():
    got, unnamed = mask_one("drops to 0.706 by time step 1, then 0.690 by step 3.",
                            "1.0 0\n0.706 1\n3.0 0\n0.690 1")
    assert got == "drops to <val> by time step 1, then <val> by step 3."
    assert unnamed == 0

    got, unnamed = mask_one("starts at approximately -0.1, drops to about -0.5 by step 4.",
                            "0.1 1\n0.5 1\n4 0")
    assert "-0.1" not in got and "-0.5" not in got and "step 4" in got
    assert unnamed == 0
