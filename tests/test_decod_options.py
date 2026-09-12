import re

from decodability_rl.test_reward_design import prompts as P
from decodability_rl.rl.reward_server_decodability import letter_regexes, parse_letter

OPTS4 = ["[1.00, 2.00]", "[3.00, 4.00]", "[5.00, 6.00]", "[7.00, 8.00]"]


def test_four_options_reproduces_the_original_template():
    expected = P.DISCRIM_TEMPLATE_NEUTRAL.format(
        caption="CAP", opt_a=OPTS4[0], opt_b=OPTS4[1], opt_c=OPTS4[2], opt_d=OPTS4[3]
    )
    got = P.build_discriminator_prompt(
        "CAP", OPTS4, variant="neutral", answer_format="letter"
    )
    assert got == expected


def test_the_tag_format_differs_from_it_in_the_last_line_only():
    letter = P.build_discriminator_prompt("CAP", OPTS4, variant="neutral",
                                          answer_format="letter").splitlines()
    tag = P.build_discriminator_prompt("CAP", OPTS4, variant="neutral",
                                       answer_format="tag").splitlines()
    assert letter[:-1] == tag[:-1]
    assert "<answer>LETTER</answer>" in tag[-1]
    for L in P.letters_for(len(P.OPTION_LETTERS)):
        assert f"<answer>{L}</answer>" not in tag[-1]


def test_empty_caption_still_becomes_the_placeholder():
    assert P.EMPTY_CAPTION in P.build_discriminator_prompt("   ", OPTS4, variant="neutral")


def test_tag_mode_system_prompt_asks_for_the_same_thing_as_the_question():
    assert "<answer>" in P.SYSTEMS["tag"]
    assert "<answer>" not in P.SYSTEMS["letter"]


def test_k_way_prompt_labels_every_option_once():
    for k in (4, 6, 8, 10, 12):
        opts = [f"[{i}.00]" for i in range(k)]
        text = P.build_discriminator_prompt("CAP", opts, variant="neutral")
        letters = P.letters_for(k)
        for L, o in zip(letters, opts):
            assert f"{L}) {o}" in text
        ask = text.splitlines()[-1]
        assert ", ".join(letters[:-1]) + f", or {letters[-1]}" in ask
        if k < len(P.OPTION_LETTERS):
            assert not re.search(rf"^{P.OPTION_LETTERS[k]}\)", text, re.M)


def test_hinted_variant_refuses_anything_but_four():
    try:
        P.build_discriminator_prompt("CAP", [f"[{i}]" for i in range(6)], variant="hinted")
    except ValueError:
        return
    raise AssertionError("the hinted template names three distortions; 6 must be refused")


def test_letters_for_rejects_out_of_range():
    for bad in (1, len(P.OPTION_LETTERS) + 1):
        try:
            P.letters_for(bad)
        except ValueError:
            continue
        raise AssertionError(f"n_options={bad} should be refused")


def test_reply_parsing_at_four_options():
    bare, fb = letter_regexes(4)
    for reply, want in [("C", "C"), (" C.", "C"), ("The answer is B", "B"),
                        ("A) is correct", "A"), ("D", "D"), ("nothing here", None)]:
        assert parse_letter(reply, bare, fb) == (want, False), reply


def test_pronoun_i_does_not_win_at_ten_options():
    bare, fb = letter_regexes(10)
    assert parse_letter("I", bare, fb) == ("I", False)
    assert parse_letter("Option I", bare, fb) == ("I", False)
    assert parse_letter("I think it is C", bare, fb) == ("C", False)
    assert parse_letter("J)", bare, fb) == ("J", False)


def test_the_tag_wins_over_every_other_rule():
    bare, fb = letter_regexes(10)
    tag = RS.answer_tag_re(10)
    for reply, want in [("<answer>C</answer>", "C"),
                        ("<answer> I </answer>", "I"),
                        ("<ANSWER>j</ANSWER>", "J"),
                        ("D looks close but <answer>F</answer>", "F")]:
        assert parse_letter(reply, bare, fb, tag) == (want, True), reply


def test_an_untagged_reply_still_parses_when_the_tag_was_asked_for():
    bare, fb = letter_regexes(10)
    tag = RS.answer_tag_re(10)
    assert parse_letter("C", bare, fb, tag) == ("C", False)
    assert parse_letter("I think it is C", bare, fb, tag) == ("C", False)
    assert parse_letter("nothing here", bare, fb, tag) == (None, False)


import json
import types

from decodability_rl.rl import reward_server_decodability as RS


def _one(m):
    rewards, _ = m.get_reward([("cap", [(json.dumps(PAYLOAD), "")])], None, None)
    return rewards[0]


def _bare_model(n_options: int, **overrides):
    m = object.__new__(RS.DecodabilityRewardModel)
    m.n_options = n_options
    m.letters = P.letters_for(n_options)
    m.bare_re, m.letter_re = letter_regexes(n_options)
    m.answer_format = overrides.get("answer_format", "letter")
    m.tag_re = RS.answer_tag_re(n_options) if m.answer_format == "tag" else None
    m.n_tagged = m.n_parsed = 0
    m.picks = dict.fromkeys(m.letters, 0)
    m.n_calls = 0
    m._api = None
    m.tok = types.SimpleNamespace(apply_chat_template=lambda msgs, **kw: msgs[-1]["content"])
    args = dict(
        score_mode="generate", reward_scale=2.0, reward_transform="raw", log_every=10**9,
        number_mask_weight=0.0, batch_size=4, max_new_tokens=16, n_options=n_options,
        answer_format="letter",
    )
    args.update(overrides)
    m.args = types.SimpleNamespace(**args)
    return m


PAYLOAD = {
    "id": 1,
    "true": [1.0, 2.0, 3.0, 4.0],
    "distractors": [[float(i)] * 4 for i in range(11)],
    "decimals": 0,
}


def test_rotation_prompts_use_exactly_k_minus_one_distractors():
    for k in (4, 6, 8, 10, 12):
        m = _bare_model(k)
        prompts = m._rotation_prompts("a caption", dict(PAYLOAD))
        assert len(prompts) == k
        for text in prompts:
            assert text.count(") [") == k
        assert f"[{k - 2}, {k - 2}, {k - 2}, {k - 2}]" in prompts[0]
        assert f"[{k - 1}, {k - 1}, {k - 1}, {k - 1}]" not in prompts[0]


def test_too_few_distractors_is_an_error_not_a_smaller_question():
    m = _bare_model(10)
    thin = dict(PAYLOAD, distractors=PAYLOAD["distractors"][:3])
    try:
        m._rotation_prompts("a caption", thin)
    except ValueError as e:
        assert "distractors" in str(e)
        return
    raise AssertionError("a 10-way server must refuse a 4-way item, not silently ask 4-way")


def test_reward_is_the_mean_over_rotations_and_scales_with_k():
    for k in (4, 6, 8, 10, 12):
        m = _bare_model(k)
        m._score_generate = lambda texts, gold: ([1.0] + [0.0] * (len(texts) - 1), 0)
        rewards, _ = m.get_reward([("cap", [(json.dumps(PAYLOAD), "")])], None, None)
        assert abs(rewards[0] - 2.0 / k) < 1e-9, (k, rewards)


def test_chance_transform_follows_k():
    for k in (4, 12):
        m = _bare_model(k, reward_transform="chance")
        m._score_generate = lambda texts, gold: ([1.0] * len(texts), 0)
        rewards, _ = m.get_reward([("cap", [(json.dumps(PAYLOAD), "")])], None, None)
        assert abs(rewards[0] - 2.0) < 1e-9
        m = _bare_model(k, reward_transform="chance")
        m._score_generate = lambda texts, gold: ([1.0] + [0.0] * (len(texts) - 1), 0)
        rewards, _ = m.get_reward([("cap", [(json.dumps(PAYLOAD), "")])], None, None)
        assert abs(rewards[0]) < 1e-9


def test_centered_puts_every_k_on_one_scale():
    for k in (4, 6, 8, 10, 12):
        m = _bare_model(k, reward_transform="centered")
        m._score_generate = lambda texts, gold: ([1.0] * len(texts), 0)
        assert abs(_one(m) - 2.0) < 1e-9
        m = _bare_model(k, reward_transform="centered")
        m._score_generate = lambda texts, gold: ([1.0] + [0.0] * (len(texts) - 1), 0)
        assert abs(_one(m)) < 1e-9
        m = _bare_model(k, reward_transform="centered")
        m._score_generate = lambda texts, gold: ([0.0] * len(texts), 0)
        assert _one(m) < 0
        m = _bare_model(k, reward_transform="chance")
        m._score_generate = lambda texts, gold: ([0.0] * len(texts), 0)
        assert _one(m) == 0.0


def test_raw_accuracy_is_reported_alongside_the_transformed_reward():
    m = _bare_model(10, reward_transform="centered")
    m._score_generate = lambda texts, gold: ([1.0] + [0.0] * (len(texts) - 1), 0)
    rewards, accuracies = m.get_reward([("cap", [(json.dumps(PAYLOAD), "")])], None, None)
    assert abs(accuracies[0] - 0.1) < 1e-9
    assert abs(rewards[0]) < 1e-9


def test_masking_is_skipped_when_its_weight_is_zero():
    m = _bare_model(4)
    caps = ["peaks at 3650 around step 18"]
    assert m._mask_batch(caps) == (caps, 0, 0)


def test_rloo_cancels_the_chance_floor_but_not_the_scale():
    def rloo(rs):
        n = len(rs)
        return [r - (sum(rs) - r) / (n - 1) for r in rs]

    accs = [0.2, 0.4, 0.6, 1.0]
    for k in (4, 6, 8, 10, 12):
        raw = [2 * a for a in accs]
        centered = [2 * (a - 1 / k) / (1 - 1 / k) for a in accs]
        a_raw, a_cent = rloo(raw), rloo(centered)
        expected = 1 / (1 - 1 / k)
        for r, c in zip(a_raw, a_cent):
            assert abs(c - expected * r) < 1e-9, (k, r, c)
        shifted = rloo([r + 7.0 for r in raw])
        for x, y in zip(a_raw, shifted):
            assert abs(x - y) < 1e-9
    assert [round(1 / (1 - 1 / k), 3) for k in (4, 6, 8, 10)] == [1.333, 1.2, 1.143, 1.111]


def test_per_arm_reward_scale_equalises_the_advantage_across_k():
    def rloo(rs):
        n = len(rs)
        return [r - (sum(rs) - r) / (n - 1) for r in rs]

    def scale_for(k):
        return (2.0 / 0.75) * (1.0 - 1.0 / k)

    assert abs(scale_for(4) - 2.0) < 1e-9
    assert [round(scale_for(k), 4) for k in (6, 8, 10)] == [2.2222, 2.3333, 2.4]

    accs = [0.2, 0.4, 0.6, 1.0]
    reference = rloo([(8 / 3) * a for a in accs])
    for k in (4, 6, 8, 10, 12):
        s = scale_for(k)
        rewards = [s * (a - 1 / k) / (1 - 1 / k) for a in accs]
        for got, want in zip(rloo(rewards), reference):
            assert abs(got - want) < 1e-9, (k, got, want)

    naive = rloo([2.0 * (a - 1 / 10) / (1 - 1 / 10) for a in accs])
    assert any(abs(g - w) > 1e-3 for g, w in zip(naive, reference))
