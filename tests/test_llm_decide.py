"""Tests for llm-decide.

No network or API keys: `llm.get_model` is monkeypatched (in the llm_decide
module namespace) with a fake model that returns canned responses.
"""

import json

import pytest
from click.testing import CliRunner

import llm_decide
from llm_decide import (
    DecideError,
    _aggregate_criteria,
    _coerce_probability,
    _extract_json,
    _parse_score,
    _parse_yes_no,
    decide,
)


class FakeResponse:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class FakeModel:
    """Returns canned responses in order; records the prompts it was given."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def prompt(self, prompt_text):
        self.prompts.append(prompt_text)
        if not self.responses:
            raise AssertionError("FakeModel ran out of canned responses")
        return FakeResponse(self.responses.pop(0))


@pytest.fixture
def install_model(monkeypatch):
    """Return a helper that installs a FakeModel with the given responses."""

    def _install(*responses):
        fake = FakeModel(*responses)
        seen = {}

        def fake_get_model(model_id=None):
            seen["model_id"] = model_id
            return fake

        monkeypatch.setattr(llm_decide.llm, "get_model", fake_get_model)
        return fake, seen

    return _install


def combined_output(result):
    out = result.output or ""
    try:
        err = result.stderr or ""
    except Exception:
        err = ""
    return out + err


YES_NO_JSON = json.dumps(
    {"answer": True, "probability": 0.87, "reason": "The word 'invoice' appears."}
)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_embedded_in_prose():
    text = 'Sure! Here is the result: {"a": 1} Hope that helps.'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_garbage_raises():
    with pytest.raises(ValueError):
        _extract_json("I cannot answer that.")


def test_extract_json_non_object_raises():
    with pytest.raises(ValueError):
        _extract_json("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Probability coercion / parsers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.5, 0.5),
        (1, 1.0),
        (1.7, 1.0),  # clamped from above
        (-0.2, 0.0),  # clamped from below
        ("0.42", 0.42),  # numeric string accepted
    ],
)
def test_coerce_probability(raw, expected):
    assert _coerce_probability(raw) == expected


def test_coerce_probability_rejects_non_number():
    with pytest.raises(ValueError):
        _coerce_probability("very likely")


def test_parse_yes_no_boolean_answer():
    result = _parse_yes_no(YES_NO_JSON)
    assert result == {
        "answer": "yes",
        "probability": 0.87,
        "reason": "The word 'invoice' appears.",
    }


def test_parse_yes_no_string_answer():
    result = _parse_yes_no('{"answer": "No", "probability": 0.6, "reason": "x"}')
    assert result["answer"] == "no"
    assert result["probability"] == 0.6


def test_parse_yes_no_missing_field_raises():
    with pytest.raises(ValueError):
        _parse_yes_no('{"answer": true}')


def test_parse_score():
    result = _parse_score('{"score": 0.25, "reason": "minor typo"}')
    assert result == {"score": 0.25, "reason": "minor typo"}


# ---------------------------------------------------------------------------
# CLI behaviour (via CliRunner, fake model)
# ---------------------------------------------------------------------------


def test_register_commands_group_shape():
    assert set(decide.commands) == {"yes-no", "choice", "score", "criteria"}


def test_yes_no_happy_path(install_model):
    install_model(YES_NO_JSON)
    result = CliRunner().invoke(
        decide,
        ["yes-no", "Is this ticket a billing issue?", "--context", "My invoice is wrong"],
    )
    assert result.exit_code == 0, combined_output(result)
    assert json.loads(result.output) == {
        "answer": "yes",
        "probability": 0.87,
        "reason": "The word 'invoice' appears.",
    }


def test_yes_no_fenced_json(install_model):
    install_model("```json\n{}\n```".format(YES_NO_JSON))
    result = CliRunner().invoke(decide, ["yes-no", "Is this billing?"])
    assert result.exit_code == 0, combined_output(result)
    assert json.loads(result.output)["answer"] == "yes"


def test_yes_no_probability_is_clamped(install_model):
    install_model('{"answer": false, "probability": 1.4, "reason": "definitely not"}')
    result = CliRunner().invoke(decide, ["yes-no", "Is this billing?"])
    assert result.exit_code == 0, combined_output(result)
    data = json.loads(result.output)
    assert data["answer"] == "no"
    assert data["probability"] == 1.0


def test_choice_happy_path(install_model):
    install_model('{"choice": "tech", "probability": 0.7, "reason": "It is a crash."}')
    result = CliRunner().invoke(
        decide,
        [
            "choice",
            "Which queue should get this ticket?",
            "--options",
            "billing,tech,account",
            "--context",
            "The app crashes on launch",
        ],
    )
    assert result.exit_code == 0, combined_output(result)
    assert json.loads(result.output) == {
        "choice": "tech",
        "probability": 0.7,
        "reason": "It is a crash.",
    }


def test_choice_case_insensitive_match(install_model):
    install_model('{"choice": "Tech", "probability": 0.7, "reason": "r"}')
    result = CliRunner().invoke(
        decide, ["choice", "Which queue?", "--options", "billing,tech,account"]
    )
    assert result.exit_code == 0, combined_output(result)
    assert json.loads(result.output)["choice"] == "tech"


def test_choice_invalid_option_errors(install_model):
    install_model('{"choice": "sales", "probability": 0.9, "reason": "r"}')
    result = CliRunner().invoke(
        decide, ["choice", "Which queue?", "--options", "billing,tech,account"]
    )
    assert result.exit_code == 1
    assert "not one of the allowed options" in combined_output(result)


def test_score_happy_path(install_model):
    install_model('{"score": 0.9, "reason": "Data loss on startup."}')
    result = CliRunner().invoke(
        decide, ["score", "How severe is this bug?", "--context", "Users lose data"]
    )
    assert result.exit_code == 0, combined_output(result)
    assert json.loads(result.output) == {
        "score": 0.9,
        "reason": "Data loss on startup.",
    }


def test_parse_failure_retries_then_succeeds(install_model):
    fake, _ = install_model("Sorry, I think yes, probably.", YES_NO_JSON)
    result = CliRunner().invoke(decide, ["yes-no", "Is this billing?"])
    assert result.exit_code == 0, combined_output(result)
    assert json.loads(result.output)["answer"] == "yes"
    # The retry prompt must demand JSON only:
    assert len(fake.prompts) == 2
    assert "ONLY the JSON object" in fake.prompts[1]


def test_parse_failure_errors_after_retry(install_model):
    install_model("total nonsense", "still not json")
    result = CliRunner().invoke(decide, ["yes-no", "Is this billing?"])
    assert result.exit_code == 1
    assert "Could not get a valid decision" in combined_output(result)


def test_context_read_from_stdin(install_model):
    fake, _ = install_model(YES_NO_JSON)
    result = CliRunner().invoke(
        decide, ["yes-no", "Is this billing?"], input="My invoice total is wrong"
    )
    assert result.exit_code == 0, combined_output(result)
    assert "My invoice total is wrong" in fake.prompts[0]


def test_model_option_passed_through(install_model):
    _, seen = install_model(YES_NO_JSON)
    result = CliRunner().invoke(decide, ["yes-no", "Q?", "-m", "some-model"])
    assert result.exit_code == 0, combined_output(result)
    assert seen["model_id"] == "some-model"


def test_ask_raises_decide_error_type(install_model):
    install_model("junk", "junk")
    with pytest.raises(DecideError):
        llm_decide._ask(None, "prompt", _parse_yes_no)


# ---------------------------------------------------------------------------
# criteria command
# ---------------------------------------------------------------------------

CRITERIA_RESPONSES = [
    '{"answer": true, "probability": 0.9, "reason": "It names the amount."}',
    '{"answer": true, "probability": 0.8, "reason": "It gives a timeline."}',
    '{"answer": false, "probability": 0.7, "reason": "No apology present."}',
]

CRITERIA_FLAGS = [
    "--criterion",
    "Does it state the refund amount?",
    "--criterion",
    "Does it give a timeline?",
    "--criterion",
    "Does it apologize?",
]


def test_criteria_happy_path(install_model):
    install_model(*CRITERIA_RESPONSES)
    result = CliRunner().invoke(
        decide,
        ["criteria", "Is this support reply good?", *CRITERIA_FLAGS,
         "--context", "We will refund $12 within 5 days."],
    )
    assert result.exit_code == 0, combined_output(result)
    data = json.loads(result.output)
    assert data["question"] == "Is this support reply good?"
    assert [c["criterion"] for c in data["criteria"]] == [
        "Does it state the refund amount?",
        "Does it give a timeline?",
        "Does it apologize?",
    ]
    assert [c["answer"] for c in data["criteria"]] == ["yes", "yes", "no"]
    assert [c["probability"] for c in data["criteria"]] == [0.9, 0.8, 0.7]
    assert data["criteria"][0]["reason"] == "It names the amount."
    agg = data["aggregate"]
    assert agg["yes_count"] == 2
    assert agg["total"] == 3
    assert agg["yes_fraction"] == pytest.approx(2 / 3)
    assert agg["mean_probability"] == pytest.approx(0.8)
    # Score: (0.9 + 0.8 + (1 - 0.7)) / 3 = 2.0 / 3
    assert agg["score"] == pytest.approx(2 / 3)
    assert agg["verdict"] == "yes"


def test_criteria_verdict_uses_score_not_majority(install_model):
    # One yes at 0.6, one no at 0.9: the yes fraction is 0.5, but the
    # score (0.6 + (1 - 0.9)) / 2 = 0.35 is below 0.5, so "no".
    install_model(
        '{"answer": true, "probability": 0.6, "reason": "r"}',
        '{"answer": false, "probability": 0.9, "reason": "r"}',
    )
    result = CliRunner().invoke(
        decide,
        ["criteria", "Q?", "--criterion", "A?", "--criterion", "B?"],
    )
    assert result.exit_code == 0, combined_output(result)
    agg = json.loads(result.output)["aggregate"]
    assert agg["yes_fraction"] == pytest.approx(0.5)
    assert agg["score"] == pytest.approx(0.35)
    assert agg["verdict"] == "no"


def test_criteria_one_call_per_criterion(install_model):
    fake, _ = install_model(*CRITERIA_RESPONSES)
    result = CliRunner().invoke(
        decide,
        ["criteria", "Is this reply good?", *CRITERIA_FLAGS,
         "--context", "Refund context here"],
    )
    assert result.exit_code == 0, combined_output(result)
    assert len(fake.prompts) == 3
    criteria = [
        "Does it state the refund amount?",
        "Does it give a timeline?",
        "Does it apologize?",
    ]
    for prompt, criterion in zip(fake.prompts, criteria):
        assert criterion in prompt
        assert "Is this reply good?" in prompt
        assert "Refund context here" in prompt


def test_criteria_requires_at_least_one_criterion():
    result = CliRunner().invoke(decide, ["criteria", "Is this good?"])
    assert result.exit_code == 2
    assert "at least one --criterion" in combined_output(result)


def test_criteria_blank_criterion_rejected():
    result = CliRunner().invoke(
        decide, ["criteria", "Q?", "--criterion", "   "]
    )
    assert result.exit_code == 2
    assert "at least one --criterion" in combined_output(result)


def test_criteria_invalid_answer_errors_and_names_criterion(install_model):
    install_model(
        '{"answer": true, "probability": 0.9, "reason": "r"}',
        '{"answer": "maybe", "probability": 0.5, "reason": "r"}',
        '{"answer": "maybe", "probability": 0.5, "reason": "r"}',
    )
    result = CliRunner().invoke(
        decide,
        ["criteria", "Q?", "--criterion", "First?", "--criterion", "Second?"],
    )
    assert result.exit_code == 1
    output = combined_output(result)
    assert "Could not get a valid decision" in output
    assert "Second?" in output
    # No partial output: the successful first criterion is not printed.
    assert '"criteria"' not in output


def test_aggregate_criteria_all_yes():
    agg = _aggregate_criteria(
        [
            {"answer": "yes", "probability": 1.0, "reason": ""},
            {"answer": "yes", "probability": 0.5, "reason": ""},
        ]
    )
    assert agg == {
        "yes_count": 2,
        "total": 2,
        "yes_fraction": 1.0,
        "mean_probability": 0.75,
        "score": 0.75,
        "verdict": "yes",
    }


def test_aggregate_criteria_tie_is_yes():
    # A score of exactly 0.5 verdicts "yes" (the rule is >= 0.5).
    agg = _aggregate_criteria(
        [
            {"answer": "yes", "probability": 0.5, "reason": ""},
            {"answer": "no", "probability": 0.5, "reason": ""},
        ]
    )
    assert agg["score"] == pytest.approx(0.5)
    assert agg["verdict"] == "yes"


def test_aggregate_criteria_empty_raises():
    with pytest.raises(ValueError):
        _aggregate_criteria([])
