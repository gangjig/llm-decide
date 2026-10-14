"""llm-decide: typed decision commands for the `llm` CLI.

Instead of asking a model a question and scraping its prose for an answer,
`llm decide` asks for a *decision* — a yes/no with a probability, a choice
from a fixed list, or a score — and prints clean, machine-readable JSON.

    llm decide yes-no "Is this ticket a billing issue?" --context "..."
    llm decide choice "Which queue?" --options billing,tech,account
    llm decide score "How severe is this bug?" --context "..."

The plugin prompts the model for a strict JSON object, then validates and
normalizes whatever comes back (markdown fences stripped, probabilities
clamped to [0, 1], choices checked against the allowed list). Unparseable
output gets one retry with a stricter prompt, then a clear error.
"""

import json
import re
import sys

import click
import llm

__version__ = "0.1.0"

_JSON_RULES = (
    "Respond with a single JSON object and nothing else — no markdown, "
    "no code fences, no explanation outside the JSON."
)


class DecideError(click.ClickException):
    """A user-facing error: message on stderr, exit code 1."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _context_block(context):
    return "\n\nContext:\n{}".format(context) if context else ""


def _yes_no_prompt(question, context):
    return (
        "Answer the following yes/no question.\n\n"
        "Question: {}{}\n\n"
        "Return JSON with exactly these fields:\n"
        '- "answer": true or false\n'
        '- "probability": your confidence that the answer is correct, '
        "a number between 0 and 1\n"
        '- "reason": one short sentence explaining the answer\n\n'
        "{}"
    ).format(question, _context_block(context), _JSON_RULES)


def _choice_prompt(question, options, context):
    return (
        "Choose exactly one of the following options.\n\n"
        "Options: {}\n\n"
        "Question: {}{}\n\n"
        "Return JSON with exactly these fields:\n"
        '- "choice": one of the options above, copied exactly\n'
        '- "probability": your confidence that this is the right choice, '
        "a number between 0 and 1\n"
        '- "reason": one short sentence explaining the choice\n\n'
        "{}"
    ).format(", ".join(options), question, _context_block(context), _JSON_RULES)


def _score_prompt(question, context):
    return (
        "Rate the following on a scale from 0 to 1, where 0 is the lowest "
        "possible value of whatever is being asked about and 1 is the "
        "highest.\n\n"
        "Question: {}{}\n\n"
        "Return JSON with exactly these fields:\n"
        '- "score": a number between 0 and 1\n'
        '- "reason": one short sentence explaining the score\n\n'
        "{}"
    ).format(question, _context_block(context), _JSON_RULES)


def _retry_prompt(original_prompt):
    return (
        original_prompt
        + "\n\nIMPORTANT: your previous reply could not be parsed as JSON. "
        "Reply with ONLY the JSON object — no prose, no markdown fences."
    )


# ---------------------------------------------------------------------------
# Response parsing / normalization
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text):
    """Pull a JSON object out of a model response, tolerating fences/prose."""
    if not text or not text.strip():
        raise ValueError("the model returned an empty response")
    candidate = text.strip()
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in the model response")
        try:
            data = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                "could not parse JSON from the model response: {}".format(exc)
            ) from exc
    if not isinstance(data, dict):
        raise ValueError("the model response was JSON, but not an object")
    return data


def _coerce_probability(value, field="probability"):
    """Coerce a model-supplied number to a float clamped to [0, 1]."""
    if isinstance(value, bool):
        raise ValueError("field {!r} was a boolean, not a number".format(field))
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError(
                "field {!r} was not a number: {!r}".format(field, value)
            ) from None
    if not isinstance(value, (int, float)):
        raise ValueError("field {!r} was not a number: {!r}".format(field, value))
    return max(0.0, min(1.0, float(value)))


def _require(data, field):
    if field not in data or data[field] is None:
        raise ValueError("response JSON is missing the {!r} field".format(field))
    return data[field]


def _parse_yes_no(text):
    data = _extract_json(text)
    answer = _require(data, "answer")
    if isinstance(answer, bool):
        normalized = "yes" if answer else "no"
    elif isinstance(answer, str) and answer.strip().lower() in (
        "yes",
        "no",
        "true",
        "false",
    ):
        normalized = "yes" if answer.strip().lower() in ("yes", "true") else "no"
    else:
        raise ValueError("field 'answer' was not a yes/no value: {!r}".format(answer))
    return {
        "answer": normalized,
        "probability": _coerce_probability(_require(data, "probability")),
        "reason": str(data.get("reason", "")),
    }


def _make_choice_parser(options):
    def parse(text):
        data = _extract_json(text)
        raw_choice = str(_require(data, "choice")).strip()
        for option in options:
            if raw_choice == option or raw_choice.lower() == option.lower():
                choice = option
                break
        else:
            raise DecideError(
                "Model chose {!r}, which is not one of the allowed options: "
                "{}".format(raw_choice, ", ".join(options))
            )
        return {
            "choice": choice,
            "probability": _coerce_probability(_require(data, "probability")),
            "reason": str(data.get("reason", "")),
        }

    return parse


def _parse_score(text):
    data = _extract_json(text)
    return {
        "score": _coerce_probability(_require(data, "score"), field="score"),
        "reason": str(data.get("reason", "")),
    }


# ---------------------------------------------------------------------------
# Model interaction
# ---------------------------------------------------------------------------


def _ask(model_id, prompt, parser):
    """Run a decision prompt, parsing the reply; retry once on bad output."""
    try:
        model = llm.get_model(model_id)
    except Exception as exc:  # unknown model id, missing plugin, ...
        raise DecideError("Could not load model: {}".format(exc)) from exc
    last_error = None
    for attempt_prompt in (prompt, _retry_prompt(prompt)):
        try:
            raw = model.prompt(attempt_prompt).text()
        except Exception as exc:  # missing API key, network error, ...
            raise DecideError("Model call failed: {}".format(exc)) from exc
        try:
            return parser(raw)
        except ValueError as exc:
            last_error = exc
    raise DecideError(
        "Could not get a valid decision from the model ({}). "
        "Try re-running, or pick a different model with -m.".format(last_error)
    )


def _resolve_context(context):
    """Use --context if given; otherwise read piped stdin, if any."""
    if context is not None:
        return context
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        return data.strip() or None
    return None


def _emit(result):
    click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# CLI: the `decide` command group
# ---------------------------------------------------------------------------


@click.group()
def decide():
    """Ask a model for a typed decision (yes/no, choice, score) as JSON."""


@decide.command(name="yes-no")
@click.argument("question")
@click.option("--context", default=None, help="Context for the question. "
              "If omitted and stdin is piped, the context is read from stdin.")
@click.option("-m", "--model", default=None,
              help="Model to use (defaults to the llm default model).")
def yes_no(question, context, model):
    """Answer a yes/no QUESTION, with a probability for the answer."""
    result = _ask(
        model,
        _yes_no_prompt(question, _resolve_context(context)),
        _parse_yes_no,
    )
    _emit(result)


@decide.command(name="choice")
@click.argument("question")
@click.option("--options", required=True,
              help="Comma-separated list of allowed choices.")
@click.option("--context", default=None, help="Context for the question. "
              "If omitted and stdin is piped, the context is read from stdin.")
@click.option("-m", "--model", default=None,
              help="Model to use (defaults to the llm default model).")
def choice(question, options, context, model):
    """Pick one of --options for a QUESTION, with a probability."""
    opts = [o.strip() for o in options.split(",") if o.strip()]
    if len(opts) < 2:
        raise click.UsageError("--options must list at least two choices")
    result = _ask(
        model,
        _choice_prompt(question, opts, _resolve_context(context)),
        _make_choice_parser(opts),
    )
    _emit(result)


@decide.command(name="score")
@click.argument("question")
@click.option("--context", default=None, help="Context for the question. "
              "If omitted and stdin is piped, the context is read from stdin.")
@click.option("-m", "--model", default=None,
              help="Model to use (defaults to the llm default model).")
def score(question, context, model):
    """Score a QUESTION from 0.0 to 1.0."""
    result = _ask(
        model,
        _score_prompt(question, _resolve_context(context)),
        _parse_score,
    )
    _emit(result)


@llm.hookimpl
def register_commands(cli):
    cli.add_command(decide)
