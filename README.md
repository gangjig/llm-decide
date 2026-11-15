# llm-decide

**Ask a model for a decision, not a paragraph.**

`llm-decide` is a plugin for Simon Willison's [`llm`](https://llm.datasette.io/) CLI that adds *typed decision commands*. Instead of prompting a model with a question and scraping its prose for the answer, you ask for one of four decision primitives — the same primitives dedicated "decision models" like Jev are built around — and get clean, machine-readable JSON back:

- **yes-no** — a boolean answer with a probability
- **choice** — one pick from a list *you* define, with a probability
- **score** — a number from 0.0 to 1.0
- **criteria** — one judgment split into atomic yes/no sub-questions, each graded separately, then aggregated

```console
$ llm decide yes-no "Is this ticket a billing issue?" \
    --context "I was charged twice for my subscription this month"
{
  "answer": "yes",
  "probability": 0.93,
  "reason": "The context describes a duplicate charge, which is a billing problem."
}
```

That output pipes straight into `jq`, a shell script, or the next step of an agent pipeline — no regexes over prose, no "the model said *probably yes*" edge cases in your code.

## Install

```console
llm install llm-decide
```

Requires Python 3.10+ and an `llm` installation with at least one model configured (an API key for a hosted model, or a local model plugin — `llm-decide` works with any model `llm` can talk to).

## Usage

Every command takes a question, an optional `--context`, and an optional `-m/--model` (defaults to your configured `llm` default model). If `--context` is omitted and stdin is piped, the context is read from stdin.

### `llm decide yes-no`

```console
$ llm decide yes-no "Does this message need a human review?" \
    --context "User is threatening legal action over a refund"
{
  "answer": "yes",
  "probability": 0.88,
  "reason": "Threatened legal action should be escalated to a human."
}

$ cat user-message.txt | llm decide yes-no "Is this spam?"
{
  "answer": "no",
  "probability": 0.76,
  "reason": "The message is a genuine support request with account details."
}
```

### `llm decide choice`

The returned choice is validated against your list — if the model invents an option that isn't there, the command fails loudly instead of passing garbage downstream.

```console
$ llm decide choice "Which queue should handle this ticket?" \
    --options billing,tech,account \
    --context "The app crashes every time I open the settings page"
{
  "choice": "tech",
  "probability": 0.81,
  "reason": "The report describes a reproducible app crash."
}
```

### `llm decide score`

```console
$ llm decide score "How severe is this bug?" \
    --context "Typo in the footer of the settings page"
{
  "score": 0.08,
  "reason": "Cosmetic issue in a low-traffic area with no functional impact."
}
```

### `llm decide criteria`

Split one judgment into several independent yes/no sub-questions, grade
each one separately, then aggregate — instead of asking for a single
overall score. Breaking a grading standard into atomic yes/no items
reduces grader subjectivity and shows *which* dimension drags the
score down, a pattern Anthropic recommends for building strong
evaluations ([Building strong safety
evaluations](https://www.anthropic.com/research/building-strong-safety-evaluations)).

```console
$ llm decide criteria "Is this support reply good?" \
    --criterion "Does it state the refund amount?" \
    --criterion "Does it give a timeline?" \
    --criterion "Does it apologize for the inconvenience?" \
    --context "We will refund $12 within 5 days."
{
  "question": "Is this support reply good?",
  "criteria": [
    {
      "criterion": "Does it state the refund amount?",
      "answer": "yes",
      "probability": 0.93,
      "reason": "The reply names the $12 refund."
    },
    {
      "criterion": "Does it give a timeline?",
      "answer": "yes",
      "probability": 0.9,
      "reason": "It promises the refund within 5 days."
    },
    {
      "criterion": "Does it apologize for the inconvenience?",
      "answer": "no",
      "probability": 0.88,
      "reason": "There is no apology in the reply."
    }
  ],
  "aggregate": {
    "yes_count": 2,
    "total": 3,
    "yes_fraction": 0.6666666666666666,
    "mean_probability": 0.9033333333333333,
    "score": 0.65,
    "verdict": "yes"
  }
}
```

The aggregation rule is explicit: each criterion contributes its
probability of being satisfied — the stated probability for a `"yes"`
answer, one minus it for a `"no"` answer (the probability is the
model's confidence that its answer is correct). The aggregate `score`
is the mean of those contributions, and the `verdict` is `"yes"` when
the score is >= 0.5. Every criterion is weighted equally, and the
per-criterion results are always shown alongside the aggregate so the
verdict can be audited.

Cost: one model call per criterion — three criteria means three calls
(plus one retry each if a reply is malformed). Each criterion is
judged independently, in its own call, so the judgments cannot anchor
on each other; the price is N calls instead of one.

At least one `--criterion` is required. If any criterion cannot be
answered yes/no even after the retry, the command fails loudly, names
the failing criterion, and prints no partial output.

### Errors

If the model can't be made to produce a valid decision, the command exits with code 1 and a clear message on stderr — it never prints partial or guessed JSON:

```console
$ llm decide yes-no "..." -m some-model
Error: Could not get a valid decision from the model (no JSON object found in the model response). Try re-running, or pick a different model with -m.
```

## How it works

1. The plugin builds a strict prompt for the decision type, naming the exact JSON fields it wants (`answer`/`probability`/`reason`, etc.) and instructing the model to reply with a single JSON object and nothing else.
2. Whatever comes back is **validated and normalized**, because models are sloppy:
   - markdown code fences are stripped, and a JSON object embedded in prose is extracted;
   - probabilities are coerced to numbers and **clamped to [0, 1]** (models happily emit `1.4` or `"very likely"`);
   - boolean answers and `"yes"`/`"no"` strings are normalized to `"yes"`/`"no"`;
   - a `choice` must match one of your options (case-insensitively) or the command errors.
3. If the first reply can't be parsed, the plugin retries **once** with a stricter "JSON only" follow-up prompt. If that also fails, you get the error above.

## Why not just prompt and `jq` it?

You can — for about a day. Then you meet the long tail: the model wraps its JSON in a fence, adds "Sure! Here's the result:", returns a probability of `85` (percent? fraction?), picks an option you never offered, or answers in a sentence. Every script that consumes model output ends up re-implementing the same defensive parsing. `llm-decide` is that parsing, written once, tested, and shared — plus a consistent interface across yes/no, choice, score, and criteria so calling code never changes shape.

## Design notes

- **Single module, one dependency.** The whole plugin is `llm_decide.py` and depends only on `llm` (which brings `click`). There is nothing to configure.
- **Validation over trust.** The model is treated as an unreliable narrator: every field is checked, coerced, or rejected. A loud failure is better than a quiet wrong answer propagating into an automated decision.
- **Probabilities are the model's own estimates.** They express how confident the model *says* it is — see Limitations.

## Limitations

- **The probabilities are not calibrated.** They are the model's self-reported confidence, not a measured frequency. A `0.9` from the model does not mean "right 90% of the time." If you need calibrated probabilities, measure them against your own labeled data (or use a dedicated decision model) — don't trust these numbers blindly for high-stakes thresholds.
- **Quality is bounded by the model.** A weak model makes weak decisions; the plugin only guarantees the *shape* of the output, not its correctness.
- **Criteria are weighted equally.** The aggregate score gives every criterion the same weight; if some criteria matter more than others, weight the per-criterion results yourself from the JSON output.
- **One retry, not infinite patience.** Persistently malformed output fails fast with an error rather than burning tokens on repeated attempts.
- Each decision is a single stateless model call — there is no memory or conversation between invocations beyond the context you pass in.

## Development

```console
pip install -e . pytest
pytest
```

Tests use a fake model with canned responses, so the suite runs fully offline with no API keys.

## Build your own

This repository is small enough to regenerate from a prompt. Paste the
following into an LLM and it should produce a closely similar project:

> Write a plugin for Simon Willison's llm command-line tool that lets
> the user ask a model for a typed decision instead of a paragraph of
> prose, in three primitives: a yes/no answer with a probability, a
> choice from a user-defined list of options with a probability, and a
> numeric score between 0 and 1, each returned as clean machine-readable
> JSON with a short reason. The results pipe straight into jq, shell
> scripts, or the next step of an agent pipeline, with a consistent
> output shape across all three decision types. Because model output is
> sloppy, the plugin validates and normalizes whatever comes back
> (stripping code fences, extracting a JSON object from surrounding
> prose, coercing probabilities into range, and rejecting a choice that
> is not one of the offered options) and retries once with a stricter
> prompt before failing loudly. The design principle is validation over
> trust: a loud error is better than a quietly wrong answer flowing into
> an automated decision.

## License

MIT — see [LICENSE](LICENSE).
