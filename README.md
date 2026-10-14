# llm-decide

**Ask a model for a decision, not a paragraph.**

`llm-decide` is a plugin for Simon Willison's [`llm`](https://llm.datasette.io/) CLI that adds *typed decision commands*. Instead of prompting a model with a question and scraping its prose for the answer, you ask for one of three decision primitives — the same primitives dedicated "decision models" like Jev are built around — and get clean, machine-readable JSON back:

- **yes-no** — a boolean answer with a probability
- **choice** — one pick from a list *you* define, with a probability
- **score** — a number from 0.0 to 1.0

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

You can — for about a day. Then you meet the long tail: the model wraps its JSON in a fence, adds "Sure! Here's the result:", returns a probability of `85` (percent? fraction?), picks an option you never offered, or answers in a sentence. Every script that consumes model output ends up re-implementing the same defensive parsing. `llm-decide` is that parsing, written once, tested, and shared — plus a consistent interface across yes/no, choice, and score so calling code never changes shape.

## Design notes

- **Single module, one dependency.** The whole plugin is `llm_decide.py` and depends only on `llm` (which brings `click`). There is nothing to configure.
- **Validation over trust.** The model is treated as an unreliable narrator: every field is checked, coerced, or rejected. A loud failure is better than a quiet wrong answer propagating into an automated decision.
- **Probabilities are the model's own estimates.** They express how confident the model *says* it is — see Limitations.

## Limitations

- **The probabilities are not calibrated.** They are the model's self-reported confidence, not a measured frequency. A `0.9` from the model does not mean "right 90% of the time." If you need calibrated probabilities, measure them against your own labeled data (or use a dedicated decision model) — don't trust these numbers blindly for high-stakes thresholds.
- **Quality is bounded by the model.** A weak model makes weak decisions; the plugin only guarantees the *shape* of the output, not its correctness.
- **One retry, not infinite patience.** Persistently malformed output fails fast with an error rather than burning tokens on repeated attempts.
- Each decision is a single stateless model call — there is no memory or conversation between invocations beyond the context you pass in.

## Development

```console
pip install -e . pytest
pytest
```

Tests use a fake model with canned responses, so the suite runs fully offline with no API keys.

## License

MIT — see [LICENSE](LICENSE).
