# Day 006 — Temperature, top-k, and top-p sampling

`src/day_006/main.py` continues a prompt with `distilgpt2`, but it samples the next token instead of always taking the most likely one. The model and tokenizer download on the first run.

The script reads one prompt, then runs three experiments. Each experiment adds 20 tokens and prints the text after every token. Experiment B starts from A's finished text, and experiment C starts from B's.

| Experiment | Temperature | Top-k | Top-p |
| --- | --- | --- | --- |
| A | 0.3 | 20 | 0.8 |
| B | 1.0 | 40 | 0.9 |
| C | 1.5 | 100 | 0.95 |

A lower temperature makes the distribution sharper, so the samples stay closer to the likely tokens. A higher temperature flattens it. A larger top-k or top-p also leaves more tokens available to sample.

## What it does

For each new token:

1. Tokenizes the text so far and runs the model.
2. Divides the last-position logits by the temperature.
3. **Top-k:** keeps the `k` largest logits and sets the rest to `-inf`.
4. **Top-p:** sorts the remaining tokens from most likely to least likely, takes a running total of their probabilities, and drops tokens once that total has already passed `p`. The token that crosses `p` stays.
5. Turns the filtered logits into probabilities with softmax and draws one token with `torch.multinomial`.
6. Appends the decoded token and prints the growing string.

Because the draw is random, a second run of the same prompt will differ.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`: `transformers` and PyTorch.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_006/main.py
```

Example prompt: `Once upon a`

One run ended each experiment like this. The script also prints every intermediate step, and the last line of an experiment is printed again before the next one starts.

```text
Experiment A: Temperature = 0.3, Top K = 20, Top P = 0.8
Once upon a second glance, the world is a very different place. The world is a very different place. The

Experiment B: Temperature = 1.0, Top K = 40, Top P = 0.9
Once upon a second glance, the world is a very different place. The world is a very different place. The world is not so much like a human, it is much better than a bird, because it is

Experiment C: Temperature = 1.5, Top K = 100, Top P = 0.95
Once upon a second glance, the world is a very different place. The world is a very different place. The world is not so much like a human, it is much better than a bird, because it is extremely new in its architecture. Moreover. With the whole community, this could mean that many people living
```
