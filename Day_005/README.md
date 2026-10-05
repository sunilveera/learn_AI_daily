# Day 005 — Next-token prediction with DistilGPT-2

This day loads `distilgpt2` and asks it what token should come next. Both scripts read one prompt from the terminal. The model and tokenizer download on the first run.

`src/day_005/top5.py` runs the model once and prints the five most likely next tokens. `src/day_005/greedy_generation.py` keeps picking the single most likely token and appends it, for 10 new tokens.

## What `top5.py` does

1. Tokenizes the prompt into a PyTorch tensor and prints those input IDs.
2. Runs a forward pass and prints the logits shape: batch size, number of tokens, vocabulary size (`50257` for this model).
3. Takes the logits for the last position and turns them into probabilities with softmax.
4. Prints the top 5 tokens and their probabilities.

## What `greedy_generation.py` does

1. Starts from the prompt.
2. For each of 10 steps, tokenizes the text so far, runs the model, and takes the last-position logits.
3. Picks the token with the highest probability (`argmax` after softmax) and appends its decoded text.
4. Prints the growing string after every token.

Each step conditions on the text produced so far, so the model never looks ahead. The decoded token usually includes a leading space, which is why the new words attach cleanly.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`: `transformers` and PyTorch.

From this folder:

```bash
uv sync
```

## Run

Top 5 next tokens:

```bash
uv run python src/day_005/top5.py
```

Greedy continuation:

```bash
uv run python src/day_005/greedy_generation.py
```

Example prompt: `Once upon a time`

`top5.py` prints input IDs for those four tokens, logits of shape `[1, 4, 50257]`, then a table like this:

```text
Top 5 most likely next tokens with their probabilities:
--------------------------------
| Token | Probability |
--------------------------------
|  of        | 0.25134119391441345 |
|  when      | 0.2189478874206543 |
| ,          | 0.15847718715667725 |
|  I         | 0.036167167127132416 |
|  in        | 0.030109139159321785 |
--------------------------------
```

`greedy_generation.py` follows the top choice (` of`) and prints each extension:

```text
Once upon a time of
Once upon a time of war
Once upon a time of war,
Once upon a time of war, the
Once upon a time of war, the United
Once upon a time of war, the United States
Once upon a time of war, the United States was
Once upon a time of war, the United States was the
Once upon a time of war, the United States was the only
Once upon a time of war, the United States was the only country
```
