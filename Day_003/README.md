# Day 003 — Scaled dot-product attention

`src/day_003/main.py` computes one step of scaled dot-product attention with NumPy. Query, key, and value matrices are fixed arrays, so the script takes no input.

Each matrix has three rows. A row is one token, and each token has two features. `Q` and `K` use the same three vectors: `[1, 0]`, `[0, 1]`, and `[1, 1]`. `V` holds the values those tokens contribute: `[10, 0]`, `[0, 10]`, and `[5, 5]`.

## What it does

1. Builds the score matrix `Q @ K.T`. Entry `(i, j)` is how much token `i` attends to token `j`.
2. Divides the scores by `sqrt(d_k)`, where `d_k` is the key width (`2`).
3. Applies softmax across each row so the weights for a token sum to 1.
4. Mixes the value rows with those weights: `attention_weights @ V`.

The third output row is `[5, 5]`. That token's query matches the third key most strongly, and the third value is already `[5, 5]`.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_003/main.py
```

Example:

```text
Scores:
 [[1. 0. 1.]
 [0. 1. 1.]
 [1. 1. 2.]]
Normalized Scores:
 [[0.70710678 0.         0.70710678]
 [0.         0.70710678 0.70710678]
 [0.70710678 0.70710678 1.41421356]]
Attention Weights:
 [[0.40111209 0.19777581 0.40111209]
 [0.19777581 0.40111209 0.40111209]
 [0.24825508 0.24825508 0.50348984]]
Output:
 [[6.01668139 3.98331861]
 [3.98331861 6.01668139]
 [5.         5.        ]]
```
