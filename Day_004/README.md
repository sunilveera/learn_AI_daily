# Day 004 — A transformer block

`src/day_004/main.py` runs one transformer block in NumPy: scaled dot-product attention, a residual connection, layer normalization, and a two-layer feed-forward network. The arrays are fixed, so the script takes no input.

`Q`, `K`, `V`, and the token matrix `X` match Day 003. Three tokens, two features each. `Q`, `K`, and `X` are `[1, 0]`, `[0, 1]`, and `[1, 1]`. `V` is `[10, 0]`, `[0, 10]`, and `[5, 5]`.

`W1` is 2×3 and expands each token to three hidden units. `W2` is 3×2 and maps that hidden vector back to two features.

## What it does

1. Computes attention the same way as Day 003: `Q @ K.T`, divide by `sqrt(2)`, softmax, then multiply by `V`.
2. Adds the attention output back onto `X` (the first residual connection).
3. Layer-normalizes each token over its two features. This version has no learned scale or shift.
4. Runs the feed-forward network: `ReLU(normalized @ W1) @ W2`.
5. Adds that output back onto the normalized residual (the second residual connection).
6. Layer-normalizes again. That matrix is the block output.

The third token stays `[0, 0]` after the first layer norm. Its residual row is `[6, 6]`, so both features share one value and the normalized result is zero. ReLU then keeps the hidden row at zero, and the second residual stays zero as well.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_004/main.py
```

Example:

```text
Attention weights:
 [[0.40111209 0.19777581 0.40111209]
 [0.19777581 0.40111209 0.40111209]
 [0.24825508 0.24825508 0.50348984]]
Attention output:
 [[6.01668139 3.98331861]
 [3.98331861 6.01668139]
 [5.         5.        ]]
After first residual:
 [[7.01668139 3.98331861]
 [3.98331861 7.01668139]
 [6.         6.        ]]
After first LayerNorm:
 [[ 0.99999783 -0.99999783]
 [-0.99999783  0.99999783]
 [ 0.          0.        ]]
Feed-forward hidden:
 [[0.49999891 0.         0.99999783]
 [0.         1.99999565 0.        ]
 [0.         0.         0.        ]]
Feed-forward output:
 [[0.99999783 0.49999891]
 [0.         1.99999565]
 [0.         0.        ]]
After second residual:
 [[ 1.99999565 -0.49999891]
 [-0.99999783  2.99999348]
 [ 0.          0.        ]]
Final output:
 [[ 0.9999968  -0.9999968 ]
 [-0.99999875  0.99999875]
 [ 0.          0.        ]]
```
