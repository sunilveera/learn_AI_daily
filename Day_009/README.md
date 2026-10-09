# Day 009 — Encoder, decoder, and cross-attention

`src/day_009/main.py` runs one attention head in three roles: encoder self-attention, causal decoder self-attention, and cross-attention from the decoder onto the encoder. The arrays come from `np.random.seed(42)`, so the script takes no input and prints the same numbers every run.

The encoder sequence has 5 tokens. The decoder sequence has 3. The model width is 8, and this head uses a width of 4.

## What it does

1. Draws the encoder token matrix `X` with shape `(5, 8)` and projects it into `Q`, `K`, and `V` for one head.
2. Runs scaled dot-product attention with no mask. Every encoder token can attend to every other token. The weight matrix is `(5, 5)` and the head output is `(5, 4)`.
3. Draws the decoder token matrix `Y` with shape `(3, 8)` and projects it the same way.
4. Builds a causal mask with `np.triu(..., k=1)`. Positions above the diagonal are set to `-inf` before the softmax, so a decoder token cannot attend to a later token. The weight matrix is `(3, 3)`. Its upper triangle is all zeros, and each row still sums to 1.
5. Projects the decoder head output into queries and the encoder head output into keys and values.
6. Runs cross-attention with no mask. Each decoder token attends over the 5 encoder tokens, so the weight matrix is `(3, 5)` and the output is `(3, 4)`.

Scores are `Q @ K.T / sqrt(4)`. Softmax subtracts the row max before the exponential. Masked positions stay at zero weight because `exp(-inf)` is 0.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_009/main.py
```

The causal decoder weights have zeros above the diagonal. The cross-attention weights have one row per decoder token and one column per encoder token.

```text
encoder weights_1:
 [[7.91509928e-06 1.44675956e-02 3.68469262e-03 6.47050000e-09
  9.81839790e-01]
 [1.38141767e-01 1.28103028e-02 2.74981652e-03 8.39602241e-01
  6.69587212e-03]
 [3.05263834e-01 1.22254475e-03 1.29218510e-03 6.88205098e-01
  4.01633846e-03]
 [1.44193526e-03 3.49981645e-02 5.16800940e-03 6.25902440e-05
  9.58329301e-01]
 [3.90129276e-03 3.34619075e-01 5.50920858e-01 1.08437224e-01
  2.12154979e-03]]
decoder weights_1:
 [[1.00000000e+00 0.00000000e+00 0.00000000e+00]
 [9.99924363e-01 7.56369289e-05 0.00000000e+00]
 [9.78746271e-01 1.86531925e-04 2.10671971e-02]]
cross-attention weights_1:
 [[7.09863033e-01 6.32456309e-18 1.51021149e-17 2.90136967e-01
  2.80401931e-31]
 [7.09834356e-01 6.38542474e-18 1.52454591e-17 2.90165644e-01
  2.83322677e-31]
 [7.08304480e-01 6.07136393e-18 1.43833859e-17 2.91695520e-01
  5.37117985e-31]]
```

The script also prints `X`, `Y`, and each head output. The checks cover the three weight shapes, the output shapes `(5, 4)`, `(3, 4)`, and `(3, 4)`, row sums of 1, and a zero upper triangle on the decoder weights.
