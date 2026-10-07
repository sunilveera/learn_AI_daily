# Day 007 — Sinusoidal positional encoding

`src/day_007/main.py` builds the sinusoidal position vectors from "Attention Is All You Need" and adds them to a fixed token embedding. The arrays are fixed, so the script takes no input.

The sequence is three tokens, `very very good`, each with 8 features. The two `very` rows are the same embedding. Position is the only thing that can tell them apart.

## What it does

1. Builds a position vector of length 3 and a dimension index of length 8.
2. Computes an angle rate for each dimension: `1 / 10000 ^ (2 * (d // 2) / 8)`. Neighboring even and odd columns share a rate.
3. Fills even columns with `sin(position * angle_rate)` and odd columns with `cos(position * angle_rate)`.
4. Adds that matrix to the token embedding.
5. Checks that both matrices are `(3, 8)`, that the two `very` embeddings match, and that the summed rows do not.

Position 0 is `sin(0)` and `cos(0)`, so its encoding is `0` in every even column and `1` in every odd column. After the addition, the first `very` and the second `very` differ.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_007/main.py
```

Example:

```text
Positional Encoding:

[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.00000000e+00
   0.00000000e+00  1.00000000e+00  0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  9.98334166e-02  9.95004165e-01
   9.99983333e-03  9.99950000e-01  9.99999833e-04  9.99999500e-01]
 [ 9.09297427e-01 -4.16146837e-01  1.98669331e-01  9.80066578e-01
   1.99986667e-02  9.99800007e-01  1.99999867e-03  9.99998000e-01]]

Token Embedding + Positional Encoding:

[[0.1        1.2        0.3        1.4        0.5        1.6
  0.7        1.8       ]
 [0.94147098 0.74030231 0.39983342 1.39500417 0.50999983 1.59995
  0.701      1.7999995 ]
 [1.80929743 0.58385316 1.29866933 2.18006658 1.31999867 2.39980001
  1.502      2.599998  ]]

Token Embedding + Positional Encoding (rounded to 3 decimal places):

[[0.1   1.2   0.3   1.4   0.5   1.6   0.7   1.8  ]
 [0.941 0.74  0.4   1.395 0.51  1.6   0.701 1.8  ]
 [1.809 0.584 1.299 2.18  1.32  2.4   1.502 2.6  ]]
```

The rounded print is only for display. The checks use the full-precision sum.
