# Day 010 — A tiny decoder that predicts the next token

`src/day_010/main.py` wires the earlier pieces into one decoder-style block and uses it to score the next token after `the cat`. Every weight matrix is random, drawn with `np.random.seed(42)`. The script takes no input and prints the same numbers every run. Nothing here is trained, so the winning token is just the one this seed ranks highest.

The vocabulary is `the`, `cat`, `sleeps`, `runs`, `dog`, `eats`, `a`, `on`, `mat`, `.`. The sequence is two tokens. The model width is 8, with two heads of width 4 and a feed-forward width of 16.

## What it does

1. Looks up embeddings for `the` and `cat` in a `(10, 8)` table and adds a `(2, 8)` positional table.
2. Projects that sum into query, key, and value for each head.
3. Runs scaled dot-product attention with one causal mask shared by both heads. The mask is `np.triu(..., k=1)`, so position 0 cannot attend to position 1. Each weight row sums to 1, and every masked entry is 0.
4. Concatenates the head outputs and projects them with `W_o` back to width 8.
5. Adds the residual connection with the embedded input and applies layer norm.
6. Runs a feed-forward network, `ReLU(x @ W_1) @ W_2`, with shapes `(8, 16)` and `(16, 8)`.
7. Adds a second residual connection and applies layer norm again. Each token's features then have mean 0.
8. Projects the block output onto the 10-word vocabulary. The last row is the distribution for the token after `the cat`.
9. Softmaxes that row, after subtracting its max, and prints each token with its probability. `argmax` picks the predicted token.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_010/main.py
```

The causal weights give the first token no attention on the second token:

```text
Head 1 attention weights:
 [[1.         0.        ]
 [0.00123014 0.99876986]]
Head 2 attention weights:
 [[1.00000000e+00 0.00000000e+00]
 [2.09742955e-19 1.00000000e+00]]
```

The next-token distribution for this seed:

```text
| Token ID | Token String | Softmax Logit |
--------------------------------
| 0        | the          | 0.00  % |
| 1        | cat          | 0.25  % |
| 2        | sleeps       | 95.83 % |
| 3        | runs         | 1.17  % |
| 4        | dog          | 0.06  % |
| 5        | eats         | 0.03  % |
| 6        | a            | 1.81  % |
| 7        | on           | 0.71  % |
| 8        | mat          | 0.01  % |
| 9        | .            | 0.14  % |
--------------------------------
Predicted token ID is [2] and token string is 'sleeps'
```

The script also prints the embedded input, the attention output, and both residual-and-layer-norm stages.
