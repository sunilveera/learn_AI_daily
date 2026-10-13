# Day 012 — Backpropagation through a hidden layer

`src/day_012/main.py` trains a two-layer network so a fixed input vector predicts the token `sleeps`. The vocabulary is `cat`, `dog`, `runs`, and `sleeps`. The input `X` is `[0.5, 0.2, 0.8]`. `W1` has shape `(3, 4)` and `W2` has shape `(4, 4)`. Both start random with `np.random.seed(42)`. The learning rate is `0.1`. The script takes no input and prints the same table every run.

## What it does

1. Computes a hidden vector `ReLU(X @ W1)`, then four logits `hidden @ W2`, then a stable softmax.
2. Computes cross-entropy loss as `-log(probability of sleeps)`. The probability is clipped away from `0`.
3. Forms the logit gradient `softmax - one_hot`.
4. Builds the output-layer gradient `np.outer(hidden, gradient_logits)`, which has the same shape as `W2`.
5. Sends that gradient backward: `gradient_logits @ W2.T`, then multiplies by the ReLU derivative. The derivative is `1` where the hidden unit is positive and `0` where it is not.
6. Builds the first-layer gradient `np.outer(X, gradient_z1)`, which has the same shape as `W1`.
7. Updates both matrices: `W = W - learning_rate * gradient`.
8. Repeats for 100 steps. Every 10th step prints the loss and the probability of `sleeps` from the forward pass at the start of that step, before the update.

The loss falls and the probability of `sleeps` rises. At step 100 that probability is about `0.99`.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_012/main.py
```

Example:

```text
| Epoch      | Loss       | Probability |
|------------|------------|-------------|
|     10     |   0.5785   |   0.5608    |
|     20     |   0.2042   |   0.8153    |
|     30     |   0.0914   |   0.9126    |
|     40     |   0.0530   |   0.9483    |
|     50     |   0.0357   |   0.9649    |
|     60     |   0.0263   |   0.9741    |
|     70     |   0.0205   |   0.9797    |
|     80     |   0.0167   |   0.9835    |
|     90     |   0.0139   |   0.9862    |
|    100     |   0.0119   |   0.9882    |
```
