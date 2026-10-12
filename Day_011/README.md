# Day 011 — One training step, repeated

`src/day_011/main.py` trains a single weight matrix so a fixed input vector predicts the token `sleeps`. The vocabulary is `cat`, `dog`, `runs`, and `sleeps`. The input `X` is the vector `[0.5, 0.2, 0.8]`. `W` starts random with `np.random.seed(42)` and has shape `(3, 4)`. The learning rate is `0.1`. The script takes no input and prints the same table every run.

## What it does

1. Builds a one-hot target with a `1` at `sleeps`.
2. On each step, multiplies `X` by `W` to get four logits and turns them into probabilities with a stable softmax.
3. Computes cross-entropy loss as `-log(probability of sleeps)`. The probability is clipped away from `0` so the log stays finite.
4. Forms the logit gradient `softmax - one_hot`, then the weight gradient with `np.outer(X, gradient_logits)`. That gradient has the same shape as `W`.
5. Updates the weights with gradient descent: `W = W - learning_rate * gradient_W`.
6. Repeats for iterations `0` through `100`. The printed loss and probability come from the forward pass at the start of that iteration, before the update. Iteration `0` is the random initialization. The table shows every 10th iteration.

The loss falls and the probability of `sleeps` rises. By iteration `100` that probability is about `0.91`.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_011/main.py
```

Example:

```text
Target token: sleeps
| Iteration  |    Loss    | Probability |
|------------|------------|-------------|
|     0      |   1.1146   |   0.3281    |
|    10      |   0.6829   |   0.5051    |
|    20      |   0.4521   |   0.6363    |
|    30      |   0.3241   |   0.7232    |
|    40      |   0.2474   |   0.7808    |
|    50      |   0.1978   |   0.8205    |
|    60      |   0.1638   |   0.8489    |
|    70      |   0.1391   |   0.8701    |
|    80      |   0.1207   |   0.8863    |
|    90      |   0.1063   |   0.8992    |
|   100      |   0.0949   |   0.9095    |
```

The script prints a separator line before every row.
