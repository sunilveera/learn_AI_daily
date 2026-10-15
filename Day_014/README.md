# Day 014 — Learning-rate schedules

`src/day_014/main.py` minimizes `(w - 3)²` from `w = 0` with momentum, and changes only the learning-rate schedule. The gradient is `2 * (w - 3)`. Momentum uses `beta = 0.9`: `velocity = 0.9 * velocity + learning_rate * gradient`, then `w = w - velocity`. Each schedule runs for 100 steps from an initial learning rate of `0.1`.

## What it does

1. **Constant.** The learning rate stays `0.1` on every step.
2. **Step decay.** The rate starts at `0.1` and is multiplied by `0.5` when `(iteration - 1)` is divisible by `20`. That happens on steps 21, 41, 61, and 81, so the rate becomes `0.05`, `0.025`, `0.0125`, and then `0.00625`.
3. **Cosine annealing.** The rate is `(0.1 / 2) * (1 + cos(pi * iteration / 100))`. It starts just under `0.1` on step 1, is `0.05` on step 50, and is `0` on step 100.

The script then opens three matplotlib windows, one after another. Close a window to see the next one.

- Loss against iteration, with a logarithmic y-axis and a dashed line at `0`.
- Weight against iteration, with a dashed line at the minimum `w = 3`.
- Learning rate against iteration, with a dashed line at `0`.

The horizontal axis is numbered `0` through `99`, which is steps `1` through `100`.

After 100 steps the losses are about `7.32e-5` for the constant rate, `9.04e-4` for step decay, and `2.60e-3` for cosine annealing. The constant schedule is the same momentum update as Day 013, so it finishes at the same weight, about `3.009`.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day needs NumPy and matplotlib.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_014/main.py
```

| Step | Constant | Step decay | Cosine annealing |
| --- | --- | --- | --- |
| 1 | 0.1 | 0.1 | 0.09998 |
| 21 | 0.1 | 0.05 | 0.0895 |
| 41 | 0.1 | 0.025 | 0.0639 |
| 61 | 0.1 | 0.0125 | 0.0331 |
| 81 | 0.1 | 0.00625 | 0.00865 |
| 100 | 0.1 | 0.00625 | 0 |

Loss at the last step:

| Schedule | Weight | Loss |
| --- | --- | --- |
| Constant | 3.00855 | 7.32e-5 |
| Step decay | 2.96993 | 9.04e-4 |
| Cosine annealing | 3.05100 | 2.60e-3 |
