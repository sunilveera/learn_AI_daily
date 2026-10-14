# Day 013 — SGD, momentum, and Adam

`src/day_013/main.py` minimizes the loss `(w - 3)²`, starting from `w = 0`, with three optimizers. The gradient is `2 * (w - 3)`. Each optimizer runs for 100 steps at learning rate `0.1`. The script then opens a matplotlib window titled `Loss History` with one curve per optimizer.

## What it does

1. **SGD** updates `w = w - learning_rate * gradient`.
2. **Momentum** keeps a velocity. Each step sets `velocity = 0.9 * velocity + learning_rate * gradient`, then `w = w - velocity`. The learning rate sits inside the velocity, and `beta` is `0.9`.
3. **Adam** tracks a first moment `m` and a second moment `v`, with `beta1 = 0.9` and `beta2 = 0.999`. It bias-corrects both moments using the step number, then steps by `learning_rate * m_hat / (epsilon + sqrt(v_hat))`. `epsilon` is `1e-8`.
4. Records the loss after every update. Adam's `m` and `v` are stored as well. The figure plots only the three loss lists.

On this loss, with these settings, the three curves do not drop at the same rate. After the first step, SGD and momentum both sit at loss `5.76`. Adam's first step is smaller, so its loss is `8.41`. By step 10, momentum is already near the minimum. By step 100, SGD's loss is about `0`, momentum's is about `7.3e-5`, and Adam's is about `3.7e-4`.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day needs NumPy and matplotlib.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_013/main.py
```

The window stays open until you close it. Loss after selected steps:

| Step | SGD | Momentum | Adam |
| --- | --- | --- | --- |
| 1 | 5.76 | 5.76 | 8.41 |
| 10 | 0.104 | 0.000174 | 4.06 |
| 50 | 1.83e-9 | 0.00837 | 0.0285 |
| 100 | 3.73e-19 | 7.32e-5 | 0.000374 |
