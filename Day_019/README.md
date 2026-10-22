# Day 019 — Train, validation, and test

`src/day_019/main.py` fits a small network to a noisy sine wave and keeps the weights with the lowest validation loss. The inputs are 150 evenly spaced values from `0` to `2π`. Targets are `sin(x)` plus Gaussian noise with standard deviation `0.2`, drawn with `np.random.seed(42)`.

The indices are shuffled, then split 70% train, 15% validation, and 15% test. That is 105 training points, 22 validation points, and 23 test points. Each array is a `float32` tensor of shape `(N, 1)`.

## What it does

1. Builds a network with two hidden layers of 32 units, each followed by ReLU, and a single output. `torch.manual_seed(42)` fixes the starting weights.
2. Trains for 300 epochs with Adam at learning rate `0.01` and mean squared error. Every epoch uses the full training set.
3. After each update, measures validation loss. Whenever that loss is the lowest so far, the script saves `best_sine_model.pth`.
4. Prints training and validation loss every 50 epochs.
5. Loads the saved weights into a new model and reports test loss.

The best validation loss is about `0.0376` at epoch 281. The test loss of those weights is about `0.0495`.

| Epoch | Training loss | Validation loss |
| --- | --- | --- |
| 50 | 0.119804 | 0.087430 |
| 100 | 0.091086 | 0.075457 |
| 150 | 0.077948 | 0.064784 |
| 200 | 0.065914 | 0.055253 |
| 250 | 0.044887 | 0.041745 |
| 300 | 0.035412 | 0.042616 |

```text
Best Epoch: 281
Best Validation Loss: 0.037592
Test Loss: 0.049496
```

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day needs NumPy, PyTorch, and matplotlib.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_019/main.py
```

The checkpoint `best_sine_model.pth` is written in the folder where you run the script.
