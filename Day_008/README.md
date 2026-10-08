# Day 008 — Multi-head attention

`src/day_008/main.py` runs two attention heads in NumPy, concatenates them, and projects the result back to the model width. The arrays come from a fixed random seed, so the script takes no input and prints the same numbers every run.

The sequence has 3 tokens and a model width of 8. Two heads each use a width of 4 (`8 / 2`).

## What it does

1. Draws the token matrix `X` with shape `(3, 8)`, using `np.random.seed(42)`.
2. Draws a query, key, and value projection for each head, each of shape `(8, 4)`, plus an output projection `W_o` of shape `(8, 8)`.
3. Projects `X` into `Q`, `K`, and `V` for head 1 and head 2.
4. Runs scaled dot-product attention in each head. Scores are `Q @ K.T / sqrt(4)`. Softmax subtracts the row max before the exponential so the weights stay finite. Each head returns an output of shape `(3, 4)` and a weight matrix of shape `(3, 3)`.
5. Checks that each head output has shape `(3, 4)` and that each weight row sums to 1.
6. Concatenates the two head outputs along the feature axis, producing a `(3, 8)` matrix.
7. Projects that matrix with `W_o`. The result `O` has shape `(3, 8)`, the same shape as `X`.

The two heads do not share projections, so their weight matrices differ. Head 1's first token attends mostly to the second token. Head 2's first token attends mostly to the third.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. This day only needs NumPy.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_008/main.py
```

Example:

```text
X:
 [[ 0.49671415 -0.1382643   0.64768854  1.52302986 -0.23415337 -0.23413696
   1.57921282  0.76743473]
 [-0.46947439  0.54256004 -0.46341769 -0.46572975  0.24196227 -1.91328024
  -1.72491783 -0.56228753]
 [-1.01283112  0.31424733 -0.90802408 -1.4123037   1.46564877 -0.2257763
   0.0675282  -1.42474819]]
head_1_output:
 [[-0.75760956 -1.81809774  2.6842234  -1.05241394]
 [ 1.80473337  2.17408302  1.81446473 -0.88405958]
 [ 1.79400759  2.16905256  1.81287276 -0.88217804]]
weights_1:
 [[2.88984994e-01 7.08252246e-01 2.76275965e-03]
 [1.26489621e-05 9.42888184e-05 9.99893062e-01]
 [2.21149227e-03 1.16477098e-03 9.96623737e-01]]
head_2_output:
 [[ 0.54511645  0.10552996  1.01033498  1.87872479]
 [ 0.58450918  6.63355735  0.77614009  0.35674765]
 [ 1.25451624 -5.72548718 -2.09831133 -2.25519259]]
weights_2:
 [[2.61557277e-02 2.53299719e-02 9.48514300e-01]
 [9.97520886e-01 2.40042428e-03 7.86897748e-05]
 [6.12418921e-09 9.98656002e-01 1.34399146e-03]]
attention:
 [[-0.75760956 -1.81809774  2.6842234  -1.05241394  0.54511645  0.10552996
   1.01033498  1.87872479]
 [ 1.80473337  2.17408302  1.81446473 -0.88405958  0.58450918  6.63355735
   0.77614009  0.35674765]
 [ 1.79400759  2.16905256  1.81287276 -0.88217804  1.25451624 -5.72548718
  -2.09831133 -2.25519259]]
O:
 [[  5.42606427  -4.52406528   4.63902322   2.69319628  -3.6374525
    0.59258318  -0.05155965   1.25729288]
 [  8.01754117  -4.56325183   5.83140275   2.91944133  -5.17015313
   -1.16241247 -22.37595604  -8.64589434]
 [ -8.76042317   8.0120548   -2.62480794  -3.13027303   8.43237332
    1.47262922  12.33630712   8.83747239]]
```

The script prints `O` a second time. That second matrix matches the one above.
