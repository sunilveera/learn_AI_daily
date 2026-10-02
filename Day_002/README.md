# Day 002 — Sentence embeddings and similarity

`src/day_002/main.py` turns two sentences into vectors with a sentence-transformer model, then scores how close those vectors are.

## What it does

1. Loads `SentenceTransformer` for `all-MiniLM-L6-v2` (downloads the model on first run).
2. Reads two lines of text from the terminal.
3. Encodes each line into an embedding.
4. Prints the embedding shape and the first 10 values of each vector.
5. Prints the cosine similarity of the two embeddings.

`all-MiniLM-L6-v2` produces 384-dimensional vectors. Cosine similarity is the dot product of the two vectors divided by the product of their lengths. Closer sentences score nearer to 1.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. `sentence-transformers` brings in the model code and PyTorch.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_002/main.py
```

Example prompts:

```text
Enter the first text: The cat sat on the mat
Enter the second text: A kitten is sitting on a rug
```

The script then prints `embedding shape: (384,)`, the first 10 values of each embedding, and a similarity score for that pair.
