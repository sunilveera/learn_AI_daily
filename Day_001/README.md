# Day 001 — BERT tokenizer walkthrough

`src/day_001/main.py` loads the `bert-base-uncased` tokenizer and walks one line of text through the usual Hugging Face steps: split into tokens, map those tokens to vocabulary IDs, encode the full sentence as a PyTorch tensor (including special tokens), decode that tensor back to text, then turn the IDs back into tokens.

## What it does

1. Loads `AutoTokenizer` for `bert-base-uncased` (downloads the tokenizer files on first run).
2. Reads a line of text from the terminal.
3. Prints WordPiece tokens from `tokenizer.tokenize`.
4. Prints vocabulary IDs from `tokenizer.convert_tokens_to_ids`.
5. Prints the model-ready tensor from `tokenizer.encode` (`return_tensors="pt"`). This adds `[CLS]` and `[SEP]`.
6. Prints the decoded string from `tokenizer.decode`.
7. Prints tokens again from `tokenizer.convert_ids_to_tokens`.

`tokenize` does not add special tokens. `encode` does, so the decoded line is wrapped as `[CLS] ... [SEP]`.

## Setup

Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml` and are locked in `uv.lock`. The `torch` extra of `transformers` installs PyTorch, which `encode` needs to return a tensor.

From this folder:

```bash
uv sync
```

## Run

```bash
uv run python src/day_001/main.py
```

Example:

```text
Enter a text: Hello, world!
['hello', ',', 'world', '!']
[7592, 1010, 2088, 999]
tensor([[ 101, 7592, 1010, 2088, 999, 102]])
[CLS] hello, world! [SEP]
['hello', ',', 'world', '!']
```

`101` is `[CLS]` and `102` is `[SEP]`. Those IDs appear only in the encoded tensor.
